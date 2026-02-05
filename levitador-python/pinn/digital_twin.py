"""
Digital Twin del Levitador Magnético usando KAN-PINN.
Wrapper para usar el modelo entrenado en simulación y control.
"""
from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from typing import Tuple

from kan_model import KANLevitator


class LevitatorDigitalTwin:
    """
    Gemelo Digital del Levitador Magnético.
    
    Usa el modelo KAN-PINN entrenado para:
    1. Simular la dinámica del sistema
    2. Predecir respuesta a acciones de control
    3. Testear controladores antes de implementarlos en hardware real
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        """
        Args:
            checkpoint_path: Ruta al modelo entrenado (.pt)
            device: 'cpu' o 'cuda'
        """
        self.device = torch.device(device)
        
        # Cargar checkpoint
        print(f"Loading digital twin from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        # Recrear modelo
        arch = ckpt['architecture']
        stats = ckpt['stats']
        
        self.model = KANLevitator(
            y_mu=stats['y_mu'],
            y_std=stats['y_std'],
            i_mu=stats['i_mu'],
            i_std=stats['i_std'],
            hidden=arch['hidden'],
            depth=arch['depth'],
            num_knots=arch['num_knots']
        ).to(self.device)
        
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        
        # Parámetros físicos identificados
        self.k0 = ckpt['physical_params']['k0']
        self.k = ckpt['physical_params']['k']
        self.a = ckpt['physical_params']['a']
        
        # Parámetros del sistema físico
        self.m = 0.018  # kg
        self.g = 9.81   # m/s²
        self.R = 2.72   # Ω
        
        # Normalización (para desnormalizar tiempo)
        self.t_scale = 1.0  # Ajustar si usaste normalización de tiempo
        
        print(f"✅ Digital Twin loaded successfully")
        print(f"   Physical params: k0={self.k0:.6e}, k={self.k:.6e}, a={self.a:.6e}")
    
    def predict_state(self, t: float, u: float) -> Tuple[float, float, float]:
        """
        Predice el estado del sistema en un tiempo dado con un voltaje aplicado.
        
        Args:
            t: Tiempo [s]
            u: Voltaje de control [V]
        
        Returns:
            y: Posición del imán [m]
            i: Corriente [A]
            dydt: Velocidad [m/s]
        """
        # Normalizar entrada
        t_norm = torch.tensor([[t / self.t_scale]], dtype=torch.float32, device=self.device)
        t_norm.requires_grad_(True)
        
        with torch.enable_grad():
            # Predicción
            y, i = self.model(t_norm)
            
            # Calcular velocidad (dy/dt)
            dydt = torch.autograd.grad(
                outputs=y,
                inputs=t_norm,
                grad_outputs=torch.ones_like(y),
                create_graph=False
            )[0]
        
        return y.item(), i.item(), dydt.item() * self.t_scale
    
    def simulate_step(
        self, 
        y_current: float, 
        dydt_current: float, 
        u_control: float, 
        dt: float = 0.001
    ) -> Tuple[float, float, float]:
        """
        Simula un paso de tiempo usando la física identificada.
        Útil para control en lazo cerrado.
        
        Args:
            y_current: Posición actual [m]
            dydt_current: Velocidad actual [m/s]
            u_control: Voltaje de control aplicado [V]
            dt: Paso de tiempo [s]
        
        Returns:
            y_next: Nueva posición [m]
            dydt_next: Nueva velocidad [m/s]
            d2ydt2: Aceleración [m/s²]
        """
        # Calcular inductancia en la posición actual
        L = self.k0 + self.k / (1 + y_current / self.a)
        
        # Calcular fuerza magnética
        # F_mag = (u²/R) · (dL/dy)
        # donde dL/dy = -k/(a(1+y/a)²)
        dL_dy = -self.k / (self.a * (1 + y_current / self.a)**2)
        F_mag = (u_control**2 / self.R) * dL_dy
        
        # Ecuación de movimiento: m·d²y/dt² = m·g - F_mag
        d2ydt2 = self.g + F_mag / self.m
        
        # Integración (Euler simple)
        dydt_next = dydt_current + d2ydt2 * dt
        y_next = y_current + dydt_next * dt
        
        return y_next, dydt_next, d2ydt2
    
    def simulate_trajectory(
        self,
        t_span: np.ndarray,
        u_control: np.ndarray,
        y0: float = 0.01,
        dydt0: float = 0.0
    ) -> dict:
        """
        Simula una trayectoria completa del sistema.
        
        Args:
            t_span: Array de tiempos [s]
            u_control: Array de voltajes de control [V] (mismo tamaño que t_span)
            y0: Posición inicial [m]
            dydt0: Velocidad inicial [m/s]
        
        Returns:
            dict con arrays de: t, y, dydt, i, u
        """
        assert len(t_span) == len(u_control), "t_span y u_control deben tener el mismo tamaño"
        
        n_steps = len(t_span)
        dt = t_span[1] - t_span[0] if n_steps > 1 else 0.001
        
        # Arrays de salida
        y_traj = np.zeros(n_steps)
        dydt_traj = np.zeros(n_steps)
        i_traj = np.zeros(n_steps)
        d2ydt2_traj = np.zeros(n_steps)
        
        # Condiciones iniciales
        y_traj[0] = y0
        dydt_traj[0] = dydt0
        
        # Calcular corriente inicial
        L0 = self.k0 + self.k / (1 + y0 / self.a)
        i_traj[0] = u_control[0] / (self.R + L0)  # Aproximación DC
        
        # Simular
        for i in range(n_steps - 1):
            y_next, dydt_next, d2ydt2 = self.simulate_step(
                y_traj[i],
                dydt_traj[i],
                u_control[i],
                dt
            )
            
            y_traj[i + 1] = y_next
            dydt_traj[i + 1] = dydt_next
            d2ydt2_traj[i] = d2ydt2
            
            # Calcular corriente (aproximación)
            L = self.k0 + self.k / (1 + y_next / self.a)
            i_traj[i + 1] = u_control[i + 1] / (self.R + L)
        
        return {
            't': t_span,
            'y': y_traj,
            'dydt': dydt_traj,
            'd2ydt2': d2ydt2_traj,
            'i': i_traj,
            'u': u_control
        }
    
    def test_controller(
        self,
        controller_func,
        t_span: np.ndarray,
        y_setpoint: float = 0.01,
        y0: float = 0.005,
        dydt0: float = 0.0
    ) -> dict:
        """
        Prueba un controlador en lazo cerrado usando el gemelo digital.
        
        Args:
            controller_func: Función que recibe (y_error, dydt, t) y devuelve u
            t_span: Array de tiempos [s]
            y_setpoint: Posición deseada [m]
            y0: Posición inicial [m]
            dydt0: Velocidad inicial [m/s]
        
        Returns:
            dict con trayectorias simuladas
        """
        n_steps = len(t_span)
        dt = t_span[1] - t_span[0] if n_steps > 1 else 0.001
        
        # Arrays
        y_traj = np.zeros(n_steps)
        dydt_traj = np.zeros(n_steps)
        u_traj = np.zeros(n_steps)
        error_traj = np.zeros(n_steps)
        
        # Inicializar
        y_traj[0] = y0
        dydt_traj[0] = dydt0
        error_traj[0] = y_setpoint - y0
        u_traj[0] = controller_func(error_traj[0], dydt_traj[0], t_span[0])
        
        # Simular con controlador
        for i in range(n_steps - 1):
            # Aplicar control
            y_next, dydt_next, _ = self.simulate_step(
                y_traj[i],
                dydt_traj[i],
                u_traj[i],
                dt
            )
            
            y_traj[i + 1] = y_next
            dydt_traj[i + 1] = dydt_next
            error_traj[i + 1] = y_setpoint - y_next
            
            # Calcular nuevo control
            u_traj[i + 1] = controller_func(error_traj[i + 1], dydt_next, t_span[i + 1])
        
        return {
            't': t_span,
            'y': y_traj,
            'dydt': dydt_traj,
            'u': u_traj,
            'error': error_traj,
            'setpoint': np.full(n_steps, y_setpoint)
        }
    
    def get_linearization(self, y_eq: float, u_eq: float) -> dict:
        """
        Obtiene la linealización del sistema alrededor de un punto de equilibrio.
        Útil para diseño de controladores lineales (PID, LQR).
        
        Args:
            y_eq: Posición de equilibrio [m]
            u_eq: Voltaje de equilibrio [V]
        
        Returns:
            dict con matrices A, B del sistema linealizado: dx/dt = Ax + Bu
        """
        # Estado: x = [y, dy/dt]
        # Entrada: u = voltaje
        
        # Calcular derivadas parciales
        eps = 1e-6
        
        # df/dy
        _, dydt_plus, f_plus = self.simulate_step(y_eq + eps, 0, u_eq, 0.001)
        _, dydt_minus, f_minus = self.simulate_step(y_eq - eps, 0, u_eq, 0.001)
        df_dy = (f_plus - f_minus) / (2 * eps)
        
        # df/du
        _, _, f_u_plus = self.simulate_step(y_eq, 0, u_eq + eps, 0.001)
        _, _, f_u_minus = self.simulate_step(y_eq, 0, u_eq - eps, 0.001)
        df_du = (f_u_plus - f_u_minus) / (2 * eps)
        
        # Matrices del sistema linealizado
        A = np.array([
            [0, 1],
            [df_dy, 0]
        ])
        
        B = np.array([
            [0],
            [df_du]
        ])
        
        return {
            'A': A,
            'B': B,
            'equilibrium': {'y': y_eq, 'u': u_eq},
            'description': 'dx/dt = Ax + Bu, where x = [y, dy/dt]'
        }


# Ejemplo de uso
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Cargar gemelo digital
    twin = LevitatorDigitalTwin('runs_physics_safe/best_model_safe.pt')
    
    # Ejemplo 1: Simular con voltaje constante
    print("\n" + "="*80)
    print("EJEMPLO 1: Simulación con voltaje constante")
    print("="*80)
    
    t = np.linspace(0, 1.0, 1000)
    u = np.full_like(t, 5.0)  # 5V constante
    
    result = twin.simulate_trajectory(t, u, y0=0.008, dydt0=0.0)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(result['t'], result['y'] * 1000)
    plt.ylabel('Posición [mm]')
    plt.grid(True)
    plt.title('Simulación del Gemelo Digital')
    
    plt.subplot(3, 1, 2)
    plt.plot(result['t'], result['i'] * 1000)
    plt.ylabel('Corriente [mA]')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(result['t'], result['u'])
    plt.ylabel('Voltaje [V]')
    plt.xlabel('Tiempo [s]')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('runs_physics_safe/digital_twin_simulation.png', dpi=150)
    print("✅ Simulación guardada en: runs_physics_safe/digital_twin_simulation.png")
    
    # Ejemplo 2: Controlador PD simple
    print("\n" + "="*80)
    print("EJEMPLO 2: Test de controlador PD")
    print("="*80)
    
    # Definir controlador PD
    Kp = 500.0
    Kd = 50.0
    
    def pd_controller(error, dydt, t):
        u = Kp * error - Kd * dydt
        return np.clip(u, 0, 10)  # Saturación 0-10V
    
    t_control = np.linspace(0, 2.0, 2000)
    result_control = twin.test_controller(
        pd_controller,
        t_control,
        y_setpoint=0.010,
        y0=0.005
    )
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(result_control['t'], result_control['y'] * 1000, label='Posición')
    plt.plot(result_control['t'], result_control['setpoint'] * 1000, 
             'r--', label='Setpoint')
    plt.ylabel('Posición [mm]')
    plt.legend()
    plt.grid(True)
    plt.title('Test de Controlador PD en Gemelo Digital')
    
    plt.subplot(3, 1, 2)
    plt.plot(result_control['t'], result_control['error'] * 1000)
    plt.ylabel('Error [mm]')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(result_control['t'], result_control['u'])
    plt.ylabel('Voltaje [V]')
    plt.xlabel('Tiempo [s]')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('runs_physics_safe/digital_twin_control.png', dpi=150)
    print("✅ Test de control guardado en: runs_physics_safe/digital_twin_control.png")
    
    print("\n" + "="*80)
    print("DIGITAL TWIN READY FOR USE")
    print("="*80)
