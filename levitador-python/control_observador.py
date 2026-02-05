"""
Control del Levitador con Observador de Posición basado en Corriente
=====================================================================

Este script implementa un observador de posición que estima la posición
de la esfera a partir de la corriente medida, usando el modelo de inductancia:

    L(y) = k0 + k/(1 + y/a)

Donde:
    - k0: inductancia base [H]
    - k: variación de inductancia [H]
    - a: parámetro de forma [m]

La relación entre corriente, voltaje e inductancia es:
    u = R*i + L(y)*di/dt + dL/dy * dy/dt * i

Integrando el flujo magnético:
    φ = ∫(u - R*i)dt = L(y)*i

Por lo tanto:
    y = a * (k*i / (φ - k0*i) - 1)

Autor: Jesús (Doctorado UAQ)
Basado en: observador.m de Valentín
"""

import time
import math
from serial_win32 import Win32Serial
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from pathlib import Path

# Importar modelo de resistencia dinámica
from modelo_resistencia import EstimadorResistencia

# Para KAN-PINN
import torch
import sys
sys.path.append(str(Path(__file__).parent / 'pinn'))

# =============================================================================
# PARÁMETROS DEL MODELO FÍSICO (del benchmark/observador.m)
# =============================================================================
# Parámetros del modelo magnético (optimizados offline con datos del C control)
K0 = 0.0657  # Inductancia base [H]
K = 0.0393   # Constante magnética [H]
A = 0.00498  # Constante geométrica [m]
R = 14.77    # Resistencia efectiva (escala relativa) [Ohm]
M = 0.018     # kg - Masa de la esfera

# =============================================================================
# PARÁMETROS DE CONTROL PID
# =============================================================================
Ts = 0.01
kp = 100
ki = 50
kd = 1.5
kpi = 12.0
kii = 3000.0
Vref = 9.86
Iref = 0.827
Rs = 2.2  # Resistencia de sensado (Valor del modelo magnético)

USE_OBSERVER_FOR_CONTROL = False  # MODO SENSORLESS: Controlar con el Observador

# =============================================================================
# OBSERVADOR KAN-PINN
# =============================================================================

class ObservadorKAN:
    """
    Observador KAN-PINN usando PyTorch directamente (más preciso).
    Optimizado con tensor pre-alocado para reducir overhead.
    """
    
    def __init__(self, model_path=None):
        self.phi = 0.0
        self.y_prev = 0.005
        self.dt = Ts
        self.model = None
        self.loaded = False
        
        # Tensor pre-alocado para entrada (evita crear tensor cada vez)
        self.X = torch.zeros(1, 3, dtype=torch.float32)
        
        if model_path is None:
            pinn_dir = Path(__file__).parent / 'pinn'
            models = list(pinn_dir.glob('kan_observador_*.pt'))
            if models:
                model_path = max(models, key=lambda p: p.stat().st_mtime)
        
        if model_path and Path(model_path).exists():
            self._cargar_modelo(model_path)
    
    def _cargar_modelo(self, path):
        """Carga el modelo KAN de PyTorch."""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent / 'pinn'))
            from kan_observador import KANObservador
            
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            self.model = KANObservador(hidden=32, depth=2, num_knots=8)
            self.model.load_state_dict(ckpt['model_state'])
            self.model.eval()
            
            # Desactivar gradientes para inferencia rápida
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.loaded = True
            mae = ckpt.get('metrics', {}).get('mae', 0) * 1000
            corr = ckpt.get('metrics', {}).get('corr', 0)
            print(f"🧠 KAN PyTorch cargado: MAE={mae:.2f}mm, Corr={corr:.3f}")
            
        except Exception as e:
            print(f"⚠️ Error cargando KAN: {e}")
            self.loaded = False
    
    def estimar(self, i, u):
        """Estima posición usando KAN. Retorna (y, dy/dt)."""
        # Integrar flujo
        dphi = (u - R * i) * self.dt
        self.phi += dphi
        
        # Calcular L = φ/i
        if i > 0.05:
            L_est = self.phi / i
            if L_est < 0.05 or L_est > 0.15:
                self.phi = 0.08 * i
                L_est = 0.08
        else:
            L_est = 0.08
        
        # Estimar posición
        if self.loaded and self.model is not None:
            # Usar tensor pre-alocado
            self.X[0, 0] = i
            self.X[0, 1] = L_est
            self.X[0, 2] = u
            
            with torch.no_grad():
                y_est = self.model(self.X).item()
        else:
            # Fallback a fórmula directa
            denom = L_est - K0
            if denom > 0.001:
                y_est = A * (K / denom - 1)
                y_est = max(0.001, min(y_est, 0.020))
            else:
                y_est = 0.005
        
        # Velocidad
        dy_est = (y_est - self.y_prev) / self.dt
        self.y_prev = y_est
        
        return y_est, dy_est
    
    def reset(self):
        self.phi = 0.0
        self.y_prev = 0.005


# =============================================================================
# OBSERVADOR CLÁSICO (respaldo)
# =============================================================================

class ObservadorPosicion:
    """
    Observador basado en integración del flujo: φ = ∫(u - R*i)dt
    Luego L = φ/i y se despeja y desde L(y).
    
    COPIADO DE control_pid_observadores.py (que SÍ funciona)
    + EstimadorResistencia agregado
    """
    
    def __init__(self, k0=K0, k=K, a=A, dt=Ts):
        self.k0 = k0
        self.k = k
        self.a = a
        self.dt = dt
        
        # Estimador de Resistencia Dinámica
        self.estimador_R = EstimadorResistencia(
            R0=16.0,            # Valor del análisis C
            alpha=0.0,          # Adaptativo puro
            beta=0.02,          # Dinámica lenta
            dt=dt, 
            gain_fusion=0.1
        )
        self.estimador_R.i_min_correction = 0.05
        
        # Estado (igual que ObservadorClasico)
        self.phi = 0.0
        self.i_prev = 0.0
        self.u_prev = 0.0
        self.y_est = 0.005
        self.y_est_prev = 0.005
        self.v_est = 0.0
        self.L_est = k0 + k
        self.initialized = False
        self.alpha = 0.25
        self.y_buffer = []
        self.buffer_size = 5
    
    def inductancia(self, y):
        """L(y) = k0 + k/(1 + y/a)"""
        y = max(0.0001, y)
        return self.k0 + self.k / (1 + y / self.a)
    
    def dL_dy(self, y):
        """Derivada de L respecto a y: dL/dy = -k/(a*(1+y/a)^2)"""
        y = max(0.0001, y)
        return -self.k / (self.a * (1 + y / self.a)**2)
    
    def posicion_desde_L(self, L):
        """Despeja y de L = k0 + k/(1 + y/a)"""
        denom = L - self.k0
        if denom > 0.0001:
            y = self.a * (self.k / denom - 1)
            return max(0.001, min(y, 0.020))
        return self.y_est
    
    def estimar(self, i, u):
        """
        Estima posición usando método híbrido:
        1. Integración del flujo (para dinámica rápida)
        2. Modelo de equilibrio (para corrección de drift)
        
        En equilibrio: F_mag = m*g
        F_mag = (1/2) * dL/dy * i^2 = (1/2) * (-k/(a*(1+y/a)^2)) * i^2
        
        Simplificado: y ≈ f(i) cuando el sistema está cerca del equilibrio
        """
        self.sample_count += 1
        
        # Actualizar resistencia dinámica (Thermal Memory)
        self.r = self.estimador_R.update(u, i)
        
        # === INICIALIZACIÓN ===
        if not self.initialized:
            if i > 0.01:
                self.initialized = True
                # Estimar posición inicial desde corriente de equilibrio
                y_init = self._estimar_desde_equilibrio(i)
                self.y_est = y_init if y_init else 0.005
                L0 = self.inductancia(self.y_est)
                self.phi = L0 * i
                print(f"📐 Observador inicializado: y={self.y_est*1000:.2f}mm, i={i:.3f}A")
            self.i_prev = i
            self.u_prev = u
            return self.y_est, self.dy_est
        
        # === SEGURIDAD: MODO CAÍDA (Low Current) ===
        # Si la corriente es casi nula, no hay fuerza magnética -> la bola cae por gravedad.
        # Esto evita que el observador se congele en posiciones "altas" (ej. 1mm) cuando u=0.
        if i < 0.05:
            # Simular caída simple hacia 22mm
            self.y_est += 0.05 * self.dt  # Caer a 5 cm/s (ajustable)
            self.y_est = min(self.y_est, 0.022) # Tope físico abajo
            self.phi = self.inductancia(self.y_est) * i # Resetear flujo consistente
            
            # Retornar estimación de caída
            self.dy_est = 0.05
            self.i_prev = i
            self.u_prev = u
            return self.y_est, self.dy_est

        # === MÉTODO 1: INTEGRACIÓN DEL FLUJO ===
        dphi_now = u - self.r * i
        dphi_prev = self.u_prev - self.r * self.i_prev
        dphi = 0.5 * (dphi_now + dphi_prev) * self.dt
        self.phi += dphi
        
        y_flujo = None
        if i > 0.05:
            self.L_est = self.phi / i
            L_min = self.k0 * 0.5
            L_max = (self.k0 + self.k) * 2.0
            
            if L_min < self.L_est < L_max:
                y_flujo = self.posicion_desde_L(self.L_est)
                self.valid_samples += 1
        
        # === MÉTODO 2: MODELO DE EQUILIBRIO ===
        y_equilibrio = self._estimar_desde_equilibrio(i)
        
        # === FUSIÓN DE ESTIMACIONES ===
        y_nuevo = None
        
        if y_flujo is not None and y_equilibrio is not None:
            # Ponderar: más peso al equilibrio cuando i es estable
            di = abs(i - self.i_prev) / self.dt
            peso_equilibrio = 1.0 / (1.0 + di * 100)  # Más peso si di/dt pequeño
            y_nuevo = peso_equilibrio * y_equilibrio + (1 - peso_equilibrio) * y_flujo
        elif y_equilibrio is not None:
            y_nuevo = y_equilibrio
        elif y_flujo is not None:
            y_nuevo = y_flujo
        
        # === ACTUALIZAR ESTIMACIÓN ===
        if y_nuevo is not None:
            self.y_buffer.append(y_nuevo)
            if len(self.y_buffer) > self.buffer_size:
                self.y_buffer.pop(0)
            
            y_promedio = sum(self.y_buffer) / len(self.y_buffer)
            error = abs(y_promedio - self.y_est)
            alpha_adapt = min(0.4, self.alpha + error * 5)
            self.y_est = alpha_adapt * y_promedio + (1 - alpha_adapt) * self.y_est
            
            # Reset suave del integrador para evitar drift
            L_esperada = self.inductancia(self.y_est)
            self.phi = 0.9 * self.phi + 0.1 * L_esperada * i
        
        # === VELOCIDAD ===
        dy_raw = (self.y_est - self.y_est_prev) / self.dt
        self.dy_est = self.alpha_vel * dy_raw + (1 - self.alpha_vel) * self.dy_est
        self.y_est_prev = self.y_est
        
        self.i_prev = i
        self.u_prev = u
        
        return self.y_est, self.dy_est
    
    def _estimar_desde_equilibrio(self, i):
        """
        Estima y desde el modelo de equilibrio:
        En equilibrio: F_mag = m*g
        F_mag = (k * i^2) / (2*a*(1+y/a)^2)
        
        Despejando y (aproximación):
        y ≈ a * (sqrt(k*i^2 / (2*a*m*g)) - 1)
        """
        if i < 0.05:
            return None
        
        # Constantes
        m = 0.018  # kg (masa esfera)
        g = 9.81   # m/s^2
        
        # F_mag = m*g en equilibrio
        # (k * i^2) / (2*a*(1+y/a)^2) = m*g
        # (1+y/a)^2 = k*i^2 / (2*a*m*g)
        # 1+y/a = sqrt(k*i^2 / (2*a*m*g))
        
        try:
            termino = self.k * i**2 / (2 * self.a * m * g)
            if termino > 0:
                raiz = termino ** 0.5
                if raiz > 1:
                    y = self.a * (raiz - 1)
                    if 0.001 < y < 0.020:
                        return y
        except:
            pass
        
        return None
    
    def get_diagnostics(self):
        """Retorna información de diagnóstico."""
        return {
            'phi': self.phi,
            'L_est': self.L_est,
            'resets': self.phi_reset_count,
            'valid_samples': self.valid_samples,
            'total_samples': self.sample_count
        }
    
    def reset(self):
        """Reinicia el observador."""
        self.phi = 0.0
        self.y_est = 0.005
        self.dy_est = 0.0
        self.y_buffer = []
        self.initialized = False
        self.sample_count = 0
        self.valid_samples = 0


# =============================================================================
# CONTROL CON OBSERVADOR
# =============================================================================

data_queue = queue.Queue()
exit_event = threading.Event()


def control_thread_observador(port='COM1', baudrate=115200):
    """Hilo de control usando el observador de posición."""
    
    # Variables de control
    pv, i, y, y_1, ef, ef_1, u, t = 0, 0, 0, 0, 0, 0, 0, 0
    esc = 0.05 / 1023.0
    esci = 5.0 / (Rs * 1023.0)
    escs = 254.0 / Vref
    iTs = 1 / Ts
    
    # Variables PID (como en control_pid_observadores.py)
    t = 0
    yd = 0.005
    ef, ef_1, y_prev = 0, 0, 0
    integral, intei = 0, 0
    u = 0
    
    # Usar observador Clásico (Física explícita) porque KAN está fallando
    # observador = ObservadorKAN()
    # if not observador.loaded:
    #     print("⚠️ KAN no cargó, usando observador clásico")
    #     observador = ObservadorPosicion()
    print("⚡ Usando Observador Clásico (Física explícita)")
    observador = ObservadorPosicion()
    
    ser = None
    try:
        ser = Win32Serial(port, baudrate)
        print(f"✅ Puerto {port} abierto para control con observador.")
        
        # Esperar switch
        print("⏳ Esperando switch del microcontrolador (byte 0xAA)...")
        while not exit_event.is_set():
            recibido = ser.read(1)
            if recibido == b'\xAA':
                print("✅ ¡Switch activado! Iniciando control con observador...")
                break
        else:
            print("❌ Cierre solicitado antes de la activación.")
            return
        
        # Archivo de salida
        with open("MONIT_observador.txt", "w+") as fp:
            fp.write("# t\tyd\ty_sensor\ty_obs\tdy_obs\tie\tu\tR_est\n")
            
            flagcom = 0
            pv = 0
            i = 0
            
            while not exit_event.is_set():
                b = ser.read(1)
                if not b:
                    continue
                
                recibido = b[0]
                
                if flagcom != 0:
                    flagcom += 1
                
                if (recibido == 0xAA) and (flagcom == 0):
                    pv = 0
                    i = 0
                    flagcom = 1
                    continue
                
                if flagcom == 2:
                    pv = recibido << 8
                    continue
                
                if flagcom == 3:
                    pv = pv + recibido
                    continue
                
                if flagcom == 4:
                    i = recibido << 8
                    continue
                
                if flagcom == 5:
                    i = i + recibido
                    
                    if pv > 1023 or i > 1023:
                        flagcom = 0
                        continue
                    
                    # Referencia constante (evitar pegado)
                    yd = 0.005  # 5mm constante
                    
                    # Posición del sensor (para comparación)
                    y_sensor = esc * pv
                    
                    # Corriente medida
                    ie = esci * i
                    
                    # === OBSERVADOR ===
                    # Estimar posición a partir de corriente y voltaje
                    y_obs, dy_obs = observador.estimar(ie, u)
                    
                    # Seleccionar realimentación para el control
                    y = y_obs if USE_OBSERVER_FOR_CONTROL else y_sensor
                    
                    # === CONTROL PID (Simple, como en control_pid_observadores.py) ===
                    ef_1 = ef
                    y_1 = y_prev
                    
                    ef = yd - y
                    proporcional = kp * ef
                    derivativa = kd * (ef - ef_1) * iTs
                    
                    if -Iref < integral < Iref:
                        integral += ki * Ts * ef
                    else:
                        integral = 0.95 * Iref if integral >= Iref else -0.95 * Iref
                    
                    id_val = proporcional + integral + derivativa
                    if id_val > 0:
                        id_val = 0
                    if id_val <= -Iref:
                        id_val = -Iref
                    
                    y_prev = y

                    # === LAZO DE CORRIENTE ===
                    ied = -id_val
                    ei = ied - ie
                    propi = kpi * ei
                    
                    if -Vref < intei < Vref:
                        intei += kii * Ts * ei
                    else:
                        intei = 0.95 * Vref if intei >= Vref else -0.95 * Vref
                    
                    u = propi + intei
                    if u > Vref:
                        u = Vref
                    if u <= 0:
                        u = 0
                    
                    pwmf = escs * u
                    pwm = int(abs(pwmf))
                    ser.write(bytes([pwm]))
                    
                    # Enviar datos a la gráfica
                    data_queue.put((t, y_sensor, y_obs, dy_obs, yd, u, ie, observador.r))
                    
                    # Guardar
                    fp.write(f"{t:.4f}\t{yd:.6f}\t{y_sensor:.6f}\t{y_obs:.6f}\t{dy_obs:.6f}\t{ie:.4f}\t{u:.4f}\t{observador.r:.4f}\n")
                    fp.flush()
                    
                    flagcom = 0
                    t += Ts
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ser and ser.is_open:
            ser.close()
        print("🔌 Hilo de control finalizado.")


def main():
    """Función principal sin GUI para mayor robustez."""
    
    print("="*60)
    print("🔬 CONTROL CON OBSERVADOR DE POSICIÓN (MODO CONSOLA)")
    print("="*60)
    print(f"Parámetros del modelo:")
    print(f"  k0 = {K0:.4f} H (inductancia base)")
    print(f"  k  = {K:.4f} H (variación)")
    print(f"  a  = {A:.4f} m (parámetro de forma)")
    print(f"  R  = {R:.2f} Ω (resistencia)")
    print(f"  Control: {'OBSERVADOR' if USE_OBSERVER_FOR_CONTROL else 'SENSOR'}")
    print(f"  PID: BASE (simple)")
    print("="*60)
    
    # Iniciar hilo de control
    control = threading.Thread(target=control_thread_observador, daemon=True)
    control.start()
    
    try:
        # Correr por 30 segundos
        duration = 30
        start_time = time.time()
        print(f"🚀 Corriendo prueba por {duration} segundos...")
        
        while time.time() - start_time < duration:
            # Mostrar estado cada segundo
            if not data_queue.empty():
                # Vaciar cola para no saturar memoria
                while not data_queue.empty():
                    t_val, y_s, y_o, dy, yd, u_val, ie, r_val = data_queue.get()
                
                print(f"⏱️ t={t_val:.1f}s | Ref={yd*1000:.1f}mm | Sensor={y_s*1000:.1f}mm | Obs={y_o*1000:.1f}mm | R={r_val:.3f}Ω")
            
            time.sleep(0.5)
            
            # Verificar si el hilo sigue vivo
            if not control.is_alive():
                print("⚠️ Hilo de control murió prematuramente.")
                break
                
    except KeyboardInterrupt:
        print("\n⏹️  Detenido por el usuario")
    finally:
        exit_event.set()
        control.join(timeout=2)
        print("✅ Programa finalizado.")


if __name__ == '__main__':
    main()
