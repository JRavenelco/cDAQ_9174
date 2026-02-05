"""
KAN-PINN Corrector para Observador Sensorless
==============================================

Estrategia híbrida:
1. Observador 2023 (física) → estimación base sensorless
2. KAN-PINN → aprende a corregir errores del observador

y_corregido = y_obs2023 + KAN(i, di/dt, u, y_obs2023)

Esto combina:
- Conocimiento físico del observador 2023
- Capacidad de aprendizaje del KAN-PINN para corregir errores

Autor: José de Jesús Santana Ramírez
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import math

# =============================================================================
# PARÁMETROS FÍSICOS
# =============================================================================
K0 = 0.0704
KG = 0.0327
A = 0.0052
R = 2.72
M = 0.018
G = 9.81
Ts = 0.01

# =============================================================================
# RED KAN COMPACTA (optimizada para tiempo real)
# =============================================================================

class KANLayerFast(nn.Module):
    """KAN layer optimizado para inferencia rápida."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.basis_weights = nn.Parameter(torch.randn(out_features, in_features, 3) * 0.1)
    
    def forward(self, x):
        # Parte lineal
        out = self.linear(x)
        
        # Bases no lineales (tanh, x², sin)
        bases = torch.stack([
            torch.tanh(x),
            x ** 2,
            torch.sin(x * 3.14159)
        ], dim=-1)
        
        # Combinar
        nonlin = torch.einsum('oin,bin->bo', self.basis_weights, bases)
        return out + nonlin * 0.1


class KANCorrector(nn.Module):
    """
    Red KAN para corregir estimaciones del observador sensorless.
    
    Entrada: [i, di/dt, u, y_obs] (4 features)
    Salida: Δy (corrección a aplicar)
    """
    
    def __init__(self):
        super().__init__()
        self.layer1 = KANLayerFast(4, 16)
        self.layer2 = KANLayerFast(16, 8)
        self.layer3 = nn.Linear(8, 1)
        
        # Escala de corrección (pequeña inicialmente)
        self.correction_scale = nn.Parameter(torch.tensor(0.001))
    
    def forward(self, x):
        h = torch.tanh(self.layer1(x))
        h = torch.tanh(self.layer2(h))
        delta_y = self.layer3(h)
        
        # Limitar corrección a ±5mm
        delta_y = torch.tanh(delta_y) * 0.005
        
        return delta_y


# =============================================================================
# OBSERVADOR 2023 SENSORLESS (versión simplificada)
# =============================================================================

class Observador2023Simple:
    """Observador sensorless basado en equilibrio de fuerzas."""
    
    def __init__(self, dt=Ts):
        self.dt = dt
        self.y_est = 0.005
        self.i_prev = 0.0
        self.phi = 0.0
        self.initialized = False
    
    def _L_from_y(self, y):
        y = max(0.0001, y)
        return K0 + KG / (1 + y / A)
    
    def _y_from_equilibrio(self, i):
        if i < 0.05:
            return None
        try:
            factor = KG * i**2 / (2 * A * M * G)
            if factor > 0:
                y = A * (math.sqrt(factor) - 1)
                return max(0.001, min(0.020, y))
        except:
            pass
        return None
    
    def estimar(self, i, u):
        if not self.initialized and i > 0.03:
            self.initialized = True
            self.y_est = self._y_from_equilibrio(i) or 0.005
            self.phi = self._L_from_y(self.y_est) * i
        
        if not self.initialized:
            self.i_prev = i
            return self.y_est
        
        # Integrar flujo
        z = u - R * i
        z_prev = getattr(self, 'z_prev', z)
        self.phi += 0.5 * (z + z_prev) * self.dt
        self.z_prev = z
        
        # Estimar desde equilibrio
        y_eq = self._y_from_equilibrio(i)
        
        # Estimar desde flujo
        y_flujo = None
        if i > 0.05:
            L_est = self.phi / i
            if 0.05 < L_est < 0.15:
                denom = L_est - K0
                if denom > 0.001:
                    y_flujo = A * (KG / denom - 1)
                    y_flujo = max(0.001, min(0.020, y_flujo))
        
        # Fusionar
        if y_eq is not None and y_flujo is not None:
            y_new = 0.7 * y_eq + 0.3 * y_flujo
        elif y_eq is not None:
            y_new = y_eq
        elif y_flujo is not None:
            y_new = y_flujo
        else:
            y_new = self.y_est
        
        # Filtrar
        self.y_est = 0.3 * y_new + 0.7 * self.y_est
        self.y_est = max(0.0005, min(0.022, self.y_est))
        
        # Corregir drift
        L_esp = self._L_from_y(self.y_est)
        self.phi = 0.9 * self.phi + 0.1 * L_esp * i
        
        self.i_prev = i
        return self.y_est


# =============================================================================
# ENTRENAMIENTO DEL CORRECTOR
# =============================================================================

def cargar_datos_con_observador(use_fresh=True):
    """Carga datos y calcula estimaciones del observador 2023."""
    
    all_t, all_y, all_i, all_u = [], [], [], []
    
    if use_fresh:
        # Usar datos de monitoreo recientes (más representativos)
        monit_files = [
            ('MONIT_pid_obs.txt', 2, 7, 8),        # t, yd, y_s, y_l, v_l, y_c, v_c, ie, u
            ('MONIT_observador.txt', 2, 7, 8),
            ('MONIT_kan_hibrido.txt', 1, 4, 5),   # t, y_s, y_kan, delta, ie, u
        ]
        
        for fname, y_col, i_col, u_col in monit_files:
            if Path(fname).exists():
                try:
                    data = np.loadtxt(fname)
                    print(f"Cargando {fname}: {len(data)} muestras")
                    all_t.append(data[:, 0])
                    all_y.append(data[:, y_col])
                    all_i.append(data[:, i_col])
                    all_u.append(data[:, u_col])
                except Exception as e:
                    print(f"  Error: {e}")
    
    # También cargar datasets si existen
    dataset_dir = Path('datasets')
    for file in dataset_dir.glob('dataset_*.txt'):
        try:
            data = np.loadtxt(file, skiprows=9)
            print(f"Cargando {file.name}: {len(data)} muestras")
            all_t.append(data[:, 0])
            all_y.append(data[:, 1])
            all_i.append(data[:, 4])
            all_u.append(data[:, 5])
        except:
            pass
    
    if not all_t:
        raise ValueError("No se encontraron datos")
    
    return {
        't': np.concatenate(all_t),
        'y_real': np.concatenate(all_y),
        'i': np.concatenate(all_i),
        'u': np.concatenate(all_u)
    }


def entrenar_corrector(epochs=300, batch_size=128, lr=0.002):
    """Entrena el KAN corrector."""
    
    print("="*60)
    print("ENTRENAMIENTO KAN CORRECTOR SENSORLESS")
    print("="*60)
    
    # Cargar datos
    data = cargar_datos_con_observador()
    
    # Asegurar que todos los arrays tengan el mismo tamaño
    n = min(len(data['y_real']), len(data['i']), len(data['u']))
    data['y_real'] = data['y_real'][:n]
    data['i'] = data['i'][:n]
    data['u'] = data['u'][:n]
    data['t'] = data['t'][:n]
    print(f"Muestras totales: {n}")
    
    # Simular observador 2023 sobre los datos
    print("Simulando observador 2023...")
    obs = Observador2023Simple()
    y_obs = np.zeros(n)
    
    for k in range(n):
        y_obs[k] = obs.estimar(float(data['i'][k]), float(data['u'][k]))
    
    # Calcular error del observador
    error_obs = data['y_real'] - y_obs
    print(f"Error observador 2023: MAE={np.abs(error_obs).mean()*1000:.2f}mm")
    
    # Calcular derivadas
    dt = 0.01
    di_dt = np.zeros(n)
    di_dt[1:] = (data['i'][1:] - data['i'][:-1]) / dt
    di_dt = np.clip(di_dt, -100, 100)
    
    # Filtrar datos válidos
    mask = (data['y_real'] > 0.001) & (data['y_real'] < 0.018) & (data['i'] > 0.02)
    
    # Preparar tensores
    X = torch.tensor(np.stack([
        data['i'][mask],
        di_dt[mask],
        data['u'][mask],
        y_obs[mask]
    ], axis=1), dtype=torch.float32)
    
    # Target: error que debe corregir
    y_error = torch.tensor(error_obs[mask], dtype=torch.float32).unsqueeze(1)
    
    n_train = len(y_error)
    print(f"Muestras para entrenamiento: {n_train}")
    
    # Normalizar
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True) + 1e-6
    X_norm = (X - X_mean) / X_std
    
    # Modelo
    model = KANCorrector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_loss = float('inf')
    best_state = model.state_dict().copy()
    
    for epoch in range(epochs):
        model.train()
        
        # Mini-batch
        idx = torch.randperm(n_train)[:batch_size]
        X_batch = X_norm[idx]
        y_batch = y_error[idx]
        
        # Forward
        delta_pred = model(X_batch)
        
        # Pérdida MSE
        loss = torch.mean((delta_pred - y_batch)**2)
        
        # Regularización de suavidad
        if delta_pred.shape[0] > 2:
            d2 = delta_pred[2:] - 2*delta_pred[1:-1] + delta_pred[:-2]
            loss += 0.01 * torch.mean(d2**2)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                delta_all = model(X_norm)
                y_corregido = torch.tensor(y_obs[mask], dtype=torch.float32).unsqueeze(1) + delta_all
                y_real_t = torch.tensor(data['y_real'][mask], dtype=torch.float32).unsqueeze(1)
                
                mae_antes = torch.mean(torch.abs(y_error)).item() * 1000
                mae_despues = torch.mean(torch.abs(y_corregido - y_real_t)).item() * 1000
                mejora = (mae_antes - mae_despues) / mae_antes * 100
                
            print(f"Epoch {epoch+1}: Loss={loss.item():.6f} | "
                  f"MAE: {mae_antes:.2f}→{mae_despues:.2f}mm ({mejora:+.1f}%)")
    
    # Cargar mejor modelo
    model.load_state_dict(best_state)
    
    # Evaluación final
    model.eval()
    with torch.no_grad():
        delta_all = model(X_norm)
        y_corregido = torch.tensor(y_obs[mask], dtype=torch.float32).unsqueeze(1) + delta_all
        y_real_t = torch.tensor(data['y_real'][mask], dtype=torch.float32).unsqueeze(1)
        
        mae_obs = np.abs(error_obs[mask]).mean() * 1000
        mae_kan = torch.mean(torch.abs(y_corregido - y_real_t)).item() * 1000
        corr_obs = np.corrcoef(y_obs[mask], data['y_real'][mask])[0,1]
        corr_kan = np.corrcoef(y_corregido.numpy().flatten(), data['y_real'][mask])[0,1]
    
    print(f"\n{'='*60}")
    print("RESULTADOS FINALES")
    print(f"{'='*60}")
    print(f"{'Método':<25} {'MAE [mm]':<12} {'Correlación':<12}")
    print(f"{'-'*49}")
    print(f"{'Observador 2023 (solo)':<25} {mae_obs:<12.2f} {corr_obs:<12.4f}")
    print(f"{'Obs2023 + KAN Corrector':<25} {mae_kan:<12.2f} {corr_kan:<12.4f}")
    print(f"{'-'*49}")
    mejora_final = (mae_obs - mae_kan) / mae_obs * 100
    print(f"{'Mejora':<25} {mejora_final:+.1f}%")
    
    # Guardar
    model_path = 'kan_corrector_sensorless.pt'
    torch.save({
        'model_state': model.state_dict(),
        'X_mean': X_mean,
        'X_std': X_std,
        'metrics': {
            'mae_obs': mae_obs,
            'mae_kan': mae_kan,
            'corr_obs': corr_obs,
            'corr_kan': corr_kan
        }
    }, model_path)
    print(f"\n💾 Modelo guardado: {model_path}")
    
    return model, X_mean, X_std


# =============================================================================
# OBSERVADOR HÍBRIDO (Obs2023 + KAN)
# =============================================================================

class ObservadorHibridoKAN:
    """
    Observador sensorless híbrido:
    1. Obs2023 → estimación física
    2. KAN → corrección aprendida
    
    y_final = y_obs2023 + KAN(i, di/dt, u, y_obs2023)
    """
    
    def __init__(self, model_path='kan_corrector_sensorless.pt'):
        self.obs = Observador2023Simple()
        self.model = KANCorrector()
        self.X_mean = torch.zeros(4)
        self.X_std = torch.ones(4)
        
        self.i_prev = 0.0
        self.dt = Ts
        self.model_loaded = False
        
        if Path(model_path).exists():
            self._cargar_modelo(model_path)
    
    def _cargar_modelo(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.X_mean = checkpoint['X_mean']
        self.X_std = checkpoint['X_std']
        self.model.eval()
        self.model_loaded = True
        print(f"✅ KAN Corrector cargado: MAE mejorado a {checkpoint['metrics']['mae_kan']:.2f}mm")
    
    def estimar(self, i, u):
        # Paso 1: Observador 2023 (física)
        y_obs = self.obs.estimar(i, u)
        
        if not self.model_loaded:
            self.i_prev = i
            return y_obs, 0.0
        
        # Paso 2: KAN corrector
        di_dt = (i - self.i_prev) / self.dt
        di_dt = max(-100, min(100, di_dt))
        
        X = torch.tensor([[i, di_dt, u, y_obs]], dtype=torch.float32)
        X_norm = (X - self.X_mean) / self.X_std
        
        with torch.no_grad():
            delta_y = self.model(X_norm).item()
        
        # Paso 3: Corrección
        y_final = y_obs + delta_y
        y_final = max(0.0005, min(0.022, y_final))
        
        self.i_prev = i
        return y_final, delta_y


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'entrenar':
        entrenar_corrector(epochs=300)
    else:
        print("Uso:")
        print("  python kan_corrector_sensorless.py entrenar")
        print("\nEsto entrena un KAN para corregir el observador 2023.")
