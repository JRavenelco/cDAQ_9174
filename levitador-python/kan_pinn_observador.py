"""
KAN-PINN Observador de Posición para Levitador Magnético
=========================================================

Este script implementa un observador de posición usando:
- KAN (Kolmogorov-Arnold Networks): Para aproximar funciones no lineales
- PINN (Physics-Informed Neural Networks): Para respetar la física del sistema

Física del Levitador:
---------------------
1. Inductancia:     L(y) = k0 + k/(1 + y/a)
2. Flujo magnético: φ = L(y) * i
3. Ecuación eléctrica: u = R*i + dφ/dt = R*i + L*di/dt + dL/dy * dy/dt * i
4. Fuerza magnética: F_mag = (1/2) * dL/dy * i² = -k*i² / (2*a*(1+y/a)²)
5. Dinámica: m*ÿ = m*g - F_mag

El observador aprende: y = f(i, di/dt, u) respetando estas ecuaciones.

Autor: Jesús (Doctorado UAQ)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# =============================================================================
# PARÁMETROS FÍSICOS DEL LEVITADOR
# =============================================================================
K0 = 0.0704   # H - Inductancia base
K = 0.0327    # H - Variación de inductancia  
A = 0.0052    # m - Parámetro de forma
R = 2.72      # Ω - Resistencia
M = 0.018     # kg - Masa de la esfera
G = 9.81      # m/s² - Gravedad

# =============================================================================
# RED KAN (Kolmogorov-Arnold Network)
# =============================================================================

class KANLayer(nn.Module):
    """
    Capa KAN simplificada: Usa funciones base aprendibles.
    
    Implementa: y = Σ w_ij * φ_ij(x_j) + b_i
    donde φ_ij son funciones aprendibles (combinación de bases)
    """
    
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        # Peso lineal base
        self.linear = nn.Linear(in_features, out_features)
        
        # Funciones no lineales aprendibles (una por cada conexión)
        # Usamos una combinación de bases: tanh, sin, x², etc.
        self.n_bases = 4
        self.basis_weights = nn.Parameter(
            torch.randn(out_features, in_features, self.n_bases) * 0.1
        )
        
        # Escala para las funciones base
        self.scale = nn.Parameter(torch.ones(out_features, in_features) * 0.5)
    
    def basis_functions(self, x):
        """Calcula funciones base para cada entrada."""
        # x: [batch, in_features]
        # Retorna: [batch, in_features, n_bases]
        bases = torch.stack([
            torch.tanh(x),           # Base 1: tanh
            torch.sin(x * 3.14159),  # Base 2: sin
            x ** 2,                   # Base 3: cuadrática
            torch.sigmoid(x * 2) - 0.5  # Base 4: sigmoid centrada
        ], dim=-1)
        return bases
    
    def forward(self, x):
        # x: [batch, in_features]
        
        # Parte lineal
        linear_out = self.linear(x)
        
        # Parte no lineal (funciones aprendibles)
        bases = self.basis_functions(x)  # [batch, in_features, n_bases]
        
        # Combinar bases con pesos aprendibles
        # basis_weights: [out_features, in_features, n_bases]
        # bases: [batch, in_features, n_bases]
        nonlinear_out = torch.einsum('oin,bin->bo', self.basis_weights, bases)
        nonlinear_out = nonlinear_out * self.scale.mean(dim=1)
        
        return linear_out + nonlinear_out


class KAN(nn.Module):
    """Red KAN completa con múltiples capas."""
    
    def __init__(self, layers_dims, grid_size=5):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layers_dims) - 1):
            self.layers.append(
                KANLayer(layers_dims[i], layers_dims[i+1], grid_size=grid_size)
            )
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)  # Activación entre capas
        return x


# =============================================================================
# KAN-PINN OBSERVADOR
# =============================================================================

class KANPINNObservador(nn.Module):
    """
    Observador de posición basado en KAN-PINN.
    
    Entrada: [i, di/dt, u, du/dt] (o subconjunto)
    Salida: y (posición estimada)
    
    Pérdida física:
    1. Consistencia con modelo de inductancia
    2. Ecuación de equilibrio de fuerzas
    3. Límites físicos (y > 0, y < y_max)
    """
    
    def __init__(self, input_dim=4, hidden_dims=[32, 32], grid_size=5):
        super().__init__()
        
        # Red KAN principal
        layers = [input_dim] + hidden_dims + [1]
        self.kan = KAN(layers, grid_size=grid_size)
        
        # Parámetros físicos (pueden ser aprendibles)
        self.k0 = nn.Parameter(torch.tensor(K0), requires_grad=False)
        self.k = nn.Parameter(torch.tensor(K), requires_grad=False)
        self.a = nn.Parameter(torch.tensor(A), requires_grad=False)
        self.r = nn.Parameter(torch.tensor(R), requires_grad=False)
        self.m = nn.Parameter(torch.tensor(M), requires_grad=False)
        self.g = nn.Parameter(torch.tensor(G), requires_grad=False)
        
        # Escala de salida
        self.y_scale = nn.Parameter(torch.tensor(0.01))  # ~10mm típico
        self.y_offset = nn.Parameter(torch.tensor(0.005))  # ~5mm centro
    
    def forward(self, x):
        """
        x: [batch, input_dim] donde input_dim puede ser:
           - [i] (solo corriente)
           - [i, di/dt] (corriente y derivada)
           - [i, di/dt, u] (con voltaje)
           - [i, di/dt, u, du/dt] (completo)
        """
        y_raw = self.kan(x)
        
        # Transformar a rango físico [0.001, 0.020] m
        y = torch.sigmoid(y_raw) * 0.019 + 0.001
        
        return y
    
    def inductancia(self, y):
        """L(y) = k0 + k/(1 + y/a)"""
        return self.k0 + self.k / (1 + y / self.a)
    
    def dL_dy(self, y):
        """dL/dy = -k / (a * (1 + y/a)²)"""
        return -self.k / (self.a * (1 + y / self.a)**2)
    
    def fuerza_magnetica(self, y, i):
        """F_mag = (1/2) * |dL/dy| * i²"""
        return 0.5 * torch.abs(self.dL_dy(y)) * i**2
    
    def physics_loss(self, i, di_dt, u, y_pred):
        """
        Calcula pérdida basada en física del sistema.
        
        1. Equilibrio de fuerzas: F_mag ≈ m*g (cerca del equilibrio)
        2. Consistencia de inductancia
        """
        losses = {}
        
        # 1. Pérdida de equilibrio (cuando di/dt pequeño)
        F_mag = self.fuerza_magnetica(y_pred, i)
        F_grav = self.m * self.g
        
        # Máscara para estados cercanos al equilibrio
        equilibrio_mask = torch.abs(di_dt) < 0.5
        if equilibrio_mask.sum() > 0:
            loss_equilibrio = torch.mean((F_mag[equilibrio_mask] - F_grav)**2)
        else:
            loss_equilibrio = torch.tensor(0.0)
        losses['equilibrio'] = loss_equilibrio
        
        # 2. Pérdida de límites físicos (soft constraints)
        loss_limites = torch.mean(torch.relu(-y_pred + 0.001) + torch.relu(y_pred - 0.020))
        losses['limites'] = loss_limites * 100
        
        # 3. Suavidad (regularización de segunda derivada)
        if y_pred.shape[0] > 2:
            dy = y_pred[1:] - y_pred[:-1]
            ddy = dy[1:] - dy[:-1]
            loss_suavidad = torch.mean(ddy**2)
        else:
            loss_suavidad = torch.tensor(0.0)
        losses['suavidad'] = loss_suavidad * 1000
        
        return losses


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def cargar_datos(dataset_dir='datasets'):
    """Carga y preprocesa todos los datasets."""
    dataset_path = Path(dataset_dir)
    all_data = []
    
    for file in dataset_path.glob('dataset_*.txt'):
        print(f"Cargando {file.name}...")
        data = np.loadtxt(file, skiprows=9)  # Saltar header
        all_data.append(data)
    
    if not all_data:
        raise ValueError(f"No se encontraron datasets en {dataset_dir}")
    
    data = np.vstack(all_data)
    print(f"Total muestras: {len(data)}")
    
    # Columnas: t, y, y_obs, dy_obs, i, u, yd
    t = data[:, 0]
    y = data[:, 1]  # Posición del sensor (ground truth)
    i = data[:, 4]  # Corriente
    u = data[:, 5]  # Voltaje
    
    # Calcular derivadas de forma segura
    dt = 0.01  # Timestep fijo
    di_dt = np.zeros_like(i)
    du_dt = np.zeros_like(u)
    di_dt[1:] = (i[1:] - i[:-1]) / dt
    du_dt[1:] = (u[1:] - u[:-1]) / dt
    
    # Limpiar NaN e Inf
    di_dt = np.nan_to_num(di_dt, nan=0.0, posinf=0.0, neginf=0.0)
    du_dt = np.nan_to_num(du_dt, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Filtrar datos válidos
    mask = (y > 0.001) & (y < 0.018) & (i > 0.02) & (u > 0.1)
    mask &= np.isfinite(di_dt) & np.isfinite(du_dt)
    
    return {
        't': t[mask],
        'y': y[mask],
        'i': i[mask],
        'di_dt': di_dt[mask],
        'u': u[mask],
        'du_dt': du_dt[mask]
    }


def entrenar(epochs=500, batch_size=64, lr=0.001, physics_weight=0.1):
    """Entrena el KAN-PINN observador."""
    
    print("="*60)
    print("ENTRENAMIENTO KAN-PINN OBSERVADOR")
    print("="*60)
    
    # Cargar datos
    data = cargar_datos()
    n_samples = len(data['y'])
    print(f"Muestras válidas para entrenamiento: {n_samples}")
    
    if n_samples < 100:
        print("⚠️ Muy pocas muestras. Necesitas más datos.")
        return None
    
    # Preparar tensores
    X = torch.tensor(np.stack([
        data['i'],
        data['di_dt'],
        data['u'],
        data['du_dt']
    ], axis=1), dtype=torch.float32)
    
    y_true = torch.tensor(data['y'], dtype=torch.float32).unsqueeze(1)
    
    # Normalizar entrada
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True) + 1e-6
    X_norm = (X - X_mean) / X_std
    
    # Crear modelo
    model = KANPINNObservador(input_dim=4, hidden_dims=[32, 32], grid_size=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    
    # Entrenamiento
    history = {'loss': [], 'data_loss': [], 'physics_loss': []}
    best_loss = float('inf')
    best_model_state = model.state_dict().copy()  # Inicializar
    
    for epoch in range(epochs):
        model.train()
        
        # Mini-batch
        indices = torch.randperm(n_samples)[:batch_size]
        X_batch = X_norm[indices]
        y_batch = y_true[indices]
        i_batch = X[indices, 0]
        di_dt_batch = X[indices, 1]
        u_batch = X[indices, 2]
        
        # Forward
        y_pred = model(X_batch)
        
        # Pérdida de datos
        data_loss = torch.mean((y_pred - y_batch)**2)
        
        # Pérdida física
        physics_losses = model.physics_loss(i_batch, di_dt_batch, u_batch, y_pred.squeeze())
        physics_loss = sum(physics_losses.values())
        
        # Pérdida total
        total_loss = data_loss + physics_weight * physics_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(total_loss)
        
        # Registro
        history['loss'].append(total_loss.item())
        history['data_loss'].append(data_loss.item())
        history['physics_loss'].append(physics_loss.item())
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 50 == 0:
            # Evaluar en todo el dataset
            model.eval()
            with torch.no_grad():
                y_pred_all = model(X_norm)
                mse = torch.mean((y_pred_all - y_true)**2).item()
                mae = torch.mean(torch.abs(y_pred_all - y_true)).item()
                corr = np.corrcoef(y_pred_all.numpy().flatten(), y_true.numpy().flatten())[0,1]
            
            print(f"Epoch {epoch+1}/{epochs}: Loss={total_loss.item():.6f}, "
                  f"MAE={mae*1000:.2f}mm, Corr={corr:.3f}")
    
    # Cargar mejor modelo
    model.load_state_dict(best_model_state)
    
    # Evaluación final
    model.eval()
    with torch.no_grad():
        y_pred_all = model(X_norm)
        mse = torch.mean((y_pred_all - y_true)**2).item()
        mae = torch.mean(torch.abs(y_pred_all - y_true)).item()
        corr = np.corrcoef(y_pred_all.numpy().flatten(), y_true.numpy().flatten())[0,1]
    
    print(f"\n{'='*60}")
    print("RESULTADOS FINALES")
    print(f"{'='*60}")
    print(f"MSE: {mse:.8f}")
    print(f"MAE: {mae*1000:.2f} mm")
    print(f"Correlación: {corr:.4f}")
    
    # Guardar modelo
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"kan_pinn_observador_{timestamp}.pt"
    torch.save({
        'model_state': model.state_dict(),
        'X_mean': X_mean,
        'X_std': X_std,
        'history': history,
        'metrics': {'mse': mse, 'mae': mae, 'corr': corr}
    }, model_path)
    print(f"\n💾 Modelo guardado: {model_path}")
    
    # Graficar
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Pérdida
    axes[0, 0].plot(history['loss'], label='Total')
    axes[0, 0].plot(history['data_loss'], label='Datos')
    axes[0, 0].plot(history['physics_loss'], label='Física')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].set_title('Pérdidas durante entrenamiento')
    axes[0, 0].grid(True)
    
    # Predicción vs Real
    axes[0, 1].scatter(y_true.numpy()*1000, y_pred_all.numpy()*1000, alpha=0.3, s=1)
    axes[0, 1].plot([0, 20], [0, 20], 'r--', label='Ideal')
    axes[0, 1].set_xlabel('y real [mm]')
    axes[0, 1].set_ylabel('y predicho [mm]')
    axes[0, 1].set_title(f'Predicción vs Real (corr={corr:.3f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Serie temporal (primeras 500 muestras)
    n_plot = min(500, len(y_true))
    axes[1, 0].plot(data['t'][:n_plot], y_true[:n_plot].numpy()*1000, 'b-', label='Real', alpha=0.7)
    axes[1, 0].plot(data['t'][:n_plot], y_pred_all[:n_plot].numpy()*1000, 'r-', label='KAN-PINN', alpha=0.7)
    axes[1, 0].set_xlabel('Tiempo [s]')
    axes[1, 0].set_ylabel('Posición [mm]')
    axes[1, 0].set_title('Serie temporal')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Histograma de errores
    error = (y_pred_all - y_true).numpy() * 1000
    axes[1, 1].hist(error, bins=50, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Error [mm]')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title(f'Distribución del error (MAE={mae*1000:.2f}mm)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'kan_pinn_resultados_{timestamp}.png', dpi=150)
    print(f"📊 Gráfica guardada: kan_pinn_resultados_{timestamp}.png")
    plt.show()
    
    return model, X_mean, X_std, history


# =============================================================================
# CLASE PARA USO EN TIEMPO REAL
# =============================================================================

class KANPINNObservadorRT:
    """
    Wrapper para usar el KAN-PINN en tiempo real.
    """
    
    def __init__(self, model_path=None):
        self.model = KANPINNObservador(input_dim=4, hidden_dims=[32, 32])
        self.X_mean = torch.zeros(4)
        self.X_std = torch.ones(4)
        
        self.i_prev = 0.0
        self.u_prev = 0.0
        self.dt = 0.01
        
        if model_path and Path(model_path).exists():
            self.cargar(model_path)
    
    def cargar(self, model_path):
        """Carga modelo entrenado."""
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.X_mean = checkpoint['X_mean']
        self.X_std = checkpoint['X_std']
        self.model.eval()
        print(f"✅ Modelo cargado: {model_path}")
        print(f"   Métricas: MAE={checkpoint['metrics']['mae']*1000:.2f}mm, "
              f"Corr={checkpoint['metrics']['corr']:.3f}")
    
    def estimar(self, i, u):
        """Estima posición en tiempo real."""
        # Calcular derivadas
        di_dt = (i - self.i_prev) / self.dt
        du_dt = (u - self.u_prev) / self.dt
        
        # Preparar entrada
        X = torch.tensor([[i, di_dt, u, du_dt]], dtype=torch.float32)
        X_norm = (X - self.X_mean) / self.X_std
        
        # Predecir
        with torch.no_grad():
            y_pred = self.model(X_norm)
        
        # Actualizar estado
        self.i_prev = i
        self.u_prev = u
        
        return y_pred.item()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='KAN-PINN Observador')
    parser.add_argument('--entrenar', action='store_true', help='Entrenar modelo')
    parser.add_argument('--epochs', type=int, default=500, help='Épocas de entrenamiento')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--physics', type=float, default=0.1, help='Peso de pérdida física')
    args = parser.parse_args()
    
    if args.entrenar:
        entrenar(epochs=args.epochs, lr=args.lr, physics_weight=args.physics)
    else:
        print("Uso: python kan_pinn_observador.py --entrenar")
        print("     python kan_pinn_observador.py --entrenar --epochs 1000 --physics 0.2")
