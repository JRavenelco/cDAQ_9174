"""
KAN-PINN Observador: Aprende i,u → y respetando física del levitador
=====================================================================

Usa el KANLayer existente para crear un observador que:
1. Entrada: [i, di/dt, u] (corriente, derivada, voltaje)
2. Salida: y (posición estimada)
3. Física: Respeta L(y) = k0 + k/(1+y/a) y equilibrio de fuerzas

Autor: Jesús (Doctorado UAQ)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Importar KANLayer del módulo existente
from kan_model import KANLayer, BSplineActivation

# Parámetros físicos
K0 = 0.0704   # H
K = 0.0327    # H
A = 0.0052    # m
R = 2.72      # Ω
M = 0.018     # kg
G = 9.81     # m/s²


class KANObservador(nn.Module):
    """
    Observador de posición basado en KAN con pérdida física.
    
    Entrada: [i, φ, u] donde φ = ∫(u - R*i)dt es el flujo integrado
    Salida: y (posición en metros)
    
    Física incorporada:
    - φ = L(y) * i, entonces L = φ/i
    - L(y) = k0 + k/(1 + y/a), despejando y = a*(k/(L-k0) - 1)
    - F_mag = 0.5 * |dL/dy| * i²
    """
    
    def __init__(self, hidden=32, depth=2, num_knots=8):
        super().__init__()
        
        # Capas KAN
        layers = []
        in_features = 3  # [i, φ, u]
        
        for _ in range(depth):
            layers.append(KANLayer(in_features, hidden, num_knots=num_knots))
            in_features = hidden
        
        # Capa final: hidden → 1 (posición)
        layers.append(KANLayer(in_features, 1, num_knots=num_knots))
        
        self.kan_layers = nn.ModuleList(layers)
        
        # Parámetros físicos (pueden ser entrenables)
        self.register_buffer('k0', torch.tensor(K0))
        self.register_buffer('k', torch.tensor(K))
        self.register_buffer('a', torch.tensor(A))
        self.register_buffer('m', torch.tensor(M))
        self.register_buffer('g', torch.tensor(G))
        
        # Normalización de entrada (se actualiza con datos)
        self.register_buffer('i_mean', torch.tensor(0.3))
        self.register_buffer('i_std', torch.tensor(0.2))
        self.register_buffer('phi_mean', torch.tensor(0.03))  # Flujo típico
        self.register_buffer('phi_std', torch.tensor(0.02))
        self.register_buffer('u_mean', torch.tensor(5.0))
        self.register_buffer('u_std', torch.tensor(3.0))
        
        # Escala de salida
        self.y_min = 0.001  # 1mm
        self.y_max = 0.018  # 18mm
    
    def forward(self, x):
        """
        x: [batch, 3] = [i, φ, u]
        Retorna: y [batch, 1] en metros
        """
        # Normalizar entrada
        x_norm = torch.zeros_like(x)
        x_norm[:, 0] = (x[:, 0] - self.i_mean) / (self.i_std + 1e-6)
        x_norm[:, 1] = (x[:, 1] - self.phi_mean) / (self.phi_std + 1e-6)
        x_norm[:, 2] = (x[:, 2] - self.u_mean) / (self.u_std + 1e-6)
        
        # Pasar por capas KAN
        h = x_norm
        for layer in self.kan_layers:
            h = layer(h)
        
        # Mapear a rango físico [y_min, y_max]
        y = torch.sigmoid(h) * (self.y_max - self.y_min) + self.y_min
        
        return y
    
    def inductancia(self, y):
        """L(y) = k0 + k/(1 + y/a)"""
        return self.k0 + self.k / (1 + y / self.a)
    
    def dL_dy(self, y):
        """dL/dy = -k / (a * (1 + y/a)²)"""
        return -self.k / (self.a * (1 + y / self.a)**2)
    
    def fuerza_magnetica(self, y, i):
        """F_mag = 0.5 * |dL/dy| * i²"""
        return 0.5 * torch.abs(self.dL_dy(y)) * i**2
    
    def physics_loss(self, i, phi, y_pred):
        """
        Pérdida basada en física del sistema.
        
        Usa la relación: φ = L(y) * i
        Por lo tanto: L = φ/i y debe cumplir L(y) = k0 + k/(1+y/a)
        """
        # Calcular L desde φ e i
        L_from_phi = phi / (i + 1e-6)
        
        # Calcular L desde y predicha
        L_from_y = self.inductancia(y_pred)
        
        # Pérdida de consistencia: L calculado de φ debe coincidir con L(y)
        loss_inductancia = torch.mean((L_from_phi - L_from_y)**2)
        
        # Pérdida de equilibrio de fuerzas
        F_mag = self.fuerza_magnetica(y_pred, i)
        F_grav = self.m * self.g
        loss_equilibrio = torch.mean((F_mag - F_grav)**2) * 0.1
        
        # Pérdida de suavidad
        if y_pred.shape[0] > 2:
            dy = y_pred[1:] - y_pred[:-1]
            loss_suavidad = torch.mean(dy**2) * 100
        else:
            loss_suavidad = torch.tensor(0.0, device=y_pred.device)
        
        return loss_inductancia + loss_equilibrio + loss_suavidad
    
    def set_normalization(self, i_data, phi_data, u_data):
        """Calcula y establece parámetros de normalización."""
        self.i_mean = torch.tensor(np.mean(i_data), dtype=torch.float32)
        self.i_std = torch.tensor(np.std(i_data) + 1e-6, dtype=torch.float32)
        self.phi_mean = torch.tensor(np.mean(phi_data), dtype=torch.float32)
        self.phi_std = torch.tensor(np.std(phi_data) + 1e-6, dtype=torch.float32)
        self.u_mean = torch.tensor(np.mean(u_data), dtype=torch.float32)
        self.u_std = torch.tensor(np.std(u_data) + 1e-6, dtype=torch.float32)


def cargar_datasets(dataset_dir='../datasets'):
    """Carga datasets y calcula L = φ/i (inductancia estimada)."""
    dataset_path = Path(dataset_dir)
    all_results = []
    
    for file in sorted(dataset_path.glob('dataset_*.txt')):
        print(f"Cargando {file.name}...")
        data = np.loadtxt(file, skiprows=9)
        
        # Columnas: t, y, y_obs, dy_obs, i, u, yd
        t = data[:, 0]
        y = data[:, 1]
        i = data[:, 4]
        u = data[:, 5]
        
        # Calcular flujo con reset periódico para evitar drift
        dt = 0.01
        phi = np.zeros_like(i)
        L_est = np.zeros_like(i)
        
        for k in range(1, len(phi)):
            dphi = (u[k] - R * i[k]) * dt
            phi[k] = phi[k-1] + dphi
            
            # Calcular L = φ/i (inductancia estimada)
            if i[k] > 0.05:
                L_est[k] = phi[k] / i[k]
                # Reset suave si L sale de rango físico
                if L_est[k] < 0.05 or L_est[k] > 0.15:
                    phi[k] = 0.08 * i[k]  # Reset a L típico
                    L_est[k] = 0.08
            else:
                L_est[k] = L_est[k-1] if k > 0 else 0.08
        
        # Filtrar datos válidos
        mask = (y > 0.001) & (y < 0.018) & (i > 0.05) & (u > 0.5)
        mask &= (L_est > 0.05) & (L_est < 0.15)
        
        all_results.append({
            't': t[mask],
            'y': y[mask],
            'i': i[mask],
            'L_est': L_est[mask],  # Usar L en lugar de φ
            'u': u[mask]
        })
    
    if not all_results:
        raise ValueError(f"No se encontraron datasets en {dataset_dir}")
    
    # Concatenar
    result = {
        't': np.concatenate([r['t'] for r in all_results]),
        'y': np.concatenate([r['y'] for r in all_results]),
        'i': np.concatenate([r['i'] for r in all_results]),
        'L_est': np.concatenate([r['L_est'] for r in all_results]),
        'u': np.concatenate([r['u'] for r in all_results])
    }
    
    print(f"Total muestras válidas: {len(result['y'])}")
    print(f"Rango L: {result['L_est'].min():.4f} - {result['L_est'].max():.4f} H")
    
    return result


def entrenar(epochs=500, batch_size=128, lr=1e-3, physics_weight=0.1):
    """Entrena el KAN observador con inductancia L estimada."""
    
    print("="*60)
    print("ENTRENAMIENTO KAN-PINN OBSERVADOR (con L = φ/i)")
    print("="*60)
    
    # Cargar datos
    data = cargar_datasets()
    n_samples = len(data['y'])
    print(f"Muestras para entrenamiento: {n_samples}")
    
    if n_samples < 100:
        print("⚠️ Muy pocas muestras")
        return None
    
    # Preparar tensores: [i, L_est, u]
    X = torch.tensor(np.stack([data['i'], data['L_est'], data['u']], axis=1), dtype=torch.float32)
    y_true = torch.tensor(data['y'], dtype=torch.float32).unsqueeze(1)
    
    # Crear modelo
    model = KANObservador(hidden=32, depth=2, num_knots=8)
    model.set_normalization(data['i'], data['L_est'], data['u'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)
    
    # Entrenamiento
    history = {'loss': [], 'data_loss': [], 'physics_loss': []}
    best_loss = float('inf')
    best_state = model.state_dict().copy()
    
    for epoch in range(epochs):
        model.train()
        
        # Mini-batch aleatorio
        idx = torch.randperm(n_samples)[:batch_size]
        X_batch = X[idx]
        y_batch = y_true[idx]
        
        # Forward
        y_pred = model(X_batch)
        
        # Pérdida de datos
        data_loss = torch.mean((y_pred - y_batch)**2)
        
        # Pérdida física
        phys_loss = model.physics_loss(X_batch[:, 0], X_batch[:, 1], y_pred.squeeze())
        
        # Total
        total_loss = data_loss + physics_weight * phys_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(total_loss)
        
        # Registro
        history['loss'].append(total_loss.item())
        history['data_loss'].append(data_loss.item())
        history['physics_loss'].append(phys_loss.item())
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                y_all = model(X)
                mae = torch.mean(torch.abs(y_all - y_true)).item()
                corr = np.corrcoef(y_all.numpy().flatten(), y_true.numpy().flatten())[0, 1]
            print(f"Epoch {epoch+1}/{epochs}: Loss={total_loss.item():.6f}, MAE={mae*1000:.2f}mm, Corr={corr:.3f}")
    
    # Cargar mejor modelo
    model.load_state_dict(best_state)
    
    # Evaluación final
    model.eval()
    with torch.no_grad():
        y_pred_all = model(X)
        mae = torch.mean(torch.abs(y_pred_all - y_true)).item()
        corr = np.corrcoef(y_pred_all.numpy().flatten(), y_true.numpy().flatten())[0, 1]
    
    print(f"\n{'='*60}")
    print("RESULTADOS FINALES")
    print(f"{'='*60}")
    print(f"MAE: {mae*1000:.2f} mm")
    print(f"Correlación: {corr:.4f}")
    
    # Guardar
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"kan_observador_{timestamp}.pt"
    torch.save({
        'model_state': model.state_dict(),
        'history': history,
        'metrics': {'mae': mae, 'corr': corr}
    }, model_path)
    print(f"\n💾 Modelo guardado: {model_path}")
    
    # Gráfica
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['loss'], label='Total')
    axes[0, 0].plot(history['data_loss'], label='Datos')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_title('Pérdidas')
    
    axes[0, 1].scatter(y_true.numpy()*1000, y_pred_all.numpy()*1000, alpha=0.3, s=1)
    axes[0, 1].plot([0, 20], [0, 20], 'r--')
    axes[0, 1].set_xlabel('y real [mm]')
    axes[0, 1].set_ylabel('y KAN [mm]')
    axes[0, 1].set_title(f'Predicción (corr={corr:.3f})')
    axes[0, 1].grid(True)
    
    n_plot = min(500, len(y_true))
    axes[1, 0].plot(data['t'][:n_plot], y_true[:n_plot].numpy()*1000, 'b-', label='Real', alpha=0.7)
    axes[1, 0].plot(data['t'][:n_plot], y_pred_all[:n_plot].numpy()*1000, 'r-', label='KAN', alpha=0.7)
    axes[1, 0].set_xlabel('Tiempo [s]')
    axes[1, 0].set_ylabel('Posición [mm]')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_title('Serie temporal')
    
    error = (y_pred_all - y_true).numpy() * 1000
    axes[1, 1].hist(error, bins=50, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Error [mm]')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title(f'Distribución error (MAE={mae*1000:.2f}mm)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'kan_observador_resultados_{timestamp}.png', dpi=150)
    print(f"📊 Gráfica: kan_observador_resultados_{timestamp}.png")
    plt.show()
    
    return model


if __name__ == '__main__':
    entrenar(epochs=500, batch_size=128, lr=1e-3, physics_weight=0.01)
