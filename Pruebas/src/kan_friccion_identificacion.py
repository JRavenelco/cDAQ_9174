#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAN-PINN PARA IDENTIFICACIÓN DE PARÁMETROS m, k, c
===================================================
Basado en el proyecto Lagrangian KAN Physics Discovery.

Modelo físico:
    F_celda = m·a + c·v + k·x + F_friccion
    
Donde F_friccion puede ser:
    - Coulomb: F_c · sign(v)
    - Viscoso: c·v (ya incluido)
    - Stribeck: F_c · sign(v) · exp(-|v|/v_s)

El KAN aprende la relación y los parámetros físicos simultáneamente.

Autor: Adaptado de Lagrangian KAN Physics Discovery
Fecha: 2025-12-03
"""

import os
import sys
import time
import glob
import json
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from scipy.integrate import cumulative_trapezoid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DATOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caracterizacion_fuerza")
SAMPLE_RATE = 2500
G = 9.81

# Detectar dispositivo
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    GPU_NAME = torch.cuda.get_device_name(0)
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🚀 GPU: {GPU_NAME} ({VRAM_GB:.1f} GB)")
else:
    DEVICE = torch.device('cpu')
    print("⚠️ Sin GPU, usando CPU")

# Configuración del modelo según hardware
CONFIG = {
    'hidden_dims': [32, 32],
    'num_knots': 8,
    'batch_size': 512,
    'epochs': 300,
    'lr': 1e-3,
    'lr_phys': 1e-2,  # Learning rate para parámetros físicos
}

# Bounds para parámetros físicos (evitar soluciones triviales)
PHYS_BOUNDS = {
    'm': (0.1, 100.0),      # kg - masa efectiva
    'k': (1e3, 1e7),        # N/m - rigidez
    'c': (1.0, 1e4),        # N·s/m - amortiguamiento
    'Fc': (0.0, 100.0),     # N - fricción de Coulomb
}

# =============================================================================
# DATASET
# =============================================================================

class FriccionDataset(Dataset):
    """Dataset para identificación de fricción."""
    
    def __init__(self, archivos: list, sensor_name: str = "sensor"):
        """
        Args:
            archivos: Lista de archivos CSV
            sensor_name: Nombre del sensor para logging
        """
        self.sensor_name = sensor_name
        self.data = []
        
        for archivo in archivos:
            df = pd.read_csv(archivo)
            nombre = os.path.basename(archivo)
            
            # Extraer frecuencia
            try:
                freq = float(nombre.split('_')[-1].replace('Hz.csv', ''))
            except:
                continue
            
            # Datos crudos
            t = df['tiempo_s'].values
            fuerza_V = df['fuerza_V'].values
            
            # Detectar columna de aceleración
            if 'aceleracion_sensor_g' in df.columns:
                acel_g = df['aceleracion_sensor_g'].values
            elif 'aceleracion_g' in df.columns:
                acel_g = df['aceleracion_g'].values
            else:
                acel_cols = [c for c in df.columns if 'acel' in c.lower()]
                if acel_cols:
                    acel_g = df[acel_cols[0]].values
                else:
                    continue
            
            # Descartar transitorio
            skip = int(0.5 * SAMPLE_RATE)
            if len(t) <= skip * 2:
                continue
            
            t = t[skip:] - t[skip]
            fuerza_V = fuerza_V[skip:]
            acel_g = acel_g[skip:]
            
            # Convertir a SI
            acel_ms2 = acel_g * G
            
            # Calcular velocidad y posición por integración
            dt = 1 / SAMPLE_RATE
            fc = max(freq / 10, 0.5)
            b, a_filt = signal.butter(2, fc / (SAMPLE_RATE / 2), btype='high')
            
            vel = cumulative_trapezoid(acel_ms2 - np.mean(acel_ms2), dx=dt, initial=0)
            vel = signal.filtfilt(b, a_filt, vel)
            
            pos = cumulative_trapezoid(vel, dx=dt, initial=0)
            pos = signal.filtfilt(b, a_filt, pos)
            
            # Guardar datos
            self.data.append({
                'freq': freq,
                't': t,
                'F': fuerza_V - np.mean(fuerza_V),  # Componente AC
                'a': acel_ms2 - np.mean(acel_ms2),
                'v': vel,
                'x': pos,
            })
        
        # Concatenar todos los datos
        if self.data:
            self.t = np.concatenate([d['t'] for d in self.data])
            self.F = np.concatenate([d['F'] for d in self.data])
            self.a = np.concatenate([d['a'] for d in self.data])
            self.v = np.concatenate([d['v'] for d in self.data])
            self.x = np.concatenate([d['x'] for d in self.data])
            self.freq = np.concatenate([np.full(len(d['t']), d['freq']) for d in self.data])
            
            # Estadísticas para normalización
            self.stats = {
                'F_mu': np.mean(self.F), 'F_std': np.std(self.F) + 1e-8,
                'a_mu': np.mean(self.a), 'a_std': np.std(self.a) + 1e-8,
                'v_mu': np.mean(self.v), 'v_std': np.std(self.v) + 1e-8,
                'x_mu': np.mean(self.x), 'x_std': np.std(self.x) + 1e-8,
            }
            
            print(f"📊 {sensor_name}: {len(self.data)} frecuencias, {len(self.t)} muestras")
            print(f"   F_std: {self.stats['F_std']*1000:.3f} mV")
            print(f"   a_std: {self.stats['a_std']:.4f} m/s²")
            print(f"   v_std: {self.stats['v_std']*1000:.3f} mm/s")
            print(f"   x_std: {self.stats['x_std']*1e6:.2f} µm")
    
    def __len__(self):
        return len(self.t)
    
    def __getitem__(self, idx):
        return {
            'F': torch.tensor(self.F[idx], dtype=torch.float32),
            'a': torch.tensor(self.a[idx], dtype=torch.float32),
            'v': torch.tensor(self.v[idx], dtype=torch.float32),
            'x': torch.tensor(self.x[idx], dtype=torch.float32),
            'freq': torch.tensor(self.freq[idx], dtype=torch.float32),
        }


# =============================================================================
# MODELO KAN
# =============================================================================

class KANLayer(nn.Module):
    """Capa KAN (Kolmogorov-Arnold Network) con B-splines."""
    
    def __init__(self, in_features: int, out_features: int, num_knots: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_knots = num_knots
        
        # Coeficientes de spline para cada par (in, out)
        self.coeffs = nn.Parameter(
            torch.randn(in_features, out_features, num_knots) * 0.1
        )
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Knots uniformes en [-1, 1]
        self.register_buffer('knots', torch.linspace(-1, 1, num_knots))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        batch_size = x.shape[0]
        
        # Evaluar B-splines
        # Usar RBF como aproximación simple de splines
        x_expanded = x.unsqueeze(-1)  # (batch, in, 1)
        knots = self.knots.unsqueeze(0).unsqueeze(0)  # (1, 1, num_knots)
        
        # Basis functions (Gaussian RBF)
        sigma = 2.0 / self.num_knots
        basis = torch.exp(-((x_expanded - knots) ** 2) / (2 * sigma ** 2))
        # basis: (batch, in_features, num_knots)
        
        # Combinar con coeficientes
        # coeffs: (in_features, out_features, num_knots)
        # Queremos: sum over in_features and num_knots
        out = torch.einsum('bin,ion->bo', basis, self.coeffs)
        
        return out + self.bias


class FriccionKAN(nn.Module):
    """
    KAN para identificación de parámetros de fricción.
    
    Modelo físico:
        F = m·a + c·v + k·x + F_friccion(v)
        
    Donde F_friccion es aprendido por la red.
    """
    
    def __init__(
        self,
        hidden_dims: list = [32, 32],
        num_knots: int = 8,
        stats: dict = None,
    ):
        super().__init__()
        
        self.stats = stats or {}
        
        # Parámetros físicos entrenables (en log-space para positividad)
        self.log_m = nn.Parameter(torch.tensor(np.log(10.0)))   # ~10 kg
        self.log_k = nn.Parameter(torch.tensor(np.log(1e5)))    # ~100 kN/m
        self.log_c = nn.Parameter(torch.tensor(np.log(100.0)))  # ~100 N·s/m
        
        # Fricción de Coulomb (puede ser 0)
        self.Fc = nn.Parameter(torch.tensor(0.0))
        
        # Red KAN para fricción no-lineal residual
        # Input: [v, |v|, sign(v), a, x]
        layers = []
        in_dim = 5
        
        for hidden in hidden_dims:
            layers.append(KANLayer(in_dim, hidden, num_knots))
            layers.append(nn.Tanh())
            in_dim = hidden
        
        # Output: F_residual (escalar)
        layers.append(KANLayer(in_dim, 1, num_knots))
        
        self.kan = nn.Sequential(*layers)
    
    def get_phys_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna parámetros físicos en espacio real."""
        m = torch.exp(self.log_m)
        k = torch.exp(self.log_k)
        c = torch.exp(self.log_c)
        Fc = torch.abs(self.Fc)  # Fricción siempre positiva
        return m, k, c, Fc
    
    def forward(
        self,
        a: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predice la fuerza total.
        
        Returns:
            F_pred: Fuerza predicha
            F_friction: Componente de fricción (para análisis)
        """
        m, k, c, Fc = self.get_phys_params()
        
        # Términos lineales
        F_inertial = m * a
        F_elastic = k * x
        F_viscous = c * v
        
        # Fricción de Coulomb
        F_coulomb = Fc * torch.sign(v)
        
        # Fricción no-lineal (aprendida por KAN)
        # Normalizar inputs
        v_norm = v / (self.stats.get('v_std', 1.0) + 1e-8)
        a_norm = a / (self.stats.get('a_std', 1.0) + 1e-8)
        x_norm = x / (self.stats.get('x_std', 1.0) + 1e-8)
        
        kan_input = torch.stack([
            v_norm,
            torch.abs(v_norm),
            torch.sign(v),
            a_norm,
            x_norm,
        ], dim=-1)
        
        F_residual = self.kan(kan_input).squeeze(-1) * self.stats.get('F_std', 1.0)
        
        # Fuerza total
        F_pred = F_inertial + F_elastic + F_viscous + F_coulomb + F_residual
        F_friction = F_coulomb + F_residual
        
        return F_pred, F_friction
    
    def clip_params(self):
        """Aplica clipping a parámetros físicos."""
        with torch.no_grad():
            self.log_m.data.clamp_(np.log(PHYS_BOUNDS['m'][0]), np.log(PHYS_BOUNDS['m'][1]))
            self.log_k.data.clamp_(np.log(PHYS_BOUNDS['k'][0]), np.log(PHYS_BOUNDS['k'][1]))
            self.log_c.data.clamp_(np.log(PHYS_BOUNDS['c'][0]), np.log(PHYS_BOUNDS['c'][1]))
            self.Fc.data.clamp_(PHYS_BOUNDS['Fc'][0], PHYS_BOUNDS['Fc'][1])


# =============================================================================
# FUNCIONES DE PÉRDIDA
# =============================================================================

def loss_physics(
    model: FriccionKAN,
    F: torch.Tensor,
    a: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """
    Pérdida física: ||F_medida - F_predicha||²
    """
    F_pred, F_friction = model(a, v, x)
    
    # MSE
    residual = F - F_pred
    loss = torch.mean(residual ** 2)
    
    # Métricas adicionales
    m, k, c, Fc = model.get_phys_params()
    
    info = {
        'loss': loss.item(),
        'residual_rms': torch.sqrt(loss).item(),
        'm': m.item(),
        'k': k.item(),
        'c': c.item(),
        'Fc': Fc.item(),
        'F_friction_rms': torch.sqrt(torch.mean(F_friction ** 2)).item(),
    }
    
    return loss, info


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def train_model(
    model: FriccionKAN,
    train_loader: DataLoader,
    epochs: int = 300,
    lr: float = 1e-3,
    lr_phys: float = 1e-2,
    device: torch.device = DEVICE,
) -> dict:
    """Entrena el modelo."""
    
    model = model.to(device)
    
    # Optimizadores separados para red y parámetros físicos
    kan_params = [p for n, p in model.named_parameters() if 'log_' not in n and 'Fc' not in n]
    phys_params = [model.log_m, model.log_k, model.log_c, model.Fc]
    
    optimizer_kan = optim.Adam(kan_params, lr=lr)
    optimizer_phys = optim.Adam(phys_params, lr=lr_phys)
    
    scheduler_kan = optim.lr_scheduler.CosineAnnealingLR(optimizer_kan, epochs)
    scheduler_phys = optim.lr_scheduler.CosineAnnealingLR(optimizer_phys, epochs)
    
    history = {
        'loss': [], 'm': [], 'k': [], 'c': [], 'Fc': [], 'F_friction_rms': []
    }
    
    best_loss = float('inf')
    best_params = None
    
    print(f"\n🏋️ Entrenando {epochs} épocas...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            F = batch['F'].to(device)
            a = batch['a'].to(device)
            v = batch['v'].to(device)
            x = batch['x'].to(device)
            
            optimizer_kan.zero_grad()
            optimizer_phys.zero_grad()
            
            loss, info = loss_physics(model, F, a, v, x)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer_kan.step()
            optimizer_phys.step()
            
            # Clipping de parámetros físicos
            model.clip_params()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler_kan.step()
        scheduler_phys.step()
        
        avg_loss = epoch_loss / n_batches
        m, k, c, Fc = model.get_phys_params()
        
        history['loss'].append(avg_loss)
        history['m'].append(m.item())
        history['k'].append(k.item())
        history['c'].append(c.item())
        history['Fc'].append(Fc.item())
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = {
                'm': m.item(),
                'k': k.item(),
                'c': c.item(),
                'Fc': Fc.item(),
            }
        
        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"  Época {epoch:3d}/{epochs} | Loss: {avg_loss:.2e} | "
                  f"m={m.item():.2f} kg | k={k.item()/1000:.1f} kN/m | "
                  f"c={c.item():.1f} N·s/m | Fc={Fc.item():.3f} | "
                  f"Time: {elapsed/60:.1f}min")
    
    return {
        'history': history,
        'best_loss': best_loss,
        'best_params': best_params,
        'model_state': model.state_dict(),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("KAN-PINN IDENTIFICACIÓN DE PARÁMETROS m, k, c")
    print("=" * 80)
    
    # Buscar datos DYMH-105
    pattern_dymh = os.path.join(DATOS_DIR, "caracterizacion_fuerza_20251202_*_exp*.csv")
    archivos_dymh = sorted(glob.glob(pattern_dymh))
    if not archivos_dymh:
        pattern_dymh = os.path.join(DATOS_DIR, "caracterizacion_fuerza_20251201_*_exp*.csv")
        archivos_dymh = sorted(glob.glob(pattern_dymh))
    
    # Buscar datos LC302-1K
    pattern_lc302 = os.path.join(DATOS_DIR, "caracterizacion_fuerza_20251203_160853_exp*.csv")
    archivos_lc302 = sorted(glob.glob(pattern_lc302))
    # Excluir 10 Hz (anómalo)
    archivos_lc302 = [a for a in archivos_lc302 if "_10Hz.csv" not in a]
    
    resultados = {}
    
    # =========================================================================
    # ENTRENAR CON DYMH-105
    # =========================================================================
    if archivos_dymh:
        print("\n" + "=" * 80)
        print("SENSOR: DYMH-105")
        print("=" * 80)
        
        dataset_dymh = FriccionDataset(archivos_dymh, "DYMH-105")
        
        if len(dataset_dymh) > 0:
            loader_dymh = DataLoader(
                dataset_dymh,
                batch_size=CONFIG['batch_size'],
                shuffle=True,
                num_workers=0,
            )
            
            model_dymh = FriccionKAN(
                hidden_dims=CONFIG['hidden_dims'],
                num_knots=CONFIG['num_knots'],
                stats=dataset_dymh.stats,
            )
            
            result_dymh = train_model(
                model_dymh,
                loader_dymh,
                epochs=CONFIG['epochs'],
                lr=CONFIG['lr'],
                lr_phys=CONFIG['lr_phys'],
            )
            
            resultados['DYMH-105'] = result_dymh
            print(f"\n✅ DYMH-105 Mejor resultado:")
            print(f"   m = {result_dymh['best_params']['m']:.2f} kg")
            print(f"   k = {result_dymh['best_params']['k']/1000:.2f} kN/m")
            print(f"   c = {result_dymh['best_params']['c']:.2f} N·s/m")
            print(f"   Fc = {result_dymh['best_params']['Fc']:.4f} N")
    
    # =========================================================================
    # ENTRENAR CON LC302-1K
    # =========================================================================
    if archivos_lc302:
        print("\n" + "=" * 80)
        print("SENSOR: LC302-1K")
        print("=" * 80)
        
        dataset_lc302 = FriccionDataset(archivos_lc302, "LC302-1K")
        
        if len(dataset_lc302) > 0:
            loader_lc302 = DataLoader(
                dataset_lc302,
                batch_size=CONFIG['batch_size'],
                shuffle=True,
                num_workers=0,
            )
            
            model_lc302 = FriccionKAN(
                hidden_dims=CONFIG['hidden_dims'],
                num_knots=CONFIG['num_knots'],
                stats=dataset_lc302.stats,
            )
            
            result_lc302 = train_model(
                model_lc302,
                loader_lc302,
                epochs=CONFIG['epochs'],
                lr=CONFIG['lr'],
                lr_phys=CONFIG['lr_phys'],
            )
            
            resultados['LC302-1K'] = result_lc302
            print(f"\n✅ LC302-1K Mejor resultado:")
            print(f"   m = {result_lc302['best_params']['m']:.2f} kg")
            print(f"   k = {result_lc302['best_params']['k']/1000:.2f} kN/m")
            print(f"   c = {result_lc302['best_params']['c']:.2f} N·s/m")
            print(f"   Fc = {result_lc302['best_params']['Fc']:.4f} N")
    
    # =========================================================================
    # COMPARACIÓN Y VISUALIZACIÓN
    # =========================================================================
    if resultados:
        print("\n" + "=" * 80)
        print("COMPARACIÓN DE RESULTADOS")
        print("=" * 80)
        
        print("\n" + "-" * 60)
        print(f"{'Parámetro':<15} {'DYMH-105':>15} {'LC302-1K':>15}")
        print("-" * 60)
        
        for param in ['m', 'k', 'c', 'Fc']:
            val_dymh = resultados.get('DYMH-105', {}).get('best_params', {}).get(param, 0)
            val_lc302 = resultados.get('LC302-1K', {}).get('best_params', {}).get(param, 0)
            
            if param == 'k':
                print(f"{param + ' (kN/m)':<15} {val_dymh/1000:>15.2f} {val_lc302/1000:>15.2f}")
            elif param == 'm':
                print(f"{param + ' (kg)':<15} {val_dymh:>15.2f} {val_lc302:>15.2f}")
            elif param == 'c':
                print(f"{param + ' (N·s/m)':<15} {val_dymh:>15.2f} {val_lc302:>15.2f}")
            else:
                print(f"{param + ' (N)':<15} {val_dymh:>15.4f} {val_lc302:>15.4f}")
        
        print("-" * 60)
        
        # Gráficas
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('KAN-PINN Identificación de Parámetros', fontsize=14, fontweight='bold')
        
        colors = {'DYMH-105': 'cyan', 'LC302-1K': 'orange'}
        
        for sensor, result in resultados.items():
            hist = result['history']
            color = colors.get(sensor, 'white')
            
            # Loss
            axes[0, 0].semilogy(hist['loss'], color=color, label=sensor, linewidth=2)
            
            # Masa
            axes[0, 1].plot(hist['m'], color=color, label=sensor, linewidth=2)
            
            # Rigidez
            axes[1, 0].plot([k/1000 for k in hist['k']], color=color, label=sensor, linewidth=2)
            
            # Amortiguamiento
            axes[1, 1].plot(hist['c'], color=color, label=sensor, linewidth=2)
        
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Convergencia')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('m (kg)')
        axes[0, 1].set_title('Masa Efectiva')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('k (kN/m)')
        axes[1, 0].set_title('Rigidez')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('c (N·s/m)')
        axes[1, 1].set_title('Amortiguamiento')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = os.path.join(DATOS_DIR, "kan_identificacion_mkc.png")
        plt.savefig(output_file, dpi=150, facecolor='white')
        print(f"\n📊 Guardado: {output_file}")
        plt.show()
        
        # Guardar resultados en JSON
        json_file = os.path.join(DATOS_DIR, "kan_identificacion_resultados.json")
        json_data = {
            sensor: {
                'best_params': result['best_params'],
                'best_loss': result['best_loss'],
            }
            for sensor, result in resultados.items()
        }
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"📄 Guardado: {json_file}")


if __name__ == "__main__":
    main()
