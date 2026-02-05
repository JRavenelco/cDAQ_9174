#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAN-PINN IDENTIFICACIÓN DE PARÁMETROS m, k, c - VERSIÓN EFICIENTE
===================================================================
Basado en train_kanpinn_efficient.py del proyecto Lagrangian KAN

MEJORAS IMPLEMENTADAS:
1. ✅ Curriculum learning (datos → física gradual)
2. ✅ Adaptive weighting de lambda_phys
3. ✅ Bounds estrictos para evitar mínimos espurios
4. ✅ Clipping de parámetros físicos

Modelo físico:
    F_celda = m·a + c·v + k·x + Fc·sign(v)

Datos: Barrido 10-40 Hz del 4 Dic 2025
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import signal
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DATOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caracterizacion_fuerza")
SAMPLE_RATE = 2500
G = 9.81

# Archivos del 4 Dic 2025
ARCHIVOS = {
    10: "caracterizacion_fuerza_20251204_110740_exp1_10Hz.csv",
    20: "caracterizacion_fuerza_20251204_110740_exp2_20Hz.csv",
    30: "caracterizacion_fuerza_20251204_110740_exp3_30Hz.csv",
    40: "caracterizacion_fuerza_20251204_110740_exp4_40Hz.csv",
}

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*80}")
print(f"🚀 KAN-PINN IDENTIFICACIÓN m, k, c")
print(f"{'='*80}")
print(f"Dispositivo: {device}")

if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")
    GPU_TIER = "small" if vram_gb < 10 else "medium"
else:
    GPU_TIER = "small"

# =============================================================================
# BOUNDS PARA PARÁMETROS FÍSICOS (EN UNIDADES DE VOLTAJE)
# =============================================================================
# Como F está en Voltios y a en m/s², los parámetros tienen unidades:
#   m_v: V/(m/s²) = V·s²/m
#   k_v: V/m
#   c_v: V/(m/s) = V·s/m

PHYS_BOUNDS = {
    'm': (1e-6, 1.0),       # V·s²/m - masa en unidades de voltaje
    'k': (1e-3, 1e3),       # V/m - rigidez en unidades de voltaje
    'c': (1e-6, 1.0),       # V·s/m - amortiguamiento en unidades de voltaje
    'Fc': (0.0, 0.01),      # V - fricción Coulomb en voltios
}

PHYS_BOUNDS_LOG = {
    'log_m': (np.log(PHYS_BOUNDS['m'][0]), np.log(PHYS_BOUNDS['m'][1])),
    'log_k': (np.log(PHYS_BOUNDS['k'][0]), np.log(PHYS_BOUNDS['k'][1])),
    'log_c': (np.log(PHYS_BOUNDS['c'][0]), np.log(PHYS_BOUNDS['c'][1])),
}

# =============================================================================
# CONFIGURACIÓN POR TIER
# =============================================================================

if GPU_TIER == "small":
    CONFIG = {
        "model": {"hidden": 64, "depth": 3},
        "training": {
            "epochs_phase1": 100,   # Solo datos
            "epochs_phase2": 300,   # Datos + física
            "batch_size": 512,
            "lr": 1e-3,
            "lr_phys": 1e-2,
        },
        "physics": {
            "lambda_data": 1.0,
            "lambda_phys_initial": 1e-4,
            "lambda_phys_final": 1e-1,  # Más fuerte para forzar física
        },
    }
else:
    CONFIG = {
        "model": {"hidden": 128, "depth": 4},
        "training": {
            "epochs_phase1": 100,
            "epochs_phase2": 300,
            "batch_size": 1024,
            "lr": 5e-4,
            "lr_phys": 1e-2,
        },
        "physics": {
            "lambda_data": 1.0,
            "lambda_phys_initial": 1e-5,
            "lambda_phys_final": 1e-2,
        },
    }

print(f"Tier: {GPU_TIER.upper()}")
print(f"{'='*80}\n")

# =============================================================================
# MODELO KAN PARA FRICCIÓN
# =============================================================================

class FriccionKAN(nn.Module):
    """
    KAN-PINN para identificar parámetros de fricción.
    
    Modelo: F = m·a + c·v + k·x + Fc·sign(v) + f_residual(a,v,x)
    
    Donde f_residual es aprendido por la red para capturar no-linealidades.
    """
    
    def __init__(self, hidden=64, depth=3, stats=None):
        super().__init__()
        
        self.stats = stats or {}
        
        # Parámetros físicos (en log-space para positividad)
        # Unidades en voltaje: F[V] = m_v·a + c_v·v + k_v·x
        # Para fn=25Hz: k/m = (2π·25)² ≈ 24600, si m~0.01 → k~246
        self.log_m = nn.Parameter(torch.tensor(np.log(0.01)))   # ~0.01 V·s²/m
        self.log_k = nn.Parameter(torch.tensor(np.log(100.0)))  # ~100 V/m → fn≈50Hz
        self.log_c = nn.Parameter(torch.tensor(np.log(0.1)))    # ~0.1 V·s/m
        self.Fc = nn.Parameter(torch.tensor(0.001))             # ~1 mV
        
        # Red para residual no-lineal
        # Input: [a_norm, v_norm, x_norm, sign(v), |v|_norm]
        layers = []
        in_dim = 5
        
        for i in range(depth):
            out_dim = hidden if i < depth - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < depth - 1:
                layers.append(nn.Tanh())
            in_dim = hidden
        
        self.residual_net = nn.Sequential(*layers)
        
        # Inicialización pequeña para que el residual empiece cerca de 0
        for m in self.residual_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def get_phys_params(self):
        """Retorna parámetros físicos en espacio real."""
        m = torch.exp(self.log_m)
        k = torch.exp(self.log_k)
        c = torch.exp(self.log_c)
        Fc = torch.abs(self.Fc)
        return m, k, c, Fc
    
    def forward(self, a, v, x):
        """
        Predice fuerza total.
        
        Args:
            a: aceleración (m/s²)
            v: velocidad (m/s)
            x: posición (m)
        
        Returns:
            F_pred: fuerza predicha (V o N según unidades de entrada)
        """
        m, k, c, Fc = self.get_phys_params()
        
        # Términos físicos lineales
        F_inertial = m * a
        F_elastic = k * x
        F_viscous = c * v
        F_coulomb = Fc * torch.sign(v + 1e-10)  # Evitar sign(0)
        
        # Residual no-lineal
        a_std = self.stats.get('a_std', 1.0)
        v_std = self.stats.get('v_std', 1.0)
        x_std = self.stats.get('x_std', 1.0)
        
        features = torch.stack([
            a / (a_std + 1e-8),
            v / (v_std + 1e-8),
            x / (x_std + 1e-8),
            torch.sign(v + 1e-10),
            torch.abs(v) / (v_std + 1e-8),
        ], dim=-1)
        
        F_residual = self.residual_net(features).squeeze(-1) * self.stats.get('F_std', 1.0) * 0.1
        
        F_total = F_inertial + F_elastic + F_viscous + F_coulomb + F_residual
        
        return F_total, F_residual
    
    def clip_params(self):
        """Clipping de parámetros físicos dentro de bounds."""
        with torch.no_grad():
            self.log_m.data.clamp_(PHYS_BOUNDS_LOG['log_m'][0], PHYS_BOUNDS_LOG['log_m'][1])
            self.log_k.data.clamp_(PHYS_BOUNDS_LOG['log_k'][0], PHYS_BOUNDS_LOG['log_k'][1])
            self.log_c.data.clamp_(PHYS_BOUNDS_LOG['log_c'][0], PHYS_BOUNDS_LOG['log_c'][1])
            self.Fc.data.clamp_(PHYS_BOUNDS['Fc'][0], PHYS_BOUNDS['Fc'][1])


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def cargar_datos():
    """Carga y preprocesa todos los datos."""
    print("📂 Cargando datos...")
    
    all_F = []
    all_a = []
    all_v = []
    all_x = []
    
    for freq, archivo in ARCHIVOS.items():
        path = os.path.join(DATOS_DIR, archivo)
        df = pd.read_csv(path)
        
        t = df['tiempo_s'].values
        fuerza_V = df['fuerza_V'].values
        acel_g = df['aceleracion_sensor_g'].values
        
        # Descartar transitorio
        skip = int(0.5 * SAMPLE_RATE)
        t = t[skip:] - t[skip]
        fuerza_V = fuerza_V[skip:]
        acel_g = acel_g[skip:]
        
        # Convertir a SI
        acel_ms2 = acel_g * G
        
        # Remover DC
        F_ac = fuerza_V - np.mean(fuerza_V)
        a_ac = acel_ms2 - np.mean(acel_ms2)
        
        # Calcular v y x por integración
        dt = 1 / SAMPLE_RATE
        fc = max(freq / 10, 0.5)
        b, a_filt = signal.butter(2, fc / (SAMPLE_RATE / 2), btype='high')
        
        v = cumulative_trapezoid(a_ac, dx=dt, initial=0)
        v = signal.filtfilt(b, a_filt, v)
        
        x = cumulative_trapezoid(v, dx=dt, initial=0)
        x = signal.filtfilt(b, a_filt, x)
        
        # Submuestrear para reducir datos (cada 10 puntos)
        step = 10
        all_F.append(F_ac[::step])
        all_a.append(a_ac[::step])
        all_v.append(v[::step])
        all_x.append(x[::step])
        
        print(f"  {freq} Hz: {len(F_ac[::step])} puntos")
    
    # Concatenar
    F = np.concatenate(all_F)
    a = np.concatenate(all_a)
    v = np.concatenate(all_v)
    x = np.concatenate(all_x)
    
    # Estadísticas para normalización
    stats = {
        'F_mean': np.mean(F), 'F_std': np.std(F),
        'a_mean': np.mean(a), 'a_std': np.std(a),
        'v_mean': np.mean(v), 'v_std': np.std(v),
        'x_mean': np.mean(x), 'x_std': np.std(x),
    }
    
    print(f"\n📊 Total: {len(F)} puntos")
    print(f"   F_std: {stats['F_std']*1000:.3f} mV")
    print(f"   a_std: {stats['a_std']:.4f} m/s²")
    print(f"   v_std: {stats['v_std']*1000:.3f} mm/s")
    print(f"   x_std: {stats['x_std']*1e6:.2f} µm")
    
    # Convertir a tensores
    F_gpu = torch.tensor(F, dtype=torch.float32, device=device)
    a_gpu = torch.tensor(a, dtype=torch.float32, device=device)
    v_gpu = torch.tensor(v, dtype=torch.float32, device=device)
    x_gpu = torch.tensor(x, dtype=torch.float32, device=device)
    
    return F_gpu, a_gpu, v_gpu, x_gpu, stats


# =============================================================================
# ENTRENAMIENTO CON CURRICULUM LEARNING
# =============================================================================

def train_with_curriculum(model, F, a, v, x, stats):
    """
    Entrenamiento en 2 fases:
    - FASE 1: Solo ajuste de datos (warm-start)
    - FASE 2: Datos + física con lambda creciente
    """
    
    epochs_phase1 = CONFIG["training"]["epochs_phase1"]
    epochs_phase2 = CONFIG["training"]["epochs_phase2"]
    total_epochs = epochs_phase1 + epochs_phase2
    
    # Optimizadores separados
    net_params = [p for n, p in model.named_parameters() if 'log_' not in n and n != 'Fc']
    phys_params = [model.log_m, model.log_k, model.log_c, model.Fc]
    
    optimizer_net = optim.Adam(net_params, lr=CONFIG["training"]["lr"])
    optimizer_phys = optim.Adam(phys_params, lr=CONFIG["training"]["lr_phys"])
    
    # Scheduler
    scheduler_net = optim.lr_scheduler.CosineAnnealingLR(optimizer_net, total_epochs)
    scheduler_phys = optim.lr_scheduler.CosineAnnealingLR(optimizer_phys, total_epochs)
    
    # Lambda adaptativo
    lambda_data = CONFIG["physics"]["lambda_data"]
    lambda_phys_init = CONFIG["physics"]["lambda_phys_initial"]
    lambda_phys_final = CONFIG["physics"]["lambda_phys_final"]
    
    # History
    history = {
        "epoch": [], "loss_total": [], "loss_data": [], "loss_phys": [],
        "m": [], "k": [], "c": [], "Fc": [], "lambda_phys": []
    }
    
    print(f"\n{'='*80}")
    print("ENTRENAMIENTO CON CURRICULUM LEARNING")
    print(f"{'='*80}")
    print(f"FASE 1: Épocas 1-{epochs_phase1} (solo datos - warm start)")
    print(f"FASE 2: Épocas {epochs_phase1+1}-{total_epochs} (datos + física gradual)")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    N = len(F)
    batch_size = CONFIG["training"]["batch_size"]
    
    for epoch in range(1, total_epochs + 1):
        model.train()
        
        # Mini-batch aleatorio
        idx = torch.randperm(N, device=device)[:batch_size]
        F_batch = F[idx]
        a_batch = a[idx]
        v_batch = v[idx]
        x_batch = x[idx]
        
        optimizer_net.zero_grad()
        optimizer_phys.zero_grad()
        
        # Forward
        F_pred, F_residual = model(a_batch, v_batch, x_batch)
        
        # Loss de datos
        loss_data = torch.mean((F_batch - F_pred) ** 2)
        
        # Loss de física (regularización del residual)
        if epoch <= epochs_phase1:
            # FASE 1: Sin física, solo datos
            loss_phys = torch.tensor(0.0, device=device)
            lambda_phys = 0.0
        else:
            # FASE 2: Lambda creciente
            progress = (epoch - epochs_phase1) / epochs_phase2
            lambda_phys = lambda_phys_init + progress * (lambda_phys_final - lambda_phys_init)
            
            # Penalizar residual grande (forzar explicación física)
            loss_phys = torch.mean(F_residual ** 2)
        
        # Loss total
        loss_total = lambda_data * loss_data + lambda_phys * loss_phys
        
        # Backprop
        loss_total.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer_net.step()
        optimizer_phys.step()
        scheduler_net.step()
        scheduler_phys.step()
        
        # Clipping de parámetros físicos
        model.clip_params()
        
        # Tracking
        m, k, c, Fc = model.get_phys_params()
        history["epoch"].append(epoch)
        history["loss_total"].append(loss_total.item())
        history["loss_data"].append(loss_data.item())
        history["loss_phys"].append(loss_phys.item() if isinstance(loss_phys, torch.Tensor) else 0.0)
        history["m"].append(m.item())
        history["k"].append(k.item())
        history["c"].append(c.item())
        history["Fc"].append(Fc.item())
        history["lambda_phys"].append(lambda_phys)
        
        # Logging
        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            phase = "DATOS" if epoch <= epochs_phase1 else "FÍSICA"
            fn = np.sqrt(k.item() / (m.item() + 1e-10)) / (2 * np.pi)
            print(f"[{phase}] Época {epoch:3d}/{total_epochs} | "
                  f"Loss: {loss_total.item():.2e} | "
                  f"m_v={m.item():.4f} | k_v={k.item():.3f} | "
                  f"c_v={c.item():.4f} | fn={fn:.1f}Hz | "
                  f"λ={lambda_phys:.1e} | {elapsed/60:.1f}min")
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"✅ ENTRENAMIENTO COMPLETADO en {total_time/60:.1f} minutos")
    print(f"{'='*80}\n")
    
    return history


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def plot_results(history, model, F, a, v, x, stats):
    """Genera gráficas de resultados."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('KAN-PINN Identificación de Parámetros m, k, c', fontsize=14, fontweight='bold')
    
    epochs = history["epoch"]
    
    # Loss
    ax = axes[0, 0]
    ax.semilogy(epochs, history["loss_total"], 'b-', linewidth=2, label='Total')
    ax.semilogy(epochs, history["loss_data"], 'r--', linewidth=1, label='Data')
    ax.set_xlabel('Época')
    ax.set_ylabel('Loss')
    ax.set_title('Convergencia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Masa (V·s²/m)
    ax = axes[0, 1]
    ax.semilogy(epochs, history["m"], 'g-', linewidth=2)
    ax.set_xlabel('Época')
    ax.set_ylabel('m_v (V·s²/m)')
    ax.set_title(f'm_v: {history["m"][-1]:.6f}')
    ax.grid(True, alpha=0.3)
    
    # Rigidez (V/m)
    ax = axes[0, 2]
    ax.semilogy(epochs, history["k"], 'r-', linewidth=2)
    ax.set_xlabel('Época')
    ax.set_ylabel('k_v (V/m)')
    ax.set_title(f'k_v: {history["k"][-1]:.4f}')
    ax.grid(True, alpha=0.3)
    
    # Amortiguamiento (V·s/m)
    ax = axes[1, 0]
    ax.semilogy(epochs, history["c"], 'purple', linewidth=2)
    ax.set_xlabel('Época')
    ax.set_ylabel('c_v (V·s/m)')
    ax.set_title(f'c_v: {history["c"][-1]:.6f}')
    ax.grid(True, alpha=0.3)
    
    # Fricción Coulomb (mV)
    ax = axes[1, 1]
    ax.plot(epochs, [fc*1000 for fc in history["Fc"]], 'orange', linewidth=2)
    ax.set_xlabel('Época')
    ax.set_ylabel('Fc (mV)')
    ax.set_title(f'Fc: {history["Fc"][-1]*1000:.4f} mV')
    ax.grid(True, alpha=0.3)
    
    # Predicción vs Real
    ax = axes[1, 2]
    model.eval()
    with torch.no_grad():
        F_pred, _ = model(a[:1000], v[:1000], x[:1000])
        F_real = F[:1000].cpu().numpy()
        F_pred = F_pred.cpu().numpy()
    
    ax.plot(F_real * 1000, 'c-', linewidth=0.5, alpha=0.7, label='Real')
    ax.plot(F_pred * 1000, 'orange', linewidth=0.5, alpha=0.7, label='Predicho')
    ax.set_xlabel('Muestra')
    ax.set_ylabel('Fuerza (mV)')
    ax.set_title('Predicción vs Real')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(DATOS_DIR, "kan_identificacion_mkc_hoy.png")
    fig.savefig(output_file, dpi=150, facecolor='white')
    print(f"📊 Guardado: {output_file}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Cargar datos
    F, a, v, x, stats = cargar_datos()
    
    # Crear modelo
    model = FriccionKAN(
        hidden=CONFIG["model"]["hidden"],
        depth=CONFIG["model"]["depth"],
        stats=stats,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Modelo: {num_params:,} parámetros")
    
    # Entrenar
    history = train_with_curriculum(model, F, a, v, x, stats)
    
    # Resultados finales
    m, k, c, Fc = model.get_phys_params()
    print("\n" + "=" * 80)
    print("PARÁMETROS IDENTIFICADOS (unidades de voltaje)")
    print("=" * 80)
    print(f"  m_v (V·s²/m):      {m.item():.6f}")
    print(f"  k_v (V/m):         {k.item():.4f}")
    print(f"  c_v (V·s/m):       {c.item():.6f}")
    print(f"  Fc  (V):           {Fc.item()*1000:.4f} mV")
    print("=" * 80)
    
    # Frecuencia natural (k_v/m_v tiene unidades de 1/s²)
    wn = np.sqrt(k.item() / m.item())
    fn = wn / (2 * np.pi)
    zeta = c.item() / (2 * np.sqrt(k.item() * m.item()))
    print(f"\nPropiedades dinámicas:")
    print(f"  Frecuencia natural: fn = {fn:.1f} Hz")
    print(f"  Factor de amortiguamiento: ζ = {zeta:.4f}")
    
    # Conversión a unidades físicas (si se conoce sensibilidad)
    # Sensibilidad LC302-1K: 7.14e-5 V/kg (de calibración)
    SENS_V_PER_KG = 7.14e-5  # V/kg
    SENS_V_PER_N = SENS_V_PER_KG / G  # V/N
    print(f"\nConversión a unidades físicas (usando sensibilidad calibrada):")
    print(f"  m_fisico = m_v / sens = {m.item() / SENS_V_PER_N:.2f} kg")
    print(f"  k_fisico = k_v / sens = {k.item() / SENS_V_PER_N / 1000:.2f} kN/m")
    print(f"  c_fisico = c_v / sens = {c.item() / SENS_V_PER_N:.2f} N·s/m")
    
    # Visualizar
    plot_results(history, model, F, a, v, x, stats)
    
    return model, history


if __name__ == "__main__":
    model, history = main()
