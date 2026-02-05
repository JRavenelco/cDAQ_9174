#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAN-PINN COMPARACIÓN DYMH-105 vs LC302-1K
==========================================
Identifica parámetros m, k, c para ambos sensores y compara.
"""

import os
import time
import glob
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

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Dispositivo: {device}")

# Bounds para parámetros (unidades de voltaje)
PHYS_BOUNDS = {
    'm': (1e-6, 1.0),
    'k': (1e-3, 1e3),
    'c': (1e-6, 1.0),
    'Fc': (0.0, 0.01),
}

PHYS_BOUNDS_LOG = {
    'log_m': (np.log(PHYS_BOUNDS['m'][0]), np.log(PHYS_BOUNDS['m'][1])),
    'log_k': (np.log(PHYS_BOUNDS['k'][0]), np.log(PHYS_BOUNDS['k'][1])),
    'log_c': (np.log(PHYS_BOUNDS['c'][0]), np.log(PHYS_BOUNDS['c'][1])),
}

CONFIG = {
    "epochs_phase1": 100,
    "epochs_phase2": 200,
    "batch_size": 512,
    "lr": 1e-3,
    "lr_phys": 1e-2,
    "lambda_phys_final": 0.1,
}

# =============================================================================
# MODELO
# =============================================================================

class FriccionKAN(nn.Module):
    def __init__(self, hidden=64, depth=3, stats=None):
        super().__init__()
        self.stats = stats or {}
        
        # Parámetros físicos
        self.log_m = nn.Parameter(torch.tensor(np.log(0.01)))
        self.log_k = nn.Parameter(torch.tensor(np.log(100.0)))
        self.log_c = nn.Parameter(torch.tensor(np.log(0.1)))
        self.Fc = nn.Parameter(torch.tensor(0.001))
        
        # Red residual
        layers = []
        in_dim = 5
        for i in range(depth):
            out_dim = hidden if i < depth - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < depth - 1:
                layers.append(nn.Tanh())
            in_dim = hidden
        self.residual_net = nn.Sequential(*layers)
        
        for m in self.residual_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def get_phys_params(self):
        m = torch.exp(self.log_m)
        k = torch.exp(self.log_k)
        c = torch.exp(self.log_c)
        Fc = torch.abs(self.Fc)
        return m, k, c, Fc
    
    def forward(self, a, v, x):
        m, k, c, Fc = self.get_phys_params()
        
        F_inertial = m * a
        F_elastic = k * x
        F_viscous = c * v
        F_coulomb = Fc * torch.sign(v + 1e-10)
        
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
        with torch.no_grad():
            self.log_m.data.clamp_(PHYS_BOUNDS_LOG['log_m'][0], PHYS_BOUNDS_LOG['log_m'][1])
            self.log_k.data.clamp_(PHYS_BOUNDS_LOG['log_k'][0], PHYS_BOUNDS_LOG['log_k'][1])
            self.log_c.data.clamp_(PHYS_BOUNDS_LOG['log_c'][0], PHYS_BOUNDS_LOG['log_c'][1])
            self.Fc.data.clamp_(PHYS_BOUNDS['Fc'][0], PHYS_BOUNDS['Fc'][1])


# =============================================================================
# FUNCIONES
# =============================================================================

def cargar_datos_sensor(archivos, sensor_name):
    """Carga datos de un sensor."""
    print(f"\n📂 Cargando {sensor_name}...")
    
    all_F, all_a, all_v, all_x = [], [], [], []
    freqs_usadas = set()
    
    for archivo in archivos:
        nombre = os.path.basename(archivo)
        try:
            freq = float(nombre.split('_')[-1].replace('Hz.csv', ''))
            if freq < 5 or freq > 100:
                continue
        except:
            continue
        
        df = pd.read_csv(archivo)
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
        
        # Convertir
        acel_ms2 = acel_g * G
        F_ac = fuerza_V - np.mean(fuerza_V)
        a_ac = acel_ms2 - np.mean(acel_ms2)
        
        # Integrar para v y x
        dt = 1 / SAMPLE_RATE
        fc = max(freq / 10, 0.5)
        b, a_filt = signal.butter(2, fc / (SAMPLE_RATE / 2), btype='high')
        
        v = cumulative_trapezoid(a_ac, dx=dt, initial=0)
        v = signal.filtfilt(b, a_filt, v)
        x = cumulative_trapezoid(v, dx=dt, initial=0)
        x = signal.filtfilt(b, a_filt, x)
        
        # Submuestrear
        step = 10
        all_F.append(F_ac[::step])
        all_a.append(a_ac[::step])
        all_v.append(v[::step])
        all_x.append(x[::step])
        freqs_usadas.add(freq)
    
    if not all_F:
        return None, None, None, None, None
    
    F = np.concatenate(all_F)
    a = np.concatenate(all_a)
    v = np.concatenate(all_v)
    x = np.concatenate(all_x)
    
    stats = {
        'F_std': np.std(F), 'a_std': np.std(a),
        'v_std': np.std(v), 'x_std': np.std(x),
    }
    
    print(f"   {len(freqs_usadas)} frecuencias: {sorted(freqs_usadas)}")
    print(f"   {len(F)} puntos | F_std={stats['F_std']*1000:.2f}mV | a_std={stats['a_std']:.3f}m/s²")
    
    F_gpu = torch.tensor(F, dtype=torch.float32, device=device)
    a_gpu = torch.tensor(a, dtype=torch.float32, device=device)
    v_gpu = torch.tensor(v, dtype=torch.float32, device=device)
    x_gpu = torch.tensor(x, dtype=torch.float32, device=device)
    
    return F_gpu, a_gpu, v_gpu, x_gpu, stats


def train_model(model, F, a, v, x, sensor_name):
    """Entrena modelo con curriculum learning."""
    
    epochs_phase1 = CONFIG["epochs_phase1"]
    epochs_phase2 = CONFIG["epochs_phase2"]
    total_epochs = epochs_phase1 + epochs_phase2
    
    net_params = [p for n, p in model.named_parameters() if 'log_' not in n and n != 'Fc']
    phys_params = [model.log_m, model.log_k, model.log_c, model.Fc]
    
    optimizer_net = optim.Adam(net_params, lr=CONFIG["lr"])
    optimizer_phys = optim.Adam(phys_params, lr=CONFIG["lr_phys"])
    
    history = {"epoch": [], "loss": [], "m": [], "k": [], "c": [], "Fc": [], "fn": []}
    
    print(f"\n🏋️ Entrenando {sensor_name}...")
    start_time = time.time()
    N = len(F)
    
    for epoch in range(1, total_epochs + 1):
        model.train()
        
        idx = torch.randperm(N, device=device)[:CONFIG["batch_size"]]
        F_batch, a_batch, v_batch, x_batch = F[idx], a[idx], v[idx], x[idx]
        
        optimizer_net.zero_grad()
        optimizer_phys.zero_grad()
        
        F_pred, F_residual = model(a_batch, v_batch, x_batch)
        loss_data = torch.mean((F_batch - F_pred) ** 2)
        
        if epoch <= epochs_phase1:
            loss_phys = torch.tensor(0.0, device=device)
            lambda_phys = 0.0
        else:
            progress = (epoch - epochs_phase1) / epochs_phase2
            lambda_phys = progress * CONFIG["lambda_phys_final"]
            loss_phys = torch.mean(F_residual ** 2)
        
        loss = loss_data + lambda_phys * loss_phys
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_net.step()
        optimizer_phys.step()
        model.clip_params()
        
        m, k, c, Fc = model.get_phys_params()
        fn = np.sqrt(k.item() / (m.item() + 1e-10)) / (2 * np.pi)
        
        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["m"].append(m.item())
        history["k"].append(k.item())
        history["c"].append(c.item())
        history["Fc"].append(Fc.item())
        history["fn"].append(fn)
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"   Época {epoch:3d}/{total_epochs} | Loss: {loss.item():.2e} | fn={fn:.1f}Hz")
    
    print(f"   ✅ Completado en {time.time() - start_time:.1f}s")
    return history


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("KAN-PINN COMPARACIÓN: DYMH-105 vs LC302-1K")
    print("=" * 80)
    
    # Buscar archivos
    archivos_dymh = sorted(glob.glob(os.path.join(DATOS_DIR, "caracterizacion_fuerza_20251202_135910_exp*.csv")))
    archivos_lc302 = sorted(glob.glob(os.path.join(DATOS_DIR, "caracterizacion_fuerza_20251203_160853_exp*.csv")))
    
    # Excluir 10 Hz de LC302 (anómalo)
    archivos_lc302 = [a for a in archivos_lc302 if "_10Hz.csv" not in a]
    
    resultados = {}
    
    # DYMH-105
    if archivos_dymh:
        F, a, v, x, stats = cargar_datos_sensor(archivos_dymh, "DYMH-105")
        if F is not None:
            model = FriccionKAN(hidden=64, depth=3, stats=stats).to(device)
            history = train_model(model, F, a, v, x, "DYMH-105")
            m, k, c, Fc = model.get_phys_params()
            resultados['DYMH-105'] = {
                'history': history,
                'm': m.item(), 'k': k.item(), 'c': c.item(), 'Fc': Fc.item(),
                'fn': history['fn'][-1],
            }
    
    # LC302-1K
    if archivos_lc302:
        F, a, v, x, stats = cargar_datos_sensor(archivos_lc302, "LC302-1K")
        if F is not None:
            model = FriccionKAN(hidden=64, depth=3, stats=stats).to(device)
            history = train_model(model, F, a, v, x, "LC302-1K")
            m, k, c, Fc = model.get_phys_params()
            resultados['LC302-1K'] = {
                'history': history,
                'm': m.item(), 'k': k.item(), 'c': c.item(), 'Fc': Fc.item(),
                'fn': history['fn'][-1],
            }
    
    # ==========================================================================
    # COMPARACIÓN
    # ==========================================================================
    print("\n" + "=" * 80)
    print("COMPARACIÓN DE PARÁMETROS IDENTIFICADOS")
    print("=" * 80)
    
    print(f"\n{'Parámetro':<20} {'DYMH-105':>15} {'LC302-1K':>15} {'Diferencia':>15}")
    print("-" * 70)
    
    if 'DYMH-105' in resultados and 'LC302-1K' in resultados:
        r1 = resultados['DYMH-105']
        r2 = resultados['LC302-1K']
        
        params = [
            ('m_v (V·s²/m)', 'm', 1e6, 'µV·s²/m'),
            ('k_v (V/m)', 'k', 1, 'V/m'),
            ('c_v (V·s/m)', 'c', 1e3, 'mV·s/m'),
            ('Fc (mV)', 'Fc', 1e3, 'mV'),
            ('fn (Hz)', 'fn', 1, 'Hz'),
        ]
        
        for nombre, key, scale, unit in params:
            v1 = r1[key] * scale
            v2 = r2[key] * scale
            diff = ((v2 - v1) / (v1 + 1e-10)) * 100
            print(f"{nombre:<20} {v1:>15.4f} {v2:>15.4f} {diff:>14.1f}%")
        
        # Factor de amortiguamiento
        zeta1 = r1['c'] / (2 * np.sqrt(r1['k'] * r1['m']))
        zeta2 = r2['c'] / (2 * np.sqrt(r2['k'] * r2['m']))
        print(f"{'ζ (amortiguamiento)':<20} {zeta1:>15.4f} {zeta2:>15.4f} {((zeta2-zeta1)/zeta1)*100:>14.1f}%")
    
    print("-" * 70)
    
    # ==========================================================================
    # GRÁFICAS
    # ==========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comparación KAN-PINN: DYMH-105 vs LC302-1K', fontsize=14, fontweight='bold')
    
    colors = {'DYMH-105': 'cyan', 'LC302-1K': 'orange'}
    
    for sensor, res in resultados.items():
        h = res['history']
        color = colors[sensor]
        
        # Loss
        axes[0, 0].semilogy(h['epoch'], h['loss'], color=color, label=sensor, linewidth=2)
        
        # m_v
        axes[0, 1].semilogy(h['epoch'], h['m'], color=color, label=sensor, linewidth=2)
        
        # k_v
        axes[0, 2].semilogy(h['epoch'], h['k'], color=color, label=sensor, linewidth=2)
        
        # c_v
        axes[1, 0].semilogy(h['epoch'], h['c'], color=color, label=sensor, linewidth=2)
        
        # fn
        axes[1, 1].plot(h['epoch'], h['fn'], color=color, label=sensor, linewidth=2)
    
    axes[0, 0].set_xlabel('Época'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Convergencia'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Época'); axes[0, 1].set_ylabel('m_v (V·s²/m)')
    axes[0, 1].set_title('Masa Efectiva'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_xlabel('Época'); axes[0, 2].set_ylabel('k_v (V/m)')
    axes[0, 2].set_title('Rigidez'); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Época'); axes[1, 0].set_ylabel('c_v (V·s/m)')
    axes[1, 0].set_title('Amortiguamiento'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Época'); axes[1, 1].set_ylabel('fn (Hz)')
    axes[1, 1].set_title('Frecuencia Natural'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    
    # Barras comparativas
    ax = axes[1, 2]
    if len(resultados) == 2:
        x_pos = np.arange(4)
        width = 0.35
        
        vals1 = [resultados['DYMH-105']['m']*1e6, resultados['DYMH-105']['k'], 
                 resultados['DYMH-105']['c']*1e3, resultados['DYMH-105']['fn']]
        vals2 = [resultados['LC302-1K']['m']*1e6, resultados['LC302-1K']['k'],
                 resultados['LC302-1K']['c']*1e3, resultados['LC302-1K']['fn']]
        
        ax.bar(x_pos - width/2, vals1, width, label='DYMH-105', color='cyan')
        ax.bar(x_pos + width/2, vals2, width, label='LC302-1K', color='orange')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['m_v\n(µV·s²/m)', 'k_v\n(V/m)', 'c_v\n(mV·s/m)', 'fn\n(Hz)'])
        ax.set_title('Parámetros Finales')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = os.path.join(DATOS_DIR, "kan_comparacion_sensores.png")
    fig.savefig(output_file, dpi=150, facecolor='white')
    print(f"\n📊 Guardado: {output_file}")
    plt.show()
    
    return resultados


if __name__ == "__main__":
    resultados = main()
