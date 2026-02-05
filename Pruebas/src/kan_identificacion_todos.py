#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAN-PINN IDENTIFICACIÓN CON TODOS LOS DATOS DISPONIBLES
========================================================
Combina datos de DYMH-105, LC302-1K y datos de hoy (4 Dic)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Dispositivo: {device}")

PHYS_BOUNDS_LOG = {
    'log_m': (np.log(1e-6), np.log(1.0)),
    'log_k': (np.log(1e-3), np.log(1e3)),
    'log_c': (np.log(1e-6), np.log(1.0)),
}

CONFIG = {
    "epochs_phase1": 150,
    "epochs_phase2": 350,
    "batch_size": 1024,
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
        
        self.log_m = nn.Parameter(torch.tensor(np.log(0.01)))
        self.log_k = nn.Parameter(torch.tensor(np.log(100.0)))
        self.log_c = nn.Parameter(torch.tensor(np.log(0.1)))
        self.Fc = nn.Parameter(torch.tensor(0.001))
        
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
        return torch.exp(self.log_m), torch.exp(self.log_k), torch.exp(self.log_c), torch.abs(self.Fc)
    
    def forward(self, a, v, x):
        m, k, c, Fc = self.get_phys_params()
        
        F_lin = m * a + k * x + c * v + Fc * torch.sign(v + 1e-10)
        
        features = torch.stack([
            a / (self.stats.get('a_std', 1.0) + 1e-8),
            v / (self.stats.get('v_std', 1.0) + 1e-8),
            x / (self.stats.get('x_std', 1.0) + 1e-8),
            torch.sign(v + 1e-10),
            torch.abs(v) / (self.stats.get('v_std', 1.0) + 1e-8),
        ], dim=-1)
        
        F_res = self.residual_net(features).squeeze(-1) * self.stats.get('F_std', 1.0) * 0.1
        return F_lin + F_res, F_res
    
    def clip_params(self):
        with torch.no_grad():
            self.log_m.data.clamp_(PHYS_BOUNDS_LOG['log_m'][0], PHYS_BOUNDS_LOG['log_m'][1])
            self.log_k.data.clamp_(PHYS_BOUNDS_LOG['log_k'][0], PHYS_BOUNDS_LOG['log_k'][1])
            self.log_c.data.clamp_(PHYS_BOUNDS_LOG['log_c'][0], PHYS_BOUNDS_LOG['log_c'][1])
            self.Fc.data.clamp_(0.0, 0.01)


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def procesar_archivo(archivo):
    """Procesa un archivo CSV y retorna F, a, v, x."""
    nombre = os.path.basename(archivo)
    try:
        freq = float(nombre.split('_')[-1].replace('Hz.csv', ''))
        if freq < 5 or freq > 100:
            return None
    except:
        return None
    
    df = pd.read_csv(archivo)
    t = df['tiempo_s'].values
    fuerza_V = df['fuerza_V'].values
    
    # Detectar aceleración
    if 'aceleracion_sensor_g' in df.columns:
        acel_g = df['aceleracion_sensor_g'].values
    elif 'aceleracion_g' in df.columns:
        acel_g = df['aceleracion_g'].values
    else:
        acel_cols = [c for c in df.columns if 'acel' in c.lower()]
        if acel_cols:
            acel_g = df[acel_cols[0]].values
        else:
            return None
    
    # Descartar transitorio
    skip = int(0.5 * SAMPLE_RATE)
    if len(t) <= skip * 2:
        return None
    
    t = t[skip:] - t[skip]
    fuerza_V = fuerza_V[skip:]
    acel_g = acel_g[skip:]
    
    # Convertir
    acel_ms2 = acel_g * G
    F_ac = fuerza_V - np.mean(fuerza_V)
    a_ac = acel_ms2 - np.mean(acel_ms2)
    
    # Integrar
    dt = 1 / SAMPLE_RATE
    fc = max(freq / 10, 0.5)
    b, a_filt = signal.butter(2, fc / (SAMPLE_RATE / 2), btype='high')
    
    v = cumulative_trapezoid(a_ac, dx=dt, initial=0)
    v = signal.filtfilt(b, a_filt, v)
    x = cumulative_trapezoid(v, dx=dt, initial=0)
    x = signal.filtfilt(b, a_filt, x)
    
    # Submuestrear
    step = 10
    return {
        'F': F_ac[::step], 'a': a_ac[::step], 
        'v': v[::step], 'x': x[::step], 'freq': freq
    }


def cargar_todos_los_datos():
    """Carga todos los datos disponibles."""
    print("\n📂 CARGANDO TODOS LOS DATOS DISPONIBLES")
    print("=" * 60)
    
    # Patrones de archivos
    patrones = [
        ("DYMH-105 (2 Dic)", "caracterizacion_fuerza_20251202_*_exp*.csv"),
        ("LC302-1K (3 Dic)", "caracterizacion_fuerza_20251203_160853_exp*.csv"),
        ("LC302-1K (4 Dic)", "caracterizacion_fuerza_20251204_110740_exp*.csv"),
    ]
    
    all_F, all_a, all_v, all_x = [], [], [], []
    total_archivos = 0
    freqs_set = set()
    
    for nombre, patron in patrones:
        archivos = sorted(glob.glob(os.path.join(DATOS_DIR, patron)))
        # Excluir 10Hz de LC302 (anómalo)
        if "LC302" in nombre:
            archivos = [a for a in archivos if "_10Hz.csv" not in a]
        
        count = 0
        for archivo in archivos:
            data = procesar_archivo(archivo)
            if data is not None:
                all_F.append(data['F'])
                all_a.append(data['a'])
                all_v.append(data['v'])
                all_x.append(data['x'])
                freqs_set.add(data['freq'])
                count += 1
        
        print(f"  {nombre}: {count} archivos")
        total_archivos += count
    
    # Concatenar
    F = np.concatenate(all_F)
    a = np.concatenate(all_a)
    v = np.concatenate(all_v)
    x = np.concatenate(all_x)
    
    stats = {
        'F_std': np.std(F), 'a_std': np.std(a),
        'v_std': np.std(v), 'x_std': np.std(x),
    }
    
    print(f"\n📊 TOTAL: {total_archivos} archivos, {len(F):,} puntos")
    print(f"   Frecuencias: {sorted(freqs_set)}")
    print(f"   F_std: {stats['F_std']*1000:.2f} mV")
    print(f"   a_std: {stats['a_std']:.3f} m/s²")
    print(f"   v_std: {stats['v_std']*1000:.2f} mm/s")
    print(f"   x_std: {stats['x_std']*1e6:.1f} µm")
    
    return (
        torch.tensor(F, dtype=torch.float32, device=device),
        torch.tensor(a, dtype=torch.float32, device=device),
        torch.tensor(v, dtype=torch.float32, device=device),
        torch.tensor(x, dtype=torch.float32, device=device),
        stats
    )


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def train_model(model, F, a, v, x):
    """Entrena con curriculum learning."""
    
    epochs_p1 = CONFIG["epochs_phase1"]
    epochs_p2 = CONFIG["epochs_phase2"]
    total = epochs_p1 + epochs_p2
    
    net_params = [p for n, p in model.named_parameters() if 'log_' not in n and n != 'Fc']
    phys_params = [model.log_m, model.log_k, model.log_c, model.Fc]
    
    opt_net = optim.Adam(net_params, lr=CONFIG["lr"])
    opt_phys = optim.Adam(phys_params, lr=CONFIG["lr_phys"])
    
    sched_net = optim.lr_scheduler.CosineAnnealingLR(opt_net, total)
    sched_phys = optim.lr_scheduler.CosineAnnealingLR(opt_phys, total)
    
    history = {"epoch": [], "loss": [], "m": [], "k": [], "c": [], "Fc": [], "fn": []}
    
    print(f"\n{'='*60}")
    print("ENTRENAMIENTO CON CURRICULUM LEARNING")
    print(f"{'='*60}")
    print(f"Fase 1: Épocas 1-{epochs_p1} (solo datos)")
    print(f"Fase 2: Épocas {epochs_p1+1}-{total} (datos + física)")
    print(f"{'='*60}\n")
    
    start = time.time()
    N = len(F)
    
    for epoch in range(1, total + 1):
        model.train()
        
        idx = torch.randperm(N, device=device)[:CONFIG["batch_size"]]
        F_b, a_b, v_b, x_b = F[idx], a[idx], v[idx], x[idx]
        
        opt_net.zero_grad()
        opt_phys.zero_grad()
        
        F_pred, F_res = model(a_b, v_b, x_b)
        loss_data = torch.mean((F_b - F_pred) ** 2)
        
        if epoch <= epochs_p1:
            loss_phys = torch.tensor(0.0, device=device)
            lam = 0.0
        else:
            prog = (epoch - epochs_p1) / epochs_p2
            lam = prog * CONFIG["lambda_phys_final"]
            loss_phys = torch.mean(F_res ** 2)
        
        loss = loss_data + lam * loss_phys
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_net.step()
        opt_phys.step()
        sched_net.step()
        sched_phys.step()
        model.clip_params()
        
        m, k, c, Fc = model.get_phys_params()
        fn = np.sqrt(k.item() / (m.item() + 1e-10)) / (2 * np.pi)
        zeta = c.item() / (2 * np.sqrt(k.item() * m.item() + 1e-10))
        
        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["m"].append(m.item())
        history["k"].append(k.item())
        history["c"].append(c.item())
        history["Fc"].append(Fc.item())
        history["fn"].append(fn)
        
        if epoch % 50 == 0 or epoch == 1:
            phase = "DATOS" if epoch <= epochs_p1 else "FÍSICA"
            print(f"[{phase}] Época {epoch:3d}/{total} | Loss: {loss.item():.2e} | "
                  f"fn={fn:.1f}Hz | ζ={zeta:.3f} | λ={lam:.2e}")
    
    print(f"\n✅ Completado en {time.time() - start:.1f}s")
    return history


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("KAN-PINN: IDENTIFICACIÓN CON TODOS LOS DATOS")
    print("=" * 60)
    
    # Cargar datos
    F, a, v, x, stats = cargar_todos_los_datos()
    
    # Crear modelo
    model = FriccionKAN(hidden=64, depth=3, stats=stats).to(device)
    print(f"\n🧠 Modelo: {sum(p.numel() for p in model.parameters()):,} parámetros")
    
    # Entrenar
    history = train_model(model, F, a, v, x)
    
    # Resultados
    m, k, c, Fc = model.get_phys_params()
    fn = np.sqrt(k.item() / m.item()) / (2 * np.pi)
    zeta = c.item() / (2 * np.sqrt(k.item() * m.item()))
    
    print("\n" + "=" * 60)
    print("PARÁMETROS IDENTIFICADOS (TODOS LOS DATOS)")
    print("=" * 60)
    print(f"  m_v = {m.item()*1e6:.2f} µV·s²/m")
    print(f"  k_v = {k.item():.2f} V/m")
    print(f"  c_v = {c.item()*1e3:.2f} mV·s/m")
    print(f"  Fc  = {Fc.item()*1e3:.4f} mV")
    print(f"\n  Frecuencia natural: fn = {fn:.1f} Hz")
    print(f"  Factor amortiguamiento: ζ = {zeta:.4f} ({zeta*100:.2f}%)")
    print("=" * 60)
    
    # Gráficas
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'KAN-PINN: Identificación con TODOS los datos\n'
                 f'fn = {fn:.1f} Hz | ζ = {zeta*100:.2f}%', fontsize=14, fontweight='bold')
    
    epochs = history["epoch"]
    
    axes[0,0].semilogy(epochs, history["loss"], 'b-', lw=2)
    axes[0,0].axvline(CONFIG["epochs_phase1"], color='r', ls='--', alpha=0.5, label='Inicio física')
    axes[0,0].set_xlabel('Época'); axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Convergencia'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].semilogy(epochs, [m*1e6 for m in history["m"]], 'g-', lw=2)
    axes[0,1].set_xlabel('Época'); axes[0,1].set_ylabel('m_v (µV·s²/m)')
    axes[0,1].set_title(f'm_v final: {m.item()*1e6:.2f} µV·s²/m'); axes[0,1].grid(True, alpha=0.3)
    
    axes[0,2].semilogy(epochs, history["k"], 'r-', lw=2)
    axes[0,2].set_xlabel('Época'); axes[0,2].set_ylabel('k_v (V/m)')
    axes[0,2].set_title(f'k_v final: {k.item():.2f} V/m'); axes[0,2].grid(True, alpha=0.3)
    
    axes[1,0].semilogy(epochs, [c*1e3 for c in history["c"]], 'purple', lw=2)
    axes[1,0].set_xlabel('Época'); axes[1,0].set_ylabel('c_v (mV·s/m)')
    axes[1,0].set_title(f'c_v final: {c.item()*1e3:.2f} mV·s/m'); axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(epochs, history["fn"], 'orange', lw=2)
    axes[1,1].axhline(fn, color='r', ls='--', alpha=0.5)
    axes[1,1].set_xlabel('Época'); axes[1,1].set_ylabel('fn (Hz)')
    axes[1,1].set_title(f'Frecuencia Natural: {fn:.1f} Hz'); axes[1,1].grid(True, alpha=0.3)
    
    # Predicción vs Real
    model.eval()
    with torch.no_grad():
        n_show = 2000
        F_pred, _ = model(a[:n_show], v[:n_show], x[:n_show])
        F_real = F[:n_show].cpu().numpy() * 1000
        F_pred = F_pred.cpu().numpy() * 1000
    
    axes[1,2].plot(F_real, 'c-', lw=0.5, alpha=0.7, label='Real')
    axes[1,2].plot(F_pred, 'orange', lw=0.5, alpha=0.7, label='Predicho')
    axes[1,2].set_xlabel('Muestra'); axes[1,2].set_ylabel('Fuerza (mV)')
    axes[1,2].set_title('Predicción vs Real'); axes[1,2].legend(); axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output = os.path.join(DATOS_DIR, "kan_identificacion_todos.png")
    fig.savefig(output, dpi=150, facecolor='white')
    print(f"\n📊 Guardado: {output}")
    plt.show()
    
    return model, history


if __name__ == "__main__":
    model, history = main()
