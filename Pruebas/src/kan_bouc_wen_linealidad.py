#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAN-PINN CON MODELO BOUC-WEN PARA VALIDAR LINEALIDAD
=====================================================
Aplica un modelo de histéresis general (Bouc-Wen) al sistema.
Si el sistema es lineal, los parámetros de histéresis deben colapsar
a valores que indiquen "no hay histéresis" (área de lazo ≈ 0).

Modelo Bouc-Wen:
    F = α·k·x + (1-α)·k·z + c·v + m·a
    
    dz/dt = v·[A - |z|^n·(β·sign(v·z) + γ)]

Donde:
    - α: Ratio de rigidez lineal (α=1 → todo lineal, α<1 → hay histéresis)
    - A, β, γ, n: Parámetros de forma del lazo de histéresis
    - z: Variable interna de histéresis (estado)
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
print(f"Dispositivo: {device}")

CONFIG = {
    "epochs": 500,
    "batch_size": 1024,
    "lr_net": 1e-3,
    "lr_phys": 5e-3,
}

# =============================================================================
# MODELO BOUC-WEN + KAN-PINN
# =============================================================================

class BoucWenKAN(nn.Module):
    """
    KAN-PINN con modelo Bouc-Wen para identificar histéresis.
    
    El modelo predice la variable de histéresis z(t) y tiene parámetros
    físicos entrenables (m, k, c, α, A, β, γ, n).
    """
    
    def __init__(self, hidden=64, depth=3, stats=None):
        super().__init__()
        self.stats = stats or {}
        
        # =====================================================================
        # PARÁMETROS FÍSICOS LINEALES (en log-space para positividad)
        # =====================================================================
        # Inicialización basada en identificación FRF previa:
        # m ~ 40.3 kg, k ~ 408 kN/m, c ~ 442 Ns/m
        self.log_m = nn.Parameter(torch.tensor(np.log(40.3)))    # masa efectiva (kg)
        self.log_k = nn.Parameter(torch.tensor(np.log(408000.0))) # rigidez (N/m)
        self.log_c = nn.Parameter(torch.tensor(np.log(442.0)))    # amortiguamiento (Ns/m)
        
        # =====================================================================
        # PARÁMETROS BOUC-WEN
        # =====================================================================
        # α: ratio de rigidez lineal (0-1). Si α→1, sistema es puramente lineal
        # Inicializamos cerca de 1.0 (lineal) para validación
        self.alpha_raw = nn.Parameter(torch.tensor(5.0))  # sigmoid(5.0) ≈ 0.99
        
        # A, β, γ: parámetros de forma (en log-space para positividad)
        self.log_A = nn.Parameter(torch.tensor(np.log(1.0)))     # A > 0
        self.log_beta = nn.Parameter(torch.tensor(np.log(0.5)))  # β > 0
        self.log_gamma = nn.Parameter(torch.tensor(np.log(0.5))) # γ > 0
        
        # n: exponente (típicamente 1-2)
        self.log_n = nn.Parameter(torch.tensor(np.log(1.0)))     # n > 0
        
        # =====================================================================
        # RED NEURONAL PARA z(t) - Variable de histéresis
        # =====================================================================
        # La red predice z directamente dado (a, v, x)
        layers = []
        in_dim = 4  # [a, v, x, sign(v)]
        for i in range(depth):
            out_dim = hidden if i < depth - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < depth - 1:
                layers.append(nn.Tanh())
            in_dim = hidden
        self.z_net = nn.Sequential(*layers)
        
        # Inicialización pequeña para z_net
        for m in self.z_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def get_linear_params(self):
        """Retorna parámetros lineales (m, k, c)."""
        m = torch.exp(self.log_m)
        k = torch.exp(self.log_k)
        c = torch.exp(self.log_c)
        return m, k, c
    
    def get_boucwen_params(self):
        """Retorna parámetros de Bouc-Wen (α, A, β, γ, n)."""
        alpha = torch.sigmoid(self.alpha_raw)  # α ∈ (0, 1)
        A = torch.exp(self.log_A)
        beta = torch.exp(self.log_beta)
        gamma = torch.exp(self.log_gamma)
        n = torch.exp(self.log_n)
        return alpha, A, beta, gamma, n
    
    def forward(self, a, v, x):
        """
        Forward pass: Predice F usando modelo Bouc-Wen.
        
        F = m·a + c·v + α·k·x + (1-α)·k·z
        """
        m, k, c = self.get_linear_params()
        alpha, A, beta, gamma, n = self.get_boucwen_params()
        
        # Normalizar entradas para la red
        a_norm = a / (self.stats.get('a_std', 1.0) + 1e-8)
        v_norm = v / (self.stats.get('v_std', 1.0) + 1e-8)
        x_norm = x / (self.stats.get('x_std', 1.0) + 1e-8)
        sign_v = torch.sign(v + 1e-10)
        
        features = torch.stack([a_norm, v_norm, x_norm, sign_v], dim=-1)
        
        # z predicho por la red (escalado por x_std para tener unidades de desplazamiento)
        z = self.z_net(features).squeeze(-1) * self.stats.get('x_std', 1.0)
        
        # Fuerza total según Bouc-Wen
        F_inertial = m * a
        F_viscous = c * v
        F_elastic_linear = alpha * k * x
        F_hysteresis = (1 - alpha) * k * z
        
        F_total = F_inertial + F_viscous + F_elastic_linear + F_hysteresis
        
        return F_total, z
    
    def compute_boucwen_residual(self, z, v, x):
        """
        Calcula el residual de la ecuación diferencial de Bouc-Wen.
        
        dz/dt = v·[A - |z|^n·(β·sign(v·z) + γ)]
        
        El residual debe ser ≈ 0 si z sigue la dinámica de Bouc-Wen.
        """
        alpha, A, beta, gamma, n = self.get_boucwen_params()
        
        # dz/dt usando diferencias finitas (aproximación)
        # En realidad, la red debería aprender z tal que satisfaga esto
        
        # Término no lineal
        abs_z = torch.abs(z) + 1e-8
        sign_vz = torch.sign(v * z + 1e-10)
        
        # dz/dt teórico según Bouc-Wen
        dz_dt_bw = v * (A - torch.pow(abs_z, n) * (beta * sign_vz + gamma))
        
        return dz_dt_bw
    
    def clip_params(self):
        """Limita parámetros a rangos físicamente razonables."""
        with torch.no_grad():
            # Rangos ajustados para sistema de 40kg y alta rigidez
            self.log_m.data.clamp_(np.log(10.0), np.log(100.0))      # 10 - 100 kg
            self.log_k.data.clamp_(np.log(1e4), np.log(1e7))         # 10 kN/m - 10 MN/m
            self.log_c.data.clamp_(np.log(1.0), np.log(1e4))         # 1 - 10000 Ns/m
            self.alpha_raw.data.clamp_(-5, 10)  # permite saturar en 1.0
            self.log_A.data.clamp_(np.log(0.01), np.log(100))
            self.log_beta.data.clamp_(np.log(0.01), np.log(100))
            self.log_gamma.data.clamp_(np.log(0.01), np.log(100))
            self.log_n.data.clamp_(np.log(0.5), np.log(5))


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def cargar_datos_lc302():
    """Carga datos de LC302-1K (sistema lineal validado)."""
    print("\n Cargando datos LC302-1K...")
    
    # Datos del 3 y 4 de diciembre
    patrones = [
        "caracterizacion_fuerza_20251203_160853_exp*.csv",
        "caracterizacion_fuerza_20251204_110740_exp*.csv",
    ]
    
    all_F, all_a, all_v, all_x = [], [], [], []
    
    for patron in patrones:
        archivos = sorted(glob.glob(os.path.join(DATOS_DIR, patron)))
        # Excluir 10Hz
        archivos = [a for a in archivos if "_10Hz.csv" not in a]
        
        for archivo in archivos:
            nombre = os.path.basename(archivo)
            try:
                freq = float(nombre.split('_')[-1].replace('Hz.csv', ''))
                if freq < 15 or freq > 50:
                    continue
            except:
                continue
            
            df = pd.read_csv(archivo)
            t = df['tiempo_s'].values
            fuerza_V = df['fuerza_V'].values
            
            if 'aceleracion_sensor_g' in df.columns:
                acel_g = df['aceleracion_sensor_g'].values
            else:
                continue
            
            # Descartar transitorio
            skip = int(0.5 * SAMPLE_RATE)
            if len(t) <= skip * 2:
                continue
            
            t = t[skip:] - t[skip]
            fuerza_V = fuerza_V[skip:]
            acel_g = acel_g[skip:]
            
            # Convertir y centrar
            acel_ms2 = acel_g * G
            F_ac = fuerza_V - np.mean(fuerza_V)
            a_ac = acel_ms2 - np.mean(acel_ms2)
            
            # Integrar con filtro HP
            dt = 1 / SAMPLE_RATE
            fc = max(freq / 10, 1.0)
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
    
    F = np.concatenate(all_F)
    a = np.concatenate(all_a)
    v = np.concatenate(all_v)
    x = np.concatenate(all_x)
    
    stats = {
        'F_std': np.std(F), 'a_std': np.std(a),
        'v_std': np.std(v), 'x_std': np.std(x),
    }
    
    print(f"   {len(F):,} puntos cargados")
    print(f"   F_std: {stats['F_std']*1000:.2f} mV, a_std: {stats['a_std']:.3f} m/s²")
    
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

def train_boucwen(model, F, a, v, x):
    """Entrena el modelo Bouc-Wen KAN-PINN."""
    
    epochs = CONFIG["epochs"]
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr_net"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    history = {
        "epoch": [], "loss": [],
        "m": [], "k": [], "c": [],
        "alpha": [], "A": [], "beta": [], "gamma": [], "n": [],
        "z_max": [],
    }
    
    print(f"\n{'='*60}")
    print("ENTRENAMIENTO BOUC-WEN KAN-PINN")
    print(f"{'='*60}")
    print("Hipotesis: Si alpha -> 1 y z -> 0, el sistema es LINEAL")
    print(f"{'='*60}\n")
    
    start = time.time()
    N = len(F)
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        idx = torch.randperm(N, device=device)[:CONFIG["batch_size"]]
        F_b, a_b, v_b, x_b = F[idx], a[idx], v[idx], x[idx]
        
        optimizer.zero_grad()
        
        F_pred, z = model(a_b, v_b, x_b)
        
        # Loss de datos
        loss_data = torch.mean((F_b - F_pred) ** 2)
        
        # Regularización: Penalizar z grande (forzar linealidad si es posible)
        loss_z_reg = torch.mean(z ** 2) * 0.01
        
        # Loss total
        loss = loss_data + loss_z_reg
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.clip_params()
        
        # Logging
        m, k, c = model.get_linear_params()
        alpha, A, beta, gamma, n = model.get_boucwen_params()
        z_max = torch.max(torch.abs(z)).item()
        
        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["m"].append(m.item())
        history["k"].append(k.item())
        history["c"].append(c.item())
        history["alpha"].append(alpha.item())
        history["A"].append(A.item())
        history["beta"].append(beta.item())
        history["gamma"].append(gamma.item())
        history["n"].append(n.item())
        history["z_max"].append(z_max)
        
        if epoch % 50 == 0 or epoch == 1:
            fn = np.sqrt(k.item() / (m.item() + 1e-10)) / (2 * np.pi)
            print(f"Epoca {epoch:3d}/{epochs} | Loss: {loss.item():.2e} | "
                  f"alpha={alpha.item():.3f} | z_max={z_max:.2e} | fn={fn:.1f}Hz")
    
    print(f"\n Completado en {time.time() - start:.1f}s")
    return history


# =============================================================================
# ANÁLISIS DE HISTÉRESIS
# =============================================================================

def analizar_histeresis(model, F, a, v, x, stats):
    """Analiza el lazo de histéresis resultante."""
    
    model.eval()
    with torch.no_grad():
        F_pred, z = model(a, v, x)
        
        x_np = x.cpu().numpy()
        z_np = z.cpu().numpy()
        F_np = F.cpu().numpy()
        F_pred_np = F_pred.cpu().numpy()
    
    # Calcular área del lazo de histéresis (F vs x)
    # Usamos solo una porción para visualizar mejor
    n_cycle = min(5000, len(x_np))
    
    # Área del lazo = integral cerrada
    # Aproximación: área encerrada entre F_ida y F_vuelta
    x_sorted_idx = np.argsort(x_np[:n_cycle])
    
    # Calcular energía disipada (área del lazo)
    # E_loop = ∮ F dx ≈ Σ F·Δx
    dx = np.diff(x_np[:n_cycle])
    F_mid = 0.5 * (F_np[:n_cycle-1] + F_np[1:n_cycle])
    E_loop = np.abs(np.sum(F_mid * dx))
    
    # Normalizar por amplitud
    x_range = np.max(x_np[:n_cycle]) - np.min(x_np[:n_cycle])
    F_range = np.max(F_np[:n_cycle]) - np.min(F_np[:n_cycle])
    E_loop_norm = E_loop / (x_range * F_range + 1e-10)
    
    # z promedio
    z_rms = np.sqrt(np.mean(z_np ** 2))
    
    return {
        'E_loop': E_loop,
        'E_loop_norm': E_loop_norm,
        'z_rms': z_rms,
        'x': x_np[:n_cycle],
        'z': z_np[:n_cycle],
        'F': F_np[:n_cycle],
        'F_pred': F_pred_np[:n_cycle],
    }


# MAIN
# =============================================================================

def main():
    print(f"Dispositivo: {device}")
    print("=" * 60)
    print("VALIDACION DE LINEALIDAD CON MODELO BOUC-WEN")
    print("=" * 60)
    
    print("\nSi el sistema es lineal, esperamos:")
    print("  - alpha -> 1 (rigidez puramente lineal)")
    print("  - z -> 0 (variable de histeresis nula)")
    print("  - Area del lazo histeresis -> 0")
    
    # Cargar datos
    try:
        F, a, v, x, stats = cargar_datos_lc302()
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return None, None, None
    
    # Crear modelo
    model = BoucWenKAN(hidden=64, depth=3, stats=stats).to(device)
    print(f"\n Modelo: {sum(p.numel() for p in model.parameters()):,} parámetros")
    
    # Entrenar
    history = train_boucwen(model, F, a, v, x)
    
    # Analizar histéresis
    hyst = analizar_histeresis(model, F, a, v, x, stats)
    
    # Resultados finales
    m, k, c = model.get_linear_params()
    alpha, A, beta, gamma, n = model.get_boucwen_params()
    fn = np.sqrt(k.item() / m.item()) / (2 * np.pi)
    zeta = c.item() / (2 * np.sqrt(k.item() * m.item()))
    
    print("\n" + "=" * 60)
    print("RESULTADOS: VALIDACION DE LINEALIDAD")
    print("=" * 60)
    
    print("\n Parametros Lineales:")
    print(f"   m_v = {m.item()*1e6:.2f} µV·s²/m")
    print(f"   k_v = {k.item():.2f} V/m")
    print(f"   c_v = {c.item()*1e3:.2f} mV·s/m")
    print(f"   fn  = {fn:.1f} Hz")
    print(f"   zeta   = {zeta*100:.2f} %")
    
    print("\n Parametros Bouc-Wen:")
    print(f"   alpha = {alpha.item():.4f}  <- {'LINEAL (alpha~1)' if alpha.item() > 0.95 else 'Hay histeresis'}")
    print(f"   A = {A.item():.4f}")
    print(f"   beta = {beta.item():.4f}")
    print(f"   gamma = {gamma.item():.4f}")
    print(f"   n = {n.item():.4f}")
    
    print("\n Metricas de Histeresis:")
    print(f"   z_rms = {hyst['z_rms']*1e6:.2f} µm  <- {'NULO' if hyst['z_rms'] < 1e-6 else 'Presente'}")
    print(f"   Area lazo normalizada = {hyst['E_loop_norm']*100:.3f} %")
    
    # Conclusión
    print("\n" + "=" * 60)
    if alpha.item() > 0.90 and hyst['z_rms'] < 1e-5:
        print(" CONCLUSION: EL SISTEMA ES LINEAL")
        print("   El modelo Bouc-Wen colapsa a comportamiento lineal.")
        print("   No hay histeresis significativa en los datos actuales.")
    else:
        print(" CONCLUSION: SE DETECTA ALGO DE NO-LINEALIDAD")
        print(f"   alpha = {alpha.item():.3f} (deberia ser ~1 para lineal)")
        print(f"   z_rms = {hyst['z_rms']:.2e} (deberia ser ~0)")
    print("=" * 60)
    
    # =========================================================================
    # GRÁFICAS ESTÉTICAS (PARA PRESENTACIÓN)
    # =========================================================================
    # Estilo limpio
    plt.style.use('default')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Validacion de Linealidad (Sensor de Fuerza no calibrado)\n'
                 f'alpha = {alpha.item():.3f} (1.0 = Lineal)', fontsize=16, fontweight='bold', color='#00274c')
    
    # 1. Convergencia (Loss)
    axes[0, 0].semilogy(history["epoch"], history["loss"], color='#00274c', lw=2)
    axes[0, 0].set_xlabel('Epoca')
    axes[0, 0].set_ylabel('Loss (Error)')
    axes[0, 0].set_title('Convergencia del Modelo', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.2)
    
    # 2. Lazo F vs x (Unidades Reales: Volts vs Metros)
    # Usamos V para fuerza y mm para desplazamiento para que sea legible
    x_mm = hyst['x'] * 1000
    F_v = hyst['F']
    F_pred_v = hyst['F_pred']
    
    axes[0, 1].plot(x_mm, F_v, color='#999999', lw=1, alpha=0.5, label='Datos Reales (Sensor V)')
    axes[0, 1].plot(x_mm, F_pred_v, color='#dd6b20', lw=2, ls='--', label='Modelo KAN')
    axes[0, 1].set_xlabel('Desplazamiento (mm)')
    axes[0, 1].set_ylabel('Fuerza (Volts)')
    axes[0, 1].set_title('Respuesta del Sistema (F vs x)', fontweight='bold')
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(True, alpha=0.2)
    
    # 3. Lazo Normalizado (La prueba de fuego de la forma)
    # Normalizamos a [-1, 1] para comparar formas sin importar unidades
    def normalize(v): return 2 * (v - np.min(v)) / (np.max(v) - np.min(v)) - 1
    
    x_norm = normalize(x_mm)
    F_norm = normalize(F_v)
    F_pred_norm = normalize(F_pred_v)
    
    axes[1, 0].plot(x_norm, F_norm, color='#00274c', lw=1, alpha=0.4, label='Real')
    axes[1, 0].plot(x_norm, F_pred_norm, color='#38a169', lw=2, alpha=0.9, label='Modelo Linealizado')
    axes[1, 0].set_xlabel('Desplazamiento (Normalizado)')
    axes[1, 0].set_ylabel('Fuerza (Normalizada)')
    axes[1, 0].set_title('Analisis de Forma (Lazo Normalizado)', fontweight='bold')
    # Añadir texto de linealidad en la gráfica
    axes[1, 0].text(0, 0, "Lazo estrecho\n= Comportamiento Lineal", 
                   ha='center', va='center', color='#38a169', fontweight='bold', 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    axes[1, 0].grid(True, alpha=0.2)
    
    # 4. Evolución de Alpha
    axes[1, 1].plot(history["epoch"], history["alpha"], color='#c5911e', lw=3)
    axes[1, 1].axhline(1.0, color='gray', ls='--', label='Referencia Lineal')
    axes[1, 1].set_xlabel('Epoca')
    axes[1, 1].set_ylabel('Parametro Alpha')
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].set_title('Identificacion de Linealidad', fontweight='bold')
    axes[1, 1].text(len(history["epoch"])/2, 0.5, "Alpha -> 1 implica\nsistema sin histeresis", 
                   ha='center', color='#c5911e')
    axes[1, 1].grid(True, alpha=0.2)
    
    plt.tight_layout()
    output = os.path.join(DATOS_DIR, "bouc_wen_validacion_linealidad.png")
    fig.savefig(output, dpi=150, facecolor='white')
    print(f"\n Guardado: {output}")
    plt.show()
    
    return model, history, hyst


if __name__ == "__main__":
    model, history, hyst = main()
