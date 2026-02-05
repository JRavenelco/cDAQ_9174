#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CÁLCULO DE FRICCIÓN - LC302-1K
==============================
F_fricción = F_celda - m·a - k·x

Donde x se obtiene por doble integración de a.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import cumulative_trapezoid
import os
import glob

# Configuración
DATOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caracterizacion_fuerza")
SAMPLE_RATE = 2500  # Hz
G = 9.81  # m/s²

print("=" * 70)
print("CÁLCULO DE FRICCIÓN - LC302-1K")
print("F_fricción = F_celda - m·a - k·x")
print("=" * 70)

# Buscar archivos
pattern = os.path.join(DATOS_DIR, "caracterizacion_fuerza_20251203_160853_exp*.csv")
archivos = sorted(glob.glob(pattern))

# Excluir 10 Hz (anómalo)
archivos = [a for a in archivos if "_10Hz.csv" not in a]
print(f"\nAnalizando {len(archivos)} frecuencias (excluyendo 10 Hz)")

# ============================================
# PASO 1: Identificar m y k de la FRF
# ============================================
print("\n" + "=" * 70)
print("PASO 1: IDENTIFICACIÓN DE PARÁMETROS")
print("=" * 70)

# FRF: H(ω) = F/a = m - k/ω² + jc/ω
# Para señales senoidales: |H| ≈ m para ω grande, |H| ≈ k/ω² para ω pequeño

frf_data = []

for archivo in archivos:
    nombre = os.path.basename(archivo)
    freq = float(nombre.split('_')[-1].replace('Hz.csv', ''))
    omega = 2 * np.pi * freq
    
    df = pd.read_csv(archivo)
    
    # Datos crudos
    fuerza_V = df['fuerza_V'].values
    acel_g = df['aceleracion_sensor_g'].values
    
    # Descartar transitorio
    skip = int(0.5 * SAMPLE_RATE)
    fuerza_V = fuerza_V[skip:]
    acel_g = acel_g[skip:]
    
    # Componentes AC
    F_ac = fuerza_V - np.mean(fuerza_V)
    a_ac = acel_g - np.mean(acel_g)
    
    # RMS
    F_rms = np.sqrt(np.mean(F_ac**2))
    a_rms = np.sqrt(np.mean(a_ac**2)) * G  # Convertir a m/s²
    
    if a_rms > 0.01:  # Filtrar frecuencias con poca señal
        H_mag = F_rms / a_rms  # V/(m/s²)
        frf_data.append({'freq': freq, 'omega': omega, 'H': H_mag, 'omega2': omega**2})

df_frf = pd.DataFrame(frf_data)

# Ajuste lineal: H = m - k/ω²
# Reescribir como: H = m + (-k) * (1/ω²)
# y = a + b*x donde y=H, x=1/ω², a=m, b=-k

x = 1 / df_frf['omega2'].values
y = df_frf['H'].values

# Regresión lineal
A = np.vstack([np.ones_like(x), x]).T
result = np.linalg.lstsq(A, y, rcond=None)
m_id, neg_k = result[0]
k_id = -neg_k

print(f"\nParámetros identificados (de FRF):")
print(f"  Masa efectiva:  m = {m_id:.6f} V/(m/s²)")
print(f"  Rigidez:        k = {k_id:.6f} V/m")

# Frecuencia natural
if k_id > 0 and m_id > 0:
    fn = np.sqrt(k_id / m_id) / (2 * np.pi)
    print(f"  Freq. natural:  fn = {fn:.2f} Hz")

# ============================================
# PASO 2: Calcular fricción para cada frecuencia
# ============================================
print("\n" + "=" * 70)
print("PASO 2: CÁLCULO DE FRICCIÓN")
print("=" * 70)

print("\n Freq    F_rms     m·a      k·x    F_fric   % de F")
print(" (Hz)     (mV)    (mV)     (mV)     (mV)")
print("-" * 60)

resultados_friccion = []

for archivo in archivos:
    nombre = os.path.basename(archivo)
    freq = float(nombre.split('_')[-1].replace('Hz.csv', ''))
    omega = 2 * np.pi * freq
    
    df = pd.read_csv(archivo)
    
    # Datos
    t = df['tiempo_s'].values
    fuerza_V = df['fuerza_V'].values
    acel_g = df['aceleracion_sensor_g'].values
    acel_ms2 = acel_g * G
    
    # Descartar transitorio
    skip = int(0.5 * SAMPLE_RATE)
    t = t[skip:] - t[skip]
    fuerza_V = fuerza_V[skip:]
    acel_ms2 = acel_ms2[skip:]
    
    # Componente AC de fuerza
    F_mean = np.mean(fuerza_V)
    F_ac = fuerza_V - F_mean
    
    # Componente AC de aceleración
    a_mean = np.mean(acel_ms2)
    a_ac = acel_ms2 - a_mean
    
    # Integrar aceleración para obtener velocidad (con filtro pasa-alto)
    dt = 1 / SAMPLE_RATE
    
    # Filtro pasa-alto para eliminar drift (fc = freq/10)
    fc = max(freq / 10, 0.5)
    b, a_filt = signal.butter(2, fc / (SAMPLE_RATE / 2), btype='high')
    
    # Velocidad: integrar aceleración
    vel = cumulative_trapezoid(a_ac, dx=dt, initial=0)
    vel = signal.filtfilt(b, a_filt, vel)
    
    # Posición: integrar velocidad
    pos = cumulative_trapezoid(vel, dx=dt, initial=0)
    pos = signal.filtfilt(b, a_filt, pos)
    
    # Calcular términos
    ma = m_id * a_ac  # Término inercial
    kx = k_id * pos   # Término elástico
    
    # Fricción = F - ma - kx
    F_friccion = F_ac - ma - kx
    
    # RMS de cada término (en mV)
    F_rms = np.sqrt(np.mean(F_ac**2)) * 1000
    ma_rms = np.sqrt(np.mean(ma**2)) * 1000
    kx_rms = np.sqrt(np.mean(kx**2)) * 1000
    Ff_rms = np.sqrt(np.mean(F_friccion**2)) * 1000
    
    # Porcentaje de fricción respecto a fuerza total
    pct = (Ff_rms / F_rms * 100) if F_rms > 0 else 0
    
    print(f"{freq:5.0f}   {F_rms:6.2f}   {ma_rms:6.2f}   {kx_rms:6.2f}   {Ff_rms:6.2f}   {pct:5.1f}%")
    
    resultados_friccion.append({
        'freq': freq,
        'F_rms_mV': F_rms,
        'ma_rms_mV': ma_rms,
        'kx_rms_mV': kx_rms,
        'Ff_rms_mV': Ff_rms,
        'pct_friccion': pct,
        't': t,
        'F_ac': F_ac,
        'F_friccion': F_friccion,
        'pos': pos,
        'vel': vel
    })

df_fric = pd.DataFrame([{k: v for k, v in r.items() if k not in ['t', 'F_ac', 'F_friccion', 'pos', 'vel']} 
                         for r in resultados_friccion])

# ============================================
# GRÁFICAS
# ============================================
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Cálculo de Fricción - LC302-1K', fontsize=14, fontweight='bold')

# 1. Componentes de fuerza vs frecuencia
ax1 = axes[0, 0]
ax1.plot(df_fric['freq'], df_fric['F_rms_mV'], 'o-', color='cyan', label='F_celda', linewidth=2)
ax1.plot(df_fric['freq'], df_fric['ma_rms_mV'], 's-', color='orange', label='m·a', linewidth=2)
ax1.plot(df_fric['freq'], df_fric['kx_rms_mV'], '^-', color='lime', label='k·x', linewidth=2)
ax1.plot(df_fric['freq'], df_fric['Ff_rms_mV'], 'd-', color='magenta', label='F_fricción', linewidth=2)
ax1.set_xlabel('Frecuencia (Hz)')
ax1.set_ylabel('Fuerza RMS (mV)')
ax1.set_title('Descomposición de Fuerzas')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Porcentaje de fricción
ax2 = axes[0, 1]
ax2.bar(df_fric['freq'], df_fric['pct_friccion'], color='magenta', alpha=0.7, width=3)
ax2.axhline(y=100, color='white', linestyle='--', alpha=0.5)
ax2.set_xlabel('Frecuencia (Hz)')
ax2.set_ylabel('Fricción / F_celda (%)')
ax2.set_title('Porcentaje de Fricción')
ax2.grid(True, alpha=0.3)

# 3. FRF con ajuste
ax3 = axes[0, 2]
ax3.semilogy(df_frf['freq'], df_frf['H'] * 1000, 'o', color='cyan', markersize=8, label='Medido')
# Curva ajustada
freq_fit = np.linspace(15, 90, 100)
omega_fit = 2 * np.pi * freq_fit
H_fit = m_id - k_id / omega_fit**2
ax3.semilogy(freq_fit, np.abs(H_fit) * 1000, 'r-', linewidth=2, label='Ajuste')
ax3.set_xlabel('Frecuencia (Hz)')
ax3.set_ylabel('|H| (mV/(m/s²))')
ax3.set_title(f'FRF: m={m_id*1000:.3f} mV/(m/s²)')
ax3.legend()
ax3.grid(True, alpha=0.3, which='both')

# 4. Señales temporales @ 50 Hz
ax4 = axes[1, 0]
idx_50 = next((i for i, r in enumerate(resultados_friccion) if r['freq'] == 50), len(resultados_friccion)//2)
r = resultados_friccion[idx_50]
n_show = int(5 / r['freq'] * SAMPLE_RATE)  # 5 ciclos
t_ms = r['t'][:n_show] * 1000
ax4.plot(t_ms, r['F_ac'][:n_show] * 1000, 'c-', label='F_celda', alpha=0.8)
ax4.plot(t_ms, r['F_friccion'][:n_show] * 1000, 'm-', label='F_fricción', alpha=0.8)
ax4.set_xlabel('Tiempo (ms)')
ax4.set_ylabel('Fuerza (mV)')
ax4.set_title(f'Señales @ {r["freq"]:.0f} Hz')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Histéresis F vs x
ax5 = axes[1, 1]
ax5.plot(r['pos'][:n_show] * 1e6, r['F_friccion'][:n_show] * 1000, 'c-', alpha=0.5, linewidth=0.5)
ax5.scatter(r['pos'][:n_show:20] * 1e6, r['F_friccion'][:n_show:20] * 1000, 
            c=np.arange(len(r['pos'][:n_show:20])), cmap='plasma', s=15)
ax5.set_xlabel('Posición (µm)')
ax5.set_ylabel('F_fricción (mV)')
ax5.set_title(f'Histéresis @ {r["freq"]:.0f} Hz')
ax5.grid(True, alpha=0.3)

# 6. Histéresis F vs v (Stribeck)
ax6 = axes[1, 2]
ax6.plot(r['vel'][:n_show] * 1000, r['F_friccion'][:n_show] * 1000, 'c-', alpha=0.5, linewidth=0.5)
ax6.scatter(r['vel'][:n_show:20] * 1000, r['F_friccion'][:n_show:20] * 1000,
            c=np.arange(len(r['vel'][:n_show:20])), cmap='plasma', s=15)
ax6.set_xlabel('Velocidad (mm/s)')
ax6.set_ylabel('F_fricción (mV)')
ax6.set_title(f'F vs v @ {r["freq"]:.0f} Hz')
ax6.grid(True, alpha=0.3)

plt.tight_layout()

output_file = os.path.join(DATOS_DIR, "calculo_friccion_lc302.png")
plt.savefig(output_file, dpi=150, facecolor='#1e1e1e')
print(f"\n📊 Guardado: {output_file}")
plt.show()

# ============================================
# RESUMEN
# ============================================
print("\n" + "=" * 70)
print("RESUMEN DE FRICCIÓN")
print("=" * 70)
print(f"\nFricción promedio: {df_fric['Ff_rms_mV'].mean():.2f} mV RMS")
print(f"Fricción mínima:   {df_fric['Ff_rms_mV'].min():.2f} mV @ {df_fric.loc[df_fric['Ff_rms_mV'].idxmin(), 'freq']:.0f} Hz")
print(f"Fricción máxima:   {df_fric['Ff_rms_mV'].max():.2f} mV @ {df_fric.loc[df_fric['Ff_rms_mV'].idxmax(), 'freq']:.0f} Hz")
print(f"\nPorcentaje promedio: {df_fric['pct_friccion'].mean():.1f}% de F_celda")
