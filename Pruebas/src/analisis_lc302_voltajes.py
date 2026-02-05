#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS LC302 - EN VOLTAJES (SIN CONVERSIONES)
================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
import glob

# Configuración
DATOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caracterizacion_fuerza")
SAMPLE_RATE = 2500  # Hz

print("=" * 70)
print("ANÁLISIS BARRIDO LC302-1K - VOLTAJES CRUDOS")
print("=" * 70)

# Buscar archivos del barrido
pattern = os.path.join(DATOS_DIR, "caracterizacion_fuerza_20251203_160853_exp*.csv")
archivos = sorted(glob.glob(pattern))

print(f"Encontrados {len(archivos)} archivos")
print()

# Analizar cada frecuencia
resultados = []

for archivo in archivos:
    nombre = os.path.basename(archivo)
    freq_str = nombre.split('_')[-1].replace('Hz.csv', '')
    freq = float(freq_str)
    
    df = pd.read_csv(archivo)
    
    # Datos crudos
    fuerza_V = df['fuerza_V'].values
    acel_g = df['aceleracion_sensor_g'].values
    
    # Descartar transitorio (0.5s)
    skip = int(0.5 * SAMPLE_RATE)
    if len(fuerza_V) > skip * 2:
        fuerza_V = fuerza_V[skip:]
        acel_g = acel_g[skip:]
    
    # Estadísticas en voltajes/g
    F_mean = np.mean(fuerza_V)
    F_ac = fuerza_V - F_mean  # Componente AC
    F_rms = np.sqrt(np.mean(F_ac**2))
    F_pp = np.max(fuerza_V) - np.min(fuerza_V)
    
    a_mean = np.mean(acel_g)
    a_ac = acel_g - a_mean
    a_rms = np.sqrt(np.mean(a_ac**2))
    a_pp = np.max(acel_g) - np.min(acel_g)
    
    # FRF en unidades crudas: V/g
    if a_rms > 0.0001:
        H_mag = F_rms / a_rms  # V/g
    else:
        H_mag = 0
    
    # Fase usando correlación cruzada
    if len(F_ac) > 100 and a_rms > 0.0001:
        corr = np.correlate(F_ac[:1000], a_ac[:1000], mode='full')
        lag = np.argmax(corr) - len(F_ac[:1000]) + 1
        fase = (lag / SAMPLE_RATE) * freq * 360  # grados
    else:
        fase = 0
    
    resultados.append({
        'freq': freq,
        'F_mean_V': F_mean,
        'F_rms_mV': F_rms * 1000,
        'F_pp_mV': F_pp * 1000,
        'a_rms_g': a_rms,
        'a_pp_g': a_pp,
        'H_mV_g': H_mag * 1000,
        'fase': fase
    })

df_res = pd.DataFrame(resultados).sort_values('freq')

print(" Freq   F_mean   F_rms    F_pp    a_rms    a_pp   |H|=F/a    fase")
print(" (Hz)      (V)    (mV)    (mV)      (g)      (g)   (mV/g)     (°)")
print("-" * 70)

for _, row in df_res.iterrows():
    print(f"{row['freq']:5.0f}  {row['F_mean_V']:7.4f}  {row['F_rms_mV']:6.2f}  {row['F_pp_mV']:6.1f}  "
          f"{row['a_rms_g']:7.4f}  {row['a_pp_g']:6.3f}  {row['H_mV_g']:7.2f}  {row['fase']:6.1f}")

# ============================================
# GRÁFICAS
# ============================================
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Análisis LC302-1K - Voltajes Crudos', fontsize=14, fontweight='bold')

# 1. Fuerza media (V)
ax1 = axes[0, 0]
ax1.plot(df_res['freq'], df_res['F_mean_V'], 'o-', color='cyan', linewidth=2, markersize=8)
ax1.set_xlabel('Frecuencia (Hz)')
ax1.set_ylabel('Fuerza media (V)')
ax1.set_title('Voltaje DC de Fuerza')
ax1.grid(True, alpha=0.3)

# 2. Fuerza AC (mV)
ax2 = axes[0, 1]
ax2.plot(df_res['freq'], df_res['F_rms_mV'], 'o-', color='lime', linewidth=2, markersize=8)
ax2.fill_between(df_res['freq'], 0, df_res['F_rms_mV'], alpha=0.3, color='lime')
ax2.set_xlabel('Frecuencia (Hz)')
ax2.set_ylabel('Fuerza RMS (mV)')
ax2.set_title('Componente AC de Fuerza')
ax2.grid(True, alpha=0.3)

# 3. Aceleración (g)
ax3 = axes[0, 2]
ax3.semilogy(df_res['freq'], df_res['a_rms_g'], 'o-', color='orange', linewidth=2, markersize=8)
ax3.set_xlabel('Frecuencia (Hz)')
ax3.set_ylabel('Aceleración RMS (g)')
ax3.set_title('Aceleración')
ax3.grid(True, alpha=0.3, which='both')

# 4. FRF Magnitud
ax4 = axes[1, 0]
ax4.semilogy(df_res['freq'], df_res['H_mV_g'], 'o-', color='magenta', linewidth=2, markersize=8)
ax4.set_xlabel('Frecuencia (Hz)')
ax4.set_ylabel('|H| = F/a (mV/g)')
ax4.set_title('FRF: Magnitud')
ax4.grid(True, alpha=0.3, which='both')

# 5. Señales temporales @ frecuencia con mayor aceleración
ax5 = axes[1, 1]
idx_max_a = df_res['a_rms_g'].idxmax()
freq_plot = df_res.loc[idx_max_a, 'freq']
archivo_plot = [a for a in archivos if f"_{int(freq_plot)}Hz.csv" in a][0]
df_plot = pd.read_csv(archivo_plot)

# 10 ciclos
n_samples = int(10 / freq_plot * SAMPLE_RATE)
t = df_plot['tiempo_s'].values[:n_samples]
f_v = df_plot['fuerza_V'].values[:n_samples]
a_g = df_plot['aceleracion_sensor_g'].values[:n_samples]

# Normalizar para comparar
f_norm = (f_v - np.mean(f_v)) / np.std(f_v)
a_norm = (a_g - np.mean(a_g)) / np.std(a_g)

ax5.plot(t*1000, f_norm, 'c-', label='Fuerza', alpha=0.8, linewidth=1)
ax5.plot(t*1000, a_norm, 'orange', label='Aceleración', alpha=0.8, linewidth=1)
ax5.set_xlabel('Tiempo (ms)')
ax5.set_ylabel('Amplitud normalizada')
ax5.set_title(f'Señales @ {freq_plot:.0f} Hz')
ax5.legend(loc='upper right')
ax5.grid(True, alpha=0.3)

# 6. Lissajous F vs a
ax6 = axes[1, 2]
# Usar datos de 50 Hz si existe
freq_liss = 50 if 50 in df_res['freq'].values else freq_plot
archivo_liss = [a for a in archivos if f"_{int(freq_liss)}Hz.csv" in a]
if archivo_liss:
    df_liss = pd.read_csv(archivo_liss[0])
    skip = int(0.5 * SAMPLE_RATE)
    f_v = df_liss['fuerza_V'].values[skip:skip+2500]  # 1 segundo
    a_g = df_liss['aceleracion_sensor_g'].values[skip:skip+2500]
    
    # Centrar
    f_v = (f_v - np.mean(f_v)) * 1000  # mV
    a_g = a_g - np.mean(a_g)  # g
    
    ax6.plot(a_g, f_v, 'c-', alpha=0.5, linewidth=0.5)
    ax6.scatter(a_g[::50], f_v[::50], c=np.arange(len(a_g[::50])), cmap='plasma', s=10)
    ax6.set_xlabel('Aceleración (g)')
    ax6.set_ylabel('Fuerza (mV)')
    ax6.set_title(f'Lissajous @ {freq_liss:.0f} Hz')
    ax6.grid(True, alpha=0.3)

plt.tight_layout()

output_file = os.path.join(DATOS_DIR, "analisis_lc302_voltajes.png")
plt.savefig(output_file, dpi=150, facecolor='#1e1e1e')
print(f"\n📊 Guardado: {output_file}")
plt.show()

# ============================================
# RESUMEN
# ============================================
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
print(f"\nVoltaje DC (precarga): {df_res['F_mean_V'].mean():.4f} V")
print(f"Variación AC promedio: {df_res['F_rms_mV'].mean():.2f} mV")
print(f"Variación pico-pico promedio: {df_res['F_pp_mV'].mean():.1f} mV")
print(f"\nAceleración máxima: {df_res['a_rms_g'].max():.3f} g @ {df_res.loc[df_res['a_rms_g'].idxmax(), 'freq']:.0f} Hz")
print(f"FRF mínima (resonancia?): {df_res['H_mV_g'].min():.2f} mV/g @ {df_res.loc[df_res['H_mV_g'].idxmin(), 'freq']:.0f} Hz")
