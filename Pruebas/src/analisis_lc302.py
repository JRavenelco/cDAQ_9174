#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS DE BARRIDO CON CELDA LC302-1K
======================================
Analiza los datos del barrido de frecuencia para calcular fricción
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import os
import json
import glob

# Configuración
DATOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caracterizacion_fuerza")
SAMPLE_RATE = 2500  # Hz

# Cargar calibración LC302
CAL_FILE = os.path.join(DATOS_DIR, "calibracion_LC302-1K_454kg.json")
with open(CAL_FILE, 'r') as f:
    cal_data = json.load(f)

VOLTAGE_OFFSET = cal_data['voltage_offset']
VOLTAGE_PER_KG = cal_data['voltage_per_kg']

print("=" * 70)
print("ANÁLISIS BARRIDO LC302-1K")
print("=" * 70)
print(f"Offset: {VOLTAGE_OFFSET:.5f} V")
print(f"Factor: {VOLTAGE_PER_KG:.6e} V/kg")
print()

# Buscar archivos del barrido más reciente
pattern = os.path.join(DATOS_DIR, "caracterizacion_fuerza_20251203_160853_exp*.csv")
archivos = sorted(glob.glob(pattern))

print(f"Encontrados {len(archivos)} archivos de experimento")
print()

# Analizar cada frecuencia
resultados = []

for archivo in archivos:
    # Extraer frecuencia del nombre
    nombre = os.path.basename(archivo)
    freq_str = nombre.split('_')[-1].replace('Hz.csv', '')
    freq = float(freq_str)
    
    # Cargar datos
    df = pd.read_csv(archivo)
    
    # Convertir voltaje a fuerza
    fuerza_V = df['fuerza_V'].values
    fuerza_kg = (fuerza_V - VOLTAGE_OFFSET) / VOLTAGE_PER_KG
    fuerza_N = fuerza_kg * 9.81
    
    # Aceleración (usar la del sensor, no la de bancada)
    acel_g = df['aceleracion_sensor_g'].values
    acel_ms2 = acel_g * 9.81
    
    # Descartar primeros 0.5s (transitorio)
    skip = int(0.5 * SAMPLE_RATE)
    if len(fuerza_N) > skip * 2:
        fuerza_N = fuerza_N[skip:]
        acel_ms2 = acel_ms2[skip:]
    
    # Estadísticas
    F_mean = np.mean(fuerza_N)
    F_std = np.std(fuerza_N)
    F_pp = np.max(fuerza_N) - np.min(fuerza_N)
    
    a_mean = np.mean(acel_ms2)
    a_std = np.std(acel_ms2)
    a_pp = np.max(acel_ms2) - np.min(acel_ms2)
    
    # Amplitudes RMS
    F_rms = np.sqrt(np.mean((fuerza_N - F_mean)**2))
    a_rms = np.sqrt(np.mean((acel_ms2 - a_mean)**2))
    
    # FRF: H = F/a
    if a_rms > 0.001:
        H_mag = F_rms / a_rms
    else:
        H_mag = 0
    
    resultados.append({
        'freq': freq,
        'F_mean': F_mean,
        'F_rms': F_rms,
        'F_pp': F_pp,
        'a_rms': a_rms,
        'a_pp': a_pp,
        'H_mag': H_mag
    })

# Convertir a DataFrame
df_res = pd.DataFrame(resultados)
df_res = df_res.sort_values('freq')

print(" Freq    F_mean     F_rms      F_pp     a_rms      a_pp    |H|=F/a")
print(" (Hz)       (N)       (N)       (N)    (m/s²)    (m/s²)       (kg)")
print("-" * 70)

for _, row in df_res.iterrows():
    print(f"{row['freq']:5.0f}   {row['F_mean']:7.2f}   {row['F_rms']:7.3f}   {row['F_pp']:7.2f}   "
          f"{row['a_rms']:7.3f}   {row['a_pp']:7.3f}   {row['H_mag']:8.2f}")

print()

# ============================================
# GRÁFICAS
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis Barrido LC302-1K', fontsize=14, fontweight='bold')

# 1. Fuerza vs Frecuencia
ax1 = axes[0, 0]
ax1.plot(df_res['freq'], df_res['F_mean'], 'o-', color='cyan', label='F media')
ax1.fill_between(df_res['freq'], 
                  df_res['F_mean'] - df_res['F_rms'],
                  df_res['F_mean'] + df_res['F_rms'],
                  alpha=0.3, color='cyan')
ax1.set_xlabel('Frecuencia (Hz)')
ax1.set_ylabel('Fuerza (N)')
ax1.set_title('Fuerza vs Frecuencia')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Aceleración vs Frecuencia
ax2 = axes[0, 1]
ax2.plot(df_res['freq'], df_res['a_rms'], 'o-', color='orange', label='a RMS')
ax2.plot(df_res['freq'], df_res['a_pp']/2, 's--', color='red', alpha=0.5, label='a pico')
ax2.set_xlabel('Frecuencia (Hz)')
ax2.set_ylabel('Aceleración (m/s²)')
ax2.set_title('Aceleración vs Frecuencia')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. FRF Magnitud
ax3 = axes[1, 0]
ax3.semilogy(df_res['freq'], df_res['H_mag'], 'o-', color='lime', linewidth=2)
ax3.set_xlabel('Frecuencia (Hz)')
ax3.set_ylabel('|H| = F/a (kg)')
ax3.set_title('FRF: Magnitud')
ax3.grid(True, alpha=0.3, which='both')

# 4. Señales de una frecuencia (50 Hz si existe)
ax4 = axes[1, 1]
freq_plot = 50 if 50 in df_res['freq'].values else df_res['freq'].iloc[len(df_res)//2]
archivo_plot = [a for a in archivos if f"_{int(freq_plot)}Hz.csv" in a]
if archivo_plot:
    df_plot = pd.read_csv(archivo_plot[0])
    t = df_plot['tiempo_s'].values[:500]  # Primeros 200ms
    f_v = df_plot['fuerza_V'].values[:500]
    f_n = ((f_v - VOLTAGE_OFFSET) / VOLTAGE_PER_KG) * 9.81
    a_g = df_plot['aceleracion_sensor_g'].values[:500]
    a_ms2 = a_g * 9.81
    
    ax4.plot(t*1000, (f_n - np.mean(f_n))/np.std(f_n), 'c-', label='Fuerza (norm)', alpha=0.8)
    ax4.plot(t*1000, (a_ms2 - np.mean(a_ms2))/np.std(a_ms2), 'orange', label='Acel (norm)', alpha=0.8)
    ax4.set_xlabel('Tiempo (ms)')
    ax4.set_ylabel('Amplitud normalizada')
    ax4.set_title(f'Señales @ {freq_plot:.0f} Hz')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Guardar
output_file = os.path.join(DATOS_DIR, "analisis_lc302_barrido.png")
plt.savefig(output_file, dpi=150, facecolor='white')
print(f"📊 Guardado: {output_file}")
plt.show()

# ============================================
# CÁLCULO DE FRICCIÓN
# ============================================
print()
print("=" * 70)
print("ESTIMACIÓN DE FRICCIÓN")
print("=" * 70)

# La fuerza media representa la precarga + fricción estática
F_media_global = df_res['F_mean'].mean()
F_variacion = df_res['F_rms'].mean()

print(f"\nFuerza media (precarga): {F_media_global:.2f} N ({F_media_global/9.81:.2f} kg)")
print(f"Variación dinámica RMS: {F_variacion:.3f} N")
print(f"Variación pico-pico promedio: {df_res['F_pp'].mean():.3f} N")

# Si la fuerza varía poco con la frecuencia, es principalmente fricción
print(f"\nCoeficiente de variación de F_mean: {df_res['F_mean'].std()/df_res['F_mean'].mean()*100:.2f}%")

if df_res['F_mean'].std()/df_res['F_mean'].mean() < 0.1:
    print("✅ Fuerza estable - la celda mide principalmente la precarga/fricción estática")
else:
    print("⚠️ Fuerza variable - hay componente dinámica significativa")
