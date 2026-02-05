#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS COMPLETO DE FRICCIÓN - MÚLTIPLES MÉTODOS
==================================================
Compara datos de DYMH-105 vs LC302-1K
Métodos: Histéresis, Fase, Potencia Disipada, Directo
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
SAMPLE_RATE = 2500
G = 9.81

print("=" * 80)
print("ANÁLISIS COMPLETO DE FRICCIÓN - DYMH-105 vs LC302-1K")
print("=" * 80)

# ============================================
# FUNCIONES DE ANÁLISIS
# ============================================

def integrar_con_filtro(señal, dt, freq_corte):
    """Integra señal con filtro pasa-alto para eliminar drift"""
    # Filtro pasa-alto
    fc = max(freq_corte, 0.5)
    b, a = signal.butter(2, fc / (SAMPLE_RATE / 2), btype='high')
    
    # Integrar
    integral = cumulative_trapezoid(señal, dx=dt, initial=0)
    # Filtrar
    integral = signal.filtfilt(b, a, integral)
    return integral

def calcular_area_histeresis(x, y):
    """Calcula área del lazo de histéresis (integral cerrada)"""
    # Usar fórmula del área de polígono (shoelace)
    n = len(x)
    area = 0.5 * np.abs(np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]))
    return area

def calcular_fase(señal1, señal2, freq, fs):
    """Calcula desfase entre dos señales usando correlación cruzada"""
    # Correlación cruzada
    corr = np.correlate(señal1 - np.mean(señal1), 
                        señal2 - np.mean(señal2), mode='full')
    lags = np.arange(-len(señal1)+1, len(señal1))
    
    # Encontrar pico
    idx_max = np.argmax(corr)
    lag = lags[idx_max]
    
    # Convertir a fase
    fase_rad = 2 * np.pi * freq * lag / fs
    fase_deg = np.degrees(fase_rad)
    
    # Normalizar a [-180, 180]
    while fase_deg > 180:
        fase_deg -= 360
    while fase_deg < -180:
        fase_deg += 360
    
    return fase_deg

def analizar_experimento(archivo, freq_nominal):
    """Analiza un archivo de experimento"""
    df = pd.read_csv(archivo)
    
    # Detectar columnas disponibles
    tiene_dos_acel = 'aceleracion_sensor_g' in df.columns
    
    t = df['tiempo_s'].values
    fuerza_V = df['fuerza_V'].values
    
    if tiene_dos_acel:
        acel_g = df['aceleracion_sensor_g'].values
    else:
        # Buscar columna de aceleración
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
    acel_ms2 = acel_g * G
    
    dt = 1 / SAMPLE_RATE
    
    # Componentes AC
    F_mean = np.mean(fuerza_V)
    F_ac = fuerza_V - F_mean
    a_mean = np.mean(acel_ms2)
    a_ac = acel_ms2 - a_mean
    
    # RMS
    F_rms = np.sqrt(np.mean(F_ac**2))
    a_rms = np.sqrt(np.mean(a_ac**2))
    
    if a_rms < 0.001:  # Señal muy débil
        return None
    
    # Integrar para obtener velocidad y posición
    vel = integrar_con_filtro(a_ac, dt, freq_nominal / 10)
    pos = integrar_con_filtro(vel, dt, freq_nominal / 10)
    
    v_rms = np.sqrt(np.mean(vel**2))
    x_rms = np.sqrt(np.mean(pos**2))
    
    # ========== MÉTODO 1: HISTÉRESIS ==========
    # Área del lazo F vs x
    # Usar solo unos ciclos para evitar ruido
    n_ciclos = min(10, int(len(t) * freq_nominal / SAMPLE_RATE))
    n_samples = int(n_ciclos / freq_nominal * SAMPLE_RATE)
    
    if n_samples > 100:
        area = calcular_area_histeresis(pos[:n_samples], F_ac[:n_samples])
        # Energía por ciclo = área
        # F_friccion_eq = E / (π × x_amp) para fricción de Coulomb
        x_amp = np.max(np.abs(pos[:n_samples]))
        if x_amp > 1e-9:
            F_histeresis = area / (np.pi * x_amp) if x_amp > 0 else 0
        else:
            F_histeresis = 0
    else:
        F_histeresis = 0
        area = 0
    
    # ========== MÉTODO 2: FASE ==========
    fase = calcular_fase(F_ac, a_ac, freq_nominal, SAMPLE_RATE)
    # Componente en cuadratura (90° de a = en fase con v)
    # F_friccion = |F| × |sin(fase)|
    F_fase = F_rms * np.abs(np.sin(np.radians(fase)))
    
    # ========== MÉTODO 3: POTENCIA DISIPADA ==========
    # P = <F × v> (promedio temporal)
    P_disipada = np.mean(F_ac * vel)
    # F_friccion_eq = P / v_rms
    F_potencia = np.abs(P_disipada) / v_rms if v_rms > 1e-9 else 0
    
    # ========== MÉTODO 4: DIRECTO ==========
    # Asumir F_celda ≈ F_friccion (si la celda mide directamente fricción)
    F_directo = F_rms
    
    # ========== MÉTODO 5: CORRELACIÓN ==========
    # Separar componente en fase con a (inercial) y en cuadratura (fricción)
    # Proyección de F sobre a normalizado
    a_norm = a_ac / (a_rms + 1e-12)
    F_inercial = np.mean(F_ac * a_norm) * a_norm
    F_friccion_corr = F_ac - F_inercial
    F_correlacion = np.sqrt(np.mean(F_friccion_corr**2))
    
    return {
        'freq': freq_nominal,
        'F_mean_V': F_mean,
        'F_rms_mV': F_rms * 1000,
        'a_rms_g': a_rms / G,
        'v_rms_mm_s': v_rms * 1000,
        'x_rms_um': x_rms * 1e6,
        'fase_deg': fase,
        'area_histeresis': area * 1e6,  # mV×µm
        'F_histeresis_mV': F_histeresis * 1000,
        'F_fase_mV': F_fase * 1000,
        'F_potencia_mV': F_potencia * 1000,
        'F_directo_mV': F_directo * 1000,
        'F_correlacion_mV': F_correlacion * 1000,
        'P_disipada_uW': P_disipada * 1e6
    }

# ============================================
# ANALIZAR DATOS DYMH-105 (barrido anterior)
# ============================================
print("\n" + "=" * 80)
print("SENSOR: DYMH-105 (500 kg)")
print("=" * 80)

# Buscar archivos del barrido DYMH-105 (fecha anterior)
pattern_dymh = os.path.join(DATOS_DIR, "caracterizacion_fuerza_20251202_*_exp*.csv")
archivos_dymh = sorted(glob.glob(pattern_dymh))

# Si no hay de esa fecha, buscar otros
if not archivos_dymh:
    pattern_dymh = os.path.join(DATOS_DIR, "caracterizacion_fuerza_20251201_*_exp*.csv")
    archivos_dymh = sorted(glob.glob(pattern_dymh))

resultados_dymh = []

if archivos_dymh:
    print(f"Encontrados {len(archivos_dymh)} archivos DYMH-105")
    
    for archivo in archivos_dymh:
        nombre = os.path.basename(archivo)
        # Extraer frecuencia
        try:
            freq_str = nombre.split('_')[-1].replace('Hz.csv', '')
            freq = float(freq_str)
        except:
            continue
        
        if freq < 5:  # Ignorar frecuencias muy bajas
            continue
            
        resultado = analizar_experimento(archivo, freq)
        if resultado:
            resultados_dymh.append(resultado)
else:
    print("No se encontraron archivos DYMH-105")

# ============================================
# ANALIZAR DATOS LC302-1K (barrido de hoy)
# ============================================
print("\n" + "=" * 80)
print("SENSOR: LC302-1K (454 kg)")
print("=" * 80)

pattern_lc302 = os.path.join(DATOS_DIR, "caracterizacion_fuerza_20251203_160853_exp*.csv")
archivos_lc302 = sorted(glob.glob(pattern_lc302))

resultados_lc302 = []

if archivos_lc302:
    print(f"Encontrados {len(archivos_lc302)} archivos LC302-1K")
    
    for archivo in archivos_lc302:
        nombre = os.path.basename(archivo)
        try:
            freq_str = nombre.split('_')[-1].replace('Hz.csv', '')
            freq = float(freq_str)
        except:
            continue
        
        if freq == 10:  # Excluir 10 Hz (anómalo)
            continue
            
        resultado = analizar_experimento(archivo, freq)
        if resultado:
            resultados_lc302.append(resultado)
else:
    print("No se encontraron archivos LC302-1K")

# ============================================
# MOSTRAR RESULTADOS
# ============================================

def mostrar_resultados(resultados, nombre_sensor):
    if not resultados:
        print(f"No hay resultados para {nombre_sensor}")
        return None
    
    df = pd.DataFrame(resultados).sort_values('freq')
    
    print(f"\n{'='*80}")
    print(f"RESULTADOS: {nombre_sensor}")
    print(f"{'='*80}")
    print("\n Freq   F_rms   a_rms   v_rms   x_rms    Fase   F_hist  F_fase  F_pot  F_dir  F_corr")
    print(" (Hz)   (mV)     (g)   (mm/s)   (µm)     (°)    (mV)    (mV)   (mV)   (mV)   (mV)")
    print("-" * 95)
    
    for _, row in df.iterrows():
        print(f"{row['freq']:5.0f}  {row['F_rms_mV']:5.2f}  {row['a_rms_g']:6.3f}  "
              f"{row['v_rms_mm_s']:6.2f}  {row['x_rms_um']:6.1f}  {row['fase_deg']:6.1f}  "
              f"{row['F_histeresis_mV']:5.2f}   {row['F_fase_mV']:5.2f}  {row['F_potencia_mV']:5.2f}  "
              f"{row['F_directo_mV']:5.2f}  {row['F_correlacion_mV']:5.2f}")
    
    print(f"\n--- PROMEDIOS ---")
    print(f"F_histéresis:  {df['F_histeresis_mV'].mean():.3f} mV")
    print(f"F_fase:        {df['F_fase_mV'].mean():.3f} mV")
    print(f"F_potencia:    {df['F_potencia_mV'].mean():.3f} mV")
    print(f"F_directo:     {df['F_directo_mV'].mean():.3f} mV")
    print(f"F_correlación: {df['F_correlacion_mV'].mean():.3f} mV")
    
    return df

df_dymh = mostrar_resultados(resultados_dymh, "DYMH-105")
df_lc302 = mostrar_resultados(resultados_lc302, "LC302-1K")

# ============================================
# GRÁFICAS COMPARATIVAS
# ============================================
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Análisis de Fricción - Comparación DYMH-105 vs LC302-1K', fontsize=14, fontweight='bold')

# Colores
c_dymh = 'cyan'
c_lc302 = 'orange'

# 1. F_rms vs Frecuencia
ax1 = axes[0, 0]
if df_dymh is not None:
    ax1.plot(df_dymh['freq'], df_dymh['F_rms_mV'], 'o-', color=c_dymh, label='DYMH-105', linewidth=2)
if df_lc302 is not None:
    ax1.plot(df_lc302['freq'], df_lc302['F_rms_mV'], 's-', color=c_lc302, label='LC302-1K', linewidth=2)
ax1.set_xlabel('Frecuencia (Hz)')
ax1.set_ylabel('F_celda RMS (mV)')
ax1.set_title('Fuerza Medida')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Aceleración vs Frecuencia
ax2 = axes[0, 1]
if df_dymh is not None:
    ax2.semilogy(df_dymh['freq'], df_dymh['a_rms_g'], 'o-', color=c_dymh, label='DYMH-105', linewidth=2)
if df_lc302 is not None:
    ax2.semilogy(df_lc302['freq'], df_lc302['a_rms_g'], 's-', color=c_lc302, label='LC302-1K', linewidth=2)
ax2.set_xlabel('Frecuencia (Hz)')
ax2.set_ylabel('Aceleración RMS (g)')
ax2.set_title('Aceleración')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

# 3. Fase vs Frecuencia
ax3 = axes[0, 2]
if df_dymh is not None:
    ax3.plot(df_dymh['freq'], df_dymh['fase_deg'], 'o-', color=c_dymh, label='DYMH-105', linewidth=2)
if df_lc302 is not None:
    ax3.plot(df_lc302['freq'], df_lc302['fase_deg'], 's-', color=c_lc302, label='LC302-1K', linewidth=2)
ax3.axhline(y=0, color='white', linestyle='--', alpha=0.3)
ax3.axhline(y=90, color='lime', linestyle='--', alpha=0.3, label='90° (fricción pura)')
ax3.axhline(y=-90, color='lime', linestyle='--', alpha=0.3)
ax3.set_xlabel('Frecuencia (Hz)')
ax3.set_ylabel('Fase F-a (°)')
ax3.set_title('Desfase Fuerza-Aceleración')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Comparación de métodos - DYMH
ax4 = axes[1, 0]
if df_dymh is not None and len(df_dymh) > 0:
    x = np.arange(len(df_dymh))
    width = 0.15
    ax4.bar(x - 2*width, df_dymh['F_histeresis_mV'], width, label='Histéresis', color='red')
    ax4.bar(x - width, df_dymh['F_fase_mV'], width, label='Fase', color='green')
    ax4.bar(x, df_dymh['F_potencia_mV'], width, label='Potencia', color='blue')
    ax4.bar(x + width, df_dymh['F_directo_mV'], width, label='Directo', color='magenta')
    ax4.bar(x + 2*width, df_dymh['F_correlacion_mV'], width, label='Correlación', color='yellow')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{f:.0f}" for f in df_dymh['freq']], rotation=45)
    ax4.set_xlabel('Frecuencia (Hz)')
    ax4.set_ylabel('F_fricción (mV)')
    ax4.set_title('DYMH-105: Comparación Métodos')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

# 5. Comparación de métodos - LC302
ax5 = axes[1, 1]
if df_lc302 is not None and len(df_lc302) > 0:
    x = np.arange(len(df_lc302))
    width = 0.15
    ax5.bar(x - 2*width, df_lc302['F_histeresis_mV'], width, label='Histéresis', color='red')
    ax5.bar(x - width, df_lc302['F_fase_mV'], width, label='Fase', color='green')
    ax5.bar(x, df_lc302['F_potencia_mV'], width, label='Potencia', color='blue')
    ax5.bar(x + width, df_lc302['F_directo_mV'], width, label='Directo', color='magenta')
    ax5.bar(x + 2*width, df_lc302['F_correlacion_mV'], width, label='Correlación', color='yellow')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f"{f:.0f}" for f in df_lc302['freq']], rotation=45)
    ax5.set_xlabel('Frecuencia (Hz)')
    ax5.set_ylabel('F_fricción (mV)')
    ax5.set_title('LC302-1K: Comparación Métodos')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

# 6. Resumen comparativo
ax6 = axes[1, 2]
metodos = ['Histéresis', 'Fase', 'Potencia', 'Directo', 'Correlación']
x = np.arange(len(metodos))
width = 0.35

if df_dymh is not None and len(df_dymh) > 0:
    vals_dymh = [df_dymh['F_histeresis_mV'].mean(), df_dymh['F_fase_mV'].mean(),
                 df_dymh['F_potencia_mV'].mean(), df_dymh['F_directo_mV'].mean(),
                 df_dymh['F_correlacion_mV'].mean()]
    ax6.bar(x - width/2, vals_dymh, width, label='DYMH-105', color=c_dymh)

if df_lc302 is not None and len(df_lc302) > 0:
    vals_lc302 = [df_lc302['F_histeresis_mV'].mean(), df_lc302['F_fase_mV'].mean(),
                  df_lc302['F_potencia_mV'].mean(), df_lc302['F_directo_mV'].mean(),
                  df_lc302['F_correlacion_mV'].mean()]
    ax6.bar(x + width/2, vals_lc302, width, label='LC302-1K', color=c_lc302)

ax6.set_xticks(x)
ax6.set_xticklabels(metodos, rotation=45)
ax6.set_ylabel('F_fricción promedio (mV)')
ax6.set_title('Comparación de Métodos (Promedio)')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()

output_file = os.path.join(DATOS_DIR, "analisis_friccion_completo.png")
plt.savefig(output_file, dpi=150, facecolor='#1e1e1e')
print(f"\n📊 Guardado: {output_file}")
plt.show()

# ============================================
# GUARDAR RESULTADOS EN CSV
# ============================================
if df_dymh is not None:
    csv_dymh = os.path.join(DATOS_DIR, "friccion_dymh105.csv")
    df_dymh.to_csv(csv_dymh, index=False)
    print(f"📄 Guardado: {csv_dymh}")

if df_lc302 is not None:
    csv_lc302 = os.path.join(DATOS_DIR, "friccion_lc302.csv")
    df_lc302.to_csv(csv_lc302, index=False)
    print(f"📄 Guardado: {csv_lc302}")

# ============================================
# CONCLUSIONES
# ============================================
print("\n" + "=" * 80)
print("CONCLUSIONES")
print("=" * 80)

if df_lc302 is not None and len(df_lc302) > 0:
    print(f"\nLC302-1K:")
    print(f"  - Fricción por histéresis:  {df_lc302['F_histeresis_mV'].mean():.3f} mV")
    print(f"  - Fricción por fase:        {df_lc302['F_fase_mV'].mean():.3f} mV")
    print(f"  - Fricción por potencia:    {df_lc302['F_potencia_mV'].mean():.3f} mV")
    print(f"  - Método más consistente:   ", end="")
    stds = {
        'Histéresis': df_lc302['F_histeresis_mV'].std(),
        'Fase': df_lc302['F_fase_mV'].std(),
        'Potencia': df_lc302['F_potencia_mV'].std()
    }
    print(f"{min(stds, key=stds.get)} (σ = {min(stds.values()):.3f} mV)")

if df_dymh is not None and len(df_dymh) > 0:
    print(f"\nDYMH-105:")
    print(f"  - Fricción por histéresis:  {df_dymh['F_histeresis_mV'].mean():.3f} mV")
    print(f"  - Fricción por fase:        {df_dymh['F_fase_mV'].mean():.3f} mV")
    print(f"  - Fricción por potencia:    {df_dymh['F_potencia_mV'].mean():.3f} mV")
