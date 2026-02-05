#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS FFT PARA DETECCIÓN DE NO-LINEALIDADES EN FRICCIÓN
===========================================================

Este script analiza el contenido frecuencial de:
- Fuerza Total (F_total)
- Aceleración (a)  
- Fricción Calculada (F_friccion = F_total - m·a)

Si la fricción es lineal (viscosa), solo veremos el pico fundamental.
Si hay fricción de Coulomb o Histéresis, aparecerán armónicos IMPARES (3f, 5f, 7f...).

Autor: Análisis DSP para vibraciones mecánicas
Fecha: 2025-12-04
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DATOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caracterizacion_fuerza")
SAMPLE_RATE = 2500  # Hz
G = 9.81  # m/s²

# Parámetro de masa efectiva (en unidades de V/(m/s²) para mantener consistencia en voltajes)
# Esto se ajustará basado en los datos
MASA_EFECTIVA_V = 0.003  # V/(m/s²) - aproximado de análisis previo

# Rango de frecuencias para visualización
FREQ_MAX = 150  # Hz - suficiente para ver hasta 5° armónico de 25Hz

# =============================================================================
# FUNCIONES DE ANÁLISIS FFT
# =============================================================================

def calcular_fft(signal_data, fs):
    """
    Calcula la FFT de una señal.
    
    Args:
        signal_data: Array con la señal temporal
        fs: Frecuencia de muestreo (Hz)
    
    Returns:
        freqs: Frecuencias positivas (Hz)
        magnitud: Magnitud normalizada del espectro
    """
    N = len(signal_data)
    
    # Aplicar ventana para reducir leakage
    window = np.hanning(N)
    signal_windowed = signal_data * window
    
    # FFT
    fft_result = fft(signal_windowed)
    freqs = fftfreq(N, 1/fs)
    
    # Solo frecuencias positivas
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    fft_result = fft_result[pos_mask]
    
    # Magnitud normalizada (compensando ventana y N/2)
    magnitud = np.abs(fft_result) * 2 / N
    # Compensar pérdida de energía por ventana Hanning
    magnitud *= 2  # Factor de corrección aproximado
    
    return freqs, magnitud


def encontrar_picos(freqs, magnitud, freq_fundamental, threshold_db=-20):
    """
    Encuentra picos en el espectro y los clasifica como fundamental o armónicos.
    
    Args:
        freqs: Array de frecuencias
        magnitud: Array de magnitudes
        freq_fundamental: Frecuencia de excitación (Hz)
        threshold_db: Umbral en dB respecto al pico máximo
    
    Returns:
        picos: Lista de diccionarios con info de cada pico
    """
    # Convertir a dB
    mag_db = 20 * np.log10(magnitud + 1e-12)
    max_db = np.max(mag_db)
    
    # Encontrar picos
    peaks, properties = signal.find_peaks(mag_db, height=max_db + threshold_db, distance=5)
    
    picos = []
    for idx in peaks:
        freq = freqs[idx]
        mag = magnitud[idx]
        
        # Determinar si es armónico
        ratio = freq / freq_fundamental
        armónico = round(ratio)
        es_impar = armónico % 2 == 1
        error = abs(ratio - armónico)
        
        if error < 0.1 and armónico > 0:  # Tolerancia del 10%
            tipo = f"{armónico}f" if armónico > 1 else "1f (fundamental)"
            if armónico > 1:
                tipo += " (IMPAR)" if es_impar else " (par)"
        else:
            tipo = "otro"
        
        picos.append({
            'freq': freq,
            'magnitud': mag,
            'armónico': armónico,
            'es_impar': es_impar,
            'tipo': tipo,
        })
    
    return picos


def analizar_frecuencia(archivo, freq_excitacion):
    """
    Analiza una frecuencia de excitación específica.
    
    Args:
        archivo: Path al archivo CSV
        freq_excitacion: Frecuencia de excitación (Hz)
    
    Returns:
        dict con resultados del análisis FFT
    """
    # Cargar datos
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
        acel_g = df[acel_cols[0]].values if acel_cols else np.zeros_like(t)
    
    # Calcular Fs real
    dt = np.mean(np.diff(t))
    fs = 1 / dt
    
    # Descartar transitorio (0.5 segundos)
    skip = int(0.5 * fs)
    if len(t) > skip * 2:
        t = t[skip:] - t[skip]
        fuerza_V = fuerza_V[skip:]
        acel_g = acel_g[skip:]
    
    # Convertir a unidades SI
    acel_ms2 = acel_g * G
    
    # Remover DC
    fuerza_ac = fuerza_V - np.mean(fuerza_V)
    acel_ac = acel_ms2 - np.mean(acel_ms2)
    
    # Calcular fricción: F_friccion = F_total - m·a
    # Estimamos m como la relación F/a a la frecuencia fundamental
    # Para mantener unidades consistentes (V), usamos m en V/(m/s²)
    m_eff = np.std(fuerza_ac) / (np.std(acel_ac) + 1e-10)
    friccion = fuerza_ac - m_eff * acel_ac
    
    # Calcular FFT de las 3 señales
    freqs, fft_acel = calcular_fft(acel_ac, fs)
    _, fft_fuerza = calcular_fft(fuerza_ac, fs)
    _, fft_friccion = calcular_fft(friccion, fs)
    
    # Encontrar picos en el espectro de fricción
    picos = encontrar_picos(freqs, fft_friccion, freq_excitacion)
    
    # Calcular THD (Total Harmonic Distortion) de la fricción
    idx_fund = np.argmin(np.abs(freqs - freq_excitacion))
    mag_fund = fft_friccion[idx_fund]
    
    # Sumar potencia de armónicos (hasta 7°)
    armonicos_potencia = 0
    for n in [3, 5, 7]:  # Armónicos impares
        f_arm = n * freq_excitacion
        if f_arm < freqs[-1]:
            idx_arm = np.argmin(np.abs(freqs - f_arm))
            armonicos_potencia += fft_friccion[idx_arm]**2
    
    thd = np.sqrt(armonicos_potencia) / (mag_fund + 1e-10) * 100  # %
    
    return {
        't': t,
        'fuerza': fuerza_ac,
        'acel': acel_ac,
        'friccion': friccion,
        'fs': fs,
        'freqs': freqs,
        'fft_acel': fft_acel,
        'fft_fuerza': fft_fuerza,
        'fft_friccion': fft_friccion,
        'picos': picos,
        'thd': thd,
        'm_eff': m_eff,
        'freq_excitacion': freq_excitacion,
    }


def plot_espectros(resultado, titulo_extra=""):
    """
    Genera figura con 3 sub-gráficas de espectros.
    """
    freqs = resultado['freqs']
    fft_acel = resultado['fft_acel']
    fft_fuerza = resultado['fft_fuerza']
    fft_friccion = resultado['fft_friccion']
    freq_exc = resultado['freq_excitacion']
    picos = resultado['picos']
    thd = resultado['thd']
    
    # Máscara para rango de frecuencias
    mask = freqs <= FREQ_MAX
    freqs_plot = freqs[mask]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Análisis FFT - Frecuencia de Excitación: {freq_exc} Hz{titulo_extra}\n'
                 f'THD Fricción: {thd:.1f}%', fontsize=14, fontweight='bold')
    
    # Colores
    color_fund = 'lime'
    color_impar = 'red'
    color_par = 'yellow'
    
    # --- Gráfica 1: Espectro de Aceleración ---
    ax1 = axes[0]
    ax1.plot(freqs_plot, fft_acel[mask] * 1000 / G, 'c-', linewidth=1, label='A(ω)')
    ax1.axvline(freq_exc, color=color_fund, linestyle='--', alpha=0.5, label=f'f₀={freq_exc}Hz')
    ax1.set_ylabel('Aceleración (mg)', fontsize=11)
    ax1.set_title('Espectro de Aceleración A(ω)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, FREQ_MAX])
    
    # --- Gráfica 2: Espectro de Fuerza Total ---
    ax2 = axes[1]
    ax2.plot(freqs_plot, fft_fuerza[mask] * 1000, 'orange', linewidth=1, label='F(ω)')
    ax2.axvline(freq_exc, color=color_fund, linestyle='--', alpha=0.5)
    ax2.set_ylabel('Fuerza (mV)', fontsize=11)
    ax2.set_title('Espectro de Fuerza Total F(ω)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # --- Gráfica 3: Espectro de Fricción ---
    ax3 = axes[2]
    ax3.plot(freqs_plot, fft_friccion[mask] * 1000, 'magenta', linewidth=1, label='F_fricción(ω)')
    
    # Resaltar armónicos
    for n in range(1, 8):
        f_arm = n * freq_exc
        if f_arm <= FREQ_MAX:
            if n == 1:
                ax3.axvline(f_arm, color=color_fund, linestyle='--', alpha=0.7, 
                           label=f'1f (fundamental)')
            elif n % 2 == 1:
                ax3.axvline(f_arm, color=color_impar, linestyle=':', alpha=0.7,
                           label=f'{n}f (IMPAR)' if n == 3 else '')
            else:
                ax3.axvline(f_arm, color=color_par, linestyle=':', alpha=0.3,
                           label=f'{n}f (par)' if n == 2 else '')
    
    # Marcar picos detectados
    for pico in picos:
        if pico['freq'] <= FREQ_MAX and pico['armónico'] > 0:
            idx = np.argmin(np.abs(freqs - pico['freq']))
            color = color_impar if pico['es_impar'] else color_par
            ax3.plot(pico['freq'], fft_friccion[idx] * 1000, 'o', 
                    color=color, markersize=10, markerfacecolor='none', linewidth=2)
            if pico['armónico'] > 1:
                ax3.annotate(f"{pico['armónico']}f", 
                            xy=(pico['freq'], fft_friccion[idx] * 1000),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, color=color, fontweight='bold')
    
    ax3.set_xlabel('Frecuencia (Hz)', fontsize=11)
    ax3.set_ylabel('Fricción (mV)', fontsize=11)
    ax3.set_title('Espectro de Fricción F_fricción(ω) - Armónicos IMPARES indican no-linealidad', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Añadir nota
    ax3.text(0.02, 0.95, 
             '🔴 Armónicos IMPARES (3f, 5f, 7f) → Fricción de Coulomb/Histéresis\n'
             '🟡 Armónicos PARES (2f, 4f, 6f) → Asimetría en el sistema',
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='white'))
    
    plt.tight_layout()
    return fig


def analizar_multiples_frecuencias(archivos, sensor_name):
    """
    Analiza múltiples frecuencias y genera resumen.
    """
    resultados = []
    
    for archivo in archivos:
        nombre = os.path.basename(archivo)
        try:
            # Extraer frecuencia del nombre
            freq = float(nombre.split('_')[-1].replace('Hz.csv', ''))
            if freq < 5:  # Ignorar frecuencias muy bajas
                continue
        except:
            continue
        
        print(f"  Analizando {nombre}... ", end='')
        try:
            res = analizar_frecuencia(archivo, freq)
            res['archivo'] = nombre
            resultados.append(res)
            print(f"THD = {res['thd']:.1f}%")
        except Exception as e:
            print(f"Error: {e}")
    
    return resultados


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("ANÁLISIS FFT PARA DETECCIÓN DE NO-LINEALIDADES EN FRICCIÓN")
    print("=" * 80)
    print("\nSi hay fricción de Coulomb/Histéresis, aparecerán armónicos IMPARES (3f, 5f...)")
    print("=" * 80)
    
    # Buscar archivos DYMH-105
    pattern_dymh = os.path.join(DATOS_DIR, "caracterizacion_fuerza_20251202_*_exp*.csv")
    archivos_dymh = sorted(glob.glob(pattern_dymh))
    if not archivos_dymh:
        pattern_dymh = os.path.join(DATOS_DIR, "caracterizacion_fuerza_20251201_*_exp*.csv")
        archivos_dymh = sorted(glob.glob(pattern_dymh))
    
    # Buscar archivos LC302-1K
    pattern_lc302 = os.path.join(DATOS_DIR, "caracterizacion_fuerza_20251203_160853_exp*.csv")
    archivos_lc302 = sorted(glob.glob(pattern_lc302))
    # Excluir 10 Hz (anómalo)
    archivos_lc302 = [a for a in archivos_lc302 if "_10Hz.csv" not in a]
    
    todos_resultados = {}
    
    # =========================================================================
    # ANALIZAR DYMH-105
    # =========================================================================
    if archivos_dymh:
        print(f"\n{'='*80}")
        print(f"SENSOR: DYMH-105")
        print(f"{'='*80}")
        print(f"Encontrados {len(archivos_dymh)} archivos\n")
        
        resultados_dymh = analizar_multiples_frecuencias(archivos_dymh, "DYMH-105")
        todos_resultados['DYMH-105'] = resultados_dymh
    
    # =========================================================================
    # ANALIZAR LC302-1K
    # =========================================================================
    if archivos_lc302:
        print(f"\n{'='*80}")
        print(f"SENSOR: LC302-1K")
        print(f"{'='*80}")
        print(f"Encontrados {len(archivos_lc302)} archivos\n")
        
        resultados_lc302 = analizar_multiples_frecuencias(archivos_lc302, "LC302-1K")
        todos_resultados['LC302-1K'] = resultados_lc302
    
    # =========================================================================
    # GENERAR GRÁFICAS PARA FRECUENCIAS REPRESENTATIVAS
    # =========================================================================
    print(f"\n{'='*80}")
    print("GENERANDO GRÁFICAS DE ESPECTROS")
    print(f"{'='*80}")
    
    # Seleccionar frecuencias representativas (20, 40, 70 Hz)
    frecuencias_plot = [20, 40, 70, 25, 35, 50]
    
    for sensor, resultados in todos_resultados.items():
        for freq_target in frecuencias_plot:
            # Buscar resultado más cercano a la frecuencia target
            for res in resultados:
                if abs(res['freq_excitacion'] - freq_target) < 2:
                    print(f"\nGenerando espectro {sensor} @ {res['freq_excitacion']} Hz...")
                    fig = plot_espectros(res, f" - {sensor}")
                    
                    output_file = os.path.join(
                        DATOS_DIR, 
                        f"fft_espectros_{sensor.replace('-', '')}_{int(res['freq_excitacion'])}Hz.png"
                    )
                    fig.savefig(output_file, dpi=150, facecolor='white', bbox_inches='tight')
                    print(f"  📊 Guardado: {output_file}")
                    plt.close(fig)
                    break
    
    # =========================================================================
    # RESUMEN DE THD POR FRECUENCIA
    # =========================================================================
    print(f"\n{'='*80}")
    print("RESUMEN: THD (Total Harmonic Distortion) DE LA FRICCIÓN")
    print(f"{'='*80}")
    print("\nTHD alto → Mayor contenido de armónicos → Fricción más no-lineal")
    print("-" * 60)
    
    for sensor, resultados in todos_resultados.items():
        print(f"\n{sensor}:")
        print(f"{'Freq (Hz)':<12} {'THD (%)':<12} {'Tipo estimado':<20}")
        print("-" * 50)
        
        for res in sorted(resultados, key=lambda x: x['freq_excitacion']):
            thd = res['thd']
            if thd > 30:
                tipo = "Coulomb/Histéresis fuerte"
            elif thd > 15:
                tipo = "No-lineal moderado"
            elif thd > 5:
                tipo = "Ligeramente no-lineal"
            else:
                tipo = "Lineal (viscoso)"
            
            print(f"{res['freq_excitacion']:<12.0f} {thd:<12.1f} {tipo:<20}")
    
    # =========================================================================
    # GRÁFICA COMPARATIVA THD vs FRECUENCIA
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('THD de Fricción vs Frecuencia de Excitación', fontsize=14, fontweight='bold')
    
    colors = {'DYMH-105': 'cyan', 'LC302-1K': 'orange'}
    markers = {'DYMH-105': 'o', 'LC302-1K': 's'}
    
    for sensor, resultados in todos_resultados.items():
        freqs = [r['freq_excitacion'] for r in resultados]
        thds = [r['thd'] for r in resultados]
        
        ax.plot(freqs, thds, f'{markers[sensor]}-', color=colors[sensor], 
                label=sensor, linewidth=2, markersize=8)
    
    # Zonas de referencia
    ax.axhspan(0, 5, alpha=0.2, color='green', label='Lineal (viscoso)')
    ax.axhspan(5, 15, alpha=0.2, color='yellow', label='Ligeramente no-lineal')
    ax.axhspan(15, 30, alpha=0.2, color='orange', label='Moderadamente no-lineal')
    ax.axhspan(30, 100, alpha=0.2, color='red', label='Coulomb/Histéresis')
    
    ax.set_xlabel('Frecuencia de Excitación (Hz)', fontsize=12)
    ax.set_ylabel('THD de Fricción (%)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max([r['freq_excitacion'] for r in resultados]) + 10])
    ax.set_ylim([0, min(100, max([r['thd'] for sensor, resultados in todos_resultados.items() for r in resultados]) * 1.2)])
    
    plt.tight_layout()
    output_file = os.path.join(DATOS_DIR, "fft_thd_comparativo.png")
    fig.savefig(output_file, dpi=150, facecolor='white', bbox_inches='tight')
    print(f"\n📊 Guardado: {output_file}")
    plt.show()
    
    # =========================================================================
    # CONCLUSIONES
    # =========================================================================
    print(f"\n{'='*80}")
    print("CONCLUSIONES")
    print(f"{'='*80}")
    
    for sensor, resultados in todos_resultados.items():
        thd_promedio = np.mean([r['thd'] for r in resultados])
        thd_max = max([r['thd'] for r in resultados])
        freq_max_thd = [r['freq_excitacion'] for r in resultados if r['thd'] == thd_max][0]
        
        print(f"\n{sensor}:")
        print(f"  THD promedio: {thd_promedio:.1f}%")
        print(f"  THD máximo:   {thd_max:.1f}% @ {freq_max_thd} Hz")
        
        if thd_promedio > 20:
            print(f"  → Fricción claramente NO-LINEAL (Coulomb/Histéresis)")
        elif thd_promedio > 10:
            print(f"  → Fricción con componente no-lineal moderada")
        else:
            print(f"  → Fricción predominantemente lineal (viscosa)")


if __name__ == "__main__":
    main()
