#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS DE DATOS CON 2 ACELERÓMETROS
=====================================
- ai0: Acelerómetro en bancada
- ai1: Acelerómetro sobre sensor de fuerza

Análisis:
1. Señales en tiempo
2. FFT de ambos acelerómetros
3. FRF: H(f) = Acel_sensor / Fuerza
4. Transmisibilidad: T(f) = Acel_sensor / Acel_bancada
5. Coherencia
6. Fase entre señales
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import os
from glob import glob

# Configuración
SAMPLE_RATE = 2500  # Hz
DATOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caracterizacion_fuerza")

# Estilo de gráficas
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def cargar_sesion_mas_reciente():
    """Carga la sesión más reciente con 2 acelerómetros"""
    # Buscar archivos de resumen
    resumenes = sorted(glob(os.path.join(DATOS_DIR, "*_resumen.csv")), reverse=True)
    
    for resumen_path in resumenes:
        df_resumen = pd.read_csv(resumen_path)
        # Verificar si tiene columnas de 2 acelerómetros
        if 'accel0_rms_g' in df_resumen.columns and 'accel1_rms_g' in df_resumen.columns:
            print(f"✅ Sesión encontrada: {os.path.basename(resumen_path)}")
            return resumen_path, df_resumen
    
    print("❌ No se encontraron datos con 2 acelerómetros")
    return None, None


def cargar_experimentos(resumen_path):
    """Carga todos los experimentos de una sesión"""
    base_name = resumen_path.replace('_resumen.csv', '')
    experimentos = []
    
    exp_files = sorted(glob(f"{base_name}_exp*.csv"))
    for exp_file in exp_files:
        df = pd.read_csv(exp_file)
        # Extraer frecuencia del nombre
        freq = float(os.path.basename(exp_file).split('_')[-1].replace('Hz.csv', ''))
        experimentos.append({
            'frecuencia': freq,
            'data': df,
            'file': exp_file
        })
    
    print(f"📊 Cargados {len(experimentos)} experimentos")
    return experimentos


def calcular_frf_transmisibilidad(fuerza, acel_bancada, acel_sensor, fs=SAMPLE_RATE, nperseg=1024):
    """
    Calcula FRF y Transmisibilidad usando estimador H1
    
    FRF: H(f) = Acel_sensor / Fuerza
    Transmisibilidad: T(f) = Acel_sensor / Acel_bancada
    """
    # FRF: Aceleración sensor / Fuerza
    f_frf, Pxy_frf = signal.csd(fuerza, acel_sensor, fs=fs, nperseg=nperseg)
    _, Pxx_frf = signal.welch(fuerza, fs=fs, nperseg=nperseg)
    H_frf = Pxy_frf / (Pxx_frf + 1e-12)
    
    # Coherencia FRF
    _, Pyy_frf = signal.welch(acel_sensor, fs=fs, nperseg=nperseg)
    coherencia_frf = np.abs(Pxy_frf)**2 / (Pxx_frf * Pyy_frf + 1e-12)
    
    # Transmisibilidad: Acel_sensor / Acel_bancada
    _, Pxy_trans = signal.csd(acel_bancada, acel_sensor, fs=fs, nperseg=nperseg)
    _, Pxx_trans = signal.welch(acel_bancada, fs=fs, nperseg=nperseg)
    T = Pxy_trans / (Pxx_trans + 1e-12)
    
    # Coherencia Transmisibilidad
    _, Pyy_trans = signal.welch(acel_sensor, fs=fs, nperseg=nperseg)
    coherencia_trans = np.abs(Pxy_trans)**2 / (Pxx_trans * Pyy_trans + 1e-12)
    
    return {
        'freqs': f_frf,
        'H_frf': H_frf,
        'coherencia_frf': coherencia_frf,
        'T': T,
        'coherencia_trans': coherencia_trans
    }


def analizar_experimento(exp, ax_time, ax_fft, ax_frf, ax_trans):
    """Analiza un experimento individual"""
    df = exp['data']
    freq = exp['frecuencia']
    
    t = df['tiempo_s'].values
    fuerza = df['fuerza_V'].values
    acel_bancada = df['aceleracion_bancada_g'].values
    acel_sensor = df['aceleracion_sensor_g'].values
    
    # Remover DC
    fuerza = fuerza - np.mean(fuerza)
    acel_bancada = acel_bancada - np.mean(acel_bancada)
    acel_sensor = acel_sensor - np.mean(acel_sensor)
    
    # Señales en tiempo (últimos 0.5 segundos)
    n_show = int(0.5 * SAMPLE_RATE)
    ax_time.plot(t[-n_show:], fuerza[-n_show:] / np.std(fuerza), 
                 label=f'Fuerza {freq}Hz', alpha=0.7)
    ax_time.plot(t[-n_show:], acel_sensor[-n_show:] / np.std(acel_sensor), 
                 label=f'Acel Sensor {freq}Hz', alpha=0.7, linestyle='--')
    
    # FFT
    n = len(fuerza)
    freqs = fftfreq(n, 1/SAMPLE_RATE)[:n//2]
    
    fft_bancada = np.abs(fft(acel_bancada))[:n//2]
    fft_sensor = np.abs(fft(acel_sensor))[:n//2]
    
    fft_bancada_db = 20 * np.log10(fft_bancada + 1e-12)
    fft_sensor_db = 20 * np.log10(fft_sensor + 1e-12)
    
    ax_fft.plot(freqs, fft_bancada_db, label=f'Bancada {freq}Hz', alpha=0.7)
    ax_fft.plot(freqs, fft_sensor_db, label=f'Sensor {freq}Hz', alpha=0.7, linestyle='--')
    
    # Calcular FRF y Transmisibilidad
    resultados = calcular_frf_transmisibilidad(fuerza, acel_bancada, acel_sensor)
    
    # FRF
    H_mag = 20 * np.log10(np.abs(resultados['H_frf']) + 1e-12)
    ax_frf.plot(resultados['freqs'], H_mag, label=f'{freq}Hz', alpha=0.8)
    
    # Transmisibilidad
    T_mag = 20 * np.log10(np.abs(resultados['T']) + 1e-12)
    ax_trans.plot(resultados['freqs'], T_mag, label=f'{freq}Hz', alpha=0.8)
    
    return resultados


def generar_reporte(df_resumen, experimentos):
    """Genera un reporte de análisis"""
    print("\n" + "="*60)
    print("📊 REPORTE DE ANÁLISIS - 2 ACELERÓMETROS")
    print("="*60)
    
    print(f"\n📁 Total experimentos: {len(experimentos)}")
    print(f"📈 Rango de frecuencias: {df_resumen['frecuencia_Hz'].min():.0f} - {df_resumen['frecuencia_Hz'].max():.0f} Hz")
    
    print("\n📋 RESUMEN POR FRECUENCIA:")
    print("-"*60)
    print(f"{'Freq':>6} | {'Acel0 RMS':>10} | {'Acel1 RMS':>10} | {'Trans':>8} | {'Fase':>8}")
    print(f"{'(Hz)':>6} | {'(g)':>10} | {'(g)':>10} | {'':>8} | {'(°)':>8}")
    print("-"*60)
    
    for _, row in df_resumen.iterrows():
        print(f"{row['frecuencia_Hz']:>6.0f} | "
              f"{row['accel0_rms_g']:>10.6f} | "
              f"{row['accel1_rms_g']:>10.6f} | "
              f"{row['transmisibilidad']:>8.4f} | "
              f"{row['fase_deg']:>8.2f}")
    
    print("-"*60)
    
    # Estadísticas de transmisibilidad
    trans_mean = df_resumen['transmisibilidad'].mean()
    trans_std = df_resumen['transmisibilidad'].std()
    print(f"\n📈 TRANSMISIBILIDAD PROMEDIO: {trans_mean:.4f} ± {trans_std:.4f}")
    
    if trans_mean < 1:
        print("   → El sistema ATENÚA las vibraciones (T < 1)")
    else:
        print("   → El sistema AMPLIFICA las vibraciones (T > 1)")
    
    # Análisis de fase
    print(f"\n📐 FASE PROMEDIO: {df_resumen['fase_deg'].mean():.2f}°")
    
    return trans_mean, trans_std


def main():
    print("\n" + "="*60)
    print("🔬 ANÁLISIS DE DATOS CON 2 ACELERÓMETROS")
    print("="*60)
    
    # Cargar datos
    resumen_path, df_resumen = cargar_sesion_mas_reciente()
    if resumen_path is None:
        return
    
    experimentos = cargar_experimentos(resumen_path)
    if not experimentos:
        return
    
    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis Dual Acelerómetro - Transmisibilidad del Sistema', fontsize=14, fontweight='bold')
    
    ax_time = axes[0, 0]
    ax_fft = axes[0, 1]
    ax_frf = axes[1, 0]
    ax_trans = axes[1, 1]
    
    # Analizar cada experimento
    for exp in experimentos:
        analizar_experimento(exp, ax_time, ax_fft, ax_frf, ax_trans)
    
    # Configurar gráficas
    ax_time.set_xlabel('Tiempo (s)')
    ax_time.set_ylabel('Amplitud (normalizada)')
    ax_time.set_title('Señales en Tiempo (Fuerza vs Acel Sensor)')
    ax_time.legend(loc='upper right', fontsize=8)
    ax_time.grid(True, alpha=0.3)
    
    ax_fft.set_xlabel('Frecuencia (Hz)')
    ax_fft.set_ylabel('Magnitud (dB)')
    ax_fft.set_title('FFT - Comparación Acelerómetros')
    ax_fft.set_xlim(0, 200)
    ax_fft.legend(loc='upper right', fontsize=8)
    ax_fft.grid(True, alpha=0.3)
    
    ax_frf.set_xlabel('Frecuencia (Hz)')
    ax_frf.set_ylabel('|H(f)| (dB)')
    ax_frf.set_title('FRF: H(f) = Acel_Sensor / Fuerza')
    ax_frf.set_xlim(0, 200)
    ax_frf.legend(loc='upper right', fontsize=8)
    ax_frf.grid(True, alpha=0.3)
    
    ax_trans.set_xlabel('Frecuencia (Hz)')
    ax_trans.set_ylabel('|T(f)| (dB)')
    ax_trans.set_title('Transmisibilidad: T(f) = Acel_Sensor / Acel_Bancada')
    ax_trans.set_xlim(0, 200)
    ax_trans.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='T=1 (0 dB)')
    ax_trans.legend(loc='upper right', fontsize=8)
    ax_trans.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar figura
    timestamp = os.path.basename(resumen_path).split('_')[2] + '_' + os.path.basename(resumen_path).split('_')[3]
    fig_path = os.path.join(DATOS_DIR, f"analisis_dual_acel_{timestamp}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Figura guardada: {fig_path}")
    
    # Generar reporte
    generar_reporte(df_resumen, experimentos)
    
    # Mostrar
    plt.show()


if __name__ == "__main__":
    main()
