#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS DE DATOS DEL 4 DE DICIEMBRE 2025
==========================================
Barrido 10-40 Hz con LC302-1K
"""

import os
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

# Archivos de hoy (set completo 110740)
ARCHIVOS = {
    10: "caracterizacion_fuerza_20251204_110740_exp1_10Hz.csv",
    20: "caracterizacion_fuerza_20251204_110740_exp2_20Hz.csv",
    30: "caracterizacion_fuerza_20251204_110740_exp3_30Hz.csv",
    40: "caracterizacion_fuerza_20251204_110740_exp4_40Hz.csv",
}

# =============================================================================
# FUNCIONES
# =============================================================================

def cargar_datos(freq):
    """Carga y procesa datos de una frecuencia."""
    archivo = os.path.join(DATOS_DIR, ARCHIVOS[freq])
    df = pd.read_csv(archivo)
    
    t = df['tiempo_s'].values
    fuerza_V = df['fuerza_V'].values
    acel_sensor = df['aceleracion_sensor_g'].values
    acel_bancada = df['aceleracion_bancada_g'].values
    
    # Descartar transitorio (0.5 s)
    skip = int(0.5 * SAMPLE_RATE)
    t = t[skip:] - t[skip]
    fuerza_V = fuerza_V[skip:]
    acel_sensor = acel_sensor[skip:]
    acel_bancada = acel_bancada[skip:]
    
    # Remover DC
    fuerza_ac = fuerza_V - np.mean(fuerza_V)
    acel_ac = (acel_sensor - np.mean(acel_sensor)) * G  # m/s²
    
    return {
        't': t,
        'fuerza_V': fuerza_V,
        'fuerza_ac': fuerza_ac,
        'acel_g': acel_sensor,
        'acel_ms2': acel_ac,
        'fuerza_dc': np.mean(fuerza_V),
    }


def calcular_fft(signal_data, fs=SAMPLE_RATE):
    """Calcula FFT normalizada."""
    N = len(signal_data)
    window = np.hanning(N)
    fft_result = fft(signal_data * window)
    freqs = fftfreq(N, 1/fs)
    
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    magnitud = np.abs(fft_result[pos_mask]) * 4 / N
    
    return freqs, magnitud


def calcular_friccion(datos, m_eff=None):
    """Calcula fricción como F_fric = F_total - m·a."""
    if m_eff is None:
        # Estimar m de la relación F/a
        m_eff = np.std(datos['fuerza_ac']) / (np.std(datos['acel_ms2']) + 1e-10)
    
    friccion = datos['fuerza_ac'] - m_eff * datos['acel_ms2']
    return friccion, m_eff


def calcular_thd(freqs, magnitud, freq_fundamental):
    """Calcula THD (Total Harmonic Distortion)."""
    idx_fund = np.argmin(np.abs(freqs - freq_fundamental))
    mag_fund = magnitud[idx_fund]
    
    # Armónicos impares (3f, 5f, 7f)
    potencia_arm = 0
    for n in [3, 5, 7]:
        f_arm = n * freq_fundamental
        if f_arm < freqs[-1]:
            idx = np.argmin(np.abs(freqs - f_arm))
            potencia_arm += magnitud[idx]**2
    
    thd = np.sqrt(potencia_arm) / (mag_fund + 1e-10) * 100
    return thd


# =============================================================================
# MAIN
# =============================================================================

print("=" * 80)
print("ANÁLISIS DE DATOS - 4 DICIEMBRE 2025")
print("Barrido 10-40 Hz con LC302-1K")
print("=" * 80)

# Cargar todos los datos
todos_datos = {}
for freq in ARCHIVOS:
    print(f"\nCargando {freq} Hz...", end=" ")
    datos = cargar_datos(freq)
    todos_datos[freq] = datos
    print(f"OK - {len(datos['t'])} muestras, DC={datos['fuerza_dc']*1000:.2f} mV")

# =============================================================================
# FIGURA 1: SEÑALES TEMPORALES
# =============================================================================
print("\n" + "=" * 80)
print("GENERANDO GRÁFICAS")
print("=" * 80)

fig1, axes = plt.subplots(4, 2, figsize=(14, 12))
fig1.suptitle('Señales Temporales - Barrido 10-40 Hz (4 Dic 2025)', fontsize=14, fontweight='bold')

for i, freq in enumerate(sorted(ARCHIVOS.keys())):
    datos = todos_datos[freq]
    t = datos['t']
    
    # Solo mostrar 0.5 segundos
    n_show = int(0.5 * SAMPLE_RATE)
    t_show = t[:n_show] * 1000  # ms
    
    # Fuerza
    ax1 = axes[i, 0]
    ax1.plot(t_show, datos['fuerza_ac'][:n_show] * 1000, 'orange', linewidth=0.8)
    ax1.set_ylabel(f'{freq} Hz\nFuerza (mV)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 200])
    if i == 0:
        ax1.set_title('Fuerza AC')
    
    # Aceleración
    ax2 = axes[i, 1]
    ax2.plot(t_show, datos['acel_g'][:n_show] * 1000, 'cyan', linewidth=0.8)
    ax2.set_ylabel(f'Acel (mg)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 200])
    if i == 0:
        ax2.set_title('Aceleración AC')

axes[-1, 0].set_xlabel('Tiempo (ms)')
axes[-1, 1].set_xlabel('Tiempo (ms)')

plt.tight_layout()
fig1.savefig(os.path.join(DATOS_DIR, "analisis_hoy_temporal.png"), dpi=150, facecolor='white')
print("📊 Guardado: analisis_hoy_temporal.png")

# =============================================================================
# FIGURA 2: ANÁLISIS FFT
# =============================================================================
fig2, axes = plt.subplots(4, 3, figsize=(15, 12))
fig2.suptitle('Análisis FFT - Detección de Armónicos (4 Dic 2025)', fontsize=14, fontweight='bold')

resultados_thd = {}

for i, freq in enumerate(sorted(ARCHIVOS.keys())):
    datos = todos_datos[freq]
    
    # Calcular fricción
    friccion, m_eff = calcular_friccion(datos)
    
    # FFT de las 3 señales
    freqs_fft, fft_acel = calcular_fft(datos['acel_ms2'])
    _, fft_fuerza = calcular_fft(datos['fuerza_ac'])
    _, fft_friccion = calcular_fft(friccion)
    
    # THD
    thd = calcular_thd(freqs_fft, fft_friccion, freq)
    resultados_thd[freq] = thd
    
    # Máscara para plot (0-150 Hz)
    mask = freqs_fft <= 150
    
    # Aceleración
    ax1 = axes[i, 0]
    ax1.plot(freqs_fft[mask], fft_acel[mask] * 1000 / G, 'cyan', linewidth=1)
    ax1.axvline(freq, color='lime', linestyle='--', alpha=0.5)
    ax1.set_ylabel(f'{freq} Hz\nAcel (mg)')
    ax1.grid(True, alpha=0.3)
    if i == 0:
        ax1.set_title('Espectro Aceleración')
    
    # Fuerza
    ax2 = axes[i, 1]
    ax2.plot(freqs_fft[mask], fft_fuerza[mask] * 1000, 'orange', linewidth=1)
    ax2.axvline(freq, color='lime', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Fuerza (mV)')
    ax2.grid(True, alpha=0.3)
    if i == 0:
        ax2.set_title('Espectro Fuerza')
    
    # Fricción
    ax3 = axes[i, 2]
    ax3.plot(freqs_fft[mask], fft_friccion[mask] * 1000, 'magenta', linewidth=1)
    ax3.axvline(freq, color='lime', linestyle='--', alpha=0.5, label='1f')
    
    # Marcar armónicos impares
    for n in [3, 5]:
        f_arm = n * freq
        if f_arm <= 150:
            ax3.axvline(f_arm, color='red', linestyle=':', alpha=0.7)
            ax3.text(f_arm, ax3.get_ylim()[1]*0.9, f'{n}f', color='red', fontsize=8, ha='center')
    
    ax3.set_ylabel('Fricción (mV)')
    ax3.set_title(f'THD = {thd:.1f}%' if i == 0 else f'THD = {thd:.1f}%')
    ax3.grid(True, alpha=0.3)
    if i == 0:
        ax3.set_title(f'Espectro Fricción | THD = {thd:.1f}%')

axes[-1, 0].set_xlabel('Frecuencia (Hz)')
axes[-1, 1].set_xlabel('Frecuencia (Hz)')
axes[-1, 2].set_xlabel('Frecuencia (Hz)')

plt.tight_layout()
fig2.savefig(os.path.join(DATOS_DIR, "analisis_hoy_fft.png"), dpi=150, facecolor='white')
print("📊 Guardado: analisis_hoy_fft.png")

# =============================================================================
# FIGURA 3: HISTÉRESIS F vs x
# =============================================================================
fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
fig3.suptitle('Curvas de Histéresis F vs x (4 Dic 2025)', fontsize=14, fontweight='bold')

for i, freq in enumerate(sorted(ARCHIVOS.keys())):
    datos = todos_datos[freq]
    ax = axes[i // 2, i % 2]
    
    # Calcular posición por doble integración
    acel_ms2 = datos['acel_ms2']
    dt = 1 / SAMPLE_RATE
    
    # Filtro pasa-alto para eliminar drift
    fc = max(freq / 10, 0.5)
    b, a = signal.butter(2, fc / (SAMPLE_RATE / 2), btype='high')
    
    vel = cumulative_trapezoid(acel_ms2, dx=dt, initial=0)
    vel = signal.filtfilt(b, a, vel)
    
    pos = cumulative_trapezoid(vel, dx=dt, initial=0)
    pos = signal.filtfilt(b, a, pos)
    
    # Solo unos ciclos
    n_ciclos = 5
    n_puntos = int(n_ciclos * SAMPLE_RATE / freq)
    
    fuerza_plot = datos['fuerza_ac'][:n_puntos] * 1000  # mV
    pos_plot = pos[:n_puntos] * 1e6  # µm
    
    # Plot con color por tiempo
    points = np.array([pos_plot, fuerza_plot]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    ax.plot(pos_plot, fuerza_plot, 'c-', linewidth=0.8, alpha=0.7)
    ax.scatter(pos_plot[0], fuerza_plot[0], c='green', s=50, zorder=5, label='Inicio')
    ax.scatter(pos_plot[-1], fuerza_plot[-1], c='red', s=50, zorder=5, label='Fin')
    
    ax.set_xlabel('Posición (µm)')
    ax.set_ylabel('Fuerza (mV)')
    ax.set_title(f'{freq} Hz - {n_ciclos} ciclos')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
fig3.savefig(os.path.join(DATOS_DIR, "analisis_hoy_histeresis.png"), dpi=150, facecolor='white')
print("📊 Guardado: analisis_hoy_histeresis.png")

# =============================================================================
# RESUMEN
# =============================================================================
print("\n" + "=" * 80)
print("RESUMEN DE RESULTADOS")
print("=" * 80)

print(f"\n{'Freq (Hz)':<12} {'F_rms (mV)':<12} {'a_rms (mg)':<12} {'THD (%)':<12} {'Tipo':<25}")
print("-" * 70)

for freq in sorted(ARCHIVOS.keys()):
    datos = todos_datos[freq]
    f_rms = np.std(datos['fuerza_ac']) * 1000
    a_rms = np.std(datos['acel_g']) * 1000
    thd = resultados_thd[freq]
    
    if thd > 30:
        tipo = "Coulomb/Histéresis fuerte"
    elif thd > 15:
        tipo = "No-lineal moderado"
    elif thd > 5:
        tipo = "Ligeramente no-lineal"
    else:
        tipo = "Lineal (viscoso)"
    
    print(f"{freq:<12} {f_rms:<12.2f} {a_rms:<12.2f} {thd:<12.1f} {tipo:<25}")

print("-" * 70)
print(f"\nTHD promedio: {np.mean(list(resultados_thd.values())):.1f}%")

# Calcular m efectiva promedio
m_effs = []
for freq in ARCHIVOS:
    datos = todos_datos[freq]
    _, m = calcular_friccion(datos)
    m_effs.append(m)

print(f"Masa efectiva promedio: {np.mean(m_effs)*1000:.3f} mV/(m/s²)")

plt.show()

print("\n✅ Análisis completado")
