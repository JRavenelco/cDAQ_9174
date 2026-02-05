#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VISUALIZADOR DE LAZOS DE HISTÉRESIS F-x
========================================
Basado en: Deep Research "Physics-AI for CNC Milling Dynamics" (2025)

Objetivo: Validar la hipótesis "Two-Loop Hysteresis"
- Lazo RECTANGULAR → Fricción estructural (Stribeck)
- Lazo ELÍPTICO → Process damping (corte)

Método: Reconstruir desplazamiento desde aceleración mediante integración.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy import signal
from scipy.fft import fft, fftfreq
from pathlib import Path
from glob import glob
import os

# Configuración
DATOS_DIR = Path(__file__).parent / "caracterizacion_fuerza"
SAMPLE_RATE = 2500  # Hz

# Parámetros de conversión de fuerza (del sensor LC302-1K)
CELDA_CAPACIDAD_KG = 454.0  # kg (1000 lb)
CELDA_SENSIBILIDAD_MV_V = 2.0  # mV/V
INA849_GANANCIA = 601.0
V_EXCITACION = 10.0

plt.style.use('dark_background')


def voltaje_a_fuerza_N(V_medido):
    """Convierte voltaje a Newtons (normalizado)."""
    sensibilidad_total = (CELDA_SENSIBILIDAD_MV_V / 1000) * V_EXCITACION * INA849_GANANCIA
    return (V_medido / sensibilidad_total) * CELDA_CAPACIDAD_KG * 9.81


def integrar_aceleracion_omega(acel_ms2, fs, fc_low=0.5):
    """
    Integración en dominio de frecuencia (método omega).
    Más robusto que integración temporal para señales ruidosas.
    """
    n = len(acel_ms2)
    dt = 1.0 / fs
    
    # Remover DC
    acel_ac = acel_ms2 - np.mean(acel_ms2)
    
    # Filtro pasa-altos para evitar drift
    try:
        b, a = signal.butter(2, fc_low / (fs / 2), btype='high')
        acel_filtered = signal.filtfilt(b, a, acel_ac)
    except:
        acel_filtered = acel_ac
    
    # Integrar: a → v
    vel = cumulative_trapezoid(acel_filtered, dx=dt, initial=0)
    vel = signal.detrend(vel)  # Remover drift lineal
    
    # Integrar: v → x
    disp = cumulative_trapezoid(vel, dx=dt, initial=0)
    disp = signal.detrend(disp)  # Remover drift lineal
    
    return vel, disp


def calcular_area_lazo(x, F):
    """
    Calcula el área del lazo de histéresis (energía disipada por ciclo).
    Usa la fórmula del área de polígono: A = 0.5 * |Σ(x_i * F_{i+1} - x_{i+1} * F_i)|
    """
    n = len(x)
    area = 0.0
    for i in range(n - 1):
        area += x[i] * F[i + 1] - x[i + 1] * F[i]
    area = 0.5 * np.abs(area)
    return area


def clasificar_lazo(x, F, freq_hz):
    """
    Clasifica la forma del lazo de histéresis.
    - Rectangular: |correlación con rectangular| > 0.7
    - Elíptico: |correlación con elipse| > 0.7
    """
    # Normalizar
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-10)
    F_norm = (F - np.mean(F)) / (np.std(F) + 1e-10)
    
    # Generar referencias
    t = np.linspace(0, 2 * np.pi, len(x))
    
    # Elipse ideal
    x_elipse = np.sin(t)
    F_elipse = np.sin(t + np.pi/4)  # Desfase típico de amortiguamiento viscoso
    
    # Rectangular ideal (sign function suavizada)
    x_rect = np.sin(t)
    F_rect = np.tanh(5 * np.cos(t))  # Transición abrupta en cruces por cero
    
    # Correlaciones
    corr_elipse = np.abs(np.corrcoef(F_norm, F_elipse)[0, 1])
    corr_rect = np.abs(np.corrcoef(F_norm, F_rect)[0, 1])
    
    # Otra métrica: "cuadratura" del lazo
    # Ratio entre área real y área del rectángulo circunscrito
    area_real = calcular_area_lazo(x, F)
    x_range = np.max(x) - np.min(x)
    F_range = np.max(F) - np.min(F)
    area_rect = x_range * F_range
    fill_ratio = area_real / (area_rect + 1e-10)
    
    # Elipse perfecta tiene fill_ratio ≈ π/4 ≈ 0.785
    # Rectángulo perfecto tiene fill_ratio ≈ 1.0
    
    if fill_ratio > 0.9:
        forma = "RECTANGULAR (Fricción Coulomb/Stribeck)"
    elif 0.7 < fill_ratio <= 0.9:
        forma = "CUASI-RECTANGULAR (Fricción con histéresis)"
    elif 0.5 < fill_ratio <= 0.7:
        forma = "ELÍPTICO (Amortiguamiento viscoso)"
    else:
        forma = "IRREGULAR (Múltiples efectos)"
    
    return forma, fill_ratio, corr_elipse, corr_rect


def analizar_archivo(filepath, mostrar=True):
    """Analiza un archivo de datos y extrae el lazo de histéresis."""
    
    df = pd.read_csv(filepath)
    nombre = os.path.basename(filepath)
    
    # Extraer frecuencia del nombre
    try:
        freq_hz = float(nombre.split('_')[-1].replace('Hz.csv', ''))
    except:
        freq_hz = 0
    
    t = df['tiempo_s'].values
    fuerza_V = df['fuerza_V'].values
    
    # Buscar columna de aceleración
    if 'aceleracion_sensor_g' in df.columns:
        acel_g = df['aceleracion_sensor_g'].values
    elif 'aceleracion_g' in df.columns:
        acel_g = df['aceleracion_g'].values
    else:
        acel_g = df['aceleracion_bancada_g'].values
    
    # Convertir a unidades físicas
    F_N = voltaje_a_fuerza_N(fuerza_V - np.mean(fuerza_V))
    acel_ms2 = (acel_g - np.mean(acel_g)) * 9.81
    
    # Integrar para obtener desplazamiento
    vel_ms, disp_m = integrar_aceleracion_omega(acel_ms2, SAMPLE_RATE, fc_low=max(0.5, freq_hz * 0.2))
    
    # Convertir a mm
    disp_mm = disp_m * 1000
    vel_mms = vel_ms * 1000
    
    # Tomar solo los últimos N ciclos para mejor visualización
    if freq_hz > 0:
        samples_per_cycle = int(SAMPLE_RATE / freq_hz)
        n_cycles = 5
        n_samples = min(samples_per_cycle * n_cycles, len(t))
    else:
        n_samples = min(5000, len(t))
    
    # Extraer segmento
    t_seg = t[-n_samples:]
    F_seg = F_N[-n_samples:]
    x_seg = disp_mm[-n_samples:]
    v_seg = vel_mms[-n_samples:]
    
    # Clasificar lazo
    forma, fill_ratio, corr_e, corr_r = clasificar_lazo(x_seg, F_seg, freq_hz)
    
    # Calcular energía
    area_mJ = calcular_area_lazo(x_seg / 1000, F_seg)  # x en metros, F en N → Joules
    
    resultado = {
        'archivo': nombre,
        'freq_hz': freq_hz,
        'forma': forma,
        'fill_ratio': fill_ratio,
        'energia_mJ': area_mJ * 1000,
        'x_rms_um': np.std(x_seg) * 1000,
        'F_rms_N': np.std(F_seg),
        'v_rms_mms': np.std(v_seg)
    }
    
    if mostrar:
        print(f"\n{'='*60}")
        print(f"📂 {nombre}")
        print(f"{'='*60}")
        print(f"   Frecuencia: {freq_hz:.0f} Hz")
        print(f"   Forma del lazo: {forma}")
        print(f"   Fill ratio: {fill_ratio:.3f} (1.0=rect, 0.785=elipse)")
        print(f"   Energía por ciclo: {area_mJ*1000:.4f} mJ")
        print(f"   x_RMS: {np.std(x_seg)*1000:.2f} µm")
        print(f"   F_RMS: {np.std(F_seg):.3f} N")
    
    return resultado, t_seg, F_seg, x_seg, v_seg


def graficar_lazo(t, F, x, v, resultado, save_path=None):
    """Genera gráfica completa del análisis de histéresis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Análisis de Histéresis - {resultado['archivo']}\n{resultado['forma']}", 
                 fontsize=14, fontweight='bold')
    
    # 1. Lazo F vs x (PRINCIPAL)
    ax1 = axes[0, 0]
    ax1.plot(x, F, 'c-', lw=0.8, alpha=0.8)
    ax1.scatter(x[0], F[0], c='lime', s=100, zorder=5, marker='o', label='Inicio')
    ax1.set_xlabel('Desplazamiento (mm)')
    ax1.set_ylabel('Fuerza (N)')
    ax1.set_title(f'Lazo de Histéresis F-x\nÁrea = {resultado["energia_mJ"]:.4f} mJ')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('auto')
    
    # Añadir rectángulo de referencia
    x_min, x_max = np.min(x), np.max(x)
    F_min, F_max = np.min(F), np.max(F)
    rect = plt.Rectangle((x_min, F_min), x_max - x_min, F_max - F_min, 
                          fill=False, edgecolor='yellow', linestyle='--', alpha=0.5, lw=2)
    ax1.add_patch(rect)
    
    # 2. Lazo F vs v (Característica de fricción)
    ax2 = axes[0, 1]
    ax2.plot(v, F, 'm-', lw=0.8, alpha=0.8)
    ax2.scatter(v[0], F[0], c='lime', s=100, zorder=5, marker='o')
    ax2.set_xlabel('Velocidad (mm/s)')
    ax2.set_ylabel('Fuerza (N)')
    ax2.set_title('Característica F-v (Curva de Stribeck)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Señales en tiempo
    ax3 = axes[1, 0]
    t_ms = (t - t[0]) * 1000
    ax3.plot(t_ms, F / np.max(np.abs(F)), 'c-', lw=1, label='Fuerza (norm)')
    ax3.plot(t_ms, x / np.max(np.abs(x)), 'orange', lw=1, alpha=0.7, label='Desplaz. (norm)')
    ax3.set_xlabel('Tiempo (ms)')
    ax3.set_ylabel('Amplitud normalizada')
    ax3.set_title('Señales F(t) y x(t)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. FFT de fuerza
    ax4 = axes[1, 1]
    n = len(F)
    freqs = fftfreq(n, 1/SAMPLE_RATE)[:n//2]
    fft_F = np.abs(fft(F - np.mean(F)))[:n//2]
    fft_F_db = 20 * np.log10(fft_F + 1e-12)
    ax4.plot(freqs, fft_F_db, 'c-', lw=1)
    ax4.axvline(resultado['freq_hz'], color='r', linestyle='--', alpha=0.7, 
                label=f"f_exc = {resultado['freq_hz']:.0f} Hz")
    ax4.set_xlim(0, min(200, SAMPLE_RATE/2))
    ax4.set_xlabel('Frecuencia (Hz)')
    ax4.set_ylabel('Magnitud (dB)')
    ax4.set_title('FFT de Fuerza')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Guardado: {save_path}")
    
    return fig


def main():
    """Función principal: analiza todos los archivos disponibles."""
    
    print("="*70)
    print("  VISUALIZADOR DE LAZOS DE HISTÉRESIS")
    print("  Basado en: Deep Research 'Physics-AI for CNC Milling' (2025)")
    print("="*70)
    
    # Buscar archivos de datos
    archivos = sorted(glob(str(DATOS_DIR / "caracterizacion_fuerza_*_exp*_*Hz.csv")))
    
    if not archivos:
        print(f"⚠️  No se encontraron archivos en {DATOS_DIR}")
        return
    
    print(f"\n📁 Encontrados {len(archivos)} archivos de experimentos")
    
    # Analizar cada archivo
    resultados = []
    for archivo in archivos[-6:]:  # Solo los últimos 6 para no saturar
        resultado, t, F, x, v = analizar_archivo(archivo)
        resultados.append(resultado)
        
        # Graficar
        nombre_base = os.path.basename(archivo).replace('.csv', '')
        save_path = DATOS_DIR / f"lazo_histeresis_{nombre_base}.png"
        graficar_lazo(t, F, x, v, resultado, save_path)
    
    # Resumen
    print("\n" + "="*70)
    print("  RESUMEN: CLASIFICACIÓN DE LAZOS")
    print("="*70)
    print(f"\n{'Freq (Hz)':<10} {'Fill Ratio':<12} {'Energía (mJ)':<14} {'Forma':<40}")
    print("-"*76)
    
    for r in resultados:
        print(f"{r['freq_hz']:<10.0f} {r['fill_ratio']:<12.3f} {r['energia_mJ']:<14.4f} {r['forma']:<40}")
    
    # Conclusión automática
    print("\n" + "="*70)
    print("  CONCLUSIÓN (Hipótesis Two-Loop)")
    print("="*70)
    
    fill_promedio = np.mean([r['fill_ratio'] for r in resultados])
    
    if fill_promedio > 0.85:
        print("\n✅ FORMA PREDOMINANTE: RECTANGULAR")
        print("   → Indica FRICCIÓN ESTRUCTURAL (Stribeck/Coulomb)")
        print("   → Sistema operando en régimen de fricción de guías")
        print("   → Comportamiento esperado para air-cutting (sin cortador)")
    elif fill_promedio > 0.65:
        print("\n🔶 FORMA PREDOMINANTE: CUASI-RECTANGULAR/MIXTA")
        print("   → Indica combinación de fricción + amortiguamiento")
        print("   → Posible transición entre regímenes")
    else:
        print("\n🔵 FORMA PREDOMINANTE: ELÍPTICA")
        print("   → Indica AMORTIGUAMIENTO VISCOSO (Process Damping)")
        print("   → Característico de corte con disipación de energía")
    
    print(f"\n   Fill ratio promedio: {fill_promedio:.3f}")
    print(f"   (1.0 = rectangular, 0.785 = elíptico ideal)\n")
    
    plt.show()


if __name__ == "__main__":
    main()
