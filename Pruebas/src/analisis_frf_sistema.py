#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS FRF DEL SISTEMA BANCADA-SHAKER

Objetivo: Caracterizar la relación Aceleración/Fuerza en función de la frecuencia
- Entrada: Fuerza (celda de carga)
- Salida: Aceleración (PCB 352C33)
- FRF: H(f) = Aceleración(f) / Fuerza(f)

Sistema:
    Shaker → Celda de carga → Bancada (fija) ← Acelerómetro
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from glob import glob
import os

# Configuración de plots
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Directorio de datos
DATOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caracterizacion_fuerza")
SAMPLE_RATE = 2500  # Hz

def cargar_experimento(filepath):
    """Carga un archivo de experimento"""
    df = pd.read_csv(filepath)
    return df['tiempo_s'].values, df['fuerza_V'].values, df['aceleracion_g'].values

def calcular_frf(fuerza, aceleracion, fs, nperseg=2048):
    """
    Calcula la FRF usando el método H1 (minimiza ruido en salida)
    
    H1(f) = Sxy(f) / Sxx(f)
    
    donde:
    - Sxy = Cross-spectral density (fuerza × aceleración)
    - Sxx = Auto-spectral density (fuerza × fuerza)
    """
    # Cross-spectral density
    f, Pxy = signal.csd(fuerza, aceleracion, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    
    # Auto-spectral density de la entrada (fuerza)
    _, Pxx = signal.welch(fuerza, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    
    # Auto-spectral density de la salida (aceleración)
    _, Pyy = signal.welch(aceleracion, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    
    # FRF H1
    H1 = Pxy / (Pxx + 1e-12)
    
    # Coherencia
    coherencia = np.abs(Pxy)**2 / (Pxx * Pyy + 1e-12)
    
    return f, H1, coherencia

def analizar_experimento(filepath):
    """Analiza un experimento individual"""
    nombre = os.path.basename(filepath)
    print(f"\n📂 Analizando: {nombre}")
    
    t, fuerza, aceleracion = cargar_experimento(filepath)
    
    # Remover DC
    fuerza = fuerza - np.mean(fuerza)
    aceleracion = aceleracion - np.mean(aceleracion)
    
    # Calcular FRF
    f, H1, coherencia = calcular_frf(fuerza, aceleracion, SAMPLE_RATE)
    
    # Magnitud y fase
    magnitud = np.abs(H1)
    fase = np.angle(H1, deg=True)
    
    # Encontrar frecuencia dominante
    idx_max = np.argmax(magnitud[1:]) + 1
    freq_dom = f[idx_max]
    mag_max = magnitud[idx_max]
    fase_dom = fase[idx_max]
    coh_dom = coherencia[idx_max]
    
    print(f"   Frecuencia dominante: {freq_dom:.2f} Hz")
    print(f"   |H(f)|: {mag_max:.4f} g/V")
    print(f"   Fase: {fase_dom:.2f}°")
    print(f"   Coherencia: {coh_dom:.4f}")
    
    return {
        'nombre': nombre,
        'tiempo': t,
        'fuerza': fuerza,
        'aceleracion': aceleracion,
        'frecuencias': f,
        'H1': H1,
        'magnitud': magnitud,
        'fase': fase,
        'coherencia': coherencia,
        'freq_dom': freq_dom,
        'mag_max': mag_max,
        'fase_dom': fase_dom,
        'coh_dom': coh_dom
    }

def plot_frf_individual(resultado, save_path=None):
    """Genera gráficas para un experimento"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"Análisis FRF: {resultado['nombre']}", fontsize=14, fontweight='bold')
    
    t = resultado['tiempo']
    f = resultado['frecuencias']
    
    # Limitar a primeros 0.1 segundos para visualización
    n_show = min(int(0.1 * SAMPLE_RATE), len(t))
    
    # 1. Señales en tiempo
    ax1 = axes[0, 0]
    ax1.plot(t[:n_show]*1000, resultado['fuerza'][:n_show], 'b-', linewidth=0.8, label='Fuerza (V)')
    ax1.set_xlabel('Tiempo (ms)')
    ax1.set_ylabel('Fuerza (V)')
    ax1.set_title('Señal de Fuerza')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(t[:n_show]*1000, resultado['aceleracion'][:n_show], 'r-', linewidth=0.8, label='Aceleración (g)')
    ax2.set_xlabel('Tiempo (ms)')
    ax2.set_ylabel('Aceleración (g)')
    ax2.set_title('Señal de Aceleración')
    ax2.grid(True, alpha=0.3)
    
    # 2. Magnitud FRF
    ax3 = axes[1, 0]
    ax3.semilogy(f, resultado['magnitud'], 'g-', linewidth=1.5)
    ax3.axvline(resultado['freq_dom'], color='yellow', linestyle='--', alpha=0.7, 
                label=f"f = {resultado['freq_dom']:.1f} Hz")
    ax3.set_xlabel('Frecuencia (Hz)')
    ax3.set_ylabel('|H(f)| (g/V)')
    ax3.set_title('FRF - Magnitud')
    ax3.set_xlim([0, 500])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 3. Fase FRF
    ax4 = axes[1, 1]
    ax4.plot(f, resultado['fase'], 'm-', linewidth=1.5)
    ax4.axvline(resultado['freq_dom'], color='yellow', linestyle='--', alpha=0.7)
    ax4.axhline(0, color='white', linestyle=':', alpha=0.5)
    ax4.axhline(-90, color='cyan', linestyle=':', alpha=0.5, label='-90°')
    ax4.axhline(-180, color='red', linestyle=':', alpha=0.5, label='-180°')
    ax4.set_xlabel('Frecuencia (Hz)')
    ax4.set_ylabel('Fase (°)')
    ax4.set_title('FRF - Fase')
    ax4.set_xlim([0, 500])
    ax4.set_ylim([-180, 180])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 4. Coherencia
    ax5 = axes[2, 0]
    ax5.plot(f, resultado['coherencia'], 'c-', linewidth=1.5)
    ax5.axhline(0.9, color='green', linestyle='--', alpha=0.7, label='Umbral 0.9')
    ax5.axvline(resultado['freq_dom'], color='yellow', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Frecuencia (Hz)')
    ax5.set_ylabel('Coherencia γ²')
    ax5.set_title('Coherencia (calidad de la medición)')
    ax5.set_xlim([0, 500])
    ax5.set_ylim([0, 1.1])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 5. Diagrama de Lissajous
    ax6 = axes[2, 1]
    # Normalizar para visualización
    fuerza_norm = resultado['fuerza'][:n_show] / (np.max(np.abs(resultado['fuerza'][:n_show])) + 1e-12)
    accel_norm = resultado['aceleracion'][:n_show] / (np.max(np.abs(resultado['aceleracion'][:n_show])) + 1e-12)
    ax6.plot(fuerza_norm, accel_norm, 'y-', linewidth=0.5, alpha=0.7)
    ax6.set_xlabel('Fuerza (norm)')
    ax6.set_ylabel('Aceleración (norm)')
    ax6.set_title(f'Lissajous (Fase ≈ {resultado["fase_dom"]:.1f}°)')
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"   💾 Guardado: {save_path}")
    
    return fig

def plot_frf_combinada(resultados, save_path=None):
    """Genera gráfica combinada de todos los experimentos"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FRF Combinada - Caracterización del Sistema", fontsize=14, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(resultados)))
    
    for i, res in enumerate(resultados):
        freq_label = f"{res['freq_dom']:.0f} Hz"
        
        # Magnitud
        axes[0, 0].semilogy(res['frecuencias'], res['magnitud'], 
                           color=colors[i], linewidth=1.5, label=freq_label, alpha=0.8)
        
        # Fase
        axes[0, 1].plot(res['frecuencias'], res['fase'], 
                       color=colors[i], linewidth=1.5, label=freq_label, alpha=0.8)
        
        # Coherencia
        axes[1, 0].plot(res['frecuencias'], res['coherencia'], 
                       color=colors[i], linewidth=1.5, label=freq_label, alpha=0.8)
    
    # Configurar ejes
    axes[0, 0].set_xlabel('Frecuencia (Hz)')
    axes[0, 0].set_ylabel('|H(f)| (g/V)')
    axes[0, 0].set_title('Magnitud FRF')
    axes[0, 0].set_xlim([0, 500])
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Frecuencia (Hz)')
    axes[0, 1].set_ylabel('Fase (°)')
    axes[0, 1].set_title('Fase FRF')
    axes[0, 1].set_xlim([0, 500])
    axes[0, 1].set_ylim([-180, 180])
    axes[0, 1].axhline(-90, color='cyan', linestyle=':', alpha=0.5)
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Frecuencia (Hz)')
    axes[1, 0].set_ylabel('Coherencia γ²')
    axes[1, 0].set_title('Coherencia')
    axes[1, 0].set_xlim([0, 500])
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].axhline(0.9, color='green', linestyle='--', alpha=0.7)
    axes[1, 0].legend(loc='lower right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Resumen en texto
    ax_text = axes[1, 1]
    ax_text.axis('off')
    
    summary = "RESUMEN DE CARACTERIZACIÓN\n"
    summary += "=" * 40 + "\n\n"
    summary += f"{'Experimento':<20} {'f_dom':>8} {'|H|':>10} {'Fase':>8} {'Coh':>6}\n"
    summary += "-" * 52 + "\n"
    
    for res in resultados:
        nombre_corto = res['nombre'].replace('caracterizacion_fuerza_', '').replace('.csv', '')[-15:]
        summary += f"{nombre_corto:<20} {res['freq_dom']:>6.1f} Hz {res['mag_max']:>8.4f} {res['fase_dom']:>7.1f}° {res['coh_dom']:>5.3f}\n"
    
    ax_text.text(0.1, 0.9, summary, transform=ax_text.transAxes, 
                fontfamily='monospace', fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\n💾 Guardado: {save_path}")
    
    return fig

def plot_bode(resultados, save_path=None):
    """Genera diagrama de Bode del sistema"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Diagrama de Bode - Sistema Bancada", fontsize=14, fontweight='bold')
    
    # Promediar FRFs si hay múltiples experimentos
    if len(resultados) > 1:
        # Usar el que tenga mejor coherencia promedio
        best_idx = np.argmax([np.mean(r['coherencia']) for r in resultados])
        res = resultados[best_idx]
        print(f"\n📊 Usando experimento con mejor coherencia: {res['nombre']}")
    else:
        res = resultados[0]
    
    f = res['frecuencias']
    mag_db = 20 * np.log10(res['magnitud'] + 1e-12)
    fase = res['fase']
    
    # Magnitud en dB
    axes[0].plot(f, mag_db, 'g-', linewidth=2)
    axes[0].set_ylabel('Magnitud (dB)')
    axes[0].set_title('|H(f)| = Aceleración / Fuerza')
    axes[0].set_xlim([1, 500])
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].axhline(0, color='white', linestyle=':', alpha=0.5)
    
    # Marcar resonancias (picos)
    peaks, _ = signal.find_peaks(mag_db, height=-20, distance=10)
    for peak in peaks[:5]:  # Máximo 5 picos
        axes[0].axvline(f[peak], color='yellow', linestyle='--', alpha=0.5)
        axes[0].annotate(f'{f[peak]:.1f} Hz', (f[peak], mag_db[peak]), 
                        textcoords="offset points", xytext=(5, 5), fontsize=9, color='yellow')
    
    # Fase
    axes[1].plot(f, fase, 'm-', linewidth=2)
    axes[1].set_xlabel('Frecuencia (Hz)')
    axes[1].set_ylabel('Fase (°)')
    axes[1].set_xlim([1, 500])
    axes[1].set_ylim([-180, 180])
    axes[1].axhline(0, color='white', linestyle=':', alpha=0.5)
    axes[1].axhline(-90, color='cyan', linestyle='--', alpha=0.5, label='-90° (resonancia)')
    axes[1].axhline(-180, color='red', linestyle='--', alpha=0.5, label='-180°')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3, which='both')
    
    for peak in peaks[:5]:
        axes[1].axvline(f[peak], color='yellow', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"💾 Guardado: {save_path}")
    
    return fig

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔬 ANÁLISIS FRF DEL SISTEMA BANCADA-SHAKER")
    print("="*60)
    
    # Buscar el último conjunto de experimentos
    resumen_files = sorted(glob(os.path.join(DATOS_DIR, "*_resumen.csv")))
    
    if not resumen_files:
        print("❌ No se encontraron archivos de resumen")
        exit(1)
    
    ultimo_resumen = resumen_files[-1]
    timestamp = os.path.basename(ultimo_resumen).replace('caracterizacion_fuerza_', '').replace('_resumen.csv', '')
    
    print(f"\n📂 Analizando sesión: {timestamp}")
    
    # Buscar archivos de experimentos de esa sesión
    exp_files = sorted(glob(os.path.join(DATOS_DIR, f"caracterizacion_fuerza_{timestamp}_exp*.csv")))
    
    print(f"   Encontrados {len(exp_files)} experimentos")
    
    # Analizar cada experimento
    resultados = []
    for filepath in exp_files:
        try:
            res = analizar_experimento(filepath)
            resultados.append(res)
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
    
    if not resultados:
        print("❌ No se pudieron analizar los experimentos")
        exit(1)
    
    # Generar gráficas
    print("\n📊 Generando gráficas...")
    
    # FRF individual del mejor experimento
    best_idx = np.argmax([r['coh_dom'] for r in resultados])
    plot_frf_individual(resultados[best_idx], 
                       os.path.join(DATOS_DIR, f"frf_individual_{timestamp}.png"))
    
    # FRF combinada
    plot_frf_combinada(resultados, 
                      os.path.join(DATOS_DIR, f"frf_combinada_{timestamp}.png"))
    
    # Diagrama de Bode
    plot_bode(resultados, 
             os.path.join(DATOS_DIR, f"bode_{timestamp}.png"))
    
    # Resumen final
    print("\n" + "="*60)
    print("📋 RESUMEN DE CARACTERIZACIÓN")
    print("="*60)
    print(f"\n{'Experimento':<30} {'f_dom':>8} {'|H|':>10} {'Fase':>8} {'Coh':>6}")
    print("-" * 62)
    for res in resultados:
        nombre = res['nombre'].split('_')[-1].replace('.csv', '')
        print(f"{nombre:<30} {res['freq_dom']:>6.1f} Hz {res['mag_max']:>8.4f} {res['fase_dom']:>7.1f}° {res['coh_dom']:>5.3f}")
    
    print("\n" + "="*60)
    print("✅ Análisis completado")
    print(f"   Gráficas guardadas en: {DATOS_DIR}")
    print("="*60)
    
    plt.show()
