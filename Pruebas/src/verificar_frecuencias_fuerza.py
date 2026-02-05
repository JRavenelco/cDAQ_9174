#!/usr/bin/env python3
"""
Verificación: ¿El sensor de fuerza detecta las frecuencias de excitación?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from glob import glob
import os

SAMPLE_RATE = 2500
plt.style.use('dark_background')

DATOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caracterizacion_fuerza")

# Buscar sesión más reciente con 2 acelerómetros
resumenes = sorted(glob(os.path.join(DATOS_DIR, "*20251202*_resumen.csv")), reverse=True)
if resumenes:
    base = resumenes[0].replace('_resumen.csv', '')
else:
    print("No se encontraron datos")
    exit()

exp_files = sorted(glob(f'{base}_exp*.csv'))
print(f'Encontrados {len(exp_files)} experimentos')

print('='*60)
print('VERIFICACIÓN: ¿El sensor de fuerza detecta las frecuencias?')
print('='*60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('FFT del Sensor de Fuerza - Detección de Frecuencias de Excitación', fontsize=14, fontweight='bold')

resultados = []

for i, exp_file in enumerate(exp_files):
    freq_nominal = float(os.path.basename(exp_file).split('_')[-1].replace('Hz.csv', ''))
    
    df = pd.read_csv(exp_file)
    fuerza = df['fuerza_V'].values - np.mean(df['fuerza_V'].values)
    
    # FFT
    n = len(fuerza)
    freqs = fftfreq(n, 1/SAMPLE_RATE)[:n//2]
    fft_mag = np.abs(fft(fuerza))[:n//2]
    fft_db = 20 * np.log10(fft_mag + 1e-12)
    
    # Encontrar pico principal (ignorar DC y muy bajas frecuencias)
    idx_start = int(5 * n / SAMPLE_RATE)  # Desde 5 Hz
    idx_max = np.argmax(fft_mag[idx_start:]) + idx_start
    freq_detectada = freqs[idx_max]
    magnitud_pico = fft_db[idx_max]
    
    # Error de detección
    error = abs(freq_detectada - freq_nominal)
    error_pct = error / freq_nominal * 100
    
    resultados.append({
        'nominal': freq_nominal,
        'detectada': freq_detectada,
        'error': error,
        'error_pct': error_pct,
        'magnitud_dB': magnitud_pico
    })
    
    # Graficar
    ax = axes.flat[i]
    ax.plot(freqs, fft_db, 'c-', lw=1, alpha=0.8)
    ax.axvline(x=freq_nominal, color='lime', linestyle='--', lw=2, alpha=0.8, label=f'Nominal: {freq_nominal:.0f} Hz')
    ax.axvline(x=freq_detectada, color='red', linestyle=':', lw=2, alpha=0.8, label=f'Detectada: {freq_detectada:.1f} Hz')
    ax.scatter([freq_detectada], [magnitud_pico], c='yellow', s=100, zorder=5, marker='*')
    
    ax.set_xlim(0, 150)
    ax.set_ylim(-50, max(fft_db) + 10)
    ax.set_title(f'Excitación: {freq_nominal:.0f} Hz', fontsize=12, fontweight='bold')
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Magnitud (dB)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Indicador de detección
    if error < 2:
        status = '✅ DETECTADA'
        color = 'lime'
    else:
        status = '⚠️ DESVIACIÓN'
        color = 'orange'
    
    ax.text(0.02, 0.98, status, transform=ax.transAxes, fontsize=11, 
            verticalalignment='top', color=color, fontweight='bold')

# Último subplot: Resumen
ax_sum = axes.flat[5]
ax_sum.axis('off')

summary_lines = [
    'RESUMEN DE DETECCIÓN',
    '=' * 30,
    '',
    f"{'Nominal':>10} | {'Detectada':>10} | {'Error':>8}",
    '-' * 35
]

for r in resultados:
    status = '✅' if r['error'] < 2 else '⚠️'
    summary_lines.append(f"{r['nominal']:>10.0f} Hz | {r['detectada']:>10.1f} Hz | {r['error']:>6.2f} Hz {status}")

summary_lines.append('-' * 35)
summary_lines.append(f"\nError promedio: {np.mean([r['error'] for r in resultados]):.2f} Hz")

summary = '\n'.join(summary_lines)

ax_sum.text(0.1, 0.9, summary, transform=ax_sum.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))

plt.tight_layout()
fig_path = os.path.join(DATOS_DIR, 'verificacion_fuerza_frecuencias.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f'\nGuardado: {fig_path}')

print('\n' + '='*60)
print('RESULTADOS:')
print('='*60)
print(f"{'Nominal':>10} | {'Detectada':>10} | {'Error':>8} | {'Magnitud':>10}")
print('-'*50)
for r in resultados:
    status = '✅' if r['error'] < 2 else '⚠️'
    print(f"{r['nominal']:>10.0f} Hz | {r['detectada']:>10.1f} Hz | {r['error']:>6.2f} Hz | {r['magnitud_dB']:>8.1f} dB {status}")
print('-'*50)
print(f"Error promedio: {np.mean([r['error'] for r in resultados]):.2f} Hz")

if all(r['error'] < 2 for r in resultados):
    print('\n✅ El sensor de fuerza DETECTA CORRECTAMENTE todas las frecuencias de excitación')
else:
    print('\n⚠️ Algunas frecuencias tienen desviación significativa')

plt.show()
