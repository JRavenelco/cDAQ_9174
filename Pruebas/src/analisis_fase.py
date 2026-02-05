#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de frecuencia y desfase entre señales de Fuerza (entrada) y Aceleración (salida)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

# Cargar datos
archivo = "experimentos_mesa/sesion_20251126_170150.csv"
print(f"📂 Cargando: {archivo}")

# Leer CSV saltando las líneas de comentario y líneas mal formateadas
df = pd.read_csv(archivo, comment='#', on_bad_lines='skip')
print(f"📊 Columnas: {df.columns.tolist()}")
print(f"📏 Muestras: {len(df)}")

# Extraer señales
tiempo = df['Tiempo(s)'].values
aceleracion = df['Acel_ai0(g)'].values  # Salida
fuerza = df['Volt_ai0(V)'].values       # Entrada

# Calcular sample rate
dt = tiempo[1] - tiempo[0]
fs = 1 / dt
print(f"⏱️ Sample Rate: {fs:.1f} Hz")
print(f"⏱️ Duración: {tiempo[-1]:.2f} s")

# Remover DC (offset)
aceleracion = aceleracion - np.mean(aceleracion)
fuerza = fuerza - np.mean(fuerza)

# ============================================
# 1. ANÁLISIS FFT - Frecuencias dominantes
# ============================================
print("\n" + "="*50)
print("📈 ANÁLISIS DE FRECUENCIA (FFT)")
print("="*50)

N = len(tiempo)
freqs = fftfreq(N, dt)[:N//2]

# FFT de aceleración
fft_acel = np.abs(fft(aceleracion))[:N//2]
fft_acel_db = 20 * np.log10(fft_acel / np.max(fft_acel) + 1e-12)

# FFT de fuerza
fft_fuerza = np.abs(fft(fuerza))[:N//2]
fft_fuerza_db = 20 * np.log10(fft_fuerza / np.max(fft_fuerza) + 1e-12)

# Encontrar picos dominantes
# Limitar búsqueda a frecuencias > 1 Hz para evitar DC residual
mask = freqs > 1
idx_pico_acel = np.argmax(fft_acel[mask]) + np.argmax(mask)
idx_pico_fuerza = np.argmax(fft_fuerza[mask]) + np.argmax(mask)

freq_dom_acel = freqs[idx_pico_acel]
freq_dom_fuerza = freqs[idx_pico_fuerza]

print(f"\n🎯 Frecuencia dominante ACELERACIÓN: {freq_dom_acel:.2f} Hz")
print(f"🎯 Frecuencia dominante FUERZA:      {freq_dom_fuerza:.2f} Hz")

# ============================================
# 2. ANÁLISIS DE DESFASE (Cross-correlation)
# ============================================
print("\n" + "="*50)
print("🔄 ANÁLISIS DE DESFASE")
print("="*50)

# Método 1: Correlación cruzada
correlation = signal.correlate(aceleracion, fuerza, mode='full')
lags = signal.correlation_lags(len(aceleracion), len(fuerza), mode='full')
lag_idx = np.argmax(np.abs(correlation))
lag_samples = lags[lag_idx]
lag_tiempo = lag_samples / fs

# Calcular fase en grados (para la frecuencia dominante)
freq_analisis = (freq_dom_acel + freq_dom_fuerza) / 2  # Promedio
periodo = 1 / freq_analisis
fase_grados = (lag_tiempo / periodo) * 360

# Normalizar fase a [-180, 180]
while fase_grados > 180:
    fase_grados -= 360
while fase_grados < -180:
    fase_grados += 360

print(f"\n📐 Desfase temporal: {lag_tiempo*1000:.3f} ms ({lag_samples} muestras)")
print(f"📐 Desfase angular:  {fase_grados:.2f}° (a {freq_analisis:.2f} Hz)")

if fase_grados > 0:
    print(f"   → La ACELERACIÓN está ADELANTADA respecto a la FUERZA")
else:
    print(f"   → La ACELERACIÓN está RETRASADA respecto a la FUERZA")

# Método 2: Fase FFT directa
# Calcular fase en la frecuencia dominante
fft_acel_complex = fft(aceleracion)[:N//2]
fft_fuerza_complex = fft(fuerza)[:N//2]

# Encontrar índice más cercano a la frecuencia dominante
idx_freq = np.argmin(np.abs(freqs - freq_analisis))
fase_acel = np.angle(fft_acel_complex[idx_freq], deg=True)
fase_fuerza = np.angle(fft_fuerza_complex[idx_freq], deg=True)
diferencia_fase = fase_acel - fase_fuerza

# Normalizar
while diferencia_fase > 180:
    diferencia_fase -= 360
while diferencia_fase < -180:
    diferencia_fase += 360

print(f"\n📐 Desfase (método FFT): {diferencia_fase:.2f}°")

# ============================================
# 3. FUNCIÓN DE TRANSFERENCIA (FRF)
# ============================================
print("\n" + "="*50)
print("📊 FUNCIÓN DE TRANSFERENCIA (FRF)")
print("="*50)

# H(f) = Aceleración / Fuerza
H = fft_acel_complex / (fft_fuerza_complex + 1e-12)
H_mag = np.abs(H)
H_phase = np.angle(H, deg=True)

# Magnitud y fase en la frecuencia dominante
H_mag_dom = H_mag[idx_freq]
H_phase_dom = H_phase[idx_freq]

print(f"\n🔧 En f = {freq_analisis:.2f} Hz:")
print(f"   Magnitud |H|: {H_mag_dom:.4f} g/V")
print(f"   Fase ∠H:      {H_phase_dom:.2f}°")

# ============================================
# 4. GRÁFICAS
# ============================================
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle(f'Análisis Fuerza → Aceleración | Frecuencia: {freq_analisis:.1f} Hz | Desfase: {diferencia_fase:.1f}°', 
             fontsize=14, fontweight='bold')

# Señales en tiempo (zoom a 5 ciclos)
n_ciclos = 5
t_zoom = n_ciclos / freq_analisis
idx_zoom = int(t_zoom * fs)

ax1 = axes[0, 0]
ax1.plot(tiempo[:idx_zoom]*1000, fuerza[:idx_zoom], 'b-', label='Fuerza (V)', linewidth=1)
ax1.set_xlabel('Tiempo (ms)')
ax1.set_ylabel('Fuerza (V)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_title('Señales en Tiempo (Entrada vs Salida)')

ax1b = ax1.twinx()
ax1b.plot(tiempo[:idx_zoom]*1000, aceleracion[:idx_zoom], 'r-', label='Aceleración (g)', linewidth=1)
ax1b.set_ylabel('Aceleración (g)', color='r')
ax1b.tick_params(axis='y', labelcolor='r')
ax1b.legend(loc='upper right')

# FFT Fuerza
ax2 = axes[0, 1]
ax2.plot(freqs, fft_fuerza_db, 'b-', linewidth=0.8)
ax2.axvline(freq_dom_fuerza, color='b', linestyle='--', alpha=0.7, label=f'Pico: {freq_dom_fuerza:.1f} Hz')
ax2.set_xlabel('Frecuencia (Hz)')
ax2.set_ylabel('Magnitud (dB)')
ax2.set_xlim([0, 200])
ax2.set_ylim([-60, 5])
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title('FFT Fuerza (Entrada)')

# FFT Aceleración
ax3 = axes[1, 0]
ax3.plot(freqs, fft_acel_db, 'r-', linewidth=0.8)
ax3.axvline(freq_dom_acel, color='r', linestyle='--', alpha=0.7, label=f'Pico: {freq_dom_acel:.1f} Hz')
ax3.set_xlabel('Frecuencia (Hz)')
ax3.set_ylabel('Magnitud (dB)')
ax3.set_xlim([0, 200])
ax3.set_ylim([-60, 5])
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_title('FFT Aceleración (Salida)')

# Correlación cruzada
ax4 = axes[1, 1]
lag_ms = lags / fs * 1000
ax4.plot(lag_ms, correlation / np.max(np.abs(correlation)), 'g-', linewidth=0.8)
ax4.axvline(lag_tiempo*1000, color='r', linestyle='--', label=f'Desfase: {lag_tiempo*1000:.2f} ms')
ax4.set_xlabel('Lag (ms)')
ax4.set_ylabel('Correlación normalizada')
ax4.set_xlim([-50, 50])
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_title('Correlación Cruzada')

# FRF Magnitud
ax5 = axes[2, 0]
H_mag_db = 20 * np.log10(H_mag + 1e-12)
ax5.plot(freqs, H_mag_db, 'm-', linewidth=0.8)
ax5.axvline(freq_analisis, color='k', linestyle='--', alpha=0.5)
ax5.set_xlabel('Frecuencia (Hz)')
ax5.set_ylabel('|H(f)| (dB)')
ax5.set_xlim([0, 200])
ax5.set_ylim([-40, 40])
ax5.grid(True, alpha=0.3)
ax5.set_title('FRF - Magnitud')

# FRF Fase
ax6 = axes[2, 1]
ax6.plot(freqs, H_phase, 'c-', linewidth=0.8)
ax6.axvline(freq_analisis, color='k', linestyle='--', alpha=0.5)
ax6.axhline(H_phase_dom, color='r', linestyle=':', alpha=0.7, label=f'Fase @ {freq_analisis:.1f} Hz: {H_phase_dom:.1f}°')
ax6.set_xlabel('Frecuencia (Hz)')
ax6.set_ylabel('Fase (°)')
ax6.set_xlim([0, 200])
ax6.set_ylim([-180, 180])
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_title('FRF - Fase')

plt.tight_layout()
plt.savefig('experimentos_mesa/analisis_fase_resultado.png', dpi=150)
print(f"\n💾 Gráfica guardada: experimentos_mesa/analisis_fase_resultado.png")

# ============================================
# 5. DIAGRAMA DE FASE (LISSAJOUS) Y 3D
# ============================================
from mpl_toolkits.mplot3d import Axes3D

fig2 = plt.figure(figsize=(16, 6))

# --- Normalización Min-Max a [-1, 1] (como en MATLAB) ---
# Fuerza = 2 * ((FuerzaD - minForz) / (maxForz - minForz)) - 1
min_fuerza = np.min(fuerza)
max_fuerza = np.max(fuerza)
min_acel = np.min(aceleracion)
max_acel = np.max(aceleracion)

fuerza_norm = 2 * ((fuerza - min_fuerza) / (max_fuerza - min_fuerza)) - 1
acel_norm = 2 * ((aceleracion - min_acel) / (max_acel - min_acel)) - 1

print(f"\n📊 Normalización Min-Max a [-1, 1]:")
print(f"   Fuerza:      min={min_fuerza:.6f} V, max={max_fuerza:.6f} V")
print(f"   Aceleración: min={min_acel:.6f} g, max={max_acel:.6f} g")

# --- Suavizado Savitzky-Golay (como smoothdata en MATLAB) ---
from scipy.signal import savgol_filter
window_length = 51  # Debe ser impar
polyorder = 3

fuerza_smooth = savgol_filter(fuerza_norm, window_length, polyorder)
acel_smooth = savgol_filter(acel_norm, window_length, polyorder)
print(f"   Suavizado Savitzky-Golay: ventana={window_length}, orden={polyorder}")

# ============================================
# FIGURA 2: Señales Original vs Suavizado (como MATLAB)
# ============================================
fig2 = plt.figure(figsize=(16, 8))

# Zoom para visualización clara
n_ciclos_zoom = 3
idx_zoom = int(n_ciclos_zoom / freq_analisis * fs)
t_zoom_ms = tiempo[:idx_zoom] * 1000  # en ms

# --- Fuerza: Original vs Suavizado ---
ax_f = fig2.add_subplot(2, 2, 1)
ax_f.plot(t_zoom_ms, fuerza_norm[:idx_zoom], 'b-', linewidth=0.8, alpha=0.6, label='Original')
ax_f.plot(t_zoom_ms, fuerza_smooth[:idx_zoom], 'r-', linewidth=1.5, label='Suavizado')
ax_f.set_xlabel('Tiempo (ms)', fontsize=11)
ax_f.set_ylabel('Fuerza (normalizada)', fontsize=11)
ax_f.set_title('Fuerza: Original vs Suavizado', fontsize=12, fontweight='bold')
ax_f.set_ylim([-1.5, 1.5])
ax_f.legend()
ax_f.grid(True, alpha=0.3)

# --- Aceleración: Original vs Suavizado ---
ax_a = fig2.add_subplot(2, 2, 2)
ax_a.plot(t_zoom_ms, acel_norm[:idx_zoom], 'b-', linewidth=0.8, alpha=0.6, label='Original')
ax_a.plot(t_zoom_ms, acel_smooth[:idx_zoom], 'r-', linewidth=1.5, label='Suavizado')
ax_a.set_xlabel('Tiempo (ms)', fontsize=11)
ax_a.set_ylabel('Aceleración (normalizada)', fontsize=11)
ax_a.set_title('Aceleración: Original vs Suavizado', fontsize=12, fontweight='bold')
ax_a.set_ylim([-1.5, 1.5])
ax_a.legend()
ax_a.grid(True, alpha=0.3)

# --- Ambas señales suavizadas superpuestas ---
ax_both = fig2.add_subplot(2, 2, 3)
ax_both.plot(t_zoom_ms, fuerza_smooth[:idx_zoom], 'b-', linewidth=1.5, label='Fuerza')
ax_both.plot(t_zoom_ms, acel_smooth[:idx_zoom], 'r-', linewidth=1.5, label='Aceleración')
ax_both.set_xlabel('Tiempo (ms)', fontsize=11)
ax_both.set_ylabel('Amplitud (normalizada)', fontsize=11)
ax_both.set_title('Fuerza vs Aceleración (Suavizadas)', fontsize=12, fontweight='bold')
ax_both.set_ylim([-1.5, 1.5])
ax_both.legend()
ax_both.grid(True, alpha=0.3)

# --- Diagrama de Lissajous (Suavizado) ---
ax_liss = fig2.add_subplot(2, 2, 4)
n_ciclos_liss = 10
idx_liss = int(n_ciclos_liss / freq_analisis * fs)

# Colorear por tiempo para ver dirección
colors = np.linspace(0, 1, idx_liss)
scatter = ax_liss.scatter(fuerza_smooth[:idx_liss], acel_smooth[:idx_liss], 
                          c=colors, cmap='viridis', s=2, alpha=0.8)
ax_liss.plot(fuerza_smooth[:idx_liss], acel_smooth[:idx_liss], 'b-', alpha=0.3, linewidth=0.5)

ax_liss.set_xlabel('Fuerza (normalizada)', fontsize=11)
ax_liss.set_ylabel('Aceleración (normalizada)', fontsize=11)
ax_liss.set_title(f'Diagrama de Lissajous (Suavizado)\nDesfase: {diferencia_fase:.1f}°', fontsize=12, fontweight='bold')
ax_liss.set_xlim([-1.2, 1.2])
ax_liss.set_ylim([-1.2, 1.2])
ax_liss.set_aspect('equal')
ax_liss.grid(True, alpha=0.3)
ax_liss.axhline(0, color='k', linewidth=0.5)
ax_liss.axvline(0, color='k', linewidth=0.5)
cbar = plt.colorbar(scatter, ax=ax_liss, label='Tiempo (norm)')

plt.tight_layout()
plt.savefig('experimentos_mesa/analisis_fase_suavizado.png', dpi=150)
print(f"💾 Gráfica suavizado guardada: experimentos_mesa/analisis_fase_suavizado.png")

# ============================================
# FIGURA 3: Diagramas 3D (como MATLAB)
# ============================================
fig3 = plt.figure(figsize=(16, 6))

# --- 3D Original ---
ax_3d1 = fig3.add_subplot(1, 3, 1, projection='3d')
n_ciclos_3d = 5
idx_3d = int(n_ciclos_3d / freq_analisis * fs)
step = max(1, idx_3d // 2000)

t_3d = tiempo[:idx_3d:step] * 1000  # ms
f_3d = fuerza_norm[:idx_3d:step]
a_3d = acel_norm[:idx_3d:step]

ax_3d1.plot(t_3d, f_3d, a_3d, 'b-', linewidth=0.8, alpha=0.8)
ax_3d1.set_xlabel('Tiempo (ms)', fontsize=10)
ax_3d1.set_ylabel('Fuerza (norm)', fontsize=10)
ax_3d1.set_zlabel('Aceleración (norm)', fontsize=10)
ax_3d1.set_title('3D Original\nTiempo × Fuerza × Aceleración', fontsize=11, fontweight='bold')
ax_3d1.view_init(elev=21.34, azim=-57.86)  # Vista similar a MATLAB
ax_3d1.grid(True)

# --- 3D Suavizado ---
ax_3d2 = fig3.add_subplot(1, 3, 2, projection='3d')
f_3d_smooth = fuerza_smooth[:idx_3d:step]
a_3d_smooth = acel_smooth[:idx_3d:step]

ax_3d2.plot(t_3d, f_3d_smooth, a_3d_smooth, 'r-', linewidth=1.2, alpha=0.9)
ax_3d2.set_xlabel('Tiempo (ms)', fontsize=10)
ax_3d2.set_ylabel('Fuerza Suavizada', fontsize=10)
ax_3d2.set_zlabel('Aceleración Suavizada', fontsize=10)
ax_3d2.set_title('3D Suavizado\nTiempo × Fuerza × Aceleración', fontsize=11, fontweight='bold')
ax_3d2.view_init(elev=21.34, azim=-57.86)
ax_3d2.grid(True)

# --- 3D Vista alternativa ---
ax_3d3 = fig3.add_subplot(1, 3, 3, projection='3d')
colors_3d = t_3d
scatter3d = ax_3d3.scatter(t_3d, f_3d_smooth, a_3d_smooth, c=colors_3d, cmap='plasma', s=3, alpha=0.8)
ax_3d3.plot(t_3d, f_3d_smooth, a_3d_smooth, 'b-', linewidth=0.5, alpha=0.4)

# Marcar inicio y fin
ax_3d3.scatter([t_3d[0]], [f_3d_smooth[0]], [a_3d_smooth[0]], c='green', s=100, marker='o', label='Inicio')
ax_3d3.scatter([t_3d[-1]], [f_3d_smooth[-1]], [a_3d_smooth[-1]], c='red', s=100, marker='s', label='Fin')

ax_3d3.set_xlabel('Tiempo (ms)', fontsize=10)
ax_3d3.set_ylabel('Fuerza (norm)', fontsize=10)
ax_3d3.set_zlabel('Aceleración (norm)', fontsize=10)
ax_3d3.set_title(f'Vista Helicoidal Suavizada\n{n_ciclos_3d} ciclos @ {freq_analisis:.1f} Hz', fontsize=11, fontweight='bold')
ax_3d3.view_init(elev=26.71, azim=-60.46)  # Vista alternativa MATLAB
ax_3d3.legend(loc='upper left')
ax_3d3.grid(True)

plt.tight_layout()
plt.savefig('experimentos_mesa/analisis_fase_3d.png', dpi=150)
print(f"💾 Gráfica 3D guardada: experimentos_mesa/analisis_fase_3d.png")
plt.show()

# ============================================
# RESUMEN FINAL
# ============================================
print("\n" + "="*50)
print("📋 RESUMEN DEL ANÁLISIS")
print("="*50)
print(f"  Archivo:           {archivo}")
print(f"  Sample Rate:       {fs:.0f} Hz")
print(f"  Duración:          {tiempo[-1]:.2f} s")
print(f"  Frecuencia señal:  {freq_analisis:.2f} Hz")
print(f"  Desfase temporal:  {lag_tiempo*1000:.3f} ms")
print(f"  Desfase angular:   {diferencia_fase:.2f}°")
print(f"  |H(f)|:            {H_mag_dom:.4f} g/V")
print("="*50)
