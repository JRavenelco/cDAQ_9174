"""
Verificar efecto de celda solo a compresión
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

DATOS_DIR = r"c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\Pruebas\src\caracterizacion_fuerza"

# Cargar calibración
with open(os.path.join(DATOS_DIR, "calibracion_celda.json"), 'r') as f:
    cal = json.load(f)

VOLTAGE_OFFSET = cal['voltage_offset']
VOLTAGE_PER_KG = cal['voltage_per_kg']

def voltaje_a_N(v):
    return ((v - VOLTAGE_OFFSET) / VOLTAGE_PER_KG) * 9.81

plt.style.use('dark_background')

# Cargar un archivo de ejemplo (50 Hz donde hay buena aceleración)
archivo = os.path.join(DATOS_DIR, "caracterizacion_fuerza_20251203_120809_exp9_50Hz.csv")
df = pd.read_csv(archivo)

t = df['tiempo_s'].values
fuerza_V = df['fuerza_V'].values
accel_g = df['aceleracion_sensor_g'].values

F = np.array([voltaje_a_N(v) for v in fuerza_V])
a = accel_g * 9.81

# Tomar solo 5 ciclos para visualizar
fs = 2500
freq = 50
samples_per_cycle = int(fs / freq)
n_cycles = 5
n_samples = samples_per_cycle * n_cycles

t_plot = t[:n_samples] * 1000  # ms
F_plot = F[:n_samples]
a_plot = a[:n_samples]

# Crear figura
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Verificación: Celda Solo a Compresión (50 Hz)', fontsize=14, fontweight='bold')

# Fuerza
ax1 = axes[0]
ax1.plot(t_plot, F_plot, 'c-', linewidth=1)
ax1.axhline(0, color='r', linestyle='--', alpha=0.5)
ax1.axhline(np.mean(F_plot), color='yellow', linestyle=':', label=f'Media: {np.mean(F_plot):.2f} N')
ax1.set_ylabel('Fuerza (N)')
ax1.set_title('Señal de Fuerza (Celda de Carga)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Aceleración
ax2 = axes[1]
ax2.plot(t_plot, a_plot, 'orange', linewidth=1)
ax2.axhline(0, color='r', linestyle='--', alpha=0.5)
ax2.set_ylabel('Aceleración (m/s²)')
ax2.set_title('Señal de Aceleración')
ax2.grid(True, alpha=0.3)

# Superpuestas y normalizadas
ax3 = axes[2]
F_norm = (F_plot - np.mean(F_plot)) / (np.max(F_plot) - np.min(F_plot))
a_norm = (a_plot - np.mean(a_plot)) / (np.max(a_plot) - np.min(a_plot))
ax3.plot(t_plot, F_norm, 'c-', linewidth=1.5, label='Fuerza (norm)')
ax3.plot(t_plot, a_norm, 'orange', linewidth=1.5, alpha=0.7, label='Aceleración (norm)')
ax3.axhline(0, color='r', linestyle='--', alpha=0.5)
ax3.set_xlabel('Tiempo (ms)')
ax3.set_ylabel('Amplitud Normalizada')
ax3.set_title('Comparación Fuerza vs Aceleración')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(DATOS_DIR, 'verificacion_compresion.png'), dpi=150)

# Estadísticas
print("="*60)
print("ANÁLISIS DE SEÑAL DE FUERZA")
print("="*60)
print(f"\nFuerza:")
print(f"  Mínimo: {np.min(F_plot):.2f} N")
print(f"  Máximo: {np.max(F_plot):.2f} N")
print(f"  Media:  {np.mean(F_plot):.2f} N")
print(f"  Pico-Pico: {np.max(F_plot) - np.min(F_plot):.2f} N")

print(f"\nAceleración:")
print(f"  Mínimo: {np.min(a_plot):.2f} m/s²")
print(f"  Máximo: {np.max(a_plot):.2f} m/s²")
print(f"  Pico-Pico: {np.max(a_plot) - np.min(a_plot):.2f} m/s²")

# Verificar si hay recorte (clipping)
F_centered = F_plot - np.mean(F_plot)
pos_peak = np.max(F_centered)
neg_peak = np.abs(np.min(F_centered))
asimetria = pos_peak / neg_peak if neg_peak > 0 else float('inf')

print(f"\nAsimetría de la señal de fuerza:")
print(f"  Pico positivo: {pos_peak:.2f} N")
print(f"  Pico negativo: {neg_peak:.2f} N")
print(f"  Ratio (pos/neg): {asimetria:.2f}")

if asimetria > 1.5 or asimetria < 0.67:
    print(f"\n⚠️  SEÑAL ASIMÉTRICA - Posible efecto de celda solo compresión")
else:
    print(f"\n✅ Señal relativamente simétrica")

plt.show()
