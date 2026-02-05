"""
Recálculo del Barrido con Calibración Correcta
==============================================
Usa los datos existentes + factor de calibración experimental
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import os
import json

DATOS_DIR = r"c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\Pruebas\src\caracterizacion_fuerza"
SAMPLE_RATE = 2500

# Cargar calibración
with open(os.path.join(DATOS_DIR, "calibracion_celda.json"), 'r') as f:
    cal = json.load(f)

VOLTAGE_OFFSET = cal['voltage_offset']
VOLTAGE_PER_KG = cal['voltage_per_kg']

print("="*70)
print("RECÁLCULO CON CALIBRACIÓN EXPERIMENTAL")
print("="*70)
print(f"Offset: {VOLTAGE_OFFSET:.5f} V")
print(f"Factor: {VOLTAGE_PER_KG:.6f} V/kg ({VOLTAGE_PER_KG*9.81:.6f} V/N)")

plt.style.use('dark_background')


def voltaje_a_N(v):
    """Convierte voltaje a Newtons con calibración"""
    return ((v - VOLTAGE_OFFSET) / VOLTAGE_PER_KG) * 9.81


def integrar_senoidal(accel, fs, freq):
    """Integración en frecuencia"""
    n = len(accel)
    freqs = fftfreq(n, 1/fs)
    omega = 2 * np.pi * freqs
    
    A = fft(accel)
    bw = max(freq * 0.3, 2)
    bp = np.exp(-((np.abs(freqs) - freq) / bw)**2)
    hp = 1 - np.exp(-(np.abs(freqs) / 0.5)**4)
    filtro = bp * hp
    
    omega_safe = np.where(np.abs(omega) > 1e-10, omega, 1e-10)
    
    V = A * filtro / (1j * omega_safe)
    V[np.abs(omega) < 1e-10] = 0
    velocity = np.real(ifft(V))
    
    X = fft(velocity) * filtro / (1j * omega_safe)
    X[np.abs(omega) < 1e-10] = 0
    position = np.real(ifft(X))
    
    return velocity, position


def analizar_frecuencia(archivo_path, freq):
    """Analiza un archivo de frecuencia"""
    df = pd.read_csv(archivo_path)
    
    t = df['tiempo_s'].values
    fuerza_V = df['fuerza_V'].values
    accel_g = df['aceleracion_sensor_g'].values
    
    # Convertir con calibración
    F = np.array([voltaje_a_N(v) for v in fuerza_V])
    F = F - np.mean(F)
    
    a = (accel_g - np.mean(accel_g)) * 9.81
    
    # Integrar
    v, x = integrar_senoidal(a, SAMPLE_RATE, freq)
    
    # FFT para FRF
    n = len(F)
    freqs = fftfreq(n, 1/SAMPLE_RATE)
    F_fft = fft(F)
    a_fft = fft(a)
    
    idx = np.argmin(np.abs(freqs - freq))
    search = int(3 * n / SAMPLE_RATE)
    idx_peak = idx - search + np.argmax(np.abs(F_fft[idx-search:idx+search]))
    
    H = F_fft[idx_peak] / a_fft[idx_peak] if np.abs(a_fft[idx_peak]) > 1e-10 else 0
    
    return {
        'freq': freq,
        't': t, 'F': F, 'a': a, 'v': v, 'x': x,
        'F_rms': np.sqrt(np.mean(F**2)),
        'F_amp': (np.max(F) - np.min(F)) / 2,
        'a_rms': np.sqrt(np.mean(a**2)),
        'a_amp': (np.max(a) - np.min(a)) / 2,
        'x_amp': (np.max(x) - np.min(x)) / 2,
        'H_mag': np.abs(H),
        'H_fase': np.angle(H, deg=True),
        'H_real': np.real(H),
        'H_imag': np.imag(H)
    }


# Analizar todas las frecuencias
base_name = "caracterizacion_fuerza_20251203_120809"
frecuencias = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

resultados = []

print("\n{:>5} {:>10} {:>10} {:>10} {:>12} {:>10}".format(
    "Freq", "F_rms", "a_rms", "x_amp", "|H|=F/a", "fase"))
print("{:>5} {:>10} {:>10} {:>10} {:>12} {:>10}".format(
    "(Hz)", "(N)", "(m/s²)", "(µm)", "(kg)", "(°)"))
print("-"*65)

for i, freq in enumerate(frecuencias):
    archivo = f"{base_name}_exp{i+1}_{freq}Hz.csv"
    path = os.path.join(DATOS_DIR, archivo)
    
    if not os.path.exists(path):
        continue
    
    res = analizar_frecuencia(path, freq)
    resultados.append(res)
    
    print("{:5.0f} {:10.2f} {:10.3f} {:10.2f} {:12.2f} {:10.1f}".format(
        freq, res['F_rms'], res['a_rms'], res['x_amp']*1e6, 
        res['H_mag'], res['H_fase']))

df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in resultados])

# === IDENTIFICACIÓN DE PARÁMETROS ===
print("\n" + "="*70)
print("IDENTIFICACIÓN DE PARÁMETROS (FRF: H = F/a)")
print("="*70)

# Filtrar datos buenos
mask = (df['H_mag'] > 0.5) & (df['H_mag'] < 200) & (df['freq'] >= 15)
df_fit = df[mask].copy()

omega = 2 * np.pi * df_fit['freq'].values

# Re(H) = m - k/ω²  =>  y = m + (-k)*x donde x = 1/ω²
x = 1 / omega**2
y = df_fit['H_real'].values
A = np.vstack([np.ones_like(x), x]).T
coef = np.linalg.lstsq(A, y, rcond=None)[0]
m_id = coef[0]
k_id = -coef[1]

# Im(H) = c/ω
c_id = np.mean(df_fit['H_imag'].values * omega)

# Parámetros derivados
omega_n = np.sqrt(abs(k_id / m_id)) if m_id > 0 and k_id > 0 else 0
f_n = omega_n / (2 * np.pi)
zeta = c_id / (2 * np.sqrt(abs(k_id * m_id))) if k_id > 0 and m_id > 0 else 0

print(f"\nPARÁMETROS IDENTIFICADOS:")
print(f"  Masa efectiva:      m = {m_id:.2f} kg")
print(f"  Rigidez:            k = {k_id/1000:.2f} kN/m ({k_id:.0f} N/m)")
print(f"  Amortiguamiento:    c = {c_id:.2f} N·s/m")
print(f"  Frecuencia natural: f_n = {f_n:.1f} Hz")
print(f"  Factor amort.:      ζ = {zeta:.4f}")

# === CÁLCULO DE FRICCIÓN ===
print("\n" + "="*70)
print("CÁLCULO DE FRICCIÓN: F_fric = F_celda - m·a - k·x")
print("="*70)

print("\n{:>5} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "Freq", "F_celda", "m·a", "k·x", "F_fric", "% fric"))
print("{:>5} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "(Hz)", "(N)", "(N)", "(N)", "(N)", ""))
print("-"*65)

friccion_data = []

for res in resultados:
    F_celda_rms = res['F_rms']
    F_inercia_rms = abs(m_id) * res['a_rms']
    F_rigidez_rms = abs(k_id) * res['x_amp'] / np.sqrt(2)  # RMS de senoidal
    
    # F_fric = F_celda - m·a - k·x (en fase, aproximación RMS)
    F_fric_rms = np.sqrt(abs(F_celda_rms**2 - (F_inercia_rms - F_rigidez_rms)**2))
    
    pct = 100 * F_fric_rms / F_celda_rms if F_celda_rms > 0 else 0
    
    print("{:5.0f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.1f}%".format(
        res['freq'], F_celda_rms, F_inercia_rms, F_rigidez_rms, F_fric_rms, pct))
    
    friccion_data.append({
        'freq': res['freq'],
        'F_celda': F_celda_rms,
        'F_inercia': F_inercia_rms,
        'F_rigidez': F_rigidez_rms,
        'F_friccion': F_fric_rms
    })

df_fric = pd.DataFrame(friccion_data)
F_fric_promedio = df_fric['F_friccion'].mean()

print(f"\n>>> FRICCIÓN PROMEDIO: {F_fric_promedio:.2f} N (RMS)")

# === GRÁFICAS ===
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(f'Análisis con Calibración Experimental\n'
             f'm={m_id:.2f}kg, k={k_id/1000:.1f}kN/m, c={c_id:.1f}N·s/m, f_n={f_n:.1f}Hz',
             fontsize=12, fontweight='bold')

# FRF Magnitud
ax = axes[0, 0]
ax.semilogy(df['freq'], df['H_mag'], 'co-', markersize=8)
omega_m = 2*np.pi*np.linspace(10, 90, 100)
H_modelo = np.abs(m_id - k_id/omega_m**2 + 1j*c_id/omega_m)
ax.semilogy(omega_m/(2*np.pi), H_modelo, 'c--', alpha=0.5)
ax.axvline(f_n, color='r', linestyle=':', label=f'f_n={f_n:.1f}Hz')
ax.set_xlabel('Frecuencia (Hz)')
ax.set_ylabel('|H| = F/a (kg)')
ax.set_title('FRF Magnitud')
ax.legend()
ax.grid(True, alpha=0.3)

# FRF Fase
ax = axes[0, 1]
ax.plot(df['freq'], df['H_fase'], 'go-', markersize=8)
ax.set_xlabel('Frecuencia (Hz)')
ax.set_ylabel('Fase (°)')
ax.set_title('FRF Fase')
ax.grid(True, alpha=0.3)

# Nyquist
ax = axes[0, 2]
ax.plot(df['H_real'], df['H_imag'], 'o-', color='orange', markersize=8)
for _, row in df.iterrows():
    ax.annotate(f"{row['freq']:.0f}", (row['H_real'], row['H_imag']), fontsize=7)
ax.axhline(0, color='w', linestyle='--', alpha=0.3)
ax.axvline(0, color='w', linestyle='--', alpha=0.3)
ax.scatter([m_id], [0], color='r', s=100, marker='x', label=f'm={m_id:.1f}kg')
ax.set_xlabel('Re(H)')
ax.set_ylabel('Im(H)')
ax.set_title('Diagrama de Nyquist')
ax.legend()
ax.grid(True, alpha=0.3)

# Fuerzas vs frecuencia
ax = axes[1, 0]
ax.plot(df_fric['freq'], df_fric['F_celda'], 'co-', label='F_celda', markersize=8)
ax.plot(df_fric['freq'], df_fric['F_inercia'], 'r^-', label='m·a', markersize=6)
ax.plot(df_fric['freq'], df_fric['F_rigidez'], 'gs-', label='k·x', markersize=6)
ax.plot(df_fric['freq'], df_fric['F_friccion'], 'm*-', label='F_fricción', markersize=10)
ax.set_xlabel('Frecuencia (Hz)')
ax.set_ylabel('Fuerza RMS (N)')
ax.set_title('Componentes de Fuerza')
ax.legend()
ax.grid(True, alpha=0.3)

# Fricción
ax = axes[1, 1]
ax.bar(df_fric['freq'], df_fric['F_friccion'], color='magenta', alpha=0.7)
ax.axhline(F_fric_promedio, color='yellow', linestyle='--', label=f'Promedio: {F_fric_promedio:.2f} N')
ax.set_xlabel('Frecuencia (Hz)')
ax.set_ylabel('F_fricción (N)')
ax.set_title('Fuerza de Fricción por Frecuencia')
ax.legend()
ax.grid(True, alpha=0.3)

# Resumen
ax = axes[1, 2]
ax.axis('off')
texto = f"""
RESUMEN DE CALIBRACIÓN Y ANÁLISIS
{'='*40}

CALIBRACIÓN:
  Offset: {VOLTAGE_OFFSET:.5f} V
  Factor: {VOLTAGE_PER_KG:.6f} V/kg
  Masa cal: {cal['notes']}

PARÁMETROS IDENTIFICADOS:
  m = {m_id:.2f} kg
  k = {k_id/1000:.2f} kN/m
  c = {c_id:.2f} N·s/m
  f_n = {f_n:.1f} Hz
  ζ = {zeta:.4f}

FRICCIÓN:
  F_fricción = {F_fric_promedio:.2f} N (promedio)
"""
ax.text(0.05, 0.95, texto, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#333', alpha=0.8))

plt.tight_layout()
fig.savefig(os.path.join(DATOS_DIR, 'analisis_calibrado.png'), dpi=150)
print(f"\n📊 Guardado: analisis_calibrado.png")

# Guardar CSV
df_fric.to_csv(os.path.join(DATOS_DIR, 'friccion_calibrada.csv'), index=False)

plt.show()
