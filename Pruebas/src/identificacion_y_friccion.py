"""
Identificación de Parámetros y Cálculo de Fricción
===================================================
Problema: Los parámetros anteriores dan F_inercia >> F_celda, lo cual es imposible.

Enfoque correcto:
1. Identificar m, k, c directamente de la FRF: H(ω) = F(ω)/a(ω)
2. Usar esos parámetros para calcular fricción

La FRF teórica es:
H(ω) = F/a = m - k/ω² + j·c/ω

Parte Real: Re(H) = m - k/ω²
Parte Imaginaria: Im(H) = c/ω
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import curve_fit
import os

# Configuración
DATOS_DIR = r"c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\Pruebas\src\caracterizacion_fuerza"
SAMPLE_RATE = 2500

# Parámetros del sensor
CELDA_CAPACIDAD_KG = 500.0
CELDA_SENSIBILIDAD_MV_V = 1.7
INA849_GANANCIA = 601.0
V_EXCITACION = 10.0

plt.style.use('dark_background')

def voltaje_a_fuerza_N(voltaje):
    sensibilidad_total = (CELDA_SENSIBILIDAD_MV_V / 1000) * V_EXCITACION * INA849_GANANCIA
    return (voltaje / sensibilidad_total) * CELDA_CAPACIDAD_KG * 9.81


def calcular_frf(archivo_path, freq):
    """Calcula la FRF a la frecuencia de excitación"""
    df = pd.read_csv(archivo_path)
    
    fuerza_V = df['fuerza_V'].values - np.mean(df['fuerza_V'].values)
    accel_g = df['aceleracion_sensor_g'].values - np.mean(df['aceleracion_sensor_g'].values)
    
    F = np.array([voltaje_a_fuerza_N(v) for v in fuerza_V])
    a = accel_g * 9.81  # m/s²
    
    n = len(F)
    freqs = fftfreq(n, 1/SAMPLE_RATE)
    
    F_fft = fft(F)
    a_fft = fft(a)
    
    # Buscar el pico cerca de la frecuencia de excitación
    idx_pos = np.where(freqs > 0)[0]
    idx_freq = idx_pos[np.argmin(np.abs(freqs[idx_pos] - freq))]
    
    # Buscar máximo en ventana ±5 Hz
    search_range = int(5 * n / SAMPLE_RATE)
    idx_start = max(1, idx_freq - search_range)
    idx_end = min(len(freqs)//2, idx_freq + search_range)
    
    idx_peak = idx_start + np.argmax(np.abs(F_fft[idx_start:idx_end]))
    
    # FRF = F/a
    if np.abs(a_fft[idx_peak]) > 1e-10:
        H = F_fft[idx_peak] / a_fft[idx_peak]
    else:
        H = 0 + 0j
    
    return {
        'freq': freq,
        'freq_real': freqs[idx_peak],
        'H': H,
        'H_mag': np.abs(H),
        'H_fase': np.angle(H, deg=True),
        'H_real': np.real(H),
        'H_imag': np.imag(H),
        'F_amp': np.abs(F_fft[idx_peak]) * 2 / n,
        'a_amp': np.abs(a_fft[idx_peak]) * 2 / n
    }


def modelo_frf_real(omega, m, k):
    """Parte real de H(ω) = m - k/ω²"""
    return m - k / (omega**2)


def modelo_frf_imag(omega, c):
    """Parte imaginaria de H(ω) = c/ω"""
    return c / omega


def identificar_parametros(df_frf):
    """Identifica m, k, c de los datos de FRF"""
    
    omega = 2 * np.pi * df_frf['freq'].values
    H_real = df_frf['H_real'].values
    H_imag = df_frf['H_imag'].values
    
    # Filtrar datos problemáticos (muy cerca de resonancia o antiresonancia)
    # Usar solo frecuencias donde la magnitud es razonable
    H_mag = df_frf['H_mag'].values
    mask = (H_mag > 0.1) & (H_mag < 500) & (df_frf['freq'] > 15)
    
    omega_fit = omega[mask]
    H_real_fit = H_real[mask]
    H_imag_fit = H_imag[mask]
    
    print(f"\nUsando {np.sum(mask)} de {len(mask)} puntos para ajuste")
    
    # Ajustar parte real: Re(H) = m - k/ω²
    try:
        popt_real, _ = curve_fit(modelo_frf_real, omega_fit, H_real_fit, 
                                  p0=[10, 1e6], bounds=([0.1, 1e3], [100, 1e8]))
        m_id = popt_real[0]
        k_id = popt_real[1]
    except:
        # Método alternativo: regresión lineal
        # Re(H) = m - k/ω² => y = m + (-k)·x donde x = 1/ω²
        x = 1 / omega_fit**2
        y = H_real_fit
        A = np.vstack([np.ones_like(x), x]).T
        result = np.linalg.lstsq(A, y, rcond=None)
        m_id = result[0][0]
        k_id = -result[0][1]
    
    # Ajustar parte imaginaria: Im(H) = c/ω
    try:
        popt_imag, _ = curve_fit(modelo_frf_imag, omega_fit, H_imag_fit,
                                  p0=[100], bounds=([0], [10000]))
        c_id = popt_imag[0]
    except:
        # Método alternativo
        c_id = np.mean(H_imag_fit * omega_fit)
    
    # Calcular frecuencia natural y factor de amortiguamiento
    if k_id > 0 and m_id > 0:
        omega_n = np.sqrt(k_id / m_id)
        f_n = omega_n / (2 * np.pi)
        zeta = c_id / (2 * np.sqrt(k_id * m_id))
    else:
        f_n = 0
        zeta = 0
    
    return {
        'm': m_id,
        'k': k_id,
        'c': c_id,
        'f_n': f_n,
        'zeta': zeta
    }


def integrar_senoidal(accel, fs, freq):
    """Integración en frecuencia para señal senoidal"""
    n = len(accel)
    freqs = fftfreq(n, 1/fs)
    omega = 2 * np.pi * freqs
    
    A = fft(accel)
    
    # Filtro pasa-banda
    bw = freq * 0.3
    bp = np.exp(-((np.abs(freqs) - freq) / bw)**2)
    hp = 1 - np.exp(-(np.abs(freqs) / 0.5)**4)
    filtro = bp * hp
    
    omega_safe = np.where(np.abs(omega) > 1e-10, omega, 1e-10)
    
    V = A * filtro / (1j * omega_safe)
    V[np.abs(omega) < 1e-10] = 0
    velocity = np.real(ifft(V))
    
    V_fft = fft(velocity)
    X = V_fft * filtro / (1j * omega_safe)
    X[np.abs(omega) < 1e-10] = 0
    position = np.real(ifft(X))
    
    return velocity, position


def calcular_friccion_con_parametros(archivo_path, freq, m, k, c):
    """Calcula fricción usando parámetros identificados"""
    df = pd.read_csv(archivo_path)
    
    t = df['tiempo_s'].values
    fuerza_V = df['fuerza_V'].values - np.mean(df['fuerza_V'].values)
    accel_g = df['aceleracion_sensor_g'].values - np.mean(df['aceleracion_sensor_g'].values)
    
    F_celda = np.array([voltaje_a_fuerza_N(v) for v in fuerza_V])
    accel = accel_g * 9.81
    
    velocity, position = integrar_senoidal(accel, SAMPLE_RATE, freq)
    
    # Componentes
    F_inercia = m * accel
    F_rigidez = k * position
    F_amort = c * velocity
    
    # Fricción
    F_friccion = F_celda - F_inercia - F_rigidez
    F_friccion_total = F_celda - F_inercia - F_rigidez - F_amort
    
    return {
        't': t,
        'F_celda': F_celda,
        'accel': accel,
        'velocity': velocity,
        'position': position,
        'F_inercia': F_inercia,
        'F_rigidez': F_rigidez,
        'F_amort': F_amort,
        'F_friccion': F_friccion,
        'F_friccion_total': F_friccion_total,
        'F_celda_rms': np.sqrt(np.mean(F_celda**2)),
        'F_inercia_rms': np.sqrt(np.mean(F_inercia**2)),
        'F_rigidez_rms': np.sqrt(np.mean(F_rigidez**2)),
        'F_amort_rms': np.sqrt(np.mean(F_amort**2)),
        'F_friccion_rms': np.sqrt(np.mean(F_friccion**2))
    }


def main():
    print("="*70)
    print("IDENTIFICACIÓN DE PARÁMETROS Y CÁLCULO DE FRICCIÓN")
    print("="*70)
    
    # Paso 1: Calcular FRF para todas las frecuencias
    print("\n📊 PASO 1: Calculando FRF...")
    
    base_name = "caracterizacion_fuerza_20251203_120809"
    frecuencias = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    
    resultados_frf = []
    
    for i, freq in enumerate(frecuencias):
        archivo = f"{base_name}_exp{i+1}_{freq}Hz.csv"
        archivo_path = os.path.join(DATOS_DIR, archivo)
        
        if os.path.exists(archivo_path):
            frf = calcular_frf(archivo_path, freq)
            resultados_frf.append(frf)
            print(f"   {freq:3d} Hz: |H|={frf['H_mag']:8.2f} kg, fase={frf['H_fase']:7.1f}°")
    
    df_frf = pd.DataFrame(resultados_frf)
    
    # Paso 2: Identificar parámetros
    print("\n📊 PASO 2: Identificando parámetros...")
    params = identificar_parametros(df_frf)
    
    print(f"\n   PARÁMETROS IDENTIFICADOS:")
    print(f"   ├─ Masa efectiva:     m = {params['m']:.4f} kg")
    print(f"   ├─ Rigidez:           k = {params['k']:.2f} N/m")
    print(f"   ├─ Amortiguamiento:   c = {params['c']:.4f} N·s/m")
    print(f"   ├─ Frecuencia natural: f_n = {params['f_n']:.2f} Hz")
    print(f"   └─ Factor amort.:     ζ = {params['zeta']:.4f}")
    
    # Paso 3: Graficar FRF con modelo ajustado
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Identificación de Parámetros del Sistema\n"
                 f"m={params['m']:.3f}kg, k={params['k']:.0f}N/m, c={params['c']:.2f}N·s/m, "
                 f"f_n={params['f_n']:.1f}Hz, ζ={params['zeta']:.3f}",
                 fontsize=12, fontweight='bold')
    
    omega = 2 * np.pi * df_frf['freq'].values
    omega_modelo = 2 * np.pi * np.linspace(10, 90, 100)
    
    # Parte Real
    ax1 = axes[0, 0]
    ax1.plot(df_frf['freq'], df_frf['H_real'], 'co', markersize=10, label='Datos')
    H_real_modelo = params['m'] - params['k'] / omega_modelo**2
    ax1.plot(omega_modelo/(2*np.pi), H_real_modelo, 'c-', linewidth=2, label='Modelo')
    ax1.axhline(y=params['m'], color='r', linestyle='--', alpha=0.5, label=f'm={params["m"]:.3f}kg')
    ax1.set_xlabel('Frecuencia (Hz)')
    ax1.set_ylabel('Re(H) = m - k/ω² (kg)')
    ax1.set_title('Parte Real de FRF')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Parte Imaginaria
    ax2 = axes[0, 1]
    ax2.plot(df_frf['freq'], df_frf['H_imag'], 'go', markersize=10, label='Datos')
    H_imag_modelo = params['c'] / omega_modelo
    ax2.plot(omega_modelo/(2*np.pi), H_imag_modelo, 'g-', linewidth=2, label='Modelo')
    ax2.set_xlabel('Frecuencia (Hz)')
    ax2.set_ylabel('Im(H) = c/ω (kg)')
    ax2.set_title('Parte Imaginaria de FRF')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Magnitud
    ax3 = axes[1, 0]
    ax3.semilogy(df_frf['freq'], df_frf['H_mag'], 'mo', markersize=10, label='Datos')
    H_modelo = (params['m'] - params['k']/omega_modelo**2) + 1j*(params['c']/omega_modelo)
    ax3.semilogy(omega_modelo/(2*np.pi), np.abs(H_modelo), 'm-', linewidth=2, label='Modelo')
    ax3.axvline(x=params['f_n'], color='r', linestyle='--', alpha=0.5, label=f'f_n={params["f_n"]:.1f}Hz')
    ax3.set_xlabel('Frecuencia (Hz)')
    ax3.set_ylabel('|H| (kg)')
    ax3.set_title('Magnitud de FRF')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Nyquist
    ax4 = axes[1, 1]
    ax4.plot(df_frf['H_real'], df_frf['H_imag'], 'o-', markersize=8, color='orange')
    ax4.plot(np.real(H_modelo), np.imag(H_modelo), '-', linewidth=2, color='cyan', alpha=0.7)
    for _, row in df_frf.iterrows():
        ax4.annotate(f"{row['freq']:.0f}", (row['H_real'], row['H_imag']), fontsize=7)
    ax4.axhline(y=0, color='w', linestyle='--', alpha=0.3)
    ax4.axvline(x=0, color='w', linestyle='--', alpha=0.3)
    ax4.scatter([params['m']], [0], color='red', s=100, marker='x', label=f'm={params["m"]:.3f}')
    ax4.set_xlabel('Re(H)')
    ax4.set_ylabel('Im(H)')
    ax4.set_title('Diagrama de Nyquist')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(DATOS_DIR, 'identificacion_parametros.png'), dpi=150)
    print(f"\n📊 Guardado: identificacion_parametros.png")
    
    # Paso 4: Calcular fricción con nuevos parámetros
    print("\n📊 PASO 3: Calculando fricción con parámetros identificados...")
    
    resultados_friccion = []
    
    for i, freq in enumerate(frecuencias):
        archivo = f"{base_name}_exp{i+1}_{freq}Hz.csv"
        archivo_path = os.path.join(DATOS_DIR, archivo)
        
        if os.path.exists(archivo_path):
            res = calcular_friccion_con_parametros(archivo_path, freq, 
                                                    params['m'], params['k'], params['c'])
            print(f"   {freq:3d} Hz: F_celda={res['F_celda_rms']:.3f}N, "
                  f"F_inercia={res['F_inercia_rms']:.3f}N, "
                  f"F_rigidez={res['F_rigidez_rms']:.3f}N, "
                  f"F_fric={res['F_friccion_rms']:.3f}N")
            
            resultados_friccion.append({
                'freq_Hz': freq,
                'F_celda_rms': res['F_celda_rms'],
                'F_inercia_rms': res['F_inercia_rms'],
                'F_rigidez_rms': res['F_rigidez_rms'],
                'F_amort_rms': res['F_amort_rms'],
                'F_friccion_rms': res['F_friccion_rms']
            })
    
    df_friccion = pd.DataFrame(resultados_friccion)
    
    # Gráfica de fricción
    fig2, ax = plt.subplots(figsize=(12, 6))
    ax.semilogy(df_friccion['freq_Hz'], df_friccion['F_celda_rms'], 'co-', markersize=8, label='F_celda')
    ax.semilogy(df_friccion['freq_Hz'], df_friccion['F_inercia_rms'], 'r^-', markersize=8, label='m·a')
    ax.semilogy(df_friccion['freq_Hz'], df_friccion['F_rigidez_rms'], 'bs-', markersize=8, label='k·x')
    ax.semilogy(df_friccion['freq_Hz'], df_friccion['F_friccion_rms'], 'm*-', markersize=12, label='F_fricción')
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Fuerza RMS (N)')
    ax.set_title(f'Componentes de Fuerza (m={params["m"]:.3f}kg, k={params["k"]:.0f}N/m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig2.savefig(os.path.join(DATOS_DIR, 'friccion_con_parametros_correctos.png'), dpi=150)
    print(f"\n📊 Guardado: friccion_con_parametros_correctos.png")
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"\nParámetros del sistema:")
    print(f"  m = {params['m']:.4f} kg")
    print(f"  k = {params['k']:.2f} N/m ({params['k']/1000:.2f} kN/m)")
    print(f"  c = {params['c']:.4f} N·s/m")
    print(f"  f_n = {params['f_n']:.2f} Hz")
    print(f"  ζ = {params['zeta']:.4f}")
    
    F_fric_promedio = df_friccion['F_friccion_rms'].mean()
    print(f"\nFricción promedio: {F_fric_promedio:.4f} N (RMS)")
    
    # Guardar
    df_friccion.to_csv(os.path.join(DATOS_DIR, 'friccion_final.csv'), index=False)
    
    plt.show()

if __name__ == "__main__":
    main()
