"""
Análisis Visual Completo: Barrido + Triangular
===============================================
Genera gráficas detalladas de señales en tiempo, FFT, Lissajous y FRF
para ambos tipos de experimento.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.integrate import cumulative_trapezoid
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

def cargar_y_procesar(archivo_path):
    """Carga y procesa un archivo de datos"""
    df = pd.read_csv(archivo_path)
    
    t = df['tiempo_s'].values
    fuerza_V = df['fuerza_V'].values
    accel_bancada_g = df['aceleracion_bancada_g'].values
    accel_sensor_g = df['aceleracion_sensor_g'].values
    
    # Remover DC
    fuerza_V = fuerza_V - np.mean(fuerza_V)
    accel_bancada_g = accel_bancada_g - np.mean(accel_bancada_g)
    accel_sensor_g = accel_sensor_g - np.mean(accel_sensor_g)
    
    # Convertir unidades
    force_N = np.array([voltaje_a_fuerza_N(v) for v in fuerza_V])
    accel_bancada_ms2 = accel_bancada_g * 9.81
    accel_sensor_ms2 = accel_sensor_g * 9.81
    
    return {
        't': t,
        'force_N': force_N,
        'accel_bancada_g': accel_bancada_g,
        'accel_sensor_g': accel_sensor_g,
        'accel_bancada_ms2': accel_bancada_ms2,
        'accel_sensor_ms2': accel_sensor_ms2
    }

def analizar_barrido():
    """Analiza y grafica datos de barrido senoidal"""
    print("\n" + "="*60)
    print("ANALISIS DE BARRIDO SENOIDAL (10-90 Hz)")
    print("="*60)
    
    # Buscar archivos de barrido
    base_name = "caracterizacion_fuerza_20251203_120809"
    
    # Frecuencias reales del experimento (pasos de 5 Hz)
    frecuencias = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    
    # Crear figura
    n_freq = len(frecuencias)
    fig, axes = plt.subplots(n_freq, 4, figsize=(20, 3*n_freq))
    fig.suptitle('Analisis Barrido Senoidal - Fuerza y Aceleracion', fontsize=14, fontweight='bold')
    
    resultados_frf = []
    
    for i, freq in enumerate(frecuencias):
        archivo = f"{base_name}_exp{i+1}_{freq}Hz.csv"
        archivo_path = os.path.join(DATOS_DIR, archivo)
        
        if not os.path.exists(archivo_path):
            print(f"  No encontrado: {archivo}")
            continue
        
        print(f"  Procesando {freq} Hz...")
        datos = cargar_y_procesar(archivo_path)
        
        t = datos['t']
        force = datos['force_N']
        accel = datos['accel_sensor_ms2']
        accel_g = datos['accel_sensor_g']
        accel_bancada = datos['accel_bancada_ms2']
        
        # Normalizar para visualización
        force_norm = force / np.max(np.abs(force)) if np.max(np.abs(force)) > 0 else force
        accel_norm = accel / np.max(np.abs(accel)) if np.max(np.abs(accel)) > 0 else accel
        
        # Columna 1: Señales en tiempo (zoom)
        ax1 = axes[i, 0]
        n_ciclos = 5
        periodo = 1/freq
        n_muestras = int(n_ciclos * periodo * SAMPLE_RATE)
        idx_start = len(t)//2
        idx_end = min(idx_start + n_muestras, len(t))
        
        ax1.plot(t[idx_start:idx_end], force_norm[idx_start:idx_end], 'c-', linewidth=0.8, label='Fuerza')
        ax1.plot(t[idx_start:idx_end], accel_norm[idx_start:idx_end], 'orange', linewidth=0.8, label='Acel Sensor')
        ax1.set_ylabel('Norm')
        ax1.set_title(f'{freq} Hz - Tiempo')
        ax1.legend(loc='upper right', fontsize=7)
        ax1.grid(True, alpha=0.3)
        
        # Columna 2: FFT
        ax2 = axes[i, 1]
        n = len(force)
        freqs_fft = fftfreq(n, 1/SAMPLE_RATE)[:n//2]
        fft_force = np.abs(fft(force))[:n//2]
        fft_force_db = 20 * np.log10(fft_force + 1e-10)
        
        ax2.plot(freqs_fft, fft_force_db, 'c-', linewidth=0.5)
        ax2.axvline(x=freq, color='r', linestyle='--', alpha=0.7, label=f'f={freq}Hz')
        
        # Encontrar pico
        idx_peak = np.argmax(fft_force[1:]) + 1
        freq_peak = freqs_fft[idx_peak]
        ax2.axvline(x=freq_peak, color='orange', linestyle=':', alpha=0.7, label=f'Pico: {freq_peak:.1f}Hz')
        
        ax2.set_xlim([0, 150])
        ax2.set_ylabel('dB')
        ax2.set_title(f'{freq} Hz - FFT Fuerza')
        ax2.legend(loc='upper right', fontsize=7)
        ax2.grid(True, alpha=0.3)
        
        # Columna 3: Lissajous F vs Acel
        ax3 = axes[i, 2]
        ax3.plot(force_norm, accel_norm, 'orange', linewidth=0.3, alpha=0.7)
        ax3.scatter([0], [0], color='lime', s=50, zorder=5)
        ax3.set_xlabel('Fuerza')
        ax3.set_ylabel('Acel Sensor')
        ax3.set_title(f'{freq} Hz - Lissajous F vs Acel')
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal', adjustable='box')
        
        # Columna 4: Transmisibilidad (Acel Sensor vs Acel Bancada)
        ax4 = axes[i, 3]
        accel_bancada_norm = accel_bancada / np.max(np.abs(accel_bancada)) if np.max(np.abs(accel_bancada)) > 0 else accel_bancada
        ax4.plot(accel_bancada_norm, accel_norm, 'm-', linewidth=0.3, alpha=0.7)
        ax4.plot([-1, 1], [-1, 1], 'w--', alpha=0.5, linewidth=1)  # Línea 1:1
        ax4.scatter([0], [0], color='lime', s=50, zorder=5)
        ax4.set_xlabel('Acel Bancada')
        ax4.set_ylabel('Acel Sensor')
        ax4.set_title(f'{freq} Hz - Transmisibilidad')
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal', adjustable='box')
        
        # Calcular FRF
        idx_freq = np.argmin(np.abs(freqs_fft - freq))
        search_range = max(1, int(5 * n / SAMPLE_RATE))
        idx_start_search = max(1, idx_freq - search_range)
        idx_end_search = min(len(freqs_fft)-1, idx_freq + search_range)
        
        fft_accel = fft(accel)[:n//2]
        fft_force_complex = fft(force)[:n//2]
        
        idx_peak_frf = idx_start_search + np.argmax(np.abs(fft_force_complex[idx_start_search:idx_end_search]))
        
        if np.abs(fft_accel[idx_peak_frf]) > 1e-10:
            H = fft_force_complex[idx_peak_frf] / fft_accel[idx_peak_frf]
            resultados_frf.append({
                'freq': freq,
                'H_mag': np.abs(H),
                'H_fase': np.angle(H, deg=True),
                'H_real': np.real(H),
                'H_imag': np.imag(H)
            })
    
    axes[-1, 0].set_xlabel('Tiempo (s)')
    axes[-1, 1].set_xlabel('Frecuencia (Hz)')
    
    plt.tight_layout()
    fig.savefig(os.path.join(DATOS_DIR, 'analisis_barrido_detallado.png'), dpi=150, bbox_inches='tight')
    print(f"\nGuardado: analisis_barrido_detallado.png")
    
    return pd.DataFrame(resultados_frf)

def analizar_triangular():
    """Analiza y grafica datos de fricción triangular"""
    print("\n" + "="*60)
    print("ANALISIS DE FRICCION TRIANGULAR (0.5-2.5 Hz)")
    print("="*60)
    
    # Buscar archivos de triangular
    base_name = "caracterizacion_fuerza_20251203_125007"
    
    # Cargar resumen
    resumen_path = os.path.join(DATOS_DIR, f"{base_name}_resumen.csv")
    df_resumen = pd.read_csv(resumen_path)
    df_tri = df_resumen[df_resumen['experiment_type'] == 'TRIANGLE'].copy()
    
    if len(df_tri) == 0:
        print("No hay datos triangulares")
        return None
    
    n_exp = len(df_tri)
    fig, axes = plt.subplots(n_exp, 4, figsize=(20, 3*n_exp))
    fig.suptitle('Analisis Friccion Triangular - Lazos de Histeresis', fontsize=14, fontweight='bold')
    
    if n_exp == 1:
        axes = axes.reshape(1, -1)
    
    resultados = []
    
    for i, (idx, row) in enumerate(df_tri.iterrows()):
        freq = row['frecuencia_Hz']
        exp_num = i + 1
        
        # Buscar archivo
        archivos = [f for f in os.listdir(DATOS_DIR) if base_name in f and f"exp{exp_num}_" in f and 'resumen' not in f]
        
        if not archivos:
            continue
        
        archivo_path = os.path.join(DATOS_DIR, archivos[0])
        print(f"  Procesando {freq:.1f} Hz ({archivos[0]})...")
        
        datos = cargar_y_procesar(archivo_path)
        
        t = datos['t']
        force = datos['force_N']
        accel = datos['accel_sensor_ms2']
        accel_g = datos['accel_sensor_g']
        
        dt = 1.0 / SAMPLE_RATE
        
        # Filtro pasa-altos e integración
        try:
            b, a = signal.butter(2, 0.2 / (SAMPLE_RATE / 2), btype='high')
            accel_filtered = signal.filtfilt(b, a, accel_g)
        except:
            accel_filtered = accel_g
        
        accel_ms2 = accel_filtered * 9.81
        velocity = signal.detrend(cumulative_trapezoid(accel_ms2, dx=dt, initial=0))
        displacement = signal.detrend(cumulative_trapezoid(velocity, dx=dt, initial=0))
        
        velocity_mms = velocity * 1000
        displacement_mm = displacement * 1000
        
        # Columna 1: Señales en tiempo
        ax1 = axes[i, 0]
        t_plot = t[:min(len(t), int(5/freq * SAMPLE_RATE))]  # 5 ciclos
        n_plot = len(t_plot)
        
        force_norm = force[:n_plot] / np.max(np.abs(force[:n_plot])) if np.max(np.abs(force[:n_plot])) > 0 else force[:n_plot]
        disp_norm = displacement_mm[:n_plot] / np.max(np.abs(displacement_mm[:n_plot])) if np.max(np.abs(displacement_mm[:n_plot])) > 0 else displacement_mm[:n_plot]
        
        ax1.plot(t_plot, force_norm, 'c-', linewidth=0.8, label='Fuerza')
        ax1.plot(t_plot, disp_norm, 'orange', linewidth=0.8, label='Desplaz.')
        ax1.set_ylabel('Norm')
        ax1.set_title(f'{freq:.1f} Hz - Tiempo')
        ax1.legend(loc='upper right', fontsize=7)
        ax1.grid(True, alpha=0.3)
        
        # Columna 2: Lazo de histéresis F vs x
        ax2 = axes[i, 1]
        ax2.plot(displacement_mm, force, 'orange', linewidth=0.5, alpha=0.7)
        ax2.axhline(y=0, color='w', linestyle='--', alpha=0.3)
        ax2.axvline(x=0, color='w', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Desplazamiento (mm)')
        ax2.set_ylabel('Fuerza (N)')
        ax2.set_title(f'{freq:.1f} Hz - Histeresis F vs x')
        ax2.grid(True, alpha=0.3)
        
        # Columna 3: F vs Velocidad
        ax3 = axes[i, 2]
        ax3.plot(velocity_mms, force, 'lime', linewidth=0.3, alpha=0.7)
        ax3.axhline(y=0, color='w', linestyle='--', alpha=0.3)
        ax3.axvline(x=0, color='w', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Velocidad (mm/s)')
        ax3.set_ylabel('Fuerza (N)')
        ax3.set_title(f'{freq:.1f} Hz - F vs Velocidad')
        ax3.grid(True, alpha=0.3)
        
        # Columna 4: Métricas
        ax4 = axes[i, 3]
        ax4.axis('off')
        
        F_max = np.max(force)
        F_min = np.min(force)
        F_friction = (F_max - F_min) / 2
        stroke = np.max(displacement_mm) - np.min(displacement_mm)
        v_mean = np.mean(np.abs(velocity_mms))
        
        # Zero-crossing
        F_friction_zc = 0
        x_threshold = stroke * 0.15 if stroke > 0.01 else 0.1
        near_zero = np.abs(displacement_mm) < x_threshold
        if np.sum(near_zero) > 10:
            fwd = near_zero & (velocity_mms > 0.1)
            bwd = near_zero & (velocity_mms < -0.1)
            if np.sum(fwd) > 5 and np.sum(bwd) > 5:
                F_friction_zc = (np.mean(force[fwd]) - np.mean(force[bwd])) / 2
        
        texto = f"METRICAS {freq:.1f} Hz\n"
        texto += "="*25 + "\n\n"
        texto += f"F_max:     {F_max:.3f} N\n"
        texto += f"F_min:     {F_min:.3f} N\n"
        texto += f"F_fric (P2P): {F_friction:.3f} N\n"
        texto += f"F_fric (ZC):  {F_friction_zc:.3f} N\n\n"
        texto += f"Carrera:   {stroke:.2f} mm\n"
        texto += f"v_media:   {v_mean:.2f} mm/s\n"
        
        ax4.text(0.1, 0.9, texto, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))
        
        resultados.append({
            'freq': freq,
            'F_friction_peak': F_friction,
            'F_friction_zc': F_friction_zc,
            'stroke_mm': stroke,
            'v_mean_mms': v_mean
        })
    
    axes[-1, 0].set_xlabel('Tiempo (s)')
    
    plt.tight_layout()
    fig.savefig(os.path.join(DATOS_DIR, 'analisis_triangular_detallado.png'), dpi=150, bbox_inches='tight')
    print(f"\nGuardado: analisis_triangular_detallado.png")
    
    return pd.DataFrame(resultados)

def graficar_resumen_frf(df_frf):
    """Gráfica resumen de FRF"""
    if df_frf is None or len(df_frf) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Resumen FRF - Identificacion de Parametros', fontsize=14, fontweight='bold')
    
    # Magnitud
    ax1 = axes[0, 0]
    ax1.semilogy(df_frf['freq'], df_frf['H_mag'], 'co-', markersize=10, linewidth=2)
    ax1.set_xlabel('Frecuencia (Hz)')
    ax1.set_ylabel('|H| = |F/a| (kg)')
    ax1.set_title('FRF - Magnitud')
    ax1.grid(True, alpha=0.3)
    
    # Fase
    ax2 = axes[0, 1]
    ax2.plot(df_frf['freq'], df_frf['H_fase'], 'go-', markersize=10, linewidth=2)
    ax2.set_xlabel('Frecuencia (Hz)')
    ax2.set_ylabel('Fase (grados)')
    ax2.set_title('FRF - Fase')
    ax2.axhline(y=0, color='w', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Nyquist
    ax3 = axes[1, 0]
    ax3.plot(df_frf['H_real'], df_frf['H_imag'], 'mo-', markersize=10, linewidth=2)
    for _, row in df_frf.iterrows():
        ax3.annotate(f"{row['freq']:.0f}Hz", (row['H_real'], row['H_imag']), 
                    fontsize=8, xytext=(5, 5), textcoords='offset points')
    ax3.set_xlabel('Re(H)')
    ax3.set_ylabel('Im(H)')
    ax3.set_title('Diagrama de Nyquist')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='w', linestyle='--', alpha=0.3)
    ax3.axvline(x=0, color='w', linestyle='--', alpha=0.3)
    
    # Tabla de valores
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    texto = "VALORES DE FRF\n"
    texto += "="*40 + "\n\n"
    texto += f"{'Freq':>6} {'|H|':>10} {'Fase':>10}\n"
    texto += f"{'(Hz)':>6} {'(kg)':>10} {'(deg)':>10}\n"
    texto += "-"*40 + "\n"
    
    for _, row in df_frf.iterrows():
        texto += f"{row['freq']:>6.0f} {row['H_mag']:>10.2f} {row['H_fase']:>10.1f}\n"
    
    # Identificar mínimo (antiresonancia)
    idx_min = df_frf['H_mag'].idxmin()
    f_min = df_frf.loc[idx_min, 'freq']
    H_min = df_frf.loc[idx_min, 'H_mag']
    
    texto += "\n" + "-"*40 + "\n"
    texto += f"Minimo |H|: {H_min:.2f} kg @ {f_min:.0f} Hz\n"
    texto += f"(Antiresonancia)\n"
    
    ax4.text(0.05, 0.95, texto, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(os.path.join(DATOS_DIR, 'resumen_frf.png'), dpi=150, bbox_inches='tight')
    print(f"\nGuardado: resumen_frf.png")

def main():
    print("="*60)
    print("ANALISIS VISUAL COMPLETO")
    print("Barrido Senoidal + Friccion Triangular")
    print("="*60)
    
    # Analizar barrido
    df_frf = analizar_barrido()
    
    # Analizar triangular
    df_friccion = analizar_triangular()
    
    # Resumen FRF
    if df_frf is not None and len(df_frf) > 0:
        graficar_resumen_frf(df_frf)
    
    # Mostrar resultados
    if df_frf is not None:
        print("\n" + "="*60)
        print("RESUMEN FRF")
        print("="*60)
        print(df_frf.to_string(index=False))
    
    if df_friccion is not None:
        print("\n" + "="*60)
        print("RESUMEN FRICCION")
        print("="*60)
        print(df_friccion.to_string(index=False))
    
    plt.show()
    print("\nAnalisis completado!")

if __name__ == "__main__":
    main()
