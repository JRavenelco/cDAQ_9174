"""
Cálculo Correcto de Fuerza de Fricción
======================================
F_celda = m·a + k·x + F_fricción

Por lo tanto:
F_fricción = F_celda - m·a - k·x

Donde:
- F_celda: Fuerza medida por la celda de carga
- m: Masa efectiva identificada del barrido (12.42 kg)
- a: Aceleración medida por el acelerómetro
- k: Rigidez identificada del barrido (997548 N/m)
- x: Posición obtenida por doble integración de aceleración

Métodos de integración:
1. Trapezoidal con filtrado pasa-altos
2. Integración en frecuencia (más robusto para drift)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.integrate import cumulative_trapezoid
import os

# Configuración
DATOS_DIR = r"c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\Pruebas\src\caracterizacion_fuerza"
SAMPLE_RATE = 2500

# Parámetros del sensor de fuerza
CELDA_CAPACIDAD_KG = 500.0
CELDA_SENSIBILIDAD_MV_V = 1.7
INA849_GANANCIA = 601.0
V_EXCITACION = 10.0

# PARÁMETROS IDENTIFICADOS DEL BARRIDO DE FRECUENCIA
M_EFECTIVA = 12.42  # kg (masa efectiva del sistema)
K_RIGIDEZ = 997548  # N/m (rigidez del sistema)
C_AMORT = 1117.28   # N·s/m (amortiguamiento)

plt.style.use('dark_background')

def voltaje_a_fuerza_N(voltaje):
    """Convierte voltaje a fuerza en Newtons"""
    sensibilidad_total = (CELDA_SENSIBILIDAD_MV_V / 1000) * V_EXCITACION * INA849_GANANCIA
    return (voltaje / sensibilidad_total) * CELDA_CAPACIDAD_KG * 9.81


def integrar_frecuencia(accel, fs, fc_low=0.3):
    """
    Integración en el dominio de la frecuencia.
    Más robusto contra drift que integración en tiempo.
    
    a(t) -> FFT -> A(ω) -> V(ω) = A(ω)/(jω) -> IFFT -> v(t)
    v(t) -> FFT -> V(ω) -> X(ω) = V(ω)/(jω) -> IFFT -> x(t)
    
    Se aplica filtro pasa-altos para evitar división por cero en DC.
    """
    n = len(accel)
    freqs = fftfreq(n, 1/fs)
    omega = 2 * np.pi * freqs
    
    # FFT de aceleración
    A = fft(accel)
    
    # Filtro pasa-altos en frecuencia (evitar DC y muy bajas frecuencias)
    hp_filter = np.ones(n)
    hp_filter[np.abs(freqs) < fc_low] = 0
    # Transición suave
    transition = (np.abs(freqs) >= fc_low) & (np.abs(freqs) < fc_low * 2)
    hp_filter[transition] = (np.abs(freqs[transition]) - fc_low) / fc_low
    
    # Integración: V(ω) = A(ω) / (jω)
    # Evitar división por cero
    omega_safe = np.where(np.abs(omega) > 1e-10, omega, 1e-10)
    V = A * hp_filter / (1j * omega_safe)
    V[np.abs(omega) < 1e-10] = 0  # DC = 0
    
    # Velocidad en tiempo
    velocity = np.real(ifft(V))
    
    # Segunda integración: X(ω) = V(ω) / (jω)
    V_fft = fft(velocity)
    X = V_fft * hp_filter / (1j * omega_safe)
    X[np.abs(omega) < 1e-10] = 0
    
    # Posición en tiempo
    position = np.real(ifft(X))
    
    return velocity, position


def integrar_trapezoidal(accel, dt, fc_hp=0.5):
    """
    Integración trapezoidal con filtro pasa-altos para evitar drift.
    """
    # Filtro pasa-altos Butterworth
    b, a = signal.butter(4, fc_hp / (SAMPLE_RATE / 2), btype='high')
    
    # Primera integración: aceleración -> velocidad
    velocity = cumulative_trapezoid(accel, dx=dt, initial=0)
    velocity = signal.filtfilt(b, a, velocity)
    velocity = signal.detrend(velocity)
    
    # Segunda integración: velocidad -> posición
    position = cumulative_trapezoid(velocity, dx=dt, initial=0)
    position = signal.filtfilt(b, a, position)
    position = signal.detrend(position)
    
    return velocity, position


def calcular_friccion(archivo_path, metodo='frecuencia'):
    """
    Calcula la fuerza de fricción usando:
    F_fricción = F_celda - m·a - k·x
    
    Parámetros:
    -----------
    archivo_path : str
        Ruta al archivo CSV con datos
    metodo : str
        'frecuencia' o 'trapezoidal'
    
    Retorna:
    --------
    dict con todos los datos y resultados
    """
    # Cargar datos
    df = pd.read_csv(archivo_path)
    
    t = df['tiempo_s'].values
    fuerza_V = df['fuerza_V'].values
    accel_g = df['aceleracion_sensor_g'].values
    
    dt = 1.0 / SAMPLE_RATE
    n = len(t)
    
    # Remover DC de las señales
    fuerza_V_ac = fuerza_V - np.mean(fuerza_V)
    accel_g_ac = accel_g - np.mean(accel_g)
    
    # Convertir unidades
    F_celda = np.array([voltaje_a_fuerza_N(v) for v in fuerza_V_ac])  # N
    accel_ms2 = accel_g_ac * 9.81  # m/s²
    
    # Integrar aceleración para obtener posición
    if metodo == 'frecuencia':
        velocity, position = integrar_frecuencia(accel_ms2, SAMPLE_RATE, fc_low=0.3)
    else:
        velocity, position = integrar_trapezoidal(accel_ms2, dt, fc_hp=0.5)
    
    # Convertir a mm para visualización
    position_mm = position * 1000
    velocity_mms = velocity * 1000
    
    # ============================================
    # CÁLCULO DE FUERZA DE FRICCIÓN
    # F_fricción = F_celda - m·a - k·x
    # ============================================
    
    F_inercia = M_EFECTIVA * accel_ms2          # Componente de inercia [N]
    F_rigidez = K_RIGIDEZ * position            # Componente de rigidez [N]
    F_amortiguamiento = C_AMORT * velocity      # Componente de amortiguamiento [N]
    
    # Fricción = Lo que sobra
    F_friccion = F_celda - F_inercia - F_rigidez
    
    # También podemos incluir amortiguamiento viscoso
    F_friccion_sin_amort = F_celda - F_inercia - F_rigidez - F_amortiguamiento
    
    # ============================================
    # MÉTRICAS
    # ============================================
    
    # Fricción dinámica (promedio del valor absoluto)
    F_fric_mean = np.mean(np.abs(F_friccion))
    F_fric_rms = np.sqrt(np.mean(F_friccion**2))
    
    # Método de cruce por cero (más preciso)
    stroke = np.max(position_mm) - np.min(position_mm)
    x_threshold = stroke * 0.15 / 1000  # En metros
    
    near_zero = np.abs(position) < x_threshold
    F_fric_zc = 0
    F_forward = 0
    F_backward = 0
    
    if np.sum(near_zero) > 20:
        fwd_mask = near_zero & (velocity > 0.0001)
        bwd_mask = near_zero & (velocity < -0.0001)
        
        if np.sum(fwd_mask) > 5:
            F_forward = np.mean(F_friccion[fwd_mask])
        if np.sum(bwd_mask) > 5:
            F_backward = np.mean(F_friccion[bwd_mask])
        
        if np.sum(fwd_mask) > 0 and np.sum(bwd_mask) > 0:
            F_fric_zc = (F_forward - F_backward) / 2
    
    return {
        't': t,
        'F_celda': F_celda,
        'accel_ms2': accel_ms2,
        'velocity': velocity,
        'velocity_mms': velocity_mms,
        'position': position,
        'position_mm': position_mm,
        'F_inercia': F_inercia,
        'F_rigidez': F_rigidez,
        'F_amortiguamiento': F_amortiguamiento,
        'F_friccion': F_friccion,
        'F_friccion_sin_amort': F_friccion_sin_amort,
        'F_fric_mean': F_fric_mean,
        'F_fric_rms': F_fric_rms,
        'F_fric_zc': F_fric_zc,
        'F_forward': F_forward,
        'F_backward': F_backward,
        'stroke_mm': stroke,
        'v_mean_mms': np.mean(np.abs(velocity_mms))
    }


def graficar_analisis_friccion(datos, freq, archivo_nombre):
    """Genera gráficas detalladas del análisis de fricción"""
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f'Análisis de Fricción - {freq:.1f} Hz\n'
                 f'F_fricción = F_celda - m·a - k·x\n'
                 f'm = {M_EFECTIVA:.2f} kg, k = {K_RIGIDEZ:.0f} N/m', 
                 fontsize=12, fontweight='bold')
    
    t = datos['t']
    
    # Fila 1: Señales originales
    ax1 = axes[0, 0]
    ax1.plot(t, datos['F_celda'], 'c-', linewidth=0.5, label='F_celda')
    ax1.set_ylabel('Fuerza (N)')
    ax1.set_title('Fuerza medida (celda)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(t, datos['accel_ms2'], 'orange', linewidth=0.5)
    ax2.set_ylabel('Aceleración (m/s²)')
    ax2.set_title('Aceleración medida')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[0, 2]
    ax3.plot(t, datos['position_mm'], 'lime', linewidth=0.5)
    ax3.set_ylabel('Posición (mm)')
    ax3.set_title('Posición (integrada de aceleración)')
    ax3.grid(True, alpha=0.3)
    
    # Fila 2: Componentes de fuerza
    ax4 = axes[1, 0]
    ax4.plot(t, datos['F_inercia'], 'r-', linewidth=0.5, label=f'm·a (m={M_EFECTIVA:.1f}kg)')
    ax4.set_ylabel('Fuerza (N)')
    ax4.set_title('Componente de Inercia')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = axes[1, 1]
    ax5.plot(t, datos['F_rigidez'], 'b-', linewidth=0.5, label=f'k·x (k={K_RIGIDEZ/1000:.0f}kN/m)')
    ax5.set_ylabel('Fuerza (N)')
    ax5.set_title('Componente de Rigidez')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = axes[1, 2]
    ax6.plot(t, datos['F_friccion'], 'm-', linewidth=0.5)
    ax6.axhline(y=datos['F_fric_mean'], color='yellow', linestyle='--', 
                label=f'|F_fric| medio = {datos["F_fric_mean"]:.3f} N')
    ax6.axhline(y=-datos['F_fric_mean'], color='yellow', linestyle='--')
    ax6.set_ylabel('Fuerza (N)')
    ax6.set_title('FUERZA DE FRICCIÓN (F_celda - m·a - k·x)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Fila 3: Lazos de histéresis
    ax7 = axes[2, 0]
    ax7.plot(datos['position_mm'], datos['F_celda'], 'c-', linewidth=0.3, alpha=0.7)
    ax7.set_xlabel('Posición (mm)')
    ax7.set_ylabel('Fuerza (N)')
    ax7.set_title('F_celda vs Posición')
    ax7.grid(True, alpha=0.3)
    
    ax8 = axes[2, 1]
    ax8.plot(datos['position_mm'], datos['F_friccion'], 'm-', linewidth=0.3, alpha=0.7)
    ax8.axhline(y=0, color='w', linestyle='--', alpha=0.3)
    ax8.set_xlabel('Posición (mm)')
    ax8.set_ylabel('F_fricción (N)')
    ax8.set_title('F_fricción vs Posición (Lazo de Histéresis)')
    ax8.grid(True, alpha=0.3)
    
    ax9 = axes[2, 2]
    ax9.plot(datos['velocity_mms'], datos['F_friccion'], 'lime', linewidth=0.3, alpha=0.7)
    ax9.axhline(y=0, color='w', linestyle='--', alpha=0.3)
    ax9.axvline(x=0, color='w', linestyle='--', alpha=0.3)
    ax9.set_xlabel('Velocidad (mm/s)')
    ax9.set_ylabel('F_fricción (N)')
    ax9.set_title('F_fricción vs Velocidad')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar
    fig_path = os.path.join(DATOS_DIR, f'friccion_analisis_{freq:.1f}Hz.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Guardado: friccion_analisis_{freq:.1f}Hz.png")
    
    return fig


def main():
    print("="*70)
    print("CÁLCULO CORRECTO DE FUERZA DE FRICCIÓN")
    print("="*70)
    print(f"\nParámetros del sistema (del barrido de frecuencia):")
    print(f"  Masa efectiva:    m = {M_EFECTIVA:.2f} kg")
    print(f"  Rigidez:          k = {K_RIGIDEZ:.0f} N/m")
    print(f"  Amortiguamiento:  c = {C_AMORT:.2f} N·s/m")
    print(f"\nEcuación: F_fricción = F_celda - m·a - k·x")
    
    # Buscar archivos de fricción triangular
    base_name = "caracterizacion_fuerza_20251203_125007"
    resumen_path = os.path.join(DATOS_DIR, f"{base_name}_resumen.csv")
    
    df_resumen = pd.read_csv(resumen_path)
    df_tri = df_resumen[df_resumen['experiment_type'] == 'TRIANGLE'].copy()
    
    print(f"\n{'='*70}")
    print("ANÁLISIS DE EXPERIMENTOS TRIANGULARES")
    print(f"{'='*70}")
    
    resultados = []
    
    for i, (idx, row) in enumerate(df_tri.iterrows()):
        freq = row['frecuencia_Hz']
        exp_num = i + 1
        
        # Buscar archivo
        archivos = [f for f in os.listdir(DATOS_DIR) 
                   if base_name in f and f"exp{exp_num}_" in f and 'resumen' not in f]
        
        if not archivos:
            continue
        
        archivo_path = os.path.join(DATOS_DIR, archivos[0])
        print(f"\n📁 Procesando: {archivos[0]}")
        print(f"   Frecuencia: {freq:.1f} Hz")
        
        # Calcular fricción con método de frecuencia
        datos = calcular_friccion(archivo_path, metodo='frecuencia')
        
        print(f"\n   RESULTADOS:")
        print(f"   ├─ Carrera (stroke):     {datos['stroke_mm']:.3f} mm")
        print(f"   ├─ Velocidad media:      {datos['v_mean_mms']:.3f} mm/s")
        print(f"   ├─ F_fricción (media):   {datos['F_fric_mean']:.4f} N")
        print(f"   ├─ F_fricción (RMS):     {datos['F_fric_rms']:.4f} N")
        print(f"   ├─ F_fricción (ZC):      {datos['F_fric_zc']:.4f} N")
        print(f"   ├─ F_forward (x≈0):      {datos['F_forward']:.4f} N")
        print(f"   └─ F_backward (x≈0):     {datos['F_backward']:.4f} N")
        
        # Graficar
        graficar_analisis_friccion(datos, freq, archivos[0])
        
        resultados.append({
            'freq_Hz': freq,
            'stroke_mm': datos['stroke_mm'],
            'v_mean_mms': datos['v_mean_mms'],
            'F_fric_mean_N': datos['F_fric_mean'],
            'F_fric_rms_N': datos['F_fric_rms'],
            'F_fric_zc_N': datos['F_fric_zc'],
            'F_forward_N': datos['F_forward'],
            'F_backward_N': datos['F_backward']
        })
    
    # Resumen final
    df_resultados = pd.DataFrame(resultados)
    
    print(f"\n{'='*70}")
    print("RESUMEN FINAL DE FRICCIÓN")
    print(f"{'='*70}")
    print(df_resultados.to_string(index=False))
    
    # Guardar CSV
    csv_path = os.path.join(DATOS_DIR, 'friccion_calculada_correcta.csv')
    df_resultados.to_csv(csv_path, index=False)
    print(f"\n💾 Resultados guardados en: {csv_path}")
    
    # Gráfica resumen
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Resumen de Fricción Calculada\nF_fricción = F_celda - m·a - k·x', 
                 fontsize=12, fontweight='bold')
    
    # Filtrar datos válidos (excluir 0.5 Hz si es anómalo)
    df_valid = df_resultados[df_resultados['freq_Hz'] >= 1.0]
    
    ax1 = axes[0]
    ax1.bar(df_valid['freq_Hz'].astype(str), df_valid['F_fric_mean_N'], 
            color='magenta', alpha=0.7, label='F_fric (media)')
    ax1.bar(df_valid['freq_Hz'].astype(str), df_valid['F_fric_zc_N'], 
            color='cyan', alpha=0.5, label='F_fric (ZC)')
    ax1.set_xlabel('Frecuencia (Hz)')
    ax1.set_ylabel('Fuerza de Fricción (N)')
    ax1.set_title('Fricción por Frecuencia')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.scatter(df_valid['v_mean_mms'], df_valid['F_fric_mean_N'], 
               s=100, c='magenta', label='F_fric (media)')
    for _, row in df_valid.iterrows():
        ax2.annotate(f"{row['freq_Hz']:.1f}Hz", 
                    (row['v_mean_mms'], row['F_fric_mean_N']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax2.set_xlabel('Velocidad media (mm/s)')
    ax2.set_ylabel('Fuerza de Fricción (N)')
    ax2.set_title('Fricción vs Velocidad')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(DATOS_DIR, 'friccion_resumen_correcto.png'), dpi=150)
    print(f"📊 Gráfica resumen guardada")
    
    plt.show()

if __name__ == "__main__":
    main()
