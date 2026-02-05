"""
Cálculo de Fuerza de Fricción desde Barrido Senoidal
=====================================================
Usando los datos del barrido de frecuencia (10-90 Hz)

F_celda = m·a + k·x + F_fricción
F_fricción = F_celda - m·a - k·x

Donde x se obtiene integrando la aceleración dos veces.
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

# PARÁMETROS IDENTIFICADOS DEL BARRIDO
M_EFECTIVA = 12.42  # kg
K_RIGIDEZ = 997548  # N/m
C_AMORT = 1117.28   # N·s/m

plt.style.use('dark_background')

def voltaje_a_fuerza_N(voltaje):
    sensibilidad_total = (CELDA_SENSIBILIDAD_MV_V / 1000) * V_EXCITACION * INA849_GANANCIA
    return (voltaje / sensibilidad_total) * CELDA_CAPACIDAD_KG * 9.81


def integrar_senoidal(accel, fs, freq_excitacion):
    """
    Integración optimizada para señales senoidales.
    Usa filtro pasa-banda centrado en la frecuencia de excitación.
    
    Para señal senoidal: a(t) = A·sin(ωt)
    Velocidad: v(t) = -A/ω·cos(ωt) = A/ω·sin(ωt - π/2)
    Posición:  x(t) = -A/ω²·sin(ωt)
    """
    n = len(accel)
    freqs = fftfreq(n, 1/fs)
    omega = 2 * np.pi * freqs
    
    # FFT de aceleración
    A = fft(accel)
    
    # Filtro pasa-banda centrado en freq_excitacion
    # Ancho de banda: ±20% de la frecuencia
    bw = freq_excitacion * 0.3
    bp_filter = np.exp(-((np.abs(freqs) - freq_excitacion) / bw)**2)
    bp_filter += np.exp(-((np.abs(freqs) + freq_excitacion) / bw)**2)  # Frecuencias negativas
    
    # También incluir DC muy atenuado y frecuencias muy bajas
    hp_filter = 1 - np.exp(-(np.abs(freqs) / 0.5)**4)
    
    # Filtro combinado
    filtro = bp_filter * hp_filter
    
    # Evitar división por cero
    omega_safe = np.where(np.abs(omega) > 1e-10, omega, 1e-10)
    
    # Integración: V = A / (jω)
    V = A * filtro / (1j * omega_safe)
    V[np.abs(omega) < 1e-10] = 0
    velocity = np.real(ifft(V))
    
    # Segunda integración: X = V / (jω)
    V_fft = fft(velocity)
    X = V_fft * filtro / (1j * omega_safe)
    X[np.abs(omega) < 1e-10] = 0
    position = np.real(ifft(X))
    
    return velocity, position


def analizar_frecuencia(archivo_path, freq):
    """
    Analiza un archivo de barrido senoidal y calcula la fricción.
    """
    df = pd.read_csv(archivo_path)
    
    t = df['tiempo_s'].values
    fuerza_V = df['fuerza_V'].values
    accel_g = df['aceleracion_sensor_g'].values
    
    # Remover DC
    fuerza_V_ac = fuerza_V - np.mean(fuerza_V)
    accel_g_ac = accel_g - np.mean(accel_g)
    
    # Convertir unidades
    F_celda = np.array([voltaje_a_fuerza_N(v) for v in fuerza_V_ac])
    accel_ms2 = accel_g_ac * 9.81
    
    # Integrar para obtener velocidad y posición
    velocity, position = integrar_senoidal(accel_ms2, SAMPLE_RATE, freq)
    
    # Convertir a mm
    position_mm = position * 1000
    velocity_mms = velocity * 1000
    
    # ============================================
    # CÁLCULO DE COMPONENTES DE FUERZA
    # ============================================
    
    F_inercia = M_EFECTIVA * accel_ms2           # m·a [N]
    F_rigidez = K_RIGIDEZ * position             # k·x [N]
    F_amortiguamiento = C_AMORT * velocity       # c·v [N]
    
    # FRICCIÓN = Lo que sobra
    F_friccion = F_celda - F_inercia - F_rigidez
    F_friccion_con_amort = F_celda - F_inercia - F_rigidez - F_amortiguamiento
    
    # ============================================
    # MÉTRICAS
    # ============================================
    
    # RMS de cada componente
    F_celda_rms = np.sqrt(np.mean(F_celda**2))
    F_inercia_rms = np.sqrt(np.mean(F_inercia**2))
    F_rigidez_rms = np.sqrt(np.mean(F_rigidez**2))
    F_amort_rms = np.sqrt(np.mean(F_amortiguamiento**2))
    F_friccion_rms = np.sqrt(np.mean(F_friccion**2))
    
    # Amplitudes
    accel_amp = (np.max(accel_ms2) - np.min(accel_ms2)) / 2
    pos_amp = (np.max(position_mm) - np.min(position_mm)) / 2
    vel_amp = (np.max(velocity_mms) - np.min(velocity_mms)) / 2
    F_celda_amp = (np.max(F_celda) - np.min(F_celda)) / 2
    
    # Fricción por método de cruce por cero
    x_threshold = pos_amp * 0.2 if pos_amp > 0.001 else 0.01
    near_zero = np.abs(position_mm) < x_threshold
    
    F_fric_zc = 0
    F_forward = 0
    F_backward = 0
    
    if np.sum(near_zero) > 20:
        v_threshold = vel_amp * 0.1 if vel_amp > 0.01 else 0.001
        fwd = near_zero & (velocity_mms > v_threshold)
        bwd = near_zero & (velocity_mms < -v_threshold)
        
        if np.sum(fwd) > 5:
            F_forward = np.mean(F_friccion[fwd])
        if np.sum(bwd) > 5:
            F_backward = np.mean(F_friccion[bwd])
        
        if np.sum(fwd) > 0 and np.sum(bwd) > 0:
            F_fric_zc = (F_forward - F_backward) / 2
    
    return {
        't': t,
        'freq': freq,
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
        'F_friccion_con_amort': F_friccion_con_amort,
        # Métricas
        'accel_amp': accel_amp,
        'pos_amp_mm': pos_amp,
        'vel_amp_mms': vel_amp,
        'F_celda_rms': F_celda_rms,
        'F_celda_amp': F_celda_amp,
        'F_inercia_rms': F_inercia_rms,
        'F_rigidez_rms': F_rigidez_rms,
        'F_amort_rms': F_amort_rms,
        'F_friccion_rms': F_friccion_rms,
        'F_fric_zc': F_fric_zc,
        'F_forward': F_forward,
        'F_backward': F_backward
    }


def graficar_analisis(datos, freq):
    """Gráfica detallada del análisis"""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'Análisis de Fricción - Barrido Senoidal {freq:.0f} Hz\n'
                 f'F_fricción = F_celda - m·a - k·x\n'
                 f'm = {M_EFECTIVA:.2f} kg, k = {K_RIGIDEZ/1000:.0f} kN/m, c = {C_AMORT:.0f} N·s/m',
                 fontsize=12, fontweight='bold')
    
    t = datos['t']
    
    # Tomar solo unos ciclos para visualización
    n_ciclos = 5
    n_muestras = int(n_ciclos / freq * SAMPLE_RATE)
    idx = slice(len(t)//2, len(t)//2 + n_muestras)
    
    # Fila 1: Señales medidas
    ax1 = axes[0, 0]
    ax1.plot(t[idx], datos['F_celda'][idx], 'c-', linewidth=1)
    ax1.set_ylabel('Fuerza (N)')
    ax1.set_title(f'F_celda (medida) - Amp={datos["F_celda_amp"]:.2f}N')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(t[idx], datos['accel_ms2'][idx], 'orange', linewidth=1)
    ax2.set_ylabel('Aceleración (m/s²)')
    ax2.set_title(f'Aceleración - Amp={datos["accel_amp"]:.3f}m/s²')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[0, 2]
    ax3.plot(t[idx], datos['position_mm'][idx], 'lime', linewidth=1)
    ax3.set_ylabel('Posición (mm)')
    ax3.set_title(f'Posición (integrada) - Amp={datos["pos_amp_mm"]:.4f}mm')
    ax3.grid(True, alpha=0.3)
    
    # Fila 2: Componentes de fuerza
    ax4 = axes[1, 0]
    ax4.plot(t[idx], datos['F_inercia'][idx], 'r-', linewidth=1, label=f'm·a (RMS={datos["F_inercia_rms"]:.2f}N)')
    ax4.set_ylabel('Fuerza (N)')
    ax4.set_title('Componente de Inercia (m·a)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    ax5 = axes[1, 1]
    ax5.plot(t[idx], datos['F_rigidez'][idx], 'b-', linewidth=1, label=f'k·x (RMS={datos["F_rigidez_rms"]:.2f}N)')
    ax5.set_ylabel('Fuerza (N)')
    ax5.set_title('Componente de Rigidez (k·x)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    ax6 = axes[1, 2]
    ax6.plot(t[idx], datos['F_friccion'][idx], 'm-', linewidth=1, label=f'F_fric (RMS={datos["F_friccion_rms"]:.2f}N)')
    ax6.set_ylabel('Fuerza (N)')
    ax6.set_title('FRICCIÓN = F_celda - m·a - k·x')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # Fila 3: Lazos y comparación
    ax7 = axes[2, 0]
    ax7.plot(datos['position_mm'], datos['F_celda'], 'c-', linewidth=0.2, alpha=0.5)
    ax7.set_xlabel('Posición (mm)')
    ax7.set_ylabel('Fuerza (N)')
    ax7.set_title('F_celda vs Posición')
    ax7.grid(True, alpha=0.3)
    
    ax8 = axes[2, 1]
    ax8.plot(datos['position_mm'], datos['F_friccion'], 'm-', linewidth=0.2, alpha=0.5)
    ax8.axhline(y=0, color='w', linestyle='--', alpha=0.3)
    ax8.axhline(y=datos['F_forward'], color='lime', linestyle='-', alpha=0.7, label=f'F_fwd={datos["F_forward"]:.3f}N')
    ax8.axhline(y=datos['F_backward'], color='red', linestyle='-', alpha=0.7, label=f'F_bwd={datos["F_backward"]:.3f}N')
    ax8.set_xlabel('Posición (mm)')
    ax8.set_ylabel('F_fricción (N)')
    ax8.set_title(f'Lazo de Histéresis - F_fric(ZC)={datos["F_fric_zc"]:.3f}N')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)
    
    ax9 = axes[2, 2]
    ax9.plot(datos['velocity_mms'], datos['F_friccion'], 'lime', linewidth=0.2, alpha=0.5)
    ax9.axhline(y=0, color='w', linestyle='--', alpha=0.3)
    ax9.axvline(x=0, color='w', linestyle='--', alpha=0.3)
    ax9.set_xlabel('Velocidad (mm/s)')
    ax9.set_ylabel('F_fricción (N)')
    ax9.set_title('F_fricción vs Velocidad')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = os.path.join(DATOS_DIR, f'friccion_barrido_{freq:.0f}Hz.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Guardado: friccion_barrido_{freq:.0f}Hz.png")
    
    return fig


def main():
    print("="*70)
    print("CÁLCULO DE FRICCIÓN DESDE BARRIDO SENOIDAL")
    print("="*70)
    print(f"\nParámetros identificados:")
    print(f"  m = {M_EFECTIVA:.2f} kg")
    print(f"  k = {K_RIGIDEZ:.0f} N/m = {K_RIGIDEZ/1000:.0f} kN/m")
    print(f"  c = {C_AMORT:.2f} N·s/m")
    print(f"\nEcuación: F_fricción = F_celda - m·a - k·x")
    
    # Buscar archivos del barrido
    base_name = "caracterizacion_fuerza_20251203_120809"
    
    # Frecuencias del barrido
    frecuencias = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    
    print(f"\n{'='*70}")
    print("ANÁLISIS POR FRECUENCIA")
    print(f"{'='*70}")
    
    resultados = []
    
    for i, freq in enumerate(frecuencias):
        archivo = f"{base_name}_exp{i+1}_{freq}Hz.csv"
        archivo_path = os.path.join(DATOS_DIR, archivo)
        
        if not os.path.exists(archivo_path):
            print(f"  No encontrado: {archivo}")
            continue
        
        print(f"\n📁 {freq} Hz")
        
        datos = analizar_frecuencia(archivo_path, freq)
        
        print(f"   ├─ Amplitud accel:    {datos['accel_amp']:.4f} m/s²")
        print(f"   ├─ Amplitud posición: {datos['pos_amp_mm']:.4f} mm")
        print(f"   ├─ F_celda RMS:       {datos['F_celda_rms']:.3f} N")
        print(f"   ├─ F_inercia RMS:     {datos['F_inercia_rms']:.3f} N")
        print(f"   ├─ F_rigidez RMS:     {datos['F_rigidez_rms']:.3f} N")
        print(f"   ├─ F_fricción RMS:    {datos['F_friccion_rms']:.3f} N")
        print(f"   └─ F_fricción (ZC):   {datos['F_fric_zc']:.4f} N")
        
        # Graficar algunas frecuencias representativas
        if freq in [10, 30, 50, 70, 90]:
            graficar_analisis(datos, freq)
        
        resultados.append({
            'freq_Hz': freq,
            'accel_amp_ms2': datos['accel_amp'],
            'pos_amp_mm': datos['pos_amp_mm'],
            'vel_amp_mms': datos['vel_amp_mms'],
            'F_celda_rms_N': datos['F_celda_rms'],
            'F_inercia_rms_N': datos['F_inercia_rms'],
            'F_rigidez_rms_N': datos['F_rigidez_rms'],
            'F_amort_rms_N': datos['F_amort_rms'],
            'F_friccion_rms_N': datos['F_friccion_rms'],
            'F_fric_zc_N': datos['F_fric_zc']
        })
    
    df_res = pd.DataFrame(resultados)
    
    print(f"\n{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}")
    print(df_res.to_string(index=False))
    
    # Gráfica resumen
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Resumen de Componentes de Fuerza vs Frecuencia\n'
                 f'm={M_EFECTIVA:.1f}kg, k={K_RIGIDEZ/1000:.0f}kN/m, c={C_AMORT:.0f}N·s/m',
                 fontsize=12, fontweight='bold')
    
    # Componentes RMS vs frecuencia
    ax1 = axes[0, 0]
    ax1.semilogy(df_res['freq_Hz'], df_res['F_celda_rms_N'], 'co-', markersize=8, label='F_celda')
    ax1.semilogy(df_res['freq_Hz'], df_res['F_inercia_rms_N'], 'r^-', markersize=8, label='m·a (inercia)')
    ax1.semilogy(df_res['freq_Hz'], df_res['F_rigidez_rms_N'], 'bs-', markersize=8, label='k·x (rigidez)')
    ax1.semilogy(df_res['freq_Hz'], df_res['F_friccion_rms_N'], 'm*-', markersize=10, label='F_fricción')
    ax1.set_xlabel('Frecuencia (Hz)')
    ax1.set_ylabel('Fuerza RMS (N)')
    ax1.set_title('Componentes de Fuerza (RMS)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Fricción vs frecuencia
    ax2 = axes[0, 1]
    ax2.plot(df_res['freq_Hz'], df_res['F_friccion_rms_N'], 'mo-', markersize=10, linewidth=2, label='F_fric RMS')
    ax2.plot(df_res['freq_Hz'], np.abs(df_res['F_fric_zc_N']), 'c^--', markersize=8, label='|F_fric ZC|')
    ax2.set_xlabel('Frecuencia (Hz)')
    ax2.set_ylabel('Fuerza de Fricción (N)')
    ax2.set_title('Fuerza de Fricción vs Frecuencia')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Amplitudes vs frecuencia
    ax3 = axes[1, 0]
    ax3.semilogy(df_res['freq_Hz'], df_res['accel_amp_ms2'], 'o-', markersize=8, label='Aceleración (m/s²)')
    ax3.set_xlabel('Frecuencia (Hz)')
    ax3.set_ylabel('Amplitud')
    ax3.set_title('Amplitud de Aceleración')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.semilogy(df_res['freq_Hz'], df_res['pos_amp_mm'], 's-', markersize=8, color='lime', label='Posición (mm)')
    ax4.set_xlabel('Frecuencia (Hz)')
    ax4.set_ylabel('Amplitud')
    ax4.set_title('Amplitud de Posición')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(DATOS_DIR, 'friccion_barrido_resumen.png'), dpi=150)
    print(f"\n📊 Guardado: friccion_barrido_resumen.png")
    
    # Guardar CSV
    csv_path = os.path.join(DATOS_DIR, 'friccion_barrido_resultados.csv')
    df_res.to_csv(csv_path, index=False)
    print(f"💾 Guardado: {csv_path}")
    
    # Fricción promedio
    F_fric_promedio = df_res['F_friccion_rms_N'].mean()
    print(f"\n{'='*70}")
    print(f"FRICCIÓN PROMEDIO: {F_fric_promedio:.3f} N (RMS)")
    print(f"{'='*70}")
    
    plt.show()

if __name__ == "__main__":
    main()
