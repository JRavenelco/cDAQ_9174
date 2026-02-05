"""
Análisis Directo de Fricción
============================
Sin asumir modelo - análisis directo de los datos.

Observación clave:
- F_celda ≈ 0.5 N (constante en todas las frecuencias)
- Aceleración varía mucho con la frecuencia

Esto sugiere que la celda de carga mide algo diferente a m·a.
Posiblemente mide solo la fricción o una fuerza de reacción.

Enfoque: Analizar F_celda directamente como indicador de fricción.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import os

DATOS_DIR = r"c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\Pruebas\src\caracterizacion_fuerza"
SAMPLE_RATE = 2500

CELDA_CAPACIDAD_KG = 500.0
CELDA_SENSIBILIDAD_MV_V = 1.7
INA849_GANANCIA = 601.0
V_EXCITACION = 10.0

plt.style.use('dark_background')

def voltaje_a_fuerza_N(voltaje):
    sensibilidad_total = (CELDA_SENSIBILIDAD_MV_V / 1000) * V_EXCITACION * INA849_GANANCIA
    return (voltaje / sensibilidad_total) * CELDA_CAPACIDAD_KG * 9.81


def integrar_frecuencia(accel, fs, freq):
    """Integración en frecuencia con filtro pasa-banda"""
    n = len(accel)
    freqs = fftfreq(n, 1/fs)
    omega = 2 * np.pi * freqs
    
    A = fft(accel)
    
    # Filtro pasa-banda centrado en freq
    bw = max(freq * 0.3, 2)  # Ancho de banda mínimo 2 Hz
    bp = np.exp(-((np.abs(freqs) - freq) / bw)**2)
    hp = 1 - np.exp(-(np.abs(freqs) / 0.3)**4)
    filtro = bp * hp
    
    omega_safe = np.where(np.abs(omega) > 1e-10, omega, 1e-10)
    
    # Velocidad
    V = A * filtro / (1j * omega_safe)
    V[np.abs(omega) < 1e-10] = 0
    velocity = np.real(ifft(V))
    
    # Posición
    V_fft = fft(velocity)
    X = V_fft * filtro / (1j * omega_safe)
    X[np.abs(omega) < 1e-10] = 0
    position = np.real(ifft(X))
    
    return velocity, position


def analizar_archivo(archivo_path, freq):
    """Análisis completo de un archivo"""
    df = pd.read_csv(archivo_path)
    
    t = df['tiempo_s'].values
    fuerza_V = df['fuerza_V'].values
    accel_g = df['aceleracion_sensor_g'].values
    
    # Remover DC
    F_celda = np.array([voltaje_a_fuerza_N(v - np.mean(fuerza_V)) for v in fuerza_V])
    accel = (accel_g - np.mean(accel_g)) * 9.81
    
    # Integrar
    velocity, position = integrar_frecuencia(accel, SAMPLE_RATE, freq)
    
    # Métricas
    F_rms = np.sqrt(np.mean(F_celda**2))
    F_amp = (np.max(F_celda) - np.min(F_celda)) / 2
    a_rms = np.sqrt(np.mean(accel**2))
    a_amp = (np.max(accel) - np.min(accel)) / 2
    v_amp = (np.max(velocity) - np.min(velocity)) / 2
    x_amp = (np.max(position) - np.min(position)) / 2
    
    # FRF simple
    H_simple = F_rms / a_rms if a_rms > 1e-10 else 0
    
    return {
        't': t,
        'F_celda': F_celda,
        'accel': accel,
        'velocity': velocity,
        'position': position,
        'F_rms': F_rms,
        'F_amp': F_amp,
        'a_rms': a_rms,
        'a_amp': a_amp,
        'v_amp': v_amp,
        'x_amp': x_amp,
        'H_simple': H_simple
    }


def main():
    print("="*70)
    print("ANÁLISIS DIRECTO - SIN MODELO")
    print("="*70)
    
    base_name = "caracterizacion_fuerza_20251203_120809"
    frecuencias = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    
    resultados = []
    
    print("\n{:>5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Freq", "F_rms", "F_amp", "a_amp", "v_amp", "x_amp", "H=F/a"))
    print("{:>5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "(Hz)", "(N)", "(N)", "(m/s²)", "(mm/s)", "(mm)", "(kg)"))
    print("-"*70)
    
    for i, freq in enumerate(frecuencias):
        archivo = f"{base_name}_exp{i+1}_{freq}Hz.csv"
        archivo_path = os.path.join(DATOS_DIR, archivo)
        
        if not os.path.exists(archivo_path):
            continue
        
        res = analizar_archivo(archivo_path, freq)
        
        print("{:5.0f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}".format(
            freq, res['F_rms'], res['F_amp'], res['a_amp'], 
            res['v_amp']*1000, res['x_amp']*1000, res['H_simple']))
        
        resultados.append({
            'freq_Hz': freq,
            'F_rms_N': res['F_rms'],
            'F_amp_N': res['F_amp'],
            'a_amp_ms2': res['a_amp'],
            'v_amp_mms': res['v_amp'] * 1000,
            'x_amp_mm': res['x_amp'] * 1000,
            'H_simple_kg': res['H_simple']
        })
    
    df = pd.DataFrame(resultados)
    
    # Gráficas
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Análisis Directo del Barrido de Frecuencia', fontsize=14, fontweight='bold')
    
    # F_celda vs frecuencia
    ax1 = axes[0, 0]
    ax1.plot(df['freq_Hz'], df['F_rms_N'], 'co-', markersize=8, label='RMS')
    ax1.plot(df['freq_Hz'], df['F_amp_N'], 'c^--', markersize=6, alpha=0.7, label='Amplitud')
    ax1.set_xlabel('Frecuencia (Hz)')
    ax1.set_ylabel('Fuerza (N)')
    ax1.set_title('Fuerza Medida (Celda)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Aceleración vs frecuencia
    ax2 = axes[0, 1]
    ax2.semilogy(df['freq_Hz'], df['a_amp_ms2'], 'o-', color='orange', markersize=8)
    ax2.set_xlabel('Frecuencia (Hz)')
    ax2.set_ylabel('Amplitud Aceleración (m/s²)')
    ax2.set_title('Aceleración')
    ax2.grid(True, alpha=0.3)
    
    # Posición vs frecuencia
    ax3 = axes[0, 2]
    ax3.semilogy(df['freq_Hz'], df['x_amp_mm'], 's-', color='lime', markersize=8)
    ax3.set_xlabel('Frecuencia (Hz)')
    ax3.set_ylabel('Amplitud Posición (mm)')
    ax3.set_title('Posición (Integrada)')
    ax3.grid(True, alpha=0.3)
    
    # H = F/a vs frecuencia
    ax4 = axes[1, 0]
    ax4.semilogy(df['freq_Hz'], df['H_simple_kg'], 'm*-', markersize=10)
    ax4.set_xlabel('Frecuencia (Hz)')
    ax4.set_ylabel('H = F_rms/a_rms (kg)')
    ax4.set_title('FRF Simple (Magnitud)')
    ax4.grid(True, alpha=0.3)
    
    # F vs a (scatter)
    ax5 = axes[1, 1]
    ax5.scatter(df['a_amp_ms2'], df['F_amp_N'], c=df['freq_Hz'], cmap='viridis', s=100)
    for _, row in df.iterrows():
        ax5.annotate(f"{row['freq_Hz']:.0f}", (row['a_amp_ms2'], row['F_amp_N']), fontsize=8)
    ax5.set_xlabel('Amplitud Aceleración (m/s²)')
    ax5.set_ylabel('Amplitud Fuerza (N)')
    ax5.set_title('F vs a (color=frecuencia)')
    ax5.grid(True, alpha=0.3)
    
    # Interpretación
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    F_promedio = df['F_rms_N'].mean()
    F_std = df['F_rms_N'].std()
    
    texto = "OBSERVACIONES\n"
    texto += "="*40 + "\n\n"
    texto += f"F_celda promedio: {F_promedio:.3f} ± {F_std:.3f} N\n\n"
    texto += "La fuerza medida es casi CONSTANTE\n"
    texto += "independiente de la frecuencia.\n\n"
    texto += "Esto sugiere que la celda de carga\n"
    texto += "NO mide la fuerza dinámica total\n"
    texto += "(m·a + k·x), sino algo diferente.\n\n"
    texto += "POSIBLES EXPLICACIONES:\n"
    texto += "-"*40 + "\n"
    texto += "1. La celda mide solo fricción\n"
    texto += "2. La celda está en paralelo, no serie\n"
    texto += "3. Problema de montaje/configuración\n\n"
    texto += f"Si F_celda ≈ F_fricción:\n"
    texto += f"   F_fricción ≈ {F_promedio:.3f} N\n"
    
    ax6.text(0.05, 0.95, texto, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(os.path.join(DATOS_DIR, 'analisis_directo.png'), dpi=150)
    print(f"\n📊 Guardado: analisis_directo.png")
    
    # Conclusión
    print(f"\n{'='*70}")
    print("CONCLUSIÓN")
    print(f"{'='*70}")
    print(f"\nFuerza medida (celda): {F_promedio:.3f} ± {F_std:.3f} N")
    print(f"\nLa fuerza es prácticamente constante en todo el rango de frecuencias.")
    print(f"Esto indica que la celda NO mide la fuerza dinámica total.")
    print(f"\nSi interpretamos F_celda como fuerza de fricción:")
    print(f"   F_fricción ≈ {F_promedio:.3f} N")
    
    df.to_csv(os.path.join(DATOS_DIR, 'analisis_directo.csv'), index=False)
    
    plt.show()

if __name__ == "__main__":
    main()
