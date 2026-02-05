"""
Cálculo de Fuerza de Fricción - Versión 2
==========================================
Problema detectado: Los parámetros del barrido (k muy alta) no aplican
al modo triangular porque el sistema opera en diferente régimen.

Enfoque alternativo:
1. Para fricción a baja frecuencia, la inercia es despreciable (a ≈ 0)
2. La fuerza medida es principalmente: F_celda ≈ F_fricción + F_rigidez_local

Método: Usar el lazo de histéresis directamente
- La fricción es el "ancho" del lazo a x=0
- F_fricción = (F_forward - F_backward) / 2
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

plt.style.use('dark_background')

def voltaje_a_fuerza_N(voltaje):
    sensibilidad_total = (CELDA_SENSIBILIDAD_MV_V / 1000) * V_EXCITACION * INA849_GANANCIA
    return (voltaje / sensibilidad_total) * CELDA_CAPACIDAD_KG * 9.81


def integrar_omega(accel, fs, fc_low=0.1):
    """
    Integración en dominio de frecuencia con filtro pasa-altos.
    Método omega: divide por jω en frecuencia.
    """
    n = len(accel)
    freqs = fftfreq(n, 1/fs)
    omega = 2 * np.pi * freqs
    
    A = fft(accel)
    
    # Filtro pasa-altos suave
    hp = 1 - np.exp(-(np.abs(freqs) / fc_low)**2)
    
    # Evitar división por cero
    omega_safe = np.where(np.abs(omega) > 1e-10, omega, 1e-10)
    
    # V = A / (jω)
    V = A * hp / (1j * omega_safe)
    V[np.abs(omega) < 1e-10] = 0
    velocity = np.real(ifft(V))
    
    # X = V / (jω)  
    V_fft = fft(velocity)
    X = V_fft * hp / (1j * omega_safe)
    X[np.abs(omega) < 1e-10] = 0
    position = np.real(ifft(X))
    
    return velocity, position


def analizar_friccion_directa(archivo_path):
    """
    Análisis directo de fricción sin asumir parámetros del sistema.
    
    Para onda triangular a baja frecuencia:
    - La aceleración es casi cero (velocidad constante)
    - F_celda ≈ F_fricción (si no hay rigidez significativa)
    - O F_celda = k_local·x + F_fricción
    
    Método: Analizar el lazo de histéresis F vs x
    """
    df = pd.read_csv(archivo_path)
    
    t = df['tiempo_s'].values
    fuerza_V = df['fuerza_V'].values
    accel_g = df['aceleracion_sensor_g'].values
    
    dt = 1.0 / SAMPLE_RATE
    
    # Remover DC
    fuerza_V_ac = fuerza_V - np.mean(fuerza_V)
    accel_g_ac = accel_g - np.mean(accel_g)
    
    # Convertir
    F_celda = np.array([voltaje_a_fuerza_N(v) for v in fuerza_V_ac])
    accel_ms2 = accel_g_ac * 9.81
    
    # Integrar para obtener posición
    velocity, position = integrar_omega(accel_ms2, SAMPLE_RATE, fc_low=0.1)
    
    position_mm = position * 1000
    velocity_mms = velocity * 1000
    
    # ============================================
    # ANÁLISIS DEL LAZO DE HISTÉRESIS
    # ============================================
    
    # Métricas básicas
    F_max = np.max(F_celda)
    F_min = np.min(F_celda)
    x_max = np.max(position_mm)
    x_min = np.min(position_mm)
    stroke = x_max - x_min
    
    # Método 1: Pico a pico simple
    F_fric_p2p = (F_max - F_min) / 2
    
    # Método 2: Cruce por cero (más preciso)
    x_threshold = max(stroke * 0.2, 0.01)  # 20% del stroke o mínimo 0.01 mm
    near_zero = np.abs(position_mm) < x_threshold
    
    F_forward = 0
    F_backward = 0
    F_fric_zc = 0
    n_fwd = 0
    n_bwd = 0
    
    if np.sum(near_zero) > 20:
        # Separar por dirección de velocidad
        v_threshold = np.std(velocity_mms) * 0.1
        fwd_mask = near_zero & (velocity_mms > v_threshold)
        bwd_mask = near_zero & (velocity_mms < -v_threshold)
        
        n_fwd = np.sum(fwd_mask)
        n_bwd = np.sum(bwd_mask)
        
        if n_fwd > 5:
            F_forward = np.mean(F_celda[fwd_mask])
        if n_bwd > 5:
            F_backward = np.mean(F_celda[bwd_mask])
        
        if n_fwd > 0 and n_bwd > 0:
            F_fric_zc = (F_forward - F_backward) / 2
    
    # Método 3: Ajuste lineal para separar rigidez de fricción
    # F = k·x + F_fric·sign(v)
    # En el lazo, la pendiente da k, el offset da F_fric
    
    # Separar datos por dirección
    going_up = velocity_mms > np.std(velocity_mms) * 0.1
    going_down = velocity_mms < -np.std(velocity_mms) * 0.1
    
    k_up = 0
    k_down = 0
    F_offset_up = 0
    F_offset_down = 0
    
    if np.sum(going_up) > 10:
        # Ajuste lineal F = k·x + b para subida
        coef_up = np.polyfit(position_mm[going_up], F_celda[going_up], 1)
        k_up = coef_up[0]  # N/mm
        F_offset_up = coef_up[1]  # N
    
    if np.sum(going_down) > 10:
        coef_down = np.polyfit(position_mm[going_down], F_celda[going_down], 1)
        k_down = coef_down[0]
        F_offset_down = coef_down[1]
    
    # Rigidez promedio y fricción
    k_local = (k_up + k_down) / 2  # N/mm
    F_fric_fit = (F_offset_up - F_offset_down) / 2  # N
    
    return {
        't': t,
        'F_celda': F_celda,
        'accel_ms2': accel_ms2,
        'velocity_mms': velocity_mms,
        'position_mm': position_mm,
        'F_max': F_max,
        'F_min': F_min,
        'x_max': x_max,
        'x_min': x_min,
        'stroke_mm': stroke,
        'v_mean_mms': np.mean(np.abs(velocity_mms)),
        'F_fric_p2p': F_fric_p2p,
        'F_fric_zc': F_fric_zc,
        'F_forward': F_forward,
        'F_backward': F_backward,
        'n_fwd': n_fwd,
        'n_bwd': n_bwd,
        'k_local_N_mm': k_local,
        'F_fric_fit': F_fric_fit
    }


def graficar_friccion(datos, freq):
    """Gráfica detallada del análisis de fricción"""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Análisis de Fricción Directa - {freq:.1f} Hz\n'
                 f'Sin asumir parámetros del sistema', fontsize=12, fontweight='bold')
    
    t = datos['t']
    
    # Señales en tiempo
    ax1 = axes[0, 0]
    ax1.plot(t, datos['F_celda'], 'c-', linewidth=0.5)
    ax1.axhline(y=datos['F_max'], color='r', linestyle='--', alpha=0.5, label=f'F_max={datos["F_max"]:.3f}N')
    ax1.axhline(y=datos['F_min'], color='b', linestyle='--', alpha=0.5, label=f'F_min={datos["F_min"]:.3f}N')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Fuerza (N)')
    ax1.set_title('Fuerza medida (celda)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(t, datos['position_mm'], 'lime', linewidth=0.5)
    ax2.axhline(y=datos['x_max'], color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=datos['x_min'], color='b', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Posición (mm)')
    ax2.set_title(f'Posición (integrada) - Stroke={datos["stroke_mm"]:.3f}mm')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[0, 2]
    ax3.plot(t, datos['velocity_mms'], 'orange', linewidth=0.5)
    ax3.axhline(y=0, color='w', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Velocidad (mm/s)')
    ax3.set_title(f'Velocidad - Media={datos["v_mean_mms"]:.3f}mm/s')
    ax3.grid(True, alpha=0.3)
    
    # Lazo de histéresis F vs x
    ax4 = axes[1, 0]
    ax4.plot(datos['position_mm'], datos['F_celda'], 'c-', linewidth=0.3, alpha=0.7)
    ax4.axhline(y=0, color='w', linestyle='--', alpha=0.3)
    ax4.axvline(x=0, color='w', linestyle='--', alpha=0.3)
    
    # Marcar F_forward y F_backward
    ax4.axhline(y=datos['F_forward'], color='lime', linestyle='-', alpha=0.8, 
                label=f'F_fwd={datos["F_forward"]:.3f}N (n={datos["n_fwd"]})')
    ax4.axhline(y=datos['F_backward'], color='red', linestyle='-', alpha=0.8,
                label=f'F_bwd={datos["F_backward"]:.3f}N (n={datos["n_bwd"]})')
    
    ax4.set_xlabel('Posición (mm)')
    ax4.set_ylabel('Fuerza (N)')
    ax4.set_title('Lazo de Histéresis F vs x')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # F vs Velocidad
    ax5 = axes[1, 1]
    ax5.plot(datos['velocity_mms'], datos['F_celda'], 'orange', linewidth=0.3, alpha=0.7)
    ax5.axhline(y=0, color='w', linestyle='--', alpha=0.3)
    ax5.axvline(x=0, color='w', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Velocidad (mm/s)')
    ax5.set_ylabel('Fuerza (N)')
    ax5.set_title('F vs Velocidad')
    ax5.grid(True, alpha=0.3)
    
    # Resumen de métricas
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    texto = f"MÉTRICAS DE FRICCIÓN - {freq:.1f} Hz\n"
    texto += "="*40 + "\n\n"
    texto += f"Carrera (stroke):    {datos['stroke_mm']:.4f} mm\n"
    texto += f"Velocidad media:     {datos['v_mean_mms']:.4f} mm/s\n\n"
    texto += "FUERZA DE FRICCIÓN:\n"
    texto += "-"*40 + "\n"
    texto += f"Método Pico-Pico:    {datos['F_fric_p2p']:.4f} N\n"
    texto += f"Método Zero-Cross:   {datos['F_fric_zc']:.4f} N\n"
    texto += f"Método Ajuste:       {datos['F_fric_fit']:.4f} N\n\n"
    texto += "DETALLES ZERO-CROSSING:\n"
    texto += "-"*40 + "\n"
    texto += f"F_forward (v>0):     {datos['F_forward']:.4f} N\n"
    texto += f"F_backward (v<0):    {datos['F_backward']:.4f} N\n"
    texto += f"Diferencia:          {datos['F_forward']-datos['F_backward']:.4f} N\n\n"
    texto += "RIGIDEZ LOCAL:\n"
    texto += "-"*40 + "\n"
    texto += f"k_local:             {datos['k_local_N_mm']:.2f} N/mm\n"
    texto += f"                     = {datos['k_local_N_mm']*1000:.0f} N/m\n"
    
    ax6.text(0.05, 0.95, texto, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))
    
    plt.tight_layout()
    
    fig_path = os.path.join(DATOS_DIR, f'friccion_directa_{freq:.1f}Hz.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Guardado: friccion_directa_{freq:.1f}Hz.png")
    
    return fig


def main():
    print("="*70)
    print("CÁLCULO DIRECTO DE FUERZA DE FRICCIÓN")
    print("="*70)
    print("\nMétodo: Análisis del lazo de histéresis sin asumir parámetros")
    print("F_fricción = (F_forward - F_backward) / 2  en x ≈ 0")
    
    # Buscar archivos
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
        
        archivos = [f for f in os.listdir(DATOS_DIR) 
                   if base_name in f and f"exp{exp_num}_" in f and 'resumen' not in f]
        
        if not archivos:
            continue
        
        archivo_path = os.path.join(DATOS_DIR, archivos[0])
        print(f"\n📁 {archivos[0]}")
        print(f"   Frecuencia: {freq:.1f} Hz")
        
        datos = analizar_friccion_directa(archivo_path)
        
        print(f"\n   RESULTADOS:")
        print(f"   ├─ Carrera:           {datos['stroke_mm']:.4f} mm")
        print(f"   ├─ Velocidad media:   {datos['v_mean_mms']:.4f} mm/s")
        print(f"   ├─ F_fric (P2P):      {datos['F_fric_p2p']:.4f} N")
        print(f"   ├─ F_fric (ZC):       {datos['F_fric_zc']:.4f} N")
        print(f"   ├─ F_fric (Ajuste):   {datos['F_fric_fit']:.4f} N")
        print(f"   ├─ F_forward:         {datos['F_forward']:.4f} N")
        print(f"   ├─ F_backward:        {datos['F_backward']:.4f} N")
        print(f"   └─ k_local:           {datos['k_local_N_mm']:.2f} N/mm")
        
        graficar_friccion(datos, freq)
        
        resultados.append({
            'freq_Hz': freq,
            'stroke_mm': datos['stroke_mm'],
            'v_mean_mms': datos['v_mean_mms'],
            'F_fric_p2p_N': datos['F_fric_p2p'],
            'F_fric_zc_N': datos['F_fric_zc'],
            'F_fric_fit_N': datos['F_fric_fit'],
            'F_forward_N': datos['F_forward'],
            'F_backward_N': datos['F_backward'],
            'k_local_N_mm': datos['k_local_N_mm']
        })
    
    df_res = pd.DataFrame(resultados)
    
    print(f"\n{'='*70}")
    print("RESUMEN FINAL")
    print(f"{'='*70}")
    print(df_res.to_string(index=False))
    
    # Guardar
    csv_path = os.path.join(DATOS_DIR, 'friccion_directa_resultados.csv')
    df_res.to_csv(csv_path, index=False)
    print(f"\n💾 Guardado: {csv_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
