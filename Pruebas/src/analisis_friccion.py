"""
Análisis de Experimentos de Fricción (Onda Triangular)
=======================================================
Analiza los datos de caracterización de fricción usando el método
de cruce por cero y genera gráficas de histéresis.

Autor: Cascade AI
Fecha: 2025-12-03
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import cumulative_trapezoid
import os

# Configuración
DATOS_DIR = r"c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\Pruebas\src\caracterizacion_fuerza"
SAMPLE_RATE = 2500  # Hz

# Parámetros del sensor de fuerza
CELDA_CAPACIDAD_KG = 500.0
CELDA_SENSIBILIDAD_MV_V = 1.7
INA849_GANANCIA = 601.0
V_EXCITACION = 10.0

def voltaje_a_fuerza_N(voltaje):
    """Convierte voltaje a fuerza en Newtons"""
    sensibilidad_total = (CELDA_SENSIBILIDAD_MV_V / 1000) * V_EXCITACION * INA849_GANANCIA
    fuerza_kg = (voltaje / sensibilidad_total) * CELDA_CAPACIDAD_KG
    return fuerza_kg * 9.81

def procesar_archivo_friccion(archivo_path):
    """
    Procesa un archivo de datos de fricción.
    Integra aceleración para obtener desplazamiento y calcula métricas.
    """
    df = pd.read_csv(archivo_path)
    
    t = df['tiempo_s'].values
    fuerza_V = df['fuerza_V'].values
    accel_g = df['aceleracion_sensor_g'].values  # Usar acelerómetro sobre sensor
    
    dt = 1.0 / SAMPLE_RATE
    
    # Remover DC
    fuerza_V = fuerza_V - np.mean(fuerza_V)
    accel_g = accel_g - np.mean(accel_g)
    
    # Filtro pasa-altos para evitar drift (0.5 Hz)
    try:
        fc = 0.3  # Frecuencia de corte más baja para capturar movimiento lento
        b, a = signal.butter(2, fc / (SAMPLE_RATE / 2), btype='high')
        accel_filtered = signal.filtfilt(b, a, accel_g)
    except:
        accel_filtered = accel_g
    
    # Convertir a m/s²
    accel_ms2 = accel_filtered * 9.81
    
    # Primera integración: Aceleración → Velocidad
    velocity = cumulative_trapezoid(accel_ms2, dx=dt, initial=0)
    velocity = signal.detrend(velocity)
    velocity_mms = velocity * 1000
    
    # Segunda integración: Velocidad → Desplazamiento
    displacement = cumulative_trapezoid(velocity, dx=dt, initial=0)
    displacement = signal.detrend(displacement)
    displacement_mm = displacement * 1000
    
    # Convertir fuerza a Newtons
    force_N = np.array([voltaje_a_fuerza_N(v) for v in fuerza_V])
    force_N = force_N - np.mean(force_N)
    
    # Métricas
    F_max = np.max(force_N)
    F_min = np.min(force_N)
    F_friction_peak = (F_max - F_min) / 2
    
    stroke = np.max(displacement_mm) - np.min(displacement_mm)
    v_mean = np.mean(np.abs(velocity_mms))
    
    # Método de cruce por cero
    F_friction_zc = 0
    F_forward = 0
    F_backward = 0
    
    x_threshold = stroke * 0.15 if stroke > 0.01 else 0.1
    near_zero_mask = np.abs(displacement_mm) < x_threshold
    
    if np.sum(near_zero_mask) > 10:
        forward_mask = near_zero_mask & (velocity_mms > 0.1)
        backward_mask = near_zero_mask & (velocity_mms < -0.1)
        
        if np.sum(forward_mask) > 5:
            F_forward = np.mean(force_N[forward_mask])
        if np.sum(backward_mask) > 5:
            F_backward = np.mean(force_N[backward_mask])
        
        if np.sum(forward_mask) > 0 and np.sum(backward_mask) > 0:
            F_friction_zc = (F_forward - F_backward) / 2
    
    # Energía disipada (área del lazo)
    energy_mJ = 0.5 * np.abs(np.sum(displacement_mm[:-1] * force_N[1:] - 
                                    displacement_mm[1:] * force_N[:-1]))
    
    return {
        't': t,
        'force_N': force_N,
        'displacement_mm': displacement_mm,
        'velocity_mms': velocity_mms,
        'accel_g': accel_g,
        'F_max': F_max,
        'F_min': F_min,
        'F_friction_peak': F_friction_peak,
        'F_friction_zc': F_friction_zc,
        'F_forward': F_forward,
        'F_backward': F_backward,
        'stroke': stroke,
        'v_mean': v_mean,
        'energy_mJ': energy_mJ
    }

def analizar_sesion_friccion(base_name):
    """Analiza todos los experimentos de una sesión de fricción"""
    
    # Cargar resumen
    resumen_path = os.path.join(DATOS_DIR, f"{base_name}_resumen.csv")
    df_resumen = pd.read_csv(resumen_path)
    
    # Filtrar solo experimentos TRIANGLE
    df_triangle = df_resumen[df_resumen['experiment_type'] == 'TRIANGLE'].copy()
    
    if len(df_triangle) == 0:
        print("❌ No hay experimentos de tipo TRIANGLE")
        return None
    
    print(f"📊 Encontrados {len(df_triangle)} experimentos de fricción")
    print(df_triangle[['frecuencia_Hz', 'F_friction_dyn_N', 'stroke_mm', 'velocity_mean_mms']].to_string())
    
    # Procesar cada archivo
    resultados = []
    
    for i, row in df_triangle.iterrows():
        freq = row['frecuencia_Hz']
        # Buscar archivo correspondiente
        archivos = [f for f in os.listdir(DATOS_DIR) if base_name in f and f"_{freq:.0f}Hz.csv" in f and 'resumen' not in f]
        
        if not archivos:
            # Intentar con decimal
            archivos = [f for f in os.listdir(DATOS_DIR) if base_name in f and f"exp{i+1}" in f and 'resumen' not in f]
        
        if archivos:
            archivo_path = os.path.join(DATOS_DIR, archivos[0])
            print(f"\n📁 Procesando: {archivos[0]}")
            
            datos = procesar_archivo_friccion(archivo_path)
            datos['freq'] = freq
            datos['archivo'] = archivos[0]
            resultados.append(datos)
    
    return resultados

def graficar_friccion(resultados, base_name):
    """Genera gráficas de análisis de fricción"""
    
    n_exp = len(resultados)
    
    # Figura 1: Lazos de histéresis
    fig1, axes1 = plt.subplots(2, min(3, n_exp), figsize=(14, 8))
    if n_exp == 1:
        axes1 = np.array([[axes1[0]], [axes1[1]]])
    axes1 = axes1.flatten() if n_exp > 1 else axes1
    
    for i, datos in enumerate(resultados[:6]):
        if n_exp > 1:
            ax = axes1[i] if i < len(axes1) else None
        else:
            ax = axes1[0]
        
        if ax is None:
            continue
            
        ax.plot(datos['displacement_mm'], datos['force_N'], 'b-', alpha=0.7, linewidth=0.5)
        ax.set_xlabel('Desplazamiento (mm)')
        ax.set_ylabel('Fuerza (N)')
        ax.set_title(f"f = {datos['freq']:.1f} Hz\nF_fric = {datos['F_friction_peak']:.2f} N")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Marcar F_forward y F_backward
        if datos['F_forward'] != 0 or datos['F_backward'] != 0:
            ax.axhline(y=datos['F_forward'], color='g', linestyle=':', alpha=0.7, label=f'F_fwd={datos["F_forward"]:.2f}')
            ax.axhline(y=datos['F_backward'], color='r', linestyle=':', alpha=0.7, label=f'F_bwd={datos["F_backward"]:.2f}')
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    fig1.savefig(os.path.join(DATOS_DIR, f'{base_name}_histeresis.png'), dpi=150)
    print(f"\n📊 Guardado: {base_name}_histeresis.png")
    
    # Figura 2: Señales en tiempo
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Usar el primer experimento para mostrar señales
    datos = resultados[0]
    t = datos['t']
    
    axes2[0].plot(t, datos['force_N'], 'b-', linewidth=0.5)
    axes2[0].set_ylabel('Fuerza (N)')
    axes2[0].set_title(f"Señales en tiempo - {datos['freq']:.1f} Hz")
    axes2[0].grid(True, alpha=0.3)
    
    axes2[1].plot(t, datos['displacement_mm'], 'g-', linewidth=0.5)
    axes2[1].set_ylabel('Desplazamiento (mm)')
    axes2[1].grid(True, alpha=0.3)
    
    axes2[2].plot(t, datos['velocity_mms'], 'r-', linewidth=0.5)
    axes2[2].set_ylabel('Velocidad (mm/s)')
    axes2[2].set_xlabel('Tiempo (s)')
    axes2[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(DATOS_DIR, f'{base_name}_tiempo.png'), dpi=150)
    print(f"📊 Guardado: {base_name}_tiempo.png")
    
    # Figura 3: Fricción vs Velocidad
    if len(resultados) > 1:
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        
        freqs = [d['freq'] for d in resultados]
        F_peak = [d['F_friction_peak'] for d in resultados]
        F_zc = [d['F_friction_zc'] for d in resultados]
        v_mean = [d['v_mean'] for d in resultados]
        
        ax3.plot(v_mean, F_peak, 'bo-', markersize=10, label='Método Pico-Pico')
        if any(f != 0 for f in F_zc):
            ax3.plot(v_mean, F_zc, 'rs--', markersize=10, label='Método Zero-Crossing')
        
        for i, f in enumerate(freqs):
            ax3.annotate(f'{f:.1f}Hz', (v_mean[i], F_peak[i]), textcoords="offset points", 
                        xytext=(5,5), fontsize=9)
        
        ax3.set_xlabel('Velocidad media (mm/s)')
        ax3.set_ylabel('Fuerza de fricción (N)')
        ax3.set_title('Caracterización de Fricción vs Velocidad')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig3.savefig(os.path.join(DATOS_DIR, f'{base_name}_friccion_vs_vel.png'), dpi=150)
        print(f"📊 Guardado: {base_name}_friccion_vs_vel.png")
    
    plt.show()

def main():
    print("="*60)
    print("🔬 ANÁLISIS DE FRICCIÓN (ONDA TRIANGULAR)")
    print("="*60)
    
    # Buscar el archivo de resumen más reciente
    archivos = [f for f in os.listdir(DATOS_DIR) if f.endswith('_resumen.csv')]
    archivos.sort(reverse=True)
    
    if not archivos:
        print("❌ No se encontraron archivos de resumen")
        return
    
    base_name = archivos[0].replace('_resumen.csv', '')
    print(f"\n📁 Analizando: {base_name}")
    
    # Analizar sesión
    resultados = analizar_sesion_friccion(base_name)
    
    if resultados is None or len(resultados) == 0:
        print("❌ No se pudieron procesar los datos")
        return
    
    # Mostrar resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DE FRICCIÓN")
    print("="*60)
    
    for datos in resultados:
        print(f"\n🔹 Frecuencia: {datos['freq']:.1f} Hz")
        print(f"   F_fricción (P2P): {datos['F_friction_peak']:.4f} N")
        print(f"   F_fricción (ZC):  {datos['F_friction_zc']:.4f} N")
        print(f"   F_forward:        {datos['F_forward']:.4f} N")
        print(f"   F_backward:       {datos['F_backward']:.4f} N")
        print(f"   Carrera:          {datos['stroke']:.3f} mm")
        print(f"   Velocidad media:  {datos['v_mean']:.3f} mm/s")
        print(f"   Energía disipada: {datos['energy_mJ']:.3f} mJ")
    
    # Graficar
    graficar_friccion(resultados, base_name)
    
    # Guardar resumen procesado
    resumen_path = os.path.join(DATOS_DIR, f'{base_name}_friccion_analisis.csv')
    df_out = pd.DataFrame([{
        'frecuencia_Hz': d['freq'],
        'F_friction_peak_N': d['F_friction_peak'],
        'F_friction_zc_N': d['F_friction_zc'],
        'F_forward_N': d['F_forward'],
        'F_backward_N': d['F_backward'],
        'stroke_mm': d['stroke'],
        'velocity_mean_mms': d['v_mean'],
        'energy_mJ': d['energy_mJ']
    } for d in resultados])
    df_out.to_csv(resumen_path, index=False)
    print(f"\n💾 Análisis guardado en: {resumen_path}")

if __name__ == "__main__":
    main()
