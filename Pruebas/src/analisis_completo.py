"""
Análisis Completo: Barrido de Frecuencia + Fricción
====================================================
Analiza datos de barrido senoidal (10-90 Hz) y fricción triangular.

Autor: Cascade AI
Fecha: 2025-12-03
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

def voltaje_a_fuerza_N(voltaje):
    sensibilidad_total = (CELDA_SENSIBILIDAD_MV_V / 1000) * V_EXCITACION * INA849_GANANCIA
    return (voltaje / sensibilidad_total) * CELDA_CAPACIDAD_KG * 9.81

# ============================================
# ANÁLISIS DE BARRIDO DE FRECUENCIA
# ============================================

def calcular_frf_archivo(archivo_path, freq_nominal):
    """Calcula FRF desde un archivo de datos"""
    df = pd.read_csv(archivo_path)
    
    t = df['tiempo_s'].values
    fuerza_V = df['fuerza_V'].values
    accel_g = df['aceleracion_sensor_g'].values
    
    # Remover DC
    fuerza_V = fuerza_V - np.mean(fuerza_V)
    accel_g = accel_g - np.mean(accel_g)
    
    # Convertir unidades
    force_N = np.array([voltaje_a_fuerza_N(v) for v in fuerza_V])
    accel_ms2 = accel_g * 9.81
    
    # FFT
    n = len(t)
    freqs = fftfreq(n, 1/SAMPLE_RATE)[:n//2]
    
    fft_force = fft(force_N)[:n//2]
    fft_accel = fft(accel_ms2)[:n//2]
    
    # Encontrar pico cerca de frecuencia nominal
    idx_freq = np.argmin(np.abs(freqs - freq_nominal))
    search_range = max(1, int(3 * n / SAMPLE_RATE))
    idx_start = max(1, idx_freq - search_range)
    idx_end = min(len(freqs)-1, idx_freq + search_range)
    
    idx_peak = idx_start + np.argmax(np.abs(fft_force[idx_start:idx_end]))
    freq_real = freqs[idx_peak]
    
    # FRF: H(ω) = F(ω) / a(ω)
    if np.abs(fft_accel[idx_peak]) > 1e-10:
        H = fft_force[idx_peak] / fft_accel[idx_peak]
    else:
        H = 0
    
    return {
        'freq_nominal': freq_nominal,
        'freq_real': freq_real,
        'H_mag': np.abs(H),
        'H_fase': np.angle(H, deg=True),
        'H_real': np.real(H),
        'H_imag': np.imag(H),
        'force_rms': np.sqrt(np.mean(force_N**2)),
        'accel_rms': np.sqrt(np.mean(accel_ms2**2)),
        'omega': 2 * np.pi * freq_real
    }

def modelo_frf(omega, m, c, k):
    """Modelo FRF: H(ω) = F/a para sistema m-c-k"""
    H_real = m - k / (omega**2)
    H_imag = c / omega
    return np.sqrt(H_real**2 + H_imag**2)

def analizar_barrido(base_name):
    """Analiza barrido de frecuencia"""
    print("\n" + "="*60)
    print("📈 ANÁLISIS DE BARRIDO DE FRECUENCIA")
    print("="*60)
    
    resumen_path = os.path.join(DATOS_DIR, f"{base_name}_resumen.csv")
    df = pd.read_csv(resumen_path)
    df_sweep = df[df['experiment_type'] == 'SWEEP'].copy()
    
    if len(df_sweep) == 0:
        print("❌ No hay datos de barrido (SWEEP)")
        return None
    
    print(f"📊 {len(df_sweep)} puntos de frecuencia")
    
    # Calcular FRF para cada frecuencia
    resultados = []
    for i, row in df_sweep.iterrows():
        freq = row['frecuencia_Hz']
        
        # Buscar archivo
        patron = f"{base_name}_exp"
        archivos = [f for f in os.listdir(DATOS_DIR) 
                   if patron in f and f"_{int(freq)}Hz.csv" in f and 'resumen' not in f]
        
        if archivos:
            archivo_path = os.path.join(DATOS_DIR, archivos[0])
            frf = calcular_frf_archivo(archivo_path, freq)
            resultados.append(frf)
            print(f"  {freq:.0f} Hz: |H| = {frf['H_mag']:.2f} kg, fase = {frf['H_fase']:.1f}°")
    
    if len(resultados) < 3:
        print("❌ Pocos puntos válidos")
        return None
    
    df_frf = pd.DataFrame(resultados)
    
    # Filtrar datos problemáticos
    df_frf = df_frf[(df_frf['accel_rms'] > 0.05) & (df_frf['H_mag'] < 200)].copy()
    
    # Identificar parámetros
    omega = df_frf['omega'].values
    H_mag = df_frf['H_mag'].values
    
    print(f"\n📊 Puntos válidos para ajuste: {len(df_frf)}")
    
    # Estimar masa (promedio a alta frecuencia)
    mask_alta = df_frf['freq_real'] > 40
    if mask_alta.sum() > 0:
        m_alta = np.mean(df_frf.loc[mask_alta, 'H_mag'])
        print(f"\n📐 Masa efectiva (f > 40 Hz): {m_alta:.2f} kg")
    
    # Ajuste completo
    try:
        m0 = np.median(H_mag)
        popt, pcov = curve_fit(modelo_frf, omega, H_mag, 
                              p0=[m0, 100, 1e5],
                              bounds=([0.1, 0, 0], [200, 10000, 1e8]),
                              maxfev=50000)
        m, c, k = popt
        
        omega_n = np.sqrt(k / m) if m > 0 and k > 0 else 0
        f_n = omega_n / (2 * np.pi)
        zeta = c / (2 * np.sqrt(k * m)) if k > 0 and m > 0 else 0
        
        print(f"\n🔧 PARÁMETROS IDENTIFICADOS:")
        print(f"   Masa efectiva:    m = {m:.2f} kg")
        print(f"   Amortiguamiento:  c = {c:.2f} N·s/m")
        print(f"   Rigidez:          k = {k:.0f} N/m")
        print(f"   Frecuencia nat.:  f_n = {f_n:.1f} Hz")
        print(f"   Factor amort.:    zeta = {zeta:.4f}")
        
        params = {'m': m, 'c': c, 'k': k, 'f_n': f_n, 'zeta': zeta}
    except Exception as e:
        print(f"⚠️ Error en ajuste: {e}")
        params = {'m': np.median(H_mag), 'c': 0, 'k': 0, 'f_n': 0, 'zeta': 0}
    
    return df_frf, params

# ============================================
# ANÁLISIS DE FRICCIÓN
# ============================================

def procesar_friccion(archivo_path):
    """Procesa archivo de fricción"""
    df = pd.read_csv(archivo_path)
    
    t = df['tiempo_s'].values
    fuerza_V = df['fuerza_V'].values - np.mean(df['fuerza_V'].values)
    accel_g = df['aceleracion_sensor_g'].values - np.mean(df['aceleracion_sensor_g'].values)
    
    dt = 1.0 / SAMPLE_RATE
    
    # Filtro pasa-altos
    try:
        b, a = signal.butter(2, 0.2 / (SAMPLE_RATE / 2), btype='high')
        accel_filtered = signal.filtfilt(b, a, accel_g)
    except:
        accel_filtered = accel_g
    
    # Integración doble
    accel_ms2 = accel_filtered * 9.81
    velocity = signal.detrend(cumulative_trapezoid(accel_ms2, dx=dt, initial=0))
    displacement = signal.detrend(cumulative_trapezoid(velocity, dx=dt, initial=0))
    
    velocity_mms = velocity * 1000
    displacement_mm = displacement * 1000
    
    force_N = np.array([voltaje_a_fuerza_N(v) for v in fuerza_V])
    force_N = force_N - np.mean(force_N)
    
    # Métricas
    F_friction_peak = (np.max(force_N) - np.min(force_N)) / 2
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
            F_friction_zc = (np.mean(force_N[fwd]) - np.mean(force_N[bwd])) / 2
    
    return {
        't': t,
        'force_N': force_N,
        'displacement_mm': displacement_mm,
        'velocity_mms': velocity_mms,
        'F_friction_peak': F_friction_peak,
        'F_friction_zc': F_friction_zc,
        'stroke': stroke,
        'v_mean': v_mean
    }

def analizar_friccion(base_name):
    """Analiza datos de fricción"""
    print("\n" + "="*60)
    print("🧩 ANÁLISIS DE FRICCIÓN (TRIANGULAR)")
    print("="*60)
    
    resumen_path = os.path.join(DATOS_DIR, f"{base_name}_resumen.csv")
    df = pd.read_csv(resumen_path)
    df_tri = df[df['experiment_type'] == 'TRIANGLE'].copy()
    
    if len(df_tri) == 0:
        print("❌ No hay datos de fricción (TRIANGLE)")
        return None
    
    print(f"📊 {len(df_tri)} experimentos de fricción")
    
    resultados = []
    for i, row in df_tri.iterrows():
        freq = row['frecuencia_Hz']
        
        # Buscar archivo
        archivos = [f for f in os.listdir(DATOS_DIR) 
                   if base_name in f and f"exp" in f and 'resumen' not in f]
        
        # Filtrar por índice
        idx = df_tri.index.get_loc(i) + 1
        archivo_match = [f for f in archivos if f"_exp{idx}_" in f]
        
        if archivo_match:
            archivo_path = os.path.join(DATOS_DIR, archivo_match[0])
            datos = procesar_friccion(archivo_path)
            datos['freq'] = freq
            datos['archivo'] = archivo_match[0]
            resultados.append(datos)
            
            print(f"  {freq:.1f} Hz: F_fric = {datos['F_friction_peak']:.3f} N, "
                  f"stroke = {datos['stroke']:.2f} mm, v = {datos['v_mean']:.2f} mm/s")
    
    return resultados

# ============================================
# GRÁFICAS
# ============================================

def graficar_todo(df_frf, params_frf, resultados_friccion, base_name):
    """Genera todas las gráficas"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. FRF Magnitud
    ax1 = fig.add_subplot(2, 3, 1)
    if df_frf is not None:
        ax1.semilogy(df_frf['freq_real'], df_frf['H_mag'], 'bo-', markersize=8, label='Medido')
        if params_frf and params_frf['k'] > 0:
            f_modelo = np.linspace(5, 100, 200)
            w_modelo = 2 * np.pi * f_modelo
            H_modelo = modelo_frf(w_modelo, params_frf['m'], params_frf['c'], params_frf['k'])
            ax1.semilogy(f_modelo, H_modelo, 'r-', linewidth=2, label='Modelo')
        ax1.set_xlabel('Frecuencia (Hz)')
        ax1.set_ylabel('|H| = |F/a| (kg)')
        ax1.set_title('FRF - Magnitud')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. FRF Fase
    ax2 = fig.add_subplot(2, 3, 2)
    if df_frf is not None:
        ax2.plot(df_frf['freq_real'], df_frf['H_fase'], 'go-', markersize=8)
        ax2.set_xlabel('Frecuencia (Hz)')
        ax2.set_ylabel('Fase (°)')
        ax2.set_title('FRF - Fase')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 3. Nyquist
    ax3 = fig.add_subplot(2, 3, 3)
    if df_frf is not None:
        ax3.plot(df_frf['H_real'], df_frf['H_imag'], 'mo-', markersize=8)
        for _, row in df_frf.iterrows():
            ax3.annotate(f"{row['freq_real']:.0f}", (row['H_real'], row['H_imag']), fontsize=7)
        ax3.set_xlabel('Re(H)')
        ax3.set_ylabel('Im(H)')
        ax3.set_title('Diagrama de Nyquist')
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
    
    # 4. Lazos de histéresis
    ax4 = fig.add_subplot(2, 3, 4)
    if resultados_friccion:
        colors = plt.cm.viridis(np.linspace(0, 1, len(resultados_friccion)))
        for datos, color in zip(resultados_friccion, colors):
            ax4.plot(datos['displacement_mm'], datos['force_N'], 
                    color=color, alpha=0.7, linewidth=0.5, 
                    label=f"{datos['freq']:.1f} Hz")
        ax4.set_xlabel('Desplazamiento (mm)')
        ax4.set_ylabel('Fuerza (N)')
        ax4.set_title('Lazos de Histéresis')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
    
    # 5. Fricción vs Velocidad
    ax5 = fig.add_subplot(2, 3, 5)
    if resultados_friccion:
        v_mean = [d['v_mean'] for d in resultados_friccion]
        F_peak = [d['F_friction_peak'] for d in resultados_friccion]
        F_zc = [d['F_friction_zc'] for d in resultados_friccion]
        freqs = [d['freq'] for d in resultados_friccion]
        
        ax5.plot(v_mean, F_peak, 'bo-', markersize=10, label='Pico-Pico')
        if any(f != 0 for f in F_zc):
            ax5.plot(v_mean, F_zc, 'rs--', markersize=10, label='Zero-Crossing')
        
        for i, f in enumerate(freqs):
            ax5.annotate(f'{f:.1f}Hz', (v_mean[i], F_peak[i]), fontsize=8)
        
        ax5.set_xlabel('Velocidad media (mm/s)')
        ax5.set_ylabel('Fuerza de fricción (N)')
        ax5.set_title('Fricción vs Velocidad')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Resumen de parámetros
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    texto = "PARÁMETROS IDENTIFICADOS\n" + "="*30 + "\n\n"
    
    if params_frf:
        texto += "📈 BARRIDO DE FRECUENCIA:\n"
        texto += f"   Masa efectiva: {params_frf['m']:.2f} kg\n"
        texto += f"   Rigidez: {params_frf['k']:.0f} N/m\n"
        texto += f"   Amortiguamiento: {params_frf['c']:.2f} N·s/m\n"
        texto += f"   Freq. natural: {params_frf['f_n']:.1f} Hz\n"
        texto += f"   Factor amort.: {params_frf['zeta']:.4f}\n\n"
    
    if resultados_friccion:
        texto += "🧩 FRICCIÓN:\n"
        F_mean = np.mean([d['F_friction_peak'] for d in resultados_friccion])
        texto += f"   F_fricción promedio: {F_mean:.3f} N\n"
        texto += f"   Rango velocidad: {min([d['v_mean'] for d in resultados_friccion]):.2f} - "
        texto += f"{max([d['v_mean'] for d in resultados_friccion]):.2f} mm/s\n"
    
    ax6.text(0.1, 0.9, texto, transform=ax6.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Guardar
    fig_path = os.path.join(DATOS_DIR, f'{base_name}_analisis_completo.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Gráfica guardada: {fig_path}")
    
    plt.show()

def main():
    print("="*60)
    print("🔬 ANÁLISIS COMPLETO DE CARACTERIZACIÓN")
    print("="*60)
    
    # Buscar archivos
    archivos = [f for f in os.listdir(DATOS_DIR) if f.endswith('_resumen.csv')]
    archivos.sort(reverse=True)
    
    # Analizar el más reciente con datos de barrido
    for archivo in archivos:
        base_name = archivo.replace('_resumen.csv', '')
        df = pd.read_csv(os.path.join(DATOS_DIR, archivo))
        
        n_sweep = len(df[df['experiment_type'] == 'SWEEP'])
        n_tri = len(df[df['experiment_type'] == 'TRIANGLE'])
        
        print(f"\n📁 {archivo}")
        print(f"   SWEEP: {n_sweep}, TRIANGLE: {n_tri}")
        
        if n_sweep >= 5:  # Usar este para barrido
            print(f"\n✅ Usando {base_name} para análisis de barrido")
            df_frf, params_frf = analizar_barrido(base_name)
            break
    else:
        df_frf, params_frf = None, None
    
    # Buscar datos de fricción más recientes
    resultados_friccion = None
    for archivo in archivos:
        base_name_fri = archivo.replace('_resumen.csv', '')
        df = pd.read_csv(os.path.join(DATOS_DIR, archivo))
        
        if len(df[df['experiment_type'] == 'TRIANGLE']) >= 2:
            print(f"\n✅ Usando {base_name_fri} para análisis de fricción")
            resultados_friccion = analizar_friccion(base_name_fri)
            break
    
    # Graficar todo
    if df_frf is not None or resultados_friccion:
        graficar_todo(df_frf, params_frf, resultados_friccion, 
                     base_name if df_frf is not None else base_name_fri)
    
    # Guardar resumen
    resumen_path = os.path.join(DATOS_DIR, 'resumen_parametros.txt')
    with open(resumen_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE CARACTERIZACIÓN\n")
        f.write("="*40 + "\n\n")
        
        if params_frf:
            f.write("BARRIDO DE FRECUENCIA:\n")
            f.write(f"  Masa efectiva: {params_frf['m']:.2f} kg\n")
            f.write(f"  Rigidez: {params_frf['k']:.0f} N/m\n")
            f.write(f"  Amortiguamiento: {params_frf['c']:.2f} N*s/m\n")
            f.write(f"  Freq. natural: {params_frf['f_n']:.1f} Hz\n\n")
        
        if resultados_friccion:
            f.write("FRICCIÓN:\n")
            for d in resultados_friccion:
                f.write(f"  {d['freq']:.1f} Hz: F={d['F_friction_peak']:.3f} N, "
                       f"v={d['v_mean']:.2f} mm/s\n")
    
    print(f"\n💾 Resumen guardado: {resumen_path}")

if __name__ == "__main__":
    main()
