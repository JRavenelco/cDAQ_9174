"""
Análisis de Barrido de Frecuencia para Identificación de Parámetros
====================================================================
Extrae masa efectiva, rigidez y amortiguamiento del sistema
a partir de datos de fuerza y aceleración en barrido senoidal.

Modelo: m*a + c*v + k*x = F
FRF: H(ω) = F/a = m - k/ω² + j*c/ω

Autor: Cascade AI
Fecha: 2025-12-03
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
import os

# Directorio de datos
DATOS_DIR = r"c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\Pruebas\src\caracterizacion_fuerza"
SAMPLE_RATE = 2500  # Hz

def cargar_resumen(archivo_resumen):
    """Carga el archivo de resumen del barrido"""
    df = pd.read_csv(archivo_resumen)
    print(f"📊 Cargados {len(df)} experimentos")
    print(df[['frecuencia_Hz', 'force_rms_N', 'accel1_rms_g', 'fase_deg']].to_string())
    return df

def calcular_frf_desde_archivos(base_name, frecuencias):
    """
    Calcula la FRF (Función de Respuesta en Frecuencia) desde los archivos de datos crudos.
    Retorna magnitud y fase de H(ω) = F(ω) / a(ω)
    """
    resultados = []
    
    for i, freq in enumerate(frecuencias):
        # Buscar archivo correspondiente
        archivo = os.path.join(DATOS_DIR, f"{base_name}_exp{i+1}_{int(freq)}Hz.csv")
        
        if not os.path.exists(archivo):
            print(f"⚠️ No encontrado: {archivo}")
            continue
            
        df = pd.read_csv(archivo)
        
        # Extraer señales
        t = df['tiempo_s'].values
        fuerza_V = df['fuerza_V'].values
        accel_g = df['aceleracion_sensor_g'].values  # Usar acelerómetro sobre sensor
        
        # Remover DC
        fuerza_V = fuerza_V - np.mean(fuerza_V)
        accel_g = accel_g - np.mean(accel_g)
        
        # Convertir a unidades físicas
        # Fuerza: V -> N (usando parámetros del sensor)
        CELDA_CAPACIDAD_KG = 500.0
        CELDA_SENSIBILIDAD_MV_V = 1.7
        INA849_GANANCIA = 601.0
        V_EXCITACION = 10.0
        
        sensibilidad_total = (CELDA_SENSIBILIDAD_MV_V / 1000) * V_EXCITACION * INA849_GANANCIA
        fuerza_N = (fuerza_V / sensibilidad_total) * CELDA_CAPACIDAD_KG * 9.81
        
        # Aceleración: g -> m/s²
        accel_ms2 = accel_g * 9.81
        
        # FFT
        n = len(t)
        freqs = fftfreq(n, 1/SAMPLE_RATE)[:n//2]
        
        fft_fuerza = fft(fuerza_N)[:n//2]
        fft_accel = fft(accel_ms2)[:n//2]
        
        # Encontrar pico en la frecuencia de excitación
        idx_freq = np.argmin(np.abs(freqs - freq))
        
        # Buscar el pico real cerca de la frecuencia nominal
        search_range = max(1, int(2 * n / SAMPLE_RATE))  # ±2 Hz
        idx_start = max(1, idx_freq - search_range)
        idx_end = min(len(freqs)-1, idx_freq + search_range)
        
        idx_peak = idx_start + np.argmax(np.abs(fft_fuerza[idx_start:idx_end]))
        freq_real = freqs[idx_peak]
        
        # FRF: H(ω) = F(ω) / a(ω)
        H = fft_fuerza[idx_peak] / fft_accel[idx_peak] if np.abs(fft_accel[idx_peak]) > 1e-10 else 0
        
        magnitud = np.abs(H)
        fase = np.angle(H, deg=True)
        
        # Amplitudes RMS
        fuerza_rms = np.sqrt(np.mean(fuerza_N**2))
        accel_rms = np.sqrt(np.mean(accel_ms2**2))
        
        resultados.append({
            'freq_nominal': freq,
            'freq_real': freq_real,
            'H_mag': magnitud,
            'H_fase': fase,
            'H_real': np.real(H),
            'H_imag': np.imag(H),
            'fuerza_rms': fuerza_rms,
            'accel_rms': accel_rms,
            'omega': 2 * np.pi * freq_real
        })
        
        print(f"  {freq:.0f} Hz: |H| = {magnitud:.4f} N/(m/s²), fase = {fase:.1f}°")
    
    return pd.DataFrame(resultados)

def identificar_masa_simple(df_frf):
    """
    Identificación simple de masa efectiva.
    A alta frecuencia: H(ω) ≈ m (la rigidez y amortiguamiento son despreciables)
    """
    # Usar frecuencias altas (>50 Hz) donde H ≈ m
    df_alta = df_frf[df_frf['freq_real'] > 50]
    
    if len(df_alta) > 0:
        masa_estimada = np.mean(df_alta['H_mag'])
        std_masa = np.std(df_alta['H_mag'])
        print(f"\n📐 MASA EFECTIVA (método alta frecuencia):")
        print(f"   m = {masa_estimada:.4f} ± {std_masa:.4f} kg")
        return masa_estimada
    else:
        print("⚠️ No hay suficientes datos a alta frecuencia")
        return None

def modelo_frf(omega, m, c, k):
    """
    Modelo de FRF para sistema masa-resorte-amortiguador
    H(ω) = F/a = m + k/ω² - j*c/ω  (en dominio de Laplace con s=jω)
    
    Pero físicamente: m*a + c*v + k*x = F
    En frecuencia: (-m*ω² + j*c*ω + k) * X = F
    Y como a = -ω²*X: X = -a/ω²
    
    Entonces: F/a = m - k/ω² + j*c/ω
    |H|² = (m - k/ω²)² + (c/ω)²
    """
    H_real = m - k / (omega**2)
    H_imag = c / omega
    return np.sqrt(H_real**2 + H_imag**2)

def identificar_parametros_completo(df_frf):
    """
    Identificación completa de m, c, k usando ajuste de curva.
    """
    if len(df_frf) < 3:
        print("⚠️ No hay suficientes puntos para ajuste completo")
        return None
        
    omega = df_frf['omega'].values
    H_mag = df_frf['H_mag'].values
    
    # Valores iniciales basados en los datos
    m0 = np.median(H_mag)  # Estimación inicial de masa
    k0 = 1e5  # Rigidez inicial
    c0 = 50   # Amortiguamiento inicial
    
    print(f"\n   Valores iniciales: m0={m0:.2f}, c0={c0:.2f}, k0={k0:.2f}")
    
    try:
        # Límites más amplios
        popt, pcov = curve_fit(modelo_frf, omega, H_mag, p0=[m0, c0, k0], 
                               bounds=([0.001, 0, 0], [100, 10000, 1e8]),
                               maxfev=50000)
        m, c, k = popt
        perr = np.sqrt(np.diag(pcov))
        
        print(f"\n🔧 PARÁMETROS IDENTIFICADOS (ajuste completo):")
        print(f"   Masa efectiva:    m = {m:.4f} ± {perr[0]:.4f} kg")
        print(f"   Amortiguamiento:  c = {c:.4f} ± {perr[1]:.4f} N·s/m")
        print(f"   Rigidez:          k = {k:.2f} ± {perr[2]:.2f} N/m")
        
        # Frecuencia natural y factor de amortiguamiento
        if m > 0:
            omega_n = np.sqrt(k / m)
            f_n = omega_n / (2 * np.pi)
            zeta = c / (2 * np.sqrt(k * m)) if k > 0 else 0
            
            print(f"\n   Frecuencia natural: f_n = {f_n:.2f} Hz")
            print(f"   Factor de amortiguamiento: ζ = {zeta:.4f}")
            
            return {'m': m, 'c': c, 'k': k, 'f_n': f_n, 'zeta': zeta}
        
    except Exception as e:
        print(f"⚠️ Error en ajuste: {e}")
        
        # Método alternativo: solo estimar masa
        print("\n📐 Usando método simplificado (solo masa):")
        m_simple = np.median(H_mag)
        print(f"   Masa efectiva (mediana): m ≈ {m_simple:.4f} kg")
        return {'m': m_simple, 'c': 0, 'k': 0, 'f_n': 0, 'zeta': 0}
    
    return None

def graficar_frf(df_frf, params=None):
    """Grafica la FRF medida y el modelo ajustado"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    freq = df_frf['freq_real'].values
    omega = df_frf['omega'].values
    
    # 1. Magnitud de FRF
    ax1 = axes[0, 0]
    ax1.semilogy(freq, df_frf['H_mag'], 'bo-', markersize=8, label='Medido')
    if params:
        freq_modelo = np.linspace(5, 100, 200)
        omega_modelo = 2 * np.pi * freq_modelo
        H_modelo = modelo_frf(omega_modelo, params['m'], params['c'], params['k'])
        ax1.semilogy(freq_modelo, H_modelo, 'r-', linewidth=2, label='Modelo ajustado')
    ax1.set_xlabel('Frecuencia (Hz)')
    ax1.set_ylabel('|H(ω)| = |F/a| (N/(m/s²) = kg)')
    ax1.set_title('Magnitud de FRF')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Fase de FRF
    ax2 = axes[0, 1]
    ax2.plot(freq, df_frf['H_fase'], 'go-', markersize=8)
    ax2.set_xlabel('Frecuencia (Hz)')
    ax2.set_ylabel('Fase (°)')
    ax2.set_title('Fase de FRF')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 3. Parte Real e Imaginaria
    ax3 = axes[1, 0]
    ax3.plot(freq, df_frf['H_real'], 'b^-', markersize=8, label='Re(H)')
    ax3.plot(freq, df_frf['H_imag'], 'rs-', markersize=8, label='Im(H)')
    ax3.set_xlabel('Frecuencia (Hz)')
    ax3.set_ylabel('H (N/(m/s²))')
    ax3.set_title('Parte Real e Imaginaria')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 4. Diagrama de Nyquist
    ax4 = axes[1, 1]
    ax4.plot(df_frf['H_real'], df_frf['H_imag'], 'mo-', markersize=8)
    for i, f in enumerate(freq):
        ax4.annotate(f'{f:.0f}Hz', (df_frf['H_real'].iloc[i], df_frf['H_imag'].iloc[i]),
                    fontsize=8, alpha=0.7)
    ax4.set_xlabel('Re(H)')
    ax4.set_ylabel('Im(H)')
    ax4.set_title('Diagrama de Nyquist')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(DATOS_DIR, 'analisis_frf.png'), dpi=150)
    plt.show()
    
    print(f"\n📊 Gráfica guardada en: {os.path.join(DATOS_DIR, 'analisis_frf.png')}")

def main():
    print("="*60)
    print("🔬 ANÁLISIS DE BARRIDO DE FRECUENCIA")
    print("="*60)
    
    # Buscar el archivo de resumen más reciente
    archivos = [f for f in os.listdir(DATOS_DIR) if f.endswith('_resumen.csv')]
    archivos.sort(reverse=True)
    
    if not archivos:
        print("❌ No se encontraron archivos de resumen")
        return
    
    archivo_resumen = os.path.join(DATOS_DIR, archivos[0])
    print(f"\n📁 Usando: {archivos[0]}")
    
    # Cargar resumen
    df_resumen = cargar_resumen(archivo_resumen)
    
    # Filtrar solo experimentos SWEEP
    df_sweep = df_resumen[df_resumen['experiment_type'] == 'SWEEP'].copy()
    
    if len(df_sweep) == 0:
        print("❌ No hay experimentos de tipo SWEEP")
        return
    
    # Extraer base_name del archivo
    base_name = archivos[0].replace('_resumen.csv', '')
    
    # Calcular FRF desde archivos crudos
    print("\n📈 Calculando FRF desde datos crudos...")
    frecuencias = df_sweep['frecuencia_Hz'].values
    df_frf = calcular_frf_desde_archivos(base_name, frecuencias)
    
    if len(df_frf) < 3:
        print("❌ No hay suficientes puntos para el análisis")
        return
    
    # Filtrar datos problemáticos
    # - Aceleración muy baja (< 0.1 m/s²) = sin movimiento
    # - Aceleración muy alta (> 100 m/s²) = offset/error de sensor
    # - FRF muy alta (> 100 kg) = probablemente error
    df_frf_filtrado = df_frf[
        (df_frf['accel_rms'] > 0.1) & 
        (df_frf['accel_rms'] < 100) &
        (df_frf['H_mag'] < 100)
    ].copy()
    
    print(f"\n📊 Puntos válidos: {len(df_frf_filtrado)} de {len(df_frf)}")
    
    if len(df_frf_filtrado) < 3:
        print("⚠️ Pocos puntos válidos. Mostrando todos los datos:")
        print(df_frf[['freq_real', 'H_mag', 'accel_rms', 'fuerza_rms']].to_string())
    
    # Identificar masa simple
    masa = identificar_masa_simple(df_frf_filtrado)
    
    # Identificar parámetros completos
    params = identificar_parametros_completo(df_frf_filtrado)
    
    # Graficar
    graficar_frf(df_frf_filtrado, params)
    
    # Guardar resultados
    if params:
        resultado_path = os.path.join(DATOS_DIR, f'{base_name}_parametros.txt')
        with open(resultado_path, 'w', encoding='utf-8') as f:
            f.write("PARAMETROS IDENTIFICADOS DEL SISTEMA\n")
            f.write("="*40 + "\n")
            f.write(f"Masa efectiva:       m = {params['m']:.4f} kg\n")
            f.write(f"Amortiguamiento:     c = {params['c']:.4f} N*s/m\n")
            f.write(f"Rigidez:             k = {params['k']:.2f} N/m\n")
            f.write(f"Frecuencia natural:  f_n = {params['f_n']:.2f} Hz\n")
            f.write(f"Factor amort.:       zeta = {params['zeta']:.4f}\n")
        print(f"\n💾 Parámetros guardados en: {resultado_path}")

if __name__ == "__main__":
    main()
