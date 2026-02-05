
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.integrate import cumulative_trapezoid
import os

# Configuración
DATA_DIR = r"c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\Pruebas\src\datos_automaticos"
OUTPUT_DIR = r"c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\Pruebas\src\caracterizacion_fuerza"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_FORCE = "sesion_20251009_150942_fuerza.csv"
FILE_VIB = "sesion_20251009_150942_vibracion.csv"

# Colores UAQ (Estética)
AZUL_UAQ = "#00274c"
ROJO_UAQ = "#B30000"
GRIS_UAQ = "#7a7a7a"

def cargar_y_procesar():
    print("Cargando datos...")
    path_f = os.path.join(DATA_DIR, FILE_FORCE)
    path_v = os.path.join(DATA_DIR, FILE_VIB)
    
    df_f = pd.read_csv(path_f)
    df_v = pd.read_csv(path_v)
    
    n = min(len(df_f), len(df_v))
    t = df_f['Tiempo(s)'].values[:n]
    fuerza = df_f['Fuerza_ai0(V)'].values[:n] 
    acel = df_v['Vibracion_ai0(g_or_V)'].values[:n]
    
    # Eliminar DC
    fuerza = fuerza - np.mean(fuerza)
    acel = acel - np.mean(acel)
    
    # Acel en g -> m/s^2 (aproximado, solo para forma)
    acel_m_s2 = acel * 9.81 
    
    # Integrar para velocidad y desplazamiento
    dt = t[1] - t[0]
    vel = cumulative_trapezoid(acel_m_s2, t, initial=0)
    
    # Filtro paso alto (drift removal)
    from scipy.signal import butter, filtfilt
    b, a = butter(2, 0.5, btype='highpass', fs=1/dt)
    vel = filtfilt(b, a, vel)
    
    pos = cumulative_trapezoid(vel, t, initial=0)
    pos = filtfilt(b, a, pos)
    
    return t, fuerza, pos, acel_m_s2, dt

def generar_fft_plot(t, fuerza, dt):
    print("Generando FFT...")
    N = len(fuerza)
    yf = fft(fuerza)
    xf = fftfreq(N, dt)[:N//2]
    
    plt.figure(figsize=(10, 6))
    # Usar escala logarítmica en Y para resaltar armónicos, o lineal si son muy fuertes
    # Aquí lineal para claridad básica
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]), color=AZUL_UAQ, lw=1.5)
    plt.title("Espectro de Frecuencia: Señal de Fuerza (No Calibrada)", fontsize=16, fontweight='bold', color=AZUL_UAQ)
    plt.xlabel("Frecuencia (Hz)", fontsize=14)
    plt.ylabel("Amplitud (Volts)", fontsize=14) # Honestidad: Volts
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 200) 
    
    # Estilo limpio
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Anotaciones
    plt.annotate('Armonicos Visibles\n(No-Linealidad)', xy=(60, 0.05), xytext=(80, 0.1),
                 arrowprops=dict(facecolor=ROJO_UAQ, shrink=0.05), fontsize=12, color=ROJO_UAQ, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fase2_fft_sucia.png"), dpi=120)
    plt.close()

def generar_histeresis_plot(pos, fuerza):
    print("Generando Histeresis...")
    start = len(pos) // 3
    end = start + 2000 
    
    # Datos a graficar
    x_data = pos[start:end]*1000 # mm estimado
    y_data = fuerza[start:end]   # Volts
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data, color=ROJO_UAQ, alpha=0.8, lw=2)
    
    plt.title("Lazo de Histeresis (Fase 2: Corte)", fontsize=16, fontweight='bold', color=ROJO_UAQ)
    plt.xlabel("Desplazamiento Estimado (mm)", fontsize=14)
    plt.ylabel("Fuerza Sensor (Volts)", fontsize=14) # Honestidad: Volts
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Relleno
    plt.fill(x_data, y_data, color=ROJO_UAQ, alpha=0.1)
    
    plt.text(0, np.max(y_data)*0.8, "Area > 0\nDisipacion de Energia", 
             horizontalalignment='center', color=AZUL_UAQ, fontweight='bold', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fase2_histeresis_real.png"), dpi=120)
    plt.close()

if __name__ == "__main__":
    try:
        t, f, x, a, dt = cargar_y_procesar()
        generar_fft_plot(t, f, dt)
        generar_histeresis_plot(x, f)
        print(" Imagenes generadas exitosamente en src/caracterizacion_fuerza/")
    except Exception as e:
        print(f" Error: {e}")
