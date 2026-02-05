import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt, resample
import os
import glob

def cargar_datos_mas_recientes():
    """Carga los archivos de datos más recientes de fuerza y vibración"""
    # Encuentra el archivo de fuerza más reciente
    pattern_fuerza = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "datos_automaticos", "*_fuerza.csv")
    archivos_fuerza = glob.glob(pattern_fuerza)
    archivo_fuerza_reciente = max(archivos_fuerza, key=os.path.getmtime) if archivos_fuerza else None
    
    # Encuentra el archivo de vibración más reciente
    pattern_vib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            "datos_automaticos", "*_vibracion.csv")
    archivos_vib = glob.glob(pattern_vib)
    archivo_vib_reciente = max(archivos_vib, key=os.path.getmtime) if archivos_vib else None
    
    print(f"[CARGAR_DATOS] Archivo de fuerza seleccionado: {os.path.basename(archivo_fuerza_reciente) if archivo_fuerza_reciente else 'NINGUNO'}", flush=True)
    print(f"[CARGAR_DATOS] Archivo de vibración seleccionado: {os.path.basename(archivo_vib_reciente) if archivo_vib_reciente else 'NINGUNO'}", flush=True)
    
    return archivo_fuerza_reciente, archivo_vib_reciente

def filtro_pasa_banda(datos, fs, lowcut=1.0, highcut=20.0, order=4):
    """Aplica filtro pasa banda a los datos"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, datos)

def calcular_aceleracion_absoluta(accel_x, accel_y):
    """Calcula la aceleración absoluta a partir de los componentes X e Y"""
    return np.sqrt(accel_x**2 + accel_y**2)

def preprocesar_datos(archivo_fuerza=None, archivo_vib=None, aplicar_filtro=True):
    print("[PREPROCESAR] Iniciando preprocesar_datos()", flush=True)
    """
    Preprocesa los datos de fuerza y vibración para el modelo Bouc-Wen
    
    Args:
        archivo_fuerza: Ruta al archivo CSV de datos de fuerza
        archivo_vib: Ruta al archivo CSV de datos de vibración
        aplicar_filtro: Si True, aplica filtro pasa banda a los datos
        
    Returns:
        t_comun: Vector de tiempo común para ambos tipos de datos
        aceleracion_absoluta_proc: Aceleración absoluta procesada
        fuerza_proc: Señal de fuerza procesada
    """
    # Si no se proporcionan archivos, cargar los más recientes
    if archivo_fuerza is None or archivo_vib is None:
        print("[PREPROCESAR] Llamando a cargar_datos_mas_recientes()", flush=True)
        archivo_fuerza, archivo_vib = cargar_datos_mas_recientes()
        print(f"[PREPROCESAR] cargar_datos_mas_recientes() devolvió: fuerza='{archivo_fuerza}', vib='{archivo_vib}'", flush=True)
    
    # Cargar datos de fuerza
    print(f"[PREPROCESAR] Intentando cargar datos de fuerza desde {archivo_fuerza}...", flush=True)
    try:
        datos_fuerza = pd.read_csv(archivo_fuerza)
        print(f"[PREPROCESAR] pd.read_csv para fuerza completado. Shape: {datos_fuerza.shape if 'datos_fuerza' in locals() and hasattr(datos_fuerza, 'shape') else 'N/A'}", flush=True)
    except FileNotFoundError:
        # Imprimir mensaje de error y verificar si el archivo existe
        print(f"ERROR: No se pudo encontrar el archivo {archivo_fuerza}")
        if os.path.exists(archivo_fuerza):
            print(f"El archivo existe pero no se puede leer")
        else:
            print(f"El archivo NO existe en la ubicación especificada")
            # Buscar el archivo en diferentes ubicaciones posibles
            for ruta_base in ["", ".", "..", "../datos", "datos", "Pruebas/src/datos_automaticos"]:
                ruta_prueba = os.path.join(ruta_base, os.path.basename(archivo_fuerza))
                if os.path.exists(ruta_prueba):
                    print(f"¡Archivo encontrado en {ruta_prueba}!")
                    archivo_fuerza = ruta_prueba
                    datos_fuerza = pd.read_csv(archivo_fuerza)
                    break
            else:
                raise FileNotFoundError(f"No se pudo encontrar el archivo {archivo_fuerza} en ninguna ubicación")
    tiempo_fuerza = datos_fuerza.iloc[:, 0].values
    canales_fuerza = []
    for i in range(1, 5):  # Asumiendo 4 canales de fuerza (ai0-ai3)
        if i < len(datos_fuerza.columns):
            canales_fuerza.append(datos_fuerza.iloc[:, i].values)
        else:
            canales_fuerza.append(np.zeros_like(tiempo_fuerza))
            
    # Cargar datos de vibración
    print(f"[PREPROCESAR] Intentando cargar datos de vibración desde {archivo_vib}...", flush=True)
    datos_vib = pd.read_csv(archivo_vib)
    print(f"[PREPROCESAR] pd.read_csv para vibración completado. Shape: {datos_vib.shape if 'datos_vib' in locals() and hasattr(datos_vib, 'shape') else 'N/A'}", flush=True)
    tiempo_vib = datos_vib.iloc[:, 0].values
    
    # Extraer canales de aceleración (X e Y)
    accel_x = datos_vib.iloc[:, 1].values if len(datos_vib.columns) > 1 else np.zeros_like(tiempo_vib)
    accel_y = datos_vib.iloc[:, 2].values if len(datos_vib.columns) > 2 else np.zeros_like(tiempo_vib)
    
    # Calcular aceleración absoluta
    print("Calculando aceleración absoluta...")
    aceleracion_absoluta = calcular_aceleracion_absoluta(accel_x, accel_y)
    
    # Aplicar filtro pasa banda si se solicita
    if aplicar_filtro:
        print("Aplicando filtro pasa banda a los datos...")
        # Frecuencias de muestreo (Hz)
        fs_fuerza = 2500  # Actualizada a 2500 Hz
        fs_vib = 2500     # Según el código de la interfaz
        
        # Filtrar canales de fuerza
        for i in range(len(canales_fuerza)):
            canales_fuerza[i] = filtro_pasa_banda(canales_fuerza[i], fs_fuerza)
        
        # Filtrar aceleración absoluta
        aceleracion_absoluta = filtro_pasa_banda(aceleracion_absoluta, fs_vib)
    
    # Sincronizar datos (resampling si es necesario)
    print("Sincronizando datos de fuerza y aceleración...")
    # Crear un vector de tiempo común si tienen diferentes tamaños o tasas de muestreo
    t_comun = np.linspace(0, max(tiempo_fuerza[-1], tiempo_vib[-1]), min(len(tiempo_fuerza), len(tiempo_vib)))
    
    # Remuestrear datos de fuerza a la duración común si es necesario
    if len(tiempo_fuerza) != len(t_comun):
        canales_fuerza_proc = []
        for canal in canales_fuerza:
            canal_resampled = resample(canal, len(t_comun))
            canales_fuerza_proc.append(canal_resampled)
    else:
        canales_fuerza_proc = canales_fuerza
    
    # Remuestrear aceleración absoluta a la duración común si es necesario
    if len(tiempo_vib) != len(t_comun):
        aceleracion_absoluta_proc = resample(aceleracion_absoluta, len(t_comun))
    else:
        aceleracion_absoluta_proc = aceleracion_absoluta
        
    # Para la demostración, usaremos ai0 como señal de fuerza de salida
    # En un caso real, el usuario debería seleccionar el canal apropiado
    fuerza_proc = canales_fuerza_proc[0]
        
    print(f"[PREPROCESAR] Preprocesamiento completado. Longitud de los datos procesados: {len(t_comun)}", flush=True)
    
    return t_comun, aceleracion_absoluta_proc, fuerza_proc, canales_fuerza_proc

def visualizar_datos(t, aceleracion, fuerza, canales_fuerza=None, titulo="Datos preprocesados"):
    import matplotlib.pyplot as plt
    """
    Visualiza los datos preprocesados
    
    Args:
        t: Vector de tiempo
        aceleracion: Vector de aceleración absoluta
        fuerza: Vector de fuerza
        canales_fuerza: Lista de canales de fuerza (opcional)
        titulo: Título del gráfico
    """
    plt.figure(figsize=(12, 10))
    
    # Gráfico 1: Aceleración absoluta vs tiempo
    plt.subplot(3, 1, 1)
    plt.plot(t, aceleracion, 'b-')
    plt.title(f"{titulo} - Aceleración Absoluta")
    plt.ylabel('Aceleración (g)')
    plt.grid(True)
    
    # Gráfico 2: Fuerza vs tiempo
    plt.subplot(3, 1, 2)
    plt.plot(t, fuerza, 'r-')
    plt.title("Fuerza (Canal ai0)")
    plt.ylabel('Fuerza (mV)')
    plt.grid(True)
    
    # Gráfico 3: Relación fuerza-aceleración (histéresis)
    plt.subplot(3, 1, 3)
    plt.plot(aceleracion, fuerza, 'g.', alpha=0.5, markersize=1)
    plt.title("Relación Fuerza-Aceleración (Histéresis)")
    plt.xlabel('Aceleración (g)')
    plt.ylabel('Fuerza (mV)')
    plt.grid(True)
    
    # Si se proporcionan todos los canales de fuerza, mostrar un gráfico adicional
    if canales_fuerza is not None and len(canales_fuerza) >= 4:
        plt.figure(figsize=(12, 8))
        labels = ['ai0', 'ai1', 'ai2', 'ai3']
        for i, canal in enumerate(canales_fuerza[:4]):
            plt.plot(t, canal, label=labels[i])
        plt.title("Todos los canales de fuerza")
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Fuerza (mV)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def guardar_datos_procesados(t, aceleracion, fuerza):
    """Guarda los datos preprocesados en un archivo CSV"""
    directorio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos_procesados")
    os.makedirs(directorio, exist_ok=True)
    
    archivo = os.path.join(directorio, f"datos_procesados_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    # Crear DataFrame y guardar
    df = pd.DataFrame({
        'Tiempo': t,
        'Aceleracion_Absoluta': aceleracion,
        'Fuerza': fuerza
    })
    
    df.to_csv(archivo, index=False)
    print(f"Datos procesados guardados en: {archivo}")
    return archivo

if __name__ == "__main__":
    # Ejecutar preprocesamiento
    t, aceleracion, fuerza, canales_fuerza = preprocesar_datos(aplicar_filtro=True)
    
    # Visualizar datos
    visualizar_datos(t, aceleracion, fuerza, canales_fuerza)
    
    # Guardar datos procesados
    archivo_guardado = guardar_datos_procesados(t, aceleracion, fuerza)
    
    print(f"\nResumen del preprocesamiento:")
    print(f"- Número de muestras: {len(t)}")
    print(f"- Duración total: {t[-1]:.2f} segundos")
    print(f"- Frecuencia de muestreo efectiva: {len(t)/t[-1]:.2f} Hz")
    print(f"- Rango de aceleración: [{min(aceleracion):.2f}, {max(aceleracion):.2f}] g")
    print(f"- Rango de fuerza: [{min(fuerza):.2f}, {max(fuerza):.2f}] mV")
    print(f"\nDatos guardados en: {archivo_guardado}")
    print("\nLos datos están listos para ser utilizados en el modelo Bouc-Wen.")
