import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import glob
import pprint  # Para mostrar estructuras de datos complejas en la terminal
import sys

def analizar_histeresis_terminal(archivo_pkl=None):
    """
    Analiza un archivo .pkl de resultados de histéresis y muestra las estadísticas en la terminal
    
    Args:
        archivo_pkl: Ruta al archivo .pkl de resultados. Si es None, busca el más reciente.
    """
    # Si no se especifica archivo, buscar el más reciente
    if archivo_pkl is None:
        print("Buscando archivos .pkl disponibles...")
        # Buscar en directorio de resultados
        directorio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
        if not os.path.exists(directorio):
            print("ERROR: No se encontró el directorio de resultados.")
            return
        
        archivos_pkl = glob.glob(os.path.join(directorio, "*.pkl"))
        if not archivos_pkl:
            print("ERROR: No se encontraron archivos .pkl en el directorio de resultados.")
            return
        
        archivos_pkl.sort(key=os.path.getmtime, reverse=True)
        archivo_pkl = archivos_pkl[0]
        print(f"Seleccionado el archivo más reciente: {os.path.basename(archivo_pkl)}")
    
    print(f"\n{'=' * 80}")
    print(f"ANÁLISIS DE HISTÉRESIS - {os.path.basename(archivo_pkl)}")
    print(f"{'=' * 80}\n")
    
    # Cargar resultados
    try:
        with open(archivo_pkl, 'rb') as f:
            resultados = pickle.load(f)
        print(f"Archivo de resultados cargado correctamente")
    except Exception as e:
        print(f"ERROR al cargar el archivo: {e}")
        return
    
    # Verificar estructura de datos
    if not isinstance(resultados, dict):
        print(f"AVISO: Los resultados no tienen el formato esperado. Tipo: {type(resultados)}")
        return
    
    # Extraer datos principales
    datos = resultados.get('Datos', {})
    
    if not datos:
        print("ERROR: No se encontró la clave 'Datos' en el archivo")
        print("Claves disponibles:", list(resultados.keys()))
        return
    
    # Obtener arrays con manejo de NaN
    tiempo = np.array(datos.get('tiempo', []))
    aceleracion = np.array(datos.get('aceleracion', []))
    fuerza = np.array(datos.get('fuerza', []))
    z_histerica = np.array(datos.get('z_histerica', []))
    fuerza_estimada = np.array(datos.get('fuerza_estimada', []))
    
    # Manejar valores NaN
    tiempo = tiempo[~np.isnan(tiempo)] if tiempo.size > 0 else tiempo
    aceleracion = aceleracion[~np.isnan(aceleracion)] if aceleracion.size > 0 else aceleracion
    fuerza = fuerza[~np.isnan(fuerza)] if fuerza.size > 0 else fuerza
    z_histerica = z_histerica[~np.isnan(z_histerica)] if z_histerica.size > 0 else z_histerica
    fuerza_estimada = fuerza_estimada[~np.isnan(fuerza_estimada)] if fuerza_estimada.size > 0 else fuerza_estimada
    
    # Estadísticas básicas
    print(f"ESTADÍSTICAS BÁSICAS:")
    print(f"- Duración del experimento: {resultados.get('Duracion', 'N/A')} segundos")
    print(f"- Tamaño de datos: {len(tiempo)} puntos")
    print(f"- Frecuencia de muestreo: {1/np.mean(np.diff(tiempo)) if len(tiempo) > 1 else 'N/A'} Hz")
    
    # Métricas de histéresis
    print(f"\nMÉTRICAS DE HISTÉRESIS:")
    print(f"- Área de histéresis experimental: {resultados.get('Area_Histeresis_Exp', 'N/A')}")
    print(f"- Área de histéresis del modelo: {resultados.get('Area_Histeresis_Modelo', 'N/A')}")
    print(f"- Energía disipada experimental: {resultados.get('Energia_Disipada_Exp', 'N/A')}")
    print(f"- Energía disipada del modelo: {resultados.get('Energia_Disipada_Modelo', 'N/A')}")
    print(f"- Error RMS: {resultados.get('Error_RMS', 'N/A')}")
    
    # Análisis de calidad de datos
    print(f"\nCALIDAD DE DATOS:")
    print(f"- Porcentaje de NaN en fuerza: {np.mean(np.isnan(fuerza)) * 100:.2f}%")
    print(f"- Porcentaje de NaN en aceleración: {np.mean(np.isnan(aceleracion)) * 100:.2f}%")
    print(f"- Rango de fuerza: [{np.nanmin(fuerza) if fuerza.size > 0 else 'N/A'}, {np.nanmax(fuerza) if fuerza.size > 0 else 'N/A'}] N")
    
    # Recomendaciones basadas en los datos
    print(f"\nRECOMENDACIONES:")
    if np.any(np.isnan(fuerza)) or np.any(np.isnan(aceleracion)):
        print("- Se detectaron valores NaN en los datos. Verifique la calidad de la adquisición.")
    if 'Area_Histeresis_Exp' not in resultados or np.isnan(resultados['Area_Histeresis_Exp']):
        print("- No se pudo calcular el área de histéresis experimental. Revise los datos de fuerza y desplazamiento.")
    if 'Error_RMS' not in resultados or np.isnan(resultados['Error_RMS']):
        print("- No se pudo calcular el error RMS. Verifique los modelos de histéresis.")
    
    print(f"\n{'=' * 80}")
    print(f"ANÁLISIS COMPLETADO")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    # Ejecutar el análisis con el archivo más reciente
    try:
        analizar_histeresis_terminal()
    except Exception as e:
        print(f"\nERROR NO MANEJADO: {e}")
        import traceback
        traceback.print_exc()
        print("\nIntente con otro archivo o revise la estructura de los datos guardados.")
        print("Si el error persiste, contacte al desarrollador.")
