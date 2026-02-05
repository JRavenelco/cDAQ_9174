import os
import pickle
import glob
import numpy as np  # Importar numpy para manejo de arrays

def analisis_simple():
    """
    Muestra un análisis simple de los archivos .pkl de resultados
    """
    # Buscar en directorio de resultados
    directorio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
    if not os.path.exists(directorio):
        print("ERROR: No se encontró el directorio de resultados.")
        return
    
    archivos_pkl = glob.glob(os.path.join(directorio, "*.pkl"))
    if not archivos_pkl:
        print("ERROR: No se encontraron archivos .pkl en el directorio de resultados.")
        return
    
    archivo_pkl = max(archivos_pkl, key=os.path.getmtime)
    print(f"\nAnalizando el archivo más reciente: {os.path.basename(archivo_pkl)}")
    
    # Cargar resultados
    try:
        with open(archivo_pkl, 'rb') as f:
            resultados = pickle.load(f)
    except Exception as e:
        print(f"ERROR al cargar el archivo: {e}")
        return
    
    # Mostrar estructura básica
    print("\nESTRUCTURA BÁSICA:")
    print("=" * 40)
    
    # Verificar si es un diccionario
    if isinstance(resultados, dict):
        print(f"Tipo: Diccionario con {len(resultados)} claves")
        print("Claves principales:", list(resultados.keys()))
        
        # Verificar clave 'Datos'
        if 'Datos' in resultados:
            datos = resultados['Datos']
            print("\nCONTENIDO DE 'Datos':")
            if isinstance(datos, dict):
                print(f"  - Tipo: Diccionario con {len(datos)} claves")
                print("  - Claves:", list(datos.keys()))
                
                # Mostrar tamaños de arrays
                print("\nTAMAÑOS DE ARRAYS:")
                for key, value in datos.items():
                    if isinstance(value, (list, np.ndarray)):
                        print(f"  - {key}: {len(value)} elementos")
            else:
                print(f"  - Tipo: {type(datos)}")
        else:
            print("ADVERTENCIA: No se encontró la clave 'Datos'")
    else:
        print(f"Tipo: {type(resultados)}")
    
    # Métricas disponibles
    print("\nMÉTRICAS DISPONIBLES:")
    print("=" * 40)
    metricas = ['Area_Histeresis_Exp', 'Area_Histeresis_Modelo', 
               'Energia_Disipada_Exp', 'Energia_Disipada_Modelo', 
               'Error_RMS', 'Frecuencia_Dominante']
    
    for metrica in metricas:
        valor = resultados.get(metrica, "No disponible")
        print(f"- {metrica}: {valor}")
    
    print("\nAnálisis simple completado")

if __name__ == "__main__":
    analisis_simple()
