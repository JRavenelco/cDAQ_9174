import os
import pickle
import glob
import pprint

def mostrar_estructura_pkl():
    """
    Muestra la estructura completa del archivo PKL más reciente
    """
    # Buscar en varios directorios posibles
    posibles_directorios = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos"),
        os.path.join(os.path.dirname(os.path.abspath(__file__))),
    ]
    
    archivos_encontrados = []
    for directorio in posibles_directorios:
        if os.path.exists(directorio):
            archivos_pkl = glob.glob(os.path.join(directorio, "*.pkl"))
            for archivo in archivos_pkl:
                archivos_encontrados.append(archivo)
    
    if not archivos_encontrados:
        print("ERROR: No se encontraron archivos .pkl en ninguno de los directorios conocidos.")
        return
    
    # Ordenar por fecha de modificación (el más reciente primero)
    archivos_encontrados.sort(key=os.path.getmtime, reverse=True)
    
    # Mostrar los 5 archivos más recientes
    print(f"\nArchivos .pkl encontrados (mostrando los {min(5, len(archivos_encontrados))} más recientes):")
    for i, archivo in enumerate(archivos_encontrados[:5]):
        fecha_mod = os.path.getmtime(archivo)
        fecha_str = f"{fecha_mod}"
        print(f"{i+1}. {os.path.basename(archivo)} - {fecha_str}")
        
    archivo_pkl = archivos_encontrados[0]  # Usar el más reciente
    print(f"\nSeleccionado: {os.path.basename(archivo_pkl)}")
    
    print(f"\n{'=' * 80}")
    print(f"ESTRUCTURA DEL ARCHIVO - {os.path.basename(archivo_pkl)}")
    print(f"{'=' * 80}\n")
    
    # Cargar resultados
    try:
        with open(archivo_pkl, 'rb') as f:
            resultados = pickle.load(f)
        print(f"Archivo cargado correctamente\n")
        
        # Verificar el tipo de resultados
        print(f"Tipo de datos: {type(resultados)}\n")
        
        # Si es un diccionario, mostrar sus claves
        if isinstance(resultados, dict):
            print("Claves en el nivel superior:")
            print("-" * 40)
            for clave in resultados.keys():
                valor = resultados[clave]
                tipo = type(valor).__name__
                if isinstance(valor, (list, tuple, set)) and len(valor) > 0:
                    tamaño = len(valor)
                    tipo_elemento = type(valor[0]).__name__
                    print(f"- {clave}: {tipo}[{tamaño}] de {tipo_elemento}")
                else:
                    print(f"- {clave}: {tipo}")
            
            print("\nEstructura anidada (primeros niveles):")
            print("-" * 40)
            pprint.pprint(resultados, depth=2)
        else:
            print("El contenido no es un diccionario. Mostrando información completa:")
            pprint.pprint(resultados)
    
    except Exception as e:
        print(f"ERROR al cargar o analizar el archivo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    mostrar_estructura_pkl()
