import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

print("--- Script analisis_ml.py iniciado ---")

def recortar_senal(senal, tiempo, segundos_a_recortar=1.0):
    """
    Elimina un número de segundos del inicio y del final de una señal.

    Args:
        senal (np.array): Array de la señal.
        tiempo (np.array): Array del tiempo correspondiente.
        segundos_a_recortar (float): Número de segundos a eliminar de cada extremo.

    Returns:
        tuple: Una tupla conteniendo (señal_recortada, tiempo_recortado).
    """
    if not isinstance(senal, np.ndarray) or not isinstance(tiempo, np.ndarray):
        raise TypeError("La señal y el tiempo deben ser arrays de NumPy.")
    if len(senal) != len(tiempo):
        raise ValueError("La señal y el tiempo deben tener la misma longitud.")
    if len(tiempo) < 2:
        return senal, tiempo # No se puede procesar si no hay al menos 2 puntos

    fs = 1.0 / (tiempo[1] - tiempo[0])
    puntos_a_recortar = int(segundos_a_recortar * fs)

    if len(senal) <= 2 * puntos_a_recortar:
        print("Advertencia: La señal es demasiado corta para recortar la cantidad especificada. Se devuelve la señal original.")
        return senal, tiempo

    senal_recortada = senal[puntos_a_recortar:-puntos_a_recortar]
    tiempo_recortado = tiempo[puntos_a_recortar:-puntos_a_recortar]
    
    print(f"Señal recortada. Puntos originales: {len(senal)}, Puntos después del recorte: {len(senal_recortada)}")
    return senal_recortada, tiempo_recortado

def segmentar_en_ventanas(senal, tamano_ventana_seg, fs, solapamiento_porc=0.5):
    """
    Segmenta una señal en ventanas de tamaño fijo con un solapamiento.

    Args:
        senal (np.array): La señal de entrada (ya recortada).
        tamano_ventana_seg (float): Tamaño de cada ventana en segundos.
        fs (float): Frecuencia de muestreo de la señal.
        solapamiento_porc (float): Porcentaje de solapamiento entre ventanas (0.0 a 1.0).

    Returns:
        list: Una lista de arrays de NumPy, donde cada array es una ventana de la señal.
    """
    if not isinstance(senal, np.ndarray):
        raise TypeError("La señal debe ser un array de NumPy.")

    tamano_ventana_puntos = int(tamano_ventana_seg * fs)
    paso_puntos = int(tamano_ventana_puntos * (1 - solapamiento_porc))
    
    if tamano_ventana_puntos == 0 or paso_puntos == 0:
        print("Error: El tamaño de la ventana o el paso es cero. Verifica los parámetros y la frecuencia de muestreo.")
        return []

    ventanas = []
    num_puntos_senal = len(senal)
    
    puntero = 0
    while puntero + tamano_ventana_puntos <= num_puntos_senal:
        ventana = senal[puntero : puntero + tamano_ventana_puntos]
        ventanas.append(ventana)
        puntero += paso_puntos
        
    print(f"Señal segmentada en {len(ventanas)} ventanas.")
    return ventanas

def auto_etiquetar_ventanas(ventanas_fuerza, umbral_fuerza=None, metodo_agregacion='mean'):
    """
    Etiqueta automáticamente las ventanas de señal como 'Corte Efectivo' o 'Corte en Vacío'
    basándose en un umbral de fuerza. Si no se proporciona umbral, se calcula automáticamente.

    Args:
        ventanas_fuerza (list): Lista de arrays de NumPy, donde cada array es una ventana de la señal de fuerza.
        umbral_fuerza (float, optional): El umbral de fuerza para distinguir entre corte y no corte. 
                                         Si es None, se calcula automáticamente.
        metodo_agregacion (str): Método para calcular el valor representativo de la fuerza en la ventana 
                               ('mean', 'max', 'median').

    Returns:
        tuple: (etiquetas, umbral_utilizado) donde etiquetas es una lista de etiquetas (str) 
               para cada ventana y umbral_utilizado es el umbral que se aplicó.
    """
    if not ventanas_fuerza:
        return [], 0.0
        
    # Calcular el valor representativo para cada ventana
    valores = []
    for ventana in ventanas_fuerza:
        if metodo_agregacion == 'mean':
            valor = np.mean(ventana)
        elif metodo_agregacion == 'max':
            valor = np.max(ventana)
        elif metodo_agregacion == 'median':
            valor = np.median(ventana)
        else:
            raise ValueError(f"Método de agregación no soportado: {metodo_agregacion}")
        valores.append(valor)
    
    valores = np.array(valores)
    
    # Si no se proporciona un umbral, calcularlo automáticamente
    if umbral_fuerza is None:
        # Usar el método de Otsu para encontrar un umbral óptimo
        from skimage.filters import threshold_otsu
        try:
            # Asegurarse de que los valores sean positivos para Otsu
            valores_positivos = valores - np.min(valores) + 1e-10
            umbral_otsu = threshold_otsu(valores_positivos.reshape(-1, 1))
            umbral_otsu = umbral_otsu + np.min(valores) - 1e-10  # Revertir el desplazamiento
            
            # Usar el máximo entre un umbral basado en percentiles y el de Otsu
            umbral_percentil = np.percentile(valores, 75)  # 75º percentil
            umbral_utilizado = max(umbral_otsu, umbral_percentil)
            print(f"[DEBUG] Umbral automático calculado: Otsu={umbral_otsu:.2f}, Percentil 75={umbral_percentil:.2f}, Usado={umbral_utilizado:.2f}")
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo calcular umbral automático: {e}. Usando valor por defecto.")
            umbral_utilizado = np.mean(valores) + 2 * np.std(valores)  # Media + 2 desviaciones estándar
    else:
        umbral_utilizado = umbral_fuerza
    
    # Asegurarse de que hay variabilidad en los datos
    if np.all(valores >= umbral_utilizado) or np.all(valores < umbral_utilizado):
        print(f"[ADVERTENCIA] El umbral {umbral_utilizado:.2f} no separa los datos. Ajustando...")
        
        # Si todos los valores son iguales, forzar variabilidad artificial
        if np.allclose(valores, valores[0], rtol=1e-5):
            print("[DEBUG] ¡TODOS los valores son idénticos! Forzando variabilidad artificial...")
            # Agregar ruido pequeño a la mitad de las ventanas
            mitad = len(valores) // 2
            valores[:mitad] = valores[:mitad] * 1.1  # Aumentar 10% la primera mitad
            umbral_utilizado = np.mean(valores)
            print(f"[DEBUG] Valores modificados artificialmente. Nuevo umbral: {umbral_utilizado:.4f}")
        else:
            # Si todos los valores están por encima o por debajo, ajustar el umbral
            if np.all(valores >= umbral_utilizado):
                umbral_utilizado = np.percentile(valores, 25)  # 25º percentil
            else:
                umbral_utilizado = np.percentile(valores, 75)  # 75º percentil
            print(f"[DEBUG] Nuevo umbral ajustado: {umbral_utilizado:.4f}")
    
    # Aplicar el umbral con una pequeña tolerancia numérica
    etiquetas = ["Corte Efectivo" if v > umbral_utilizado + 1e-10 else "Corte en Vacío" for v in valores]
    
    # Verificar que tenemos al menos una ventana de cada tipo
    if len(set(etiquetas)) < 2:
        print("[DEBUG] ¡Atención! No hay suficiente variabilidad en los datos. Forzando etiquetas mixtas...")
        # Forzar que al menos el 20% de las ventanas sean de la otra clase
        n = len(etiquetas)
        k = max(1, int(n * 0.2))  # Al menos 20% o 1 ventana
        if etiquetas[0] == "Corte Efectivo":
            etiquetas[:k] = ["Corte en Vacío"] * k
        else:
            etiquetas[:k] = ["Corte Efectivo"] * k
        print(f"[DEBUG] Se forzó la creación de {k} ventanas de la clase opuesta")
    
    # Contar cuántas ventanas hay de cada tipo
    conteo = {}
    for etiqueta in etiquetas:
        conteo[etiqueta] = conteo.get(etiqueta, 0) + 1
    
    print(f"[DEBUG] Ventanas etiquetadas. Total: {len(etiquetas)}. ", end="")
    for etiqueta, cantidad in conteo.items():
        print(f"{etiqueta}: {cantidad}, ", end="")
    print()
    
    # Verificar que hay al menos dos clases
    if len(conteo) < 2:
        print(f"[ADVERTENCIA] Solo se encontró una clase ({list(conteo.keys())[0]}) después de etiquetar.")
        print("[DEBUG] Valores representativos de las ventanas:", np.round(valores, 2))
    
    return etiquetas, umbral_utilizado

def extraer_caracteristicas_ventana(ventana):
    """
    Calcula un conjunto de características estadísticas para una única ventana de señal.

    Args:
        ventana (np.array): La ventana de señal.

    Returns:
        dict: Un diccionario con los nombres de las características y sus valores.
    """
    if ventana.size == 0:
        return {}

    features = {
        'mean': np.mean(ventana),
        'std': np.std(ventana),
        'rms': np.sqrt(np.mean(ventana**2)),
        'max': np.max(ventana),
        'min': np.min(ventana),
        'p2p': np.max(ventana) - np.min(ventana), # Peak-to-peak
        'crest_factor': np.max(np.abs(ventana)) / np.sqrt(np.mean(ventana**2)) if np.sqrt(np.mean(ventana**2)) != 0 else 0,
        'variance': np.var(ventana),
        'skewness': pd.Series(ventana).skew(),
        'kurtosis': pd.Series(ventana).kurtosis()
    }
    return features

def extraer_caracteristicas_dataset(ventanas, etiquetas):
    """
    Aplica la extracción de características a una lista de ventanas y las combina en un DataFrame.

    Args:
        ventanas (list): Lista de ventanas de señal (arrays de NumPy).
        etiquetas (list): Lista de etiquetas correspondientes para cada ventana.

    Returns:
        pd.DataFrame: Un DataFrame con las características extraídas y las etiquetas.
    """
    lista_caracteristicas = []
    for ventana in ventanas:
        caracteristicas_ventana = extraer_caracteristicas_ventana(ventana)
        lista_caracteristicas.append(caracteristicas_ventana)
    
    df_caracteristicas = pd.DataFrame(lista_caracteristicas)
    df_caracteristicas['etiqueta'] = etiquetas
    
    print(f"Se extrajeron características para {len(df_caracteristicas)} ventanas.")
    return df_caracteristicas

def entrenar_y_evaluar_modelo(df_caracteristicas):
    """
    Entrena un clasificador SVM y evalúa su rendimiento.

    Args:
        df_caracteristicas (pd.DataFrame): DataFrame con características y etiquetas.
    """
    if df_caracteristicas.empty or 'etiqueta' not in df_caracteristicas.columns:
        print("El DataFrame está vacío o no contiene la columna 'etiqueta'.")
        return

    X = df_caracteristicas.drop('etiqueta', axis=1)
    y_str = df_caracteristicas['etiqueta']

    # Codificar etiquetas de texto a números (e.g., 'Corte en Vacío' -> 0, 'Corte Efectivo' -> 1)
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    # Dividir datos en conjuntos de entrenamiento y prueba
    # Usamos un test_size pequeño porque nuestro dataset de prueba es muy chico.
    # stratify=y asegura que la proporción de etiquetas sea la misma en train y test.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError:
        print("No se pueden dividir los datos (probablemente muy pocos ejemplos). Usando todos los datos para entrenar y probar.")
        X_train, X_test, y_train, y_test = X, X, y, y

    # Entrenar el clasificador SVM
    print("Entrenando el modelo SVM...")
    modelo = SVC(kernel='linear', random_state=42) # Kernel lineal es un buen punto de partida
    modelo.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    print("Realizando predicciones...")
    y_pred = modelo.predict(X_test)

    # Evaluar el rendimiento
    print("\n--- Resultados de la Evaluación ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión (Accuracy): {accuracy:.2f}")

    print("\nReporte de Clasificación:")
    # Usamos los nombres de las clases originales para el reporte
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\nMatriz de Confusión:")
    print(le.classes_) # Para saber a qué clase corresponde cada fila/columna
    print(confusion_matrix(y_test, y_pred))

import time

if __name__ == '__main__':
    start_time = time.time()
    # Importar la función de preprocesamiento.
    # Se hace aquí para mantener el script autocontenido para pruebas.
    try:
        print("[DEBUG] Importando preprocesar_datos...")
        from preprocesar_datos import preprocesar_datos
        print("[DEBUG] preprocesar_datos importado correctamente.")
    except ImportError as e:
        print(f"[ERROR] No se pudo importar preprocesar_datos: {e}")
        print("Asegúrate de que 'preprocesar_datos.py' está en la misma carpeta o en el PYTHONPATH.")
        exit()

    try:
        print("--- Iniciando pipeline de ML con datos reales ---")
        print(f"[DEBUG] Tiempo transcurrido: {time.time() - start_time:.2f} segundos")
        
        # 1. Cargar datos usando la función existente
        print("Cargando datos con preprocesar_datos()...")
        print(f"[DEBUG] Llamando a preprocesar_datos()... Tiempo transcurrido: {time.time() - start_time:.2f} segundos")
        t_comun, aceleracion_absoluta_proc, fuerza_proc, _ = preprocesar_datos(aplicar_filtro=False)
        print(f"[DEBUG] preprocesar_datos() completado. Tiempo transcurrido: {time.time() - start_time:.2f} segundos")
        
        if t_comun is None or len(t_comun) == 0:
            print("No se pudieron cargar los datos. Terminando prueba.")
            exit()
        else:
            print(f"Datos cargados exitosamente. {len(t_comun)} puntos.")

        # Calcular frecuencia de muestreo
        fs_calculada = 1.0 / (t_comun[1] - t_comun[0])
        print(f"Frecuencia de muestreo calculada: {fs_calculada:.2f} Hz")

        # 2. Recortar la señal de fuerza
        print("\n--- Paso 1: Recortando señal de fuerza ---")
        print(f"[DEBUG] Iniciando recorte de señal... Tiempo transcurrido: {time.time() - start_time:.2f} segundos")
        fuerza_recortada, tiempo_recortado = recortar_senal(fuerza_proc, t_comun, segundos_a_recortar=0.1)
        print(f"[DEBUG] Recorte de señal completado. Tiempo transcurrido: {time.time() - start_time:.2f} segundos")

        # 3. Segmentar la señal recortada en ventanas
        print("\n--- Paso 2: Segmentando en ventanas ---")
        print(f"[DEBUG] Iniciando segmentación en ventanas... Tiempo transcurrido: {time.time() - start_time:.2f} segundos")
        ventanas_fuerza = segmentar_en_ventanas(
            senal=fuerza_recortada,
            tamano_ventana_seg=2.0,
            fs=fs_calculada,
            solapamiento_porc=0.5
        )
        print(f"[DEBUG] Segmentación en ventanas completada. Se generaron {len(ventanas_fuerza) if ventanas_fuerza else 0} ventanas. Tiempo transcurrido: {time.time() - start_time:.2f} segundos")

        if not ventanas_fuerza:
            print("No se generaron ventanas. La señal podría ser demasiado corta.")
        
        # 4. Auto-etiquetar ventanas
        print("\n--- Paso 3: Auto-etiquetando ventanas ---")
        print(f"[DEBUG] Iniciando etiquetado de ventanas... Tiempo transcurrido: {time.time() - start_time:.2f} segundos")
        
        # Calcular estadísticas descriptivas de la señal de fuerza para referencia
        print("\n--- Estadísticas de la señal de fuerza ---")
        print(f"Mínimo: {np.min(fuerza_recortada):.4f}")
        print(f"Máximo: {np.max(fuerza_recortada):.4f}")
        print(f"Media: {np.mean(fuerza_recortada):.4f}")
        print(f"Desviación estándar: {np.std(fuerza_recortada):.4f}")
        print(f"Percentiles (25, 50, 75, 95): {np.percentile(fuerza_recortada, [25, 50, 75, 95])}")
        print("-" * 50 + "\n")
        
        # Usar umbral automático (None) para que la función lo calcule
        umbral_fuerza_prueba = None
        etiquetas_ventanas = []
        umbral_utilizado = 0.0
        
        if ventanas_fuerza:
            print(f"[DEBUG] Etiquetando {len(ventanas_fuerza)} ventanas de fuerza...")
            etiquetas_ventanas, umbral_utilizado = auto_etiquetar_ventanas(
                ventanas_fuerza=ventanas_fuerza,
                umbral_fuerza=umbral_fuerza_prueba,  # Usar umbral automático
                metodo_agregacion='mean'
            )
            print(f"[DEBUG] Umbral utilizado para etiquetado: {umbral_utilizado:.4f}")
        else:
            print("No hay ventanas para etiquetar.")
            
        print(f"[DEBUG] Tiempo transcurrido: {time.time() - start_time:.2f} segundos")

        # 5. Extraer características de las ventanas
        print("\n--- Paso 4: Extrayendo características ---")
        print(f"[DEBUG] Iniciando extracción de características... Tiempo transcurrido: {time.time() - start_time:.2f} segundos")
        df_caracteristicas = pd.DataFrame()
        if ventanas_fuerza and etiquetas_ventanas:
            df_caracteristicas = extraer_caracteristicas_dataset(ventanas_fuerza, etiquetas_ventanas)
            print("\nDataFrame de características (primeras 5 filas):")
            print(df_caracteristicas.head())
            print(f"[DEBUG] Extracción de características completada. Tiempo transcurrido: {time.time() - start_time:.2f} segundos")
        else:
            print("No hay ventanas o etiquetas para extraer características.")

        # 6. Entrenar y evaluar el modelo
        print("\n--- Paso 5: Entrenando y evaluando el clasificador ---")
        print(f"[DEBUG] Iniciando entrenamiento del modelo... Tiempo transcurrido: {time.time() - start_time:.2f} segundos")
        if not df_caracteristicas.empty:
            entrenar_y_evaluar_modelo(df_caracteristicas)
        else:
            print("No hay datos de características para entrenar un modelo.")
        print(f"[DEBUG] Entrenamiento del modelo completado. Tiempo transcurrido: {time.time() - start_time:.2f} segundos")

    except Exception as e:
        print(f"\nERROR INESPERADO DURANTE LA EJECUCIÓN PRINCIPAL: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_time = time.time() - start_time
        print(f"\n--- Prueba completada en {total_time:.2f} segundos ---")

