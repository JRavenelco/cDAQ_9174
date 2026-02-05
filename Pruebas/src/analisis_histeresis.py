"""
Módulo para análisis de histéresis usando modelos de Dahl y Bouc-Wen.
Este módulo complementa la funcionalidad existente en interfaz_DAQ_V2.py
para el análisis de datos de histéresis.

Uso principal:
    python analisis_histeresis.py --modo [individual|doe|manual|analizar] [opciones]

Ejemplos:
    python analisis_histeresis.py --modo individual --fuerza data.csv --vib vib.csv
    python analisis_histeresis.py --modo individual --fuerza data.csv --vib vib.csv --intentar-reparar --umbral-nan 0.7
    python analisis_histeresis.py --modo manual --intentar-reparar
    python analisis_histeresis.py --help

Opciones de manejo de datos NaN:
    --umbral-nan VALOR     Umbral máximo permitido de valores NaN (0.0 a 1.0, default: 0.5)
    --intentar-reparar     Activar interpolación para reparar valores NaN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Para integración se usa numpy.trapz ya que cumtrapz no está disponible en SciPy 1.15.2
# No necesitamos imports adicionales de scipy.integrate ya que no usamos cumtrapz
from scipy.optimize import least_squares
import os
import glob
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Importar funciones útiles del preprocesamiento
import preprocesar_datos as pp

def modelo_dahl(aceleracion, params, dt=0.0004):
    """
    Implementa el modelo de Dahl para calcular la variable z histérica.
    
    El modelo de Dahl es un modelo simple de histéresis que describe la relación
    entre la entrada (aceleración absoluta) y la variable histérica z.
    Esta implementación incluye mejoras de estabilidad numérica y restricciones.
    
    Args:
        aceleracion: Vector de aceleración absoluta (entrada)
        params: Parámetros del modelo [sigma_0, sigma_1, ...]
            - sigma_0: Rigidez inicial (N/m)
            - sigma_1: Parámetro de forma (adimensional)
            - z_max: (opcional) Valor máximo permitido para z (para prevenir inestabilidad)
        dt: Intervalo de tiempo entre muestras (por defecto 0.0004s para 2500Hz)
            
    Returns:
        z: Variable histérica calculada
    """
    # Verificar la entrada para NaN o inf
    if np.any(np.isnan(aceleracion)) or np.any(np.isinf(aceleracion)):
        logging.warning("Detectados valores NaN o Inf en la señal de aceleración")
        # Reemplazar NaN/Inf con valores interpolados o ceros
        aceleracion = np.nan_to_num(aceleracion, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Extraer parámetros con valores seguros por defecto
    sigma_0 = params[0] if len(params) > 0 else 1.0  # Rigidez inicial
    sigma_1 = params[1] if len(params) > 1 else 0.5  # Parámetro de forma (aumentado para mayor estabilidad)
    z_max = params[2] if len(params) > 2 else 100.0  # Límite superior de z para prevenir desbordamiento
    
    # Asegurar que sigma_1 no sea demasiado pequeño (evita divisiones inestables)
    sigma_1 = max(sigma_1, 0.1)
    
    # Normalizar la aceleración para mejorar estabilidad numérica
    # Esto ayuda a mantener los valores dentro de un rango manejable
    accel_max = np.max(np.abs(aceleracion)) if len(aceleracion) > 0 else 1.0
    if accel_max > 1e-10:  # Evitar división por cero o valores muy pequeños
        accel_normalized = aceleracion / accel_max
    else:
        accel_normalized = aceleracion
        
    # Inicializar la variable histérica z
    n = len(aceleracion)
    z = np.zeros(n)
    
    # Implementación más robusta usando método de punto medio para integración
    for i in range(1, n):
        # Calcular el cambio de la entrada
        dx = accel_normalized[i] - accel_normalized[i-1]
        
        if abs(dx) < 1e-10:  # Si el cambio es insignificante, no hay cambio en z
            z[i] = z[i-1]
            continue
        
        # Calcular la tasa de cambio usando el modelo de Dahl
        # Método de punto medio para mayor estabilidad
        z_mid = z[i-1] + 0.5 * dx * sigma_0 * (1 - (np.sign(dx) * z[i-1]) / sigma_1)
        
        # Limitar z_mid para evitar valores extremos
        z_mid = np.clip(z_mid, -z_max, z_max)
        
        # Segundo paso usando el valor intermedio
        dz = sigma_0 * (1 - (np.sign(dx) * z_mid) / sigma_1) * dx
        
        # Aplicar la actualización con límites
        z[i] = np.clip(z[i-1] + dz, -z_max, z_max)
    
    # Desnormalizar si fue necesario
    if accel_max > 1e-10:
        # La desnormalización debe ser proporcional a la escala de la entrada
        z = z * accel_max
    
    return z

def calcular_z_histerica(aceleracion, tiempo, params_dahl=None):
    """
    Calcula la variable z histérica utilizando el modelo de Dahl.
    
    Args:
        aceleracion: Vector de aceleración absoluta
        tiempo: Vector de tiempo correspondiente
        params_dahl: Parámetros del modelo de Dahl, si es None utiliza valores por defecto
            
    Returns:
        z_histerica: Variable histérica z
    """
    if params_dahl is None:
        # Valores por defecto para parámetros de Dahl
        params_dahl = [1.0, 0.1]  # [sigma_0, sigma_1]
    
    # Calcular dt promedio (intervalo de tiempo)
    dt = np.mean(np.diff(tiempo)) if len(tiempo) > 1 else 0.0004  # Por defecto 0.0004s (2500Hz)
    
    # Aplicar modelo de Dahl
    z_histerica = modelo_dahl(aceleracion, params_dahl, dt)
    
    return z_histerica

def estimar_fuerza_dahl(z_histerica, aceleracion, params, dt=0.0004, max_fuerza=10000.0):
    """
    Estima la fuerza utilizando el modelo de Dahl y la variable z histérica.
    Esta implementación incluye verificación de errores y restricciones para mejorar
    la estabilidad numérica y evitar valores extremos.
    
    Args:
        z_histerica: Variable histérica z calculada
        aceleracion: Vector de aceleración absoluta (entrada)
        params: Parámetros adicionales [k, c, ...]
            - k: Rigidez (N/m)
            - c: Coeficiente de amortiguamiento (N·s/m)
        dt: Intervalo de tiempo entre muestras (s)
        max_fuerza: Valor máximo permitido para la fuerza estimada (N)
            
    Returns:
        fuerza_estimada: Vector de fuerza estimada
    """
    # Verificar si las entradas tienen la misma longitud
    if len(z_histerica) != len(aceleracion):
        logging.warning(f"Longitudes inconsistentes en estimar_fuerza_dahl: z={len(z_histerica)} vs. accel={len(aceleracion)}")
        # Usar la longitud más corta para evitar errores
        n = min(len(z_histerica), len(aceleracion))
        z_h = z_histerica[:n]
        accel = aceleracion[:n]
    else:
        n = len(z_histerica)
        z_h = z_histerica
        accel = aceleracion
    
    if n == 0:
        logging.warning("Arrays vacíos en estimar_fuerza_dahl")
        return np.array([])  # Devolver array vacío
    
    # Verificar valores NaN o infinitos en las entradas
    if np.any(np.isnan(z_h)) or np.any(np.isinf(z_h)):
        logging.warning("Valores NaN o infinitos detectados en z_histerica")
        # Reemplazar con ceros para evitar propagación de NaN
        z_h = np.nan_to_num(z_h, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.any(np.isnan(accel)) or np.any(np.isinf(accel)):
        logging.warning("Valores NaN o infinitos detectados en aceleracion")
        accel = np.nan_to_num(accel, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Extraer parámetros con valores razonables por defecto
    k = params[0] if len(params) > 0 else 1000.0  # Rigidez (N/m)
    c = params[1] if len(params) > 1 else 10.0    # Amortiguamiento (N·s/m)
    
    # Limitar parámetros a rangos razonables para prevenir inestabilidad
    # Estos límites son estimados y pueden ajustarse según las necesidades específicas
    k = np.clip(k, 1.0, 1e6)  # Limitar k entre 1 y 1e6 N/m
    c = np.clip(c, 0.0, 1e4)   # Limitar c entre 0 y 1e4 N·s/m
    
    # Asegurar que dt sea positivo y razonable
    dt = max(dt, 1e-6)  # Mínimo 1 microsegundo
    
    # Definir un máximo absoluto para la fuerza
    # Esto previene valores extremos que pueden afectar las visualizaciones
    # y cálculos posteriores
    max_fuerza = abs(max_fuerza)  # Asegurarse que sea positivo
    
    # Estimar fuerza utilizando un modelo lineal con término de histéresis mejorado
    # F = k * z + c * dx/dt con restricciones de magnitud
    fuerza_estimada = np.zeros(n)
    
    try:
        # Calcular la fuerza según modelo
        for i in range(n):
            # Término de rigidez con restricción
            term_k = k * z_h[i]
            
            # Término de amortiguamiento (derivada de la aceleración)
            if i > 0:
                # Cálculo más robusto de la derivada
                dx_dt = (accel[i] - accel[i-1]) / dt
                
                # Limitar la derivada a valores razonables
                if abs(dx_dt) > 1e6:  # Limitar cambios extremos
                    dx_dt = np.sign(dx_dt) * 1e6
                
                term_c = c * dx_dt
            else:
                term_c = 0.0
            
            # Fuerza total con restricción de magnitud
            fuerza_raw = term_k + term_c
            fuerza_estimada[i] = np.clip(fuerza_raw, -max_fuerza, max_fuerza)
        
        # Verificación final para detectar valores no válidos que puedan haber escapado
        invalid_mask = np.isnan(fuerza_estimada) | np.isinf(fuerza_estimada)
        invalid_count = np.sum(invalid_mask)
        
        if invalid_count > 0:
            logging.warning(f"{invalid_count} valores no válidos en fuerza estimada. Corrigiendo...")
            fuerza_estimada[invalid_mask] = 0.0
            
    except Exception as e:
        logging.error(f"Error en estimar_fuerza_dahl: {e}")
        # En caso de error, devolver vector de ceros
        fuerza_estimada = np.zeros(n)
    
    return fuerza_estimada

def calcular_area_histeresis(x, y):
    """
    Calcula el área del ciclo de histéresis con manejo robusto de datos.
    
    Esta implementación mejora el cálculo para manejar datos ruidosos,
    con NaN o con formas irregulares de los ciclos de histéresis.
    
    Args:
        x: Vector de entrada (aceleración)
        y: Vector de salida (fuerza)
        
    Returns:
        area: Área del ciclo de histéresis
    """
    # Verificar si los datos son adecuados para el cálculo
    if len(x) != len(y):
        logging.warning(f"Longitudes inconsistentes en calcular_area_histeresis: {len(x)} vs {len(y)}")
        return np.nan
    
    # Verificar si hay suficientes datos para calcular un área
    if len(x) < 3:
        logging.warning("Insuficientes datos para calcular el área de histéresis")
        return np.nan
    
    # Verificar y manejar NaN
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    if not np.any(mask):
        return np.nan
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Verificar datos válidos restantes
    if len(x_clean) < 3:
        return np.nan
    
    try:
        # Para datos ruidosos, es mejor usar un enfoque basado en loops cerrados
        # Primero ordenamos los puntos para formar un ciclo más limpio
        # Podemos hacerlo ordenándolos por la variable independiente
        
        # Método 1: Integración trapezoidal (buena para ciclos simples)
        area_trapz = np.abs(np.trapz(y_clean, x_clean))
        
        # Método 2: Cálculo de polígono orientado (más robusto para ciclos complejos)
        # Este método es mejor para ciclos que se cruzan a sí mismos
        n = len(x_clean)
        area_polygon = 0.0
        for i in range(n):
            j = (i + 1) % n
            area_polygon += x_clean[i] * y_clean[j]
            area_polygon -= y_clean[i] * x_clean[j]
        area_polygon = abs(area_polygon) / 2.0
        
        # Elegir el mayor valor entre los métodos para datos ruidosos
        # Esto tiende a ser más estable para diferentes tipos de ciclos
        area = max(area_trapz, area_polygon)
        
        # En caso de áreas extremadamente grandes, limitar a un valor razonable
        if area > 1e10 or np.isnan(area) or np.isinf(area):
            logging.warning(f"Área de histéresis no válida calculada: {area}")
            area = np.nan
            
        return area
    
    except Exception as e:
        logging.error(f"Error al calcular el área de histéresis: {e}")
        return np.nan

def calcular_energia_disipada(area_histeresis, frecuencia):
    """
    Calcula la energía disipada por ciclo basada en el área de histéresis.
    
    Args:
        area_histeresis: Área del ciclo de histéresis
        frecuencia: Frecuencia de excitación en Hz
        
    Returns:
        energia: Energía disipada por ciclo en Julios
    """
    # La energía disipada es proporcional al área de histéresis y la frecuencia
    energia = area_histeresis * frecuencia
    return energia

def calcular_frecuencia_dominante(aceleracion, tiempo):
    """
    Calcula la frecuencia dominante en una señal.
    
    Args:
        aceleracion: Vector de aceleración
        tiempo: Vector de tiempo
        
    Returns:
        frecuencia: Frecuencia dominante en Hz
    """
    from scipy.fft import rfft, rfftfreq
    
    # Tasa de muestreo
    dt = np.mean(np.diff(tiempo))
    fs = 1/dt
    
    # Calcular FFT
    y_fft = rfft(aceleracion)
    x_fft = rfftfreq(len(aceleracion), dt)
    
    # Encontrar frecuencia dominante
    idx_max = np.argmax(np.abs(y_fft))
    frecuencia = x_fft[idx_max]
    
    return frecuencia

def analizar_resultados_doe(resultados_df):
    """
    Realiza un análisis estadístico ANOVA de los resultados del DOE.
    
    Args:
        resultados_df: DataFrame con los resultados (columnas: Amplitud, Frecuencia, Carga, etc.)
        
    Returns:
        anova_results: Resultados del análisis ANOVA
    """
    try:
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
    except ImportError:
        logging.warning("Para realizar el análisis ANOVA, instala statsmodels con: pip install statsmodels")
        return None
    
    # Resultados ANOVA para diferentes variables de respuesta
    anova_results = {}
    
    # Lista de variables de respuesta para analizar
    response_vars = ['Area_Histeresis', 'Energia_Disipada']
    
    for response in response_vars:
        if response in resultados_df.columns:
            # Crear modelo de regresión
            model = ols(f'{response} ~ C(Amplitud) + C(Frecuencia) + C(Carga) + C(Amplitud):C(Frecuencia) + '
                       f'C(Amplitud):C(Carga) + C(Frecuencia):C(Carga)', 
                       data=resultados_df).fit()
            
            # Realizar ANOVA
            anova_table = anova_lm(model)
            anova_results[response] = anova_table
    
    return anova_results

def procesar_experimento_doe(archivo_fuerza, archivo_vib, params_dahl=None, umbral_nan=0.5, intentar_reparar=True):
    """
    Procesa un experimento del DOE a partir de archivos guardados.
    
    Args:
        archivo_fuerza: Ruta al archivo CSV de datos de fuerza
        archivo_vib: Ruta al archivo CSV de datos de vibración
        params_dahl: Parámetros del modelo de Dahl, si es None utiliza valores por defecto
        umbral_nan: Umbral máximo permitido de valores NaN (0.0 a 1.0). Por defecto 0.5 (50%)
        intentar_reparar: Si es True, intenta interpolar valores NaN antes de abortar
        
    Returns:
        resultados: Diccionario con resultados del experimento, o None si la calidad de los datos
                   es insuficiente y no se puede reparar
    """
    # Preprocesar datos
    t, aceleracion, fuerza, _ = pp.preprocesar_datos(
        archivo_fuerza=archivo_fuerza, 
        archivo_vib=archivo_vib,
        aplicar_filtro=True
    )
    
    # Extraer señales relevantes con manejo de errores
    try:
        # Extraer tiempo y fuerza (asumiendo que es la columna de fuerza principal)
        tiempo_fuerza = datos_fuerza['Tiempo(s)'].values
        fuerza = datos_fuerza.get('Fuerza_ai0(V)', pd.Series(np.zeros(len(tiempo_fuerza)))).values
        logging.info(f"After fuerza extraction: NaN count = {np.sum(np.isnan(fuerza))}")

        # Extraer tiempo y aceleración (asumiendo que es la columna de vibración principal)
        tiempo_vib = datos_vib['Tiempo(s)'].values
        aceleracion = datos_vib.get('Vibracion_ai0(g_or_V)', pd.Series(np.zeros(len(tiempo_vib)))).values
        logging.info(f"After aceleracion extraction: NaN count = {np.sum(np.isnan(aceleracion))}")

        # Sincronizar datos por tiempo (encontrar el tiempo común)
        tiempo_comun, idx_fuerza, idx_vib = np.intersect1d(tiempo_fuerza, tiempo_vib, return_indices=True)
        tiempo = tiempo_comun
        fuerza = fuerza[idx_fuerza]
        aceleracion = aceleracion[idx_vib]
        logging.info(f"After time synchronization: Fuerza NaN count = {np.sum(np.isnan(fuerza))}, Aceleracion NaN count = {np.sum(np.isnan(aceleracion))}")

        # Asegurar que los arrays tengan el mismo tamaño después de la sincronización
        if len(fuerza) != len(aceleracion):
            logging.warning(f"Longitudes inconsistentes después de sincronización: fuerza={len(fuerza)}, aceleracion={len(aceleracion)}")
            min_len = min(len(fuerza), len(aceleracion))
            fuerza = fuerza[:min_len]
            aceleracion = aceleracion[:min_len]
            tiempo = tiempo[:min_len]
        logging.info(f"After length adjustment: Fuerza NaN count = {np.sum(np.isnan(fuerza))}, Aceleracion NaN count = {np.sum(np.isnan(aceleracion))}")

    except Exception as e:
        logging.error(f"Error al extraer datos de los CSV: {e}")
        return None

    # Preprocesar señales (filtrado, eliminación de ruido)
    try:
        # Aplicar preprocesamiento si está disponible la librería
        if hasattr(pp, 'preprocesar_senal'):
            fuerza = pp.preprocesar_senal(fuerza, tipo='fuerza')
            logging.info(f"After fuerza preprocessing: NaN count = {np.sum(np.isnan(fuerza))}")
            aceleracion = pp.preprocesar_senal(aceleracion, tipo='aceleracion')
            logging.info(f"After aceleracion preprocessing: NaN count = {np.sum(np.isnan(aceleracion))}")
        else:
            logging.warning("Librería de preprocesamiento no disponible")
    except Exception as e:
        logging.error(f"Error en preprocesamiento de señales: {e}")
        # Continuar con datos sin preprocesar
        pass

    # Calcular variable de histéresis usando el modelo de Bouc-Wen
    try:
        z_histerica = calcular_z_histerica(aceleracion)
        logging.info(f"After z_histerica calculation: NaN count = {np.sum(np.isnan(z_histerica))}")
    except Exception as e:
        logging.error(f"Error al calcular z_histerica: {e}")
        z_histerica = np.zeros(len(aceleracion))

    # Estimar fuerza usando el modelo de Dahl
    try:
        fuerza_estimada = estimar_fuerza_dahl(z_histerica, aceleracion, params_dahl)
        logging.info(f"After fuerza_estimada calculation: NaN count = {np.sum(np.isnan(fuerza_estimada))}")
    except Exception as e:
        logging.error(f"Error al estimar fuerza: {e}")
        fuerza_estimada = np.zeros(len(aceleracion))

    # Verificar calidad de datos antes de los cálculos
    logging.info("\nVERIFICACIÓN DE CALIDAD DE DATOS:")
    logging.info("=" * 50)
    
    # Verificar valores NaN
    nan_counts = {
        'fuerza': np.isnan(fuerza).sum(),
        'aceleracion': np.isnan(aceleracion).sum(),
        'z_histerica': np.isnan(z_histerica).sum(),
        'fuerza_estimada': np.isnan(fuerza_estimada).sum()
    }
    
    for key, count in nan_counts.items():
        logging.info(f" - Valores NaN en {key}: {count}/{len(fuerza)} ({count/len(fuerza)*100:.2f}%)")
    
    # Verificar consistencia de tamaños
    tamanos = {
        'tiempo': len(t),
        'fuerza': len(fuerza),
        'aceleracion': len(aceleracion),
        'z_histerica': len(z_histerica),
        'fuerza_estimada': len(fuerza_estimada)
    }
    
    if len(set(tamanos.values())) > 1:
        logging.warning("\nERROR: Los arrays tienen tamaños inconsistentes:")
        for key, size in tamanos.items():
            logging.warning(f" - {key}: {size} elementos")
    else:
        logging.info(f"\nTODOS los arrays tienen tamaño consistente: {tamanos['tiempo']} elementos")
    
    # Verificar rangos de valores
    logging.info("\nRANGOS DE VALORES:")
    logging.info(f" - Fuerza: [{np.nanmin(fuerza):.2f}, {np.nanmax(fuerza):.2f}] N")
    logging.info(f" - Aceleración: [{np.nanmin(aceleracion):.2f}, {np.nanmax(aceleracion):.2f}] m/s²")
    logging.info(f" - z_histerica: [{np.nanmin(z_histerica):.2f}, {np.nanmax(z_histerica):.2f}]")
    logging.info(f" - Fuerza estimada: [{np.nanmin(fuerza_estimada):.2f}, {np.nanmax(fuerza_estimada):.2f}] N")
    
    # Calcular porcentajes de NaN
    nan_percentages = {k: count/len(fuerza)*100 for k, count in nan_counts.items()}
    
    # Manejar problemas críticos con valores NaN
    if nan_counts['fuerza'] > umbral_nan * len(fuerza) or nan_counts['aceleracion'] > umbral_nan * len(aceleracion):
        logging.warning(f"\nADVERTENCIA: Exceso de valores NaN detectados (umbral configurado: {umbral_nan*100:.1f}%):")
        for key, percentage in nan_percentages.items():
            if percentage > (umbral_nan * 100):
                logging.warning(f" - {key}: {percentage:.2f}% (excede el umbral)")
        
        # Intentar reparar los datos si está habilitado
        if intentar_reparar:
            logging.info("\nIntentando reparar datos con interpolación...")
            
            # Crear máscara de índices no-NaN
            valid_indices = ~(np.isnan(fuerza) | np.isnan(aceleracion) | np.isnan(z_histerica) | np.isnan(fuerza_estimada))
            valid_count = np.sum(valid_indices)
            
            # Análisis detallado por variable
            logging.info("\nDiagnóstico detallado de valores NaN por variable:")
            for var_name, var_data in {
                'fuerza': fuerza,
                'aceleracion': aceleracion, 
                'z_histerica': z_histerica, 
                'fuerza_estimada': fuerza_estimada
            }.items():
                valid_count_var = np.sum(~np.isnan(var_data))
                logging.info(f" - {var_name}: {valid_count_var}/{len(var_data)} puntos válidos ({valid_count_var/len(var_data)*100:.2f}%)")
            
            # Verificación de estructura del archivo
            logging.info("\nPrimeros 5 valores de cada variable para diagnóstico:")
            for var_name, var_data in {
                'fuerza': fuerza,
                'aceleracion': aceleracion, 
                'tiempo': t
            }.items():
                logging.info(f" - {var_name}: {var_data[:5]}")
            
            if valid_count == 0:
                logging.error("\nERROR CRÍTICO: No hay puntos válidos en el dataset. Imposible realizar interpolación.")
                logging.error("Posibles causas:")
                logging.error(" - Formato incorrecto de los datos CSV")
                logging.error(" - Problemas en la conversión de tipos de datos")
                logging.error(" - Valores no numéricos en los archivos de entrada")
                logging.error("\nRecomendación: Verificar formato y contenido de los archivos CSV de entrada.")
                return None
            elif valid_count < 100:  # Si hay muy pocos datos válidos, no podemos interpolar
                logging.error(f"\nERROR: Solo {valid_count} puntos válidos de {len(fuerza)}. Insuficiente para interpolación confiable.")
                logging.error("Se requiere al menos 100 puntos válidos para garantizar una interpolación de calidad.")
                return None
                
            logging.info(f"Interpolando con {valid_count}/{len(fuerza)} puntos válidos ({valid_count/len(fuerza)*100:.1f}%)")
            
            # Interpolar los arrays
            t_valid = t[valid_indices]
            indices = np.arange(len(t))
            
            # Interpolar cada señal
            fuerza_fixed = np.interp(indices, indices[valid_indices], fuerza[valid_indices])
            aceleracion_fixed = np.interp(indices, indices[valid_indices], aceleracion[valid_indices])
            z_histerica_fixed = np.interp(indices, indices[valid_indices], z_histerica[valid_indices])
            fuerza_estimada_fixed = np.interp(indices, indices[valid_indices], fuerza_estimada[valid_indices])
            
            # Reemplazar los arrays
            fuerza = fuerza_fixed
            aceleracion = aceleracion_fixed
            z_histerica = z_histerica_fixed
            fuerza_estimada = fuerza_estimada_fixed
            
            # Verificar resultados de interpolación
            nan_counts_after = {
                'fuerza': np.isnan(fuerza).sum(),
                'aceleracion': np.isnan(aceleracion).sum(),
                'z_histerica': np.isnan(z_histerica).sum(),
                'fuerza_estimada': np.isnan(fuerza_estimada).sum()
            }
            
            if any(count > 0 for count in nan_counts_after.values()):
                logging.error("ERROR: Aún quedan valores NaN después de interpolar:")
                for k, v in nan_counts_after.items():
                    if v > 0:
                        logging.error(f" - {k}: {v} valores NaN ({v/len(fuerza)*100:.2f}%)")
                return None
                
            logging.info("Interpolación exitosa, continuando con el procesamiento...")
        else:
            logging.error(f"\nERROR CRÍTICO: Exceso de NaN en los datos (umbral: {umbral_nan*100:.1f}%). Abortando cálculos.")
            logging.error("Use --intentar-reparar si desea intentar interpolar los valores faltantes.")
            return None
    
    logging.info("=" * 50)
    logging.info("\n")
    
    # Calcular área de histéresis
    area_exp = calcular_area_histeresis(aceleracion, fuerza)
    area_modelo = calcular_area_histeresis(aceleracion, fuerza_estimada)
    
    # Calcular frecuencia dominante
    freq_dominante = calcular_frecuencia_dominante(aceleracion, t)
    
    # Calcular energía disipada
    energia_exp = calcular_energia_disipada(area_exp, freq_dominante)
    energia_modelo = calcular_energia_disipada(area_modelo, freq_dominante)
    
    # Error RMS entre fuerza experimental y estimada
    error_rms = np.sqrt(np.mean((fuerza - fuerza_estimada)**2))
    
    # Guardar resultados
    resultados = {
        'Archivo_Fuerza': os.path.basename(archivo_fuerza),
        'Archivo_Vibracion': os.path.basename(archivo_vib),
        'Duracion': t[-1],
        'Frecuencia_Dominante': freq_dominante,
        'Area_Histeresis_Exp': area_exp,
        'Area_Histeresis_Modelo': area_modelo,
        'Energia_Disipada_Exp': energia_exp,
        'Energia_Disipada_Modelo': energia_modelo,
        'Error_RMS': error_rms,
        'Datos': {
            'tiempo': t,
            'aceleracion': aceleracion,
            'fuerza': fuerza,
            'z_histerica': z_histerica,
            'fuerza_estimada': fuerza_estimada
        }
    }
    
    return resultados

def visualizar_ciclo_histeresis(resultados, titulo="Ciclo de Histéresis"):
    """
    Visualiza el ciclo de histéresis a partir de los resultados procesados.
    
    Args:
        resultados: Diccionario con resultados del experimento
        titulo: Título para la figura
    """
    # Verificar si los resultados son válidos
    if resultados is None:
        logging.error("ERROR: No se pueden visualizar los resultados porque son nulos.")
        logging.error("Esto puede deberse a problemas con la calidad de los datos de entrada.")
        return
    
    # Verificar si existe la estructura de datos esperada
    if 'Datos' not in resultados:
        logging.error("ERROR: Los resultados no contienen la clave 'Datos'.")
        logging.error("Claves disponibles:", list(resultados.keys()))
        return
    
    # Extraer datos con manejo de errores
    try:
        tiempo = resultados['Datos'].get('tiempo', [])
        aceleracion = resultados['Datos'].get('aceleracion', [])
        fuerza = resultados['Datos'].get('fuerza', [])
        z_histerica = resultados['Datos'].get('z_histerica', [])
        fuerza_estimada = resultados['Datos'].get('fuerza_estimada', [])
        
        # Verificar que los datos no estén vacíos
        if len(tiempo) == 0 or len(aceleracion) == 0 or len(fuerza) == 0:
            logging.error("ERROR: Los arrays de datos están vacíos o no existen.")
            logging.error("Longitudes: tiempo={}, aceleracion={}, fuerza={}, z_histerica={}, fuerza_estimada={}".format(
                len(tiempo), len(aceleracion), len(fuerza), len(z_histerica), len(fuerza_estimada)
            ))
            return
    except Exception as e:
        logging.error(f"ERROR al extraer datos para visualización: {e}")
        return
    
    area_histeresis = resultados.get('Area_Histeresis_Modelo', 0)
    energia_disipada = resultados.get('Energia_Disipada_Modelo', 0)
    frecuencia_dominante = resultados.get('Frecuencia_Dominante', 0)
    error_rms = resultados.get('Error_RMS', 0)
    
    # Crear figura con 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(titulo, fontsize=14)
    
    # 1. Aceleración vs tiempo
    axs[0, 0].plot(tiempo, aceleracion, 'b', label='Experimental')
    axs[0, 0].set_xlabel('Tiempo (s)')
    axs[0, 0].set_ylabel('Aceleración (g)')
    axs[0, 0].set_title('Aceleración Absoluta')
    axs[0, 0].grid(True)
    
    # 2. Fuerza vs tiempo
    axs[0, 1].plot(tiempo, fuerza, 'r-', label='Experimental')
    axs[0, 1].plot(tiempo, fuerza_estimada, 'g--', label='Modelo Dahl')
    axs[0, 1].set_xlabel('Tiempo (s)')
    axs[0, 1].set_ylabel('Fuerza (mN)')
    axs[0, 1].set_title('Fuerza vs Tiempo')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # 3. Variable z histérica vs tiempo
    axs[1, 0].plot(tiempo, z_histerica, 'b')
    axs[1, 0].set_xlabel('Tiempo (s)')
    axs[1, 0].set_ylabel('z (adimensional)')
    axs[1, 0].set_title('Variable z Histérica')
    axs[1, 0].grid(True)
    
    # 4. Ciclo de histéresis: z vs aceleración
    axs[1, 1].plot(aceleracion, fuerza, 'r-', label='Experimental')
    axs[1, 1].plot(aceleracion, fuerza_estimada, 'g--', label='Modelo Dahl')
    axs[1, 1].set_xlabel('Aceleración (g)')
    axs[1, 1].set_ylabel('Fuerza (mN)')
    axs[1, 1].set_title('Ciclo de Histéresis')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    # Añadir métricas en texto
    plt.figtext(0.1, 0.01, f'Área Hist. Exp: {resultados["Area_Histeresis_Exp"]:.6f}\nÁrea Hist. Modelo: {area_histeresis:.6f}\nError RMS: {error_rms:.6f}\nFrecuencia: {frecuencia_dominante:.2f} Hz')
    
    # Ajustar disposición
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def procesar_doe_completo(directorio_datos, patron_fuerza="*_fuerza_*.csv", patron_vib="*_vib_*.csv", 
                       params_experimento=None, guardar_resultados=True):
    """
    Procesa todos los experimentos del DOE y genera un DataFrame con resultados.
    
    Args:
        directorio_datos: Ruta al directorio que contiene archivos de datos
        patron_fuerza: Patrón para identificar archivos de fuerza
        patron_vib: Patrón para identificar archivos de vibración
        params_experimento: Diccionario con parámetros experimentales adicionales
        guardar_resultados: Si es True, guarda resultados en archivo CSV
        
    Returns:
        df_resultados: DataFrame con todos los resultados del DOE
    """
    import os
    import glob
    import pandas as pd
    from datetime import datetime
    
    # Encontrar archivos de datos
    archivos_fuerza = glob.glob(os.path.join(directorio_datos, patron_fuerza))
    archivos_vib = glob.glob(os.path.join(directorio_datos, patron_vib))
    
    # Ordenar archivos
    archivos_fuerza.sort()
    archivos_vib.sort()
    
    # Verificar que haya la misma cantidad de archivos
    if len(archivos_fuerza) != len(archivos_vib):
        logging.warning(f"Diferente número de archivos (Fuerza: {len(archivos_fuerza)}, Vibración: {len(archivos_vib)})")
        return None
    
    if len(archivos_fuerza) == 0:
        logging.error(f"No se encontraron archivos en {directorio_datos}")
        return None
    
    # Lista para almacenar resultados
    resultados_list = []
    
    # Procesar cada par de archivos
    for i, (arch_fuerza, arch_vib) in enumerate(zip(archivos_fuerza, archivos_vib)):
        logging.info(f"Procesando experimento {i+1}/{len(archivos_fuerza)}: {os.path.basename(arch_fuerza)}")
        
        try:
            # Extraer parámetros experimentales del nombre de archivo
            nombre_exp = os.path.basename(arch_fuerza)
            params_doe = extraer_parametros_nombre(nombre_exp)
            
            # Procesar experimento
            resultados = procesar_experimento_doe(
                arch_fuerza, 
                arch_vib, 
                params_dahl=None,
                umbral_nan=args.umbral_nan,
                intentar_reparar=args.intentar_reparar
            )
            
            # Añadir parámetros experimentales a los resultados
            if params_doe:
                for key, value in params_doe.items():
                    resultados[key] = value
                    
            # Añadir parámetros adicionales si existen
            if params_experimento:
                for key, value in params_experimento.items():
                    resultados[key] = value[i] if isinstance(value, list) and len(value) > i else value
            
            # Guardar resultados en la lista (sin los datos crudos para no ocupar demasiada memoria)
            resultados_reducidos = {k: v for k, v in resultados.items() if k != 'Datos'}
            resultados_list.append(resultados_reducidos)
            
            # Visualizar resultados (opcional)
            # visualizar_ciclo_histeresis(resultados, titulo=f"Experimento {i+1}: {nombre_exp}")
            
        except Exception as e:
            logging.error(f"Error procesando {nombre_exp}: {str(e)}")
            continue
    
    # Crear DataFrame con todos los resultados
    df_resultados = pd.DataFrame(resultados_list)
    
    # Guardar resultados en CSV
    if guardar_resultados and len(resultados_list) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo_salida = os.path.join(directorio_datos, f"resultados_doe_{timestamp}.csv")
        df_resultados.to_csv(archivo_salida, index=False)
        logging.info(f"Resultados guardados en: {archivo_salida}")
    
    return df_resultados

def extraer_parametros_nombre(nombre_archivo):
    """
    Extrae parámetros experimentales del nombre de archivo.
    Formato esperado: "exp_amp[valor]_freq[valor]_carga[valor]_..."
    
    Args:
        nombre_archivo: Nombre del archivo
        
    Returns:
        params: Diccionario con parámetros extraídos
    """
    import re
    
    params = {}
    
    # Patrones para extraer parámetros comunes
    patrones = {
        'Amplitud': r'amp(?:litud)?[_-]?(\d+(?:\.\d+)?)',
        'Frecuencia': r'freq(?:uencia)?[_-]?(\d+(?:\.\d+)?)',
        'Carga': r'carga[_-]?(\d+(?:\.\d+)?)',
    }
    
    for param, patron in patrones.items():
        match = re.search(patron, nombre_archivo, re.IGNORECASE)
        if match:
            try:
                # Convertir a número
                params[param] = float(match.group(1))
            except:
                # Si no se puede convertir, guardar como texto
                params[param] = match.group(1)
    
    return params

def seleccionar_archivos_manualmente():
    """
    Permite al usuario seleccionar archivos de datos manualmente usando un diálogo
    """
    import tkinter as tk
    from tkinter import filedialog
    
    # Crear una ventana raíz oculta para el diálogo
    root = tk.Tk()
    root.withdraw()
    
    # Mostrar mensaje
    print("Por favor, seleccione el archivo de FUERZA...")
    archivo_fuerza = filedialog.askopenfilename(
        title="Seleccionar archivo de FUERZA",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if not archivo_fuerza:
        print("No se seleccionó ningún archivo de fuerza. Abortando.")
        root.destroy()
        return None, None
    
    # Mostrar mensaje
    print("Por favor, seleccione el archivo de VIBRACIÓN...")
    archivo_vib = filedialog.askopenfilename(
        title="Seleccionar archivo de VIBRACIÓN",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if not archivo_vib:
        print("No se seleccionó ningún archivo de vibración. Abortando.")
        root.destroy()
        return None, None
    
    root.destroy()
    return archivo_fuerza, archivo_vib

def main():
    """
    Función principal para análisis de histéresis y procesamiento de DOE
    """
    import argparse
    import os
    import pandas as pd
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Análisis de Histéresis y Procesamiento de DOE')
    parser.add_argument('--dir', type=str, help='Directorio con datos experimentales',
                        default=os.path.join(os.path.dirname(__file__), '..', 'datos'))
    parser.add_argument('--modo', type=str, 
                        choices=['individual', 'doe', 'analizar', 'manual'], 
                        default='individual', 
                        help='Modo de operación: individual, doe, analizar o manual (selección interactiva de archivos)')
    parser.add_argument('--fuerza', type=str, help='Archivo de datos de fuerza (para modo individual)')
    parser.add_argument('--vib', type=str, help='Archivo de datos de vibración (para modo individual)')
    parser.add_argument('--resultados', type=str, help='Archivo de resultados DOE (para modo analizar)')
    parser.add_argument('--no-visualizar', action='store_true', help='No visualizar resultados')
    parser.add_argument('--no-guardar', action='store_true', help='No guardar resultados')
    parser.add_argument('--umbral-nan', type=float, default=0.5, help='Umbral máximo permitido de valores NaN (0.0 a 1.0)')
    parser.add_argument('--intentar-reparar', action='store_true', help='Intentar reparar datos mediante interpolación de valores NaN')
    
    args = parser.parse_args()
    
    # Modo de operación: individual (un experimento)
    if args.modo == 'individual':
        if not args.fuerza or not args.vib:
            logging.error("ERROR: Se requieren archivos de fuerza y vibración para modo individual")
            logging.error("TIP: Puedes usar --modo manual para seleccionar archivos gráficamente")
            parser.print_help()
            return
            
        # Asegurar rutas absolutas
        if os.path.isabs(args.fuerza):
            archivo_fuerza = args.fuerza
        else:
            # Para evitar construir rutas incorrectas, verificamos si el archivo existe directamente
            archivo_directo = args.fuerza
            if os.path.exists(archivo_directo):
                archivo_fuerza = os.path.abspath(archivo_directo)
            else:
                # Si no existe como ruta relativa directa, probamos en el directorio de datos
                archivo_fuerza = os.path.join(args.dir, args.fuerza)
                
        # Lo mismo para el archivo de vibración
        if os.path.isabs(args.vib):
            archivo_vib = args.vib
        else:
            # Para evitar construir rutas incorrectas, verificamos si el archivo existe directamente
            archivo_directo = args.vib
            if os.path.exists(archivo_directo):
                archivo_vib = os.path.abspath(archivo_directo)
            else:
                # Si no existe como ruta relativa directa, probamos en el directorio de datos
                archivo_vib = os.path.join(args.dir, args.vib)
        
        # Procesar experimento individual
        resultados = procesar_experimento_doe(
            archivo_fuerza, 
            archivo_vib, 
            params_dahl=None,  # Parámetros por defecto
            umbral_nan=args.umbral_nan, 
            intentar_reparar=args.intentar_reparar
        )
        
        # Verificar si el procesamiento fue exitoso
        if resultados is None:
            logging.error("\nERROR: No se pudieron procesar los datos correctamente.")
            logging.error("Esto puede deberse a problemas con la calidad de los datos o a errores en el procesamiento.")
            logging.error("Intenta verificar los archivos de entrada y ajustar los parámetros.\n")
            return
        
        # Verificar que exista la estructura de datos necesaria
        if 'Datos' not in resultados:
            logging.error("\nERROR: Los resultados procesados no tienen la estructura esperada (falta la clave 'Datos').")
            logging.error("Claves disponibles:", list(resultados.keys()))
            return
            
        # Visualizar resultados
        if not args.no_visualizar:
            visualizar_ciclo_histeresis(resultados, titulo=f"Análisis Individual: {os.path.basename(archivo_fuerza)}")
        
        # Guardar resultados
        if not args.no_guardar:
            import pickle
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_base = os.path.splitext(os.path.basename(archivo_fuerza))[0]
            
            # Crear directorio resultados si no existe
            dir_resultados = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
            if not os.path.exists(dir_resultados):
                os.makedirs(dir_resultados)
                logging.info(f"Creado directorio para resultados: {dir_resultados}")
            
            # Guardar en el directorio de resultados
            archivo_salida = os.path.join(dir_resultados, f"{nombre_base}_analisis_{timestamp}.pkl")
            with open(archivo_salida, 'wb') as f:
                pickle.dump(resultados, f)
            logging.info(f"Resultados guardados en: {archivo_salida}")
            
            # Guardar una copia en formato CSV para más fácil acceso
            try:
                import pandas as pd
                # Acceder a los datos correctamente según la estructura
                datos = resultados['Datos']
                
                # Verificar existencia de las claves necesarias
                claves_requeridas = ['tiempo', 'aceleracion', 'fuerza', 'z_histerica', 'fuerza_estimada']
                for clave in claves_requeridas:
                    if clave not in datos:
                        logging.warning(f"Advertencia: La clave '{clave}' no existe en los datos. Saltando exportación CSV.")
                        raise KeyError(f"Falta la clave '{clave}' en los datos")
                
                # Convertir los arrays de NumPy a listas para facilitar la serialización
                # Primero verificamos que todos los arrays tengan la misma longitud
                min_len = min(
                    len(datos['tiempo']), 
                    len(datos['aceleracion']), 
                    len(datos['fuerza']), 
                    len(datos['z_histerica']), 
                    len(datos['fuerza_estimada'])
                )
                
                resultados_csv = {
                    'tiempo': datos['tiempo'][:min_len].tolist(),
                    'aceleracion': datos['aceleracion'][:min_len].tolist(),
                    'fuerza': datos['fuerza'][:min_len].tolist(),
                    'z_histerica': datos['z_histerica'][:min_len].tolist(),
                    'fuerza_estimada': datos['fuerza_estimada'][:min_len].tolist(),
                }
                
                # Métricas de resumen
                metricas = {
                    'area_histeresis': resultados.get('Area_Histeresis_Modelo', 0),
                    'energia_disipada': resultados.get('Energia_Disipada_Modelo', 0),
                    'frecuencia_dominante': resultados.get('Frecuencia_Dominante', 0),
                    'error_rms': resultados.get('Error_RMS', 0),
                    'parametros_modelo': str(resultados.get('Parametros_Modelo', {}))
                }
                
                # Guardar datos en CSV
                df = pd.DataFrame(resultados_csv)
                archivo_csv = os.path.join(dir_resultados, f"{nombre_base}_datos_{timestamp}.csv")
                df.to_csv(archivo_csv, index=False)
                
                # Guardar métricas en CSV
                df_metricas = pd.DataFrame([metricas])
                archivo_metricas = os.path.join(dir_resultados, f"{nombre_base}_metricas_{timestamp}.csv")
                df_metricas.to_csv(archivo_metricas, index=False)
                
                logging.info(f"Datos guardados en formato CSV: {archivo_csv}")
                logging.info(f"Métricas guardadas en formato CSV: {archivo_metricas}")
            except Exception as e:
                logging.warning(f"AVISO: No se pudieron guardar los resultados en CSV: {e}")
                pass
    
    # Modo de operación: manual (seleccionar archivos gráficamente)
    elif args.modo == 'manual':
        logging.info("Modo manual: selecciona los archivos de fuerza y vibración gráficamente")
        archivo_fuerza, archivo_vib = seleccionar_archivos_manualmente()
        
        if not archivo_fuerza or not archivo_vib:
            logging.error("No se seleccionaron los archivos necesarios. Operación cancelada.")
            return
            
        # Procesar experimento con los archivos seleccionados manualmente
        logging.info(f"Procesando archivos:\n - Fuerza: {archivo_fuerza}\n - Vibración: {archivo_vib}")
        resultados = procesar_experimento_doe(
            archivo_fuerza, 
            archivo_vib, 
            params_dahl=None, 
            umbral_nan=args.umbral_nan, 
            intentar_reparar=args.intentar_reparar
        )
        
        # Visualizar resultados
        if not args.no_visualizar:
            visualizar_ciclo_histeresis(resultados, titulo=f"Análisis Manual: {os.path.basename(archivo_fuerza)}")
        
        # Guardar resultados
        if not args.no_guardar:
            import pickle
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_base = os.path.splitext(os.path.basename(archivo_fuerza))[0]
            
            # Crear directorio resultados si no existe
            dir_resultados = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
            if not os.path.exists(dir_resultados):
                os.makedirs(dir_resultados)
                logging.info(f"Creado directorio para resultados: {dir_resultados}")
            
            # Guardar en el directorio de resultados
            archivo_salida = os.path.join(dir_resultados, f"{nombre_base}_analisis_{timestamp}.pkl")
            with open(archivo_salida, 'wb') as f:
                pickle.dump(resultados, f)
            logging.info(f"Resultados guardados en: {archivo_salida}")
            
            # Guardar una copia en formato CSV para más fácil acceso
            try:
                import pandas as pd
                # Convertir los arrays de NumPy a listas para facilitar la serialización
                # Primero verificamos que todos los arrays tengan la misma longitud
                min_len = min(len(resultados['tiempo']), len(resultados['aceleracion']), 
                              len(resultados['fuerza']), len(resultados['z_histerica']), 
                              len(resultados['fuerza_estimada']))
                
                resultados_csv = {
                    'tiempo': datos['tiempo'][:min_len].tolist(),
                    'aceleracion': datos['aceleracion'][:min_len].tolist(),
                    'fuerza': datos['fuerza'][:min_len].tolist(),
                    'z_histerica': datos['z_histerica'][:min_len].tolist(),
                    'fuerza_estimada': datos['fuerza_estimada'][:min_len].tolist(),
                }
                
                # Métricas de resumen
                metricas = {
                    'area_histeresis': resultados.get('Area_Histeresis_Modelo', 0),
                    'energia_disipada': resultados.get('Energia_Disipada_Modelo', 0),
                    'frecuencia_dominante': resultados.get('Frecuencia_Dominante', 0),
                    'error_rms': resultados.get('Error_RMS', 0),
                    'parametros_modelo': str(resultados.get('Parametros_Modelo', {}))
                }
                
                # Guardar datos en CSV
                df = pd.DataFrame(resultados_csv)
                archivo_csv = os.path.join(dir_resultados, f"{nombre_base}_datos_{timestamp}.csv")
                df.to_csv(archivo_csv, index=False)
                
                # Guardar métricas en CSV
                df_metricas = pd.DataFrame([metricas])
                archivo_metricas = os.path.join(dir_resultados, f"{nombre_base}_metricas_{timestamp}.csv")
                df_metricas.to_csv(archivo_metricas, index=False)
                
                logging.info(f"Datos guardados en formato CSV: {archivo_csv}")
                logging.info(f"Métricas guardadas en formato CSV: {archivo_metricas}")
            except Exception as e:
                logging.warning(f"AVISO: No se pudieron guardar los resultados en CSV: {e}")
                pass
            
    # Modo de operación: doe (procesar DOE completo)
    elif args.modo == 'doe':
        # Procesar todos los experimentos en el directorio
        df_resultados = procesar_doe_completo(
            directorio_datos=args.dir,
            guardar_resultados=not args.no_guardar
        )
        
        # Si hay resultados y no se desactivó la visualización
        if df_resultados is not None and not args.no_visualizar:
            logging.info("\nResumen de resultados:")
            logging.info(df_resultados.describe())
            
            # Realizar análisis ANOVA
            try:
                anova_resultados = analizar_resultados_doe(df_resultados)
                if anova_resultados:
                    logging.info("\nResultados ANOVA:")
                    for var, tabla in anova_resultados.items():
                        logging.info(f"\nANOVA para {var}:\n")
                        logging.info(tabla)
            except Exception as e:
                logging.error(f"Error en análisis ANOVA: {str(e)}")
    
    # Modo de operación: analizar (resultados existentes)
    elif args.modo == 'analizar':
        if not args.resultados:
            logging.error("ERROR: Se requiere archivo de resultados para modo analizar")
            parser.print_help()
            return
        
        # Cargar resultados
        try:
            archivo_resultados = args.resultados if os.path.isabs(args.resultados) else os.path.join(args.dir, args.resultados)
            df_resultados = pd.read_csv(archivo_resultados)
            logging.info(f"Se cargaron {len(df_resultados)} experimentos de {archivo_resultados}")
            
            # Realizar análisis ANOVA
            try:
                anova_resultados = analizar_resultados_doe(df_resultados)
                if anova_resultados:
                    logging.info("\nResultados ANOVA:")
                    for var, tabla in anova_resultados.items():
                        logging.info(f"\nANOVA para {var}:\n")
                        logging.info(tabla)
            except Exception as e:
                logging.error(f"Error en análisis ANOVA: {str(e)}")
            
            # Visualizar algunos gráficos
            if not args.no_visualizar:
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    # Gráfico de efectos principales
                    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Efectos de Amplitud
                    sns.boxplot(x='Amplitud', y='Area_Histeresis_Exp', data=df_resultados, ax=axs[0])
                    axs[0].set_title('Efecto de Amplitud en Área de Histéresis')
                    axs[0].grid(True)
                    
                    # Efectos de Frecuencia
                    sns.boxplot(x='Frecuencia', y='Area_Histeresis_Exp', data=df_resultados, ax=axs[1])
                    axs[1].set_title('Efecto de Frecuencia en Área de Histéresis')
                    axs[1].grid(True)
                    
                    # Efectos de Carga
                    sns.boxplot(x='Carga', y='Area_Histeresis_Exp', data=df_resultados, ax=axs[2])
                    axs[2].set_title('Efecto de Carga en Área de Histéresis')
                    axs[2].grid(True)
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # Interacciones (si hay suficientes datos)
                    if len(df_resultados) >= 9:  # Mínimo para poder visualizar interacciones
                        # Interacción Amplitud-Frecuencia
                        plt.figure(figsize=(12, 8))
                        sns.lineplot(x='Amplitud', y='Area_Histeresis_Exp', hue='Frecuencia',
                                   data=df_resultados, marker='o')
                        plt.title('Interacción Amplitud-Frecuencia')
                        plt.grid(True)
                        plt.show()
                except Exception as e:
                    logging.error(f"Error en visualización: {str(e)}")
                    
        except Exception as e:
            logging.error(f"Error al cargar resultados: {str(e)}")
    
    logging.info("Análisis completado")

if __name__ == "__main__":
    # Importaciones globales
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import sys
    
    # Agregar directorio raíz al path para importar módulos locales
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Importar módulo de preprocesamiento usando alias
    import src.preprocesar_datos as pp
    
    # Ejecutar función principal
    main()
