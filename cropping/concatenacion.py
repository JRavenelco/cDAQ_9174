# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 16:55:25 2025

@author: Carlos
"""

import os
import pandas as pd
import re

# -------------------------------
# PARÁMETROS DEL USUARIO
# -------------------------------

# Ruta de la carpeta caracteristicas
features_folder = 'C:/Users/Carlos/Documents/Doctorado en Ciencias de la Computación/Segundo Semestre/Modelado Matemático y Simulación/Proyecto Grupal/OPT1_Dataset/cropping/caracteristicas/'  # <-- CAMBIA AQUÍ

# Carpeta de salida para los XLSX agrupados
output_folder = os.path.join(features_folder, 'agrupados')

# -------------------------------
# PROCESAMIENTO
# -------------------------------

# Crear carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Regex para extraer sujeto y clase
# Ejemplo de archivo: esp32_data_20250529_174958_Carlos_S1_features.csv
pattern = re.compile(r'esp32_data_\d+_\d+_(.+?)_([A-Z]\d+)_features\.csv$')

# Diccionario para agrupar los DataFrames
group_dict = {}

# Obtener todos los CSV de la carpeta caracteristicas
csv_files = [f for f in os.listdir(features_folder) if f.endswith('_features.csv')]

for file_name in csv_files:
    match = pattern.search(file_name)
    if match:
        sujeto = match.group(1)
        clave_completa = match.group(2)
        clase = clave_completa[0]  # letra de la clase
        
        # Leer el archivo
        file_path = os.path.join(features_folder, file_name)
        df = pd.read_csv(file_path)
        
        # Renombrar columna 'Ventana' a 'Instancia'
        if 'Ventana' in df.columns:
            df = df.rename(columns={'Ventana': 'Instancia'})
        
        # Preparar clave de agrupación
        group_key = f'{sujeto}_{clase}'
        
        # Añadir al grupo
        if group_key not in group_dict:
            group_dict[group_key] = []
        group_dict[group_key].append(df)
    else:
        print(f'⚠️ No se pudo extraer Sujeto/Clase del archivo: {file_name}')

# Procesar cada grupo y guardar
for group_key, df_list in group_dict.items():
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Renumerar la columna 'Instancia' de 1 a N
    combined_df['Instancia'] = range(1, len(combined_df) + 1)
    
    # Guardar en XLSX
    output_file = os.path.join(output_folder, f'{group_key}.xlsx')
    combined_df.to_excel(output_file, index=False)
    
    print(f'✅ Archivo agrupado guardado: {output_file} ({len(combined_df)} instancias)')

print('🎉 Proceso de agrupamiento completado.')
