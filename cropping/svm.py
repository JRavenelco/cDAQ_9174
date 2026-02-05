
import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -------------------------------
# PARÁMETROS DEL USUARIO
# -------------------------------

# Carpeta de los XLSX agrupados
input_folder = 'C:/Users/Carlos/Documents/Doctorado en Ciencias de la Computación/Segundo Semestre/Modelado Matemático y Simulación/Proyecto Grupal/OPT1_Dataset/cropping/caracteristicas/agrupados'  # <-- CAMBIA AQUÍ

# Carpeta de resultados
output_folder = './resultados_svm'
os.makedirs(output_folder, exist_ok=True)

# Grilla de GridSearchCV
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto', 0.01, 0.001]
}

# CV params
n_folds = 5
random_state = 42

# -------------------------------
# FUNCIONES AUXILIARES
# -------------------------------

def plot_confusion_matrix(cm, classes, title, output_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_svm_cv(df, sujeto, output_folder, resumen_rows):
    print(f'\n--- Sujeto: {sujeto} ---')

    # Features y target
    feature_columns = [col for col in df.columns if col not in ['Instancia', 'Sujeto', 'Clase']]
    X = df[feature_columns].values
    y = df['Clase'].values

    # Pipeline: scaler + SVC
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(random_state=random_state))
    ])

    # Stratified K-Fold CV
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X, y)

    # Mejor modelo
    best_model = grid_search.best_estimator_

    # Cross-validated predictions for confusion matrix
    y_pred = grid_search.best_estimator_.predict(X)
    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred, average='weighted', zero_division=0)

    print(f'Accuracy : {acc:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall   : {recall:.2f}')
    print(f'Best params: {grid_search.best_params_}')

    # Save confusion matrix
    cm_filename = os.path.join(output_folder, f'ConfMatrix_{sujeto}.png')
    plot_confusion_matrix(cm, classes=np.unique(y), title=f'Matriz de Confusión - {sujeto}', output_path=cm_filename)

    # Save resumen row
    resumen_rows.append({
        'Sujeto': sujeto,
        'Accuracy': round(acc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'Best Params': str(grid_search.best_params_)
    })

# -------------------------------
# CARGA DE DATOS
# -------------------------------

pattern = re.compile(r'(.+?)_([A-Z])\.xlsx$')

all_data = []
xlsx_files = [f for f in os.listdir(input_folder) if f.endswith('.xlsx')]

for file_name in xlsx_files:
    match = pattern.match(file_name)
    if match:
        sujeto = match.group(1)
        clase = match.group(2)

        file_path = os.path.join(input_folder, file_name)
        df = pd.read_excel(file_path)

        df['Sujeto'] = sujeto
        df['Clase'] = clase

        all_data.append(df)
    else:
        print(f'⚠️ No se pudo extraer Sujeto/Clase del archivo: {file_name}')

df_total = pd.concat(all_data, ignore_index=True)
print(f'✅ Total de instancias: {len(df_total)}')

# -------------------------------
# PARTE 1️⃣ — Clasificación POR SUJETO
# -------------------------------

print('\n===== Clasificación POR SUJETO =====')

resumen_rows_sujeto = []

for sujeto in df_total['Sujeto'].unique():
    df_sujeto = df_total[df_total['Sujeto'] == sujeto]
    run_svm_cv(df_sujeto, sujeto, output_folder, resumen_rows_sujeto)

# Guardar resumen POR SUJETO
resumen_df_sujeto = pd.DataFrame(resumen_rows_sujeto)
resumen_df_sujeto.to_excel(os.path.join(output_folder, 'Resumen_Sujeto_SVM.xlsx'), index=False)

# -------------------------------
# PARTE 2️⃣ — Clasificación GLOBAL
# -------------------------------

print('\n===== Clasificación GLOBAL (todos los sujetos) =====')

resumen_rows_global = []

run_svm_cv(df_total, 'GLOBAL', output_folder, resumen_rows_global)

# Guardar resumen GLOBAL
resumen_df_global = pd.DataFrame(resumen_rows_global)
resumen_df_global.to_excel(os.path.join(output_folder, 'Resumen_Global_SVM.xlsx'), index=False)

print('\n🎉 Proceso de clasificación SVM completado.')
