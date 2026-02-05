import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Carga de datos
file_path = "penguins_size.csv"
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "larsen0966/penguins",
    file_path
)
print(df.head())
# 2. Exploración y limpieza
df.dropna(inplace=True)

# 3. Selección de características y variable objetivo
features = ["bill_length_mm", "bill_depth_mm", 
            "flipper_length_mm", "body_mass_g", 
            "island", "sex"]
X = df[features]
y = df["species"]

# 4. Codificación de variables categóricas
X = pd.get_dummies(X, columns=["island", "sex"], drop_first=True)

# 5. División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# 6. Definición del Pipeline y búsqueda de hiperparámetros
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 13],
    "knn__weights": ["uniform", "distance"]
}

grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring="accuracy", 
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 7. Evaluación
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print("Mejores hiperparámetros:", grid_search.best_params_)
print("Exactitud promedio en CV:", grid_search.best_score_)
print("Exactitud en Test:", test_accuracy)
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))
