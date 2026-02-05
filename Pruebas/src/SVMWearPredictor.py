# SVMWearPredictor.py
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class SVMWearPredictor:
    """
    Modelo SVM para predecir el desgaste de la herramienta a partir de un vector de características.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = SVC(probability=True)
        self.is_trained = False

    def train(self, X_train, y_train):
        """
        Entrena el modelo SVM con X_train (características) y y_train (etiquetas).
        """
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

    def predict(self, features):
        """
        Retorna la predicción y probabilidad a partir del vector de características.
        Si el modelo no está entrenado, retorna una etiqueta dummy.
        """
        if not self.is_trained:
            return "No entrenado", 0.0
        X_scaled = self.scaler.transform([features])
        label = self.model.predict(X_scaled)[0]
        prob = np.max(self.model.predict_proba(X_scaled))
        return label, prob
