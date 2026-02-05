# CBR.py
import numpy as np

class CBRHysteresis:
    """
    Razonador basado en casos (CBR) para ciclos de histeresis.
    Compara un vector de características con una base de casos y retorna los casos similares.
    """
    def __init__(self, case_base=None):
        if case_base is None:
            # Base de casos de ejemplo
            self.case_base = [
                {"features": np.array([40, 500, 0.1, 1000, 800, 150]), "label": "Desgaste Bajo"},
                {"features": np.array([70, 900, 0.12, 1500, 1200, 200]), "label": "Desgaste Medio"},
                {"features": np.array([120, 1500, 0.15, 2500, 2000, 300]), "label": "Desgaste Alto"}
            ]
        else:
            self.case_base = case_base

    def retrieve(self, features, threshold=100):
        """
        Busca casos en la base cuya distancia euclidiana sea menor al umbral.
        Retorna una lista de tuplas (caso, distancia) ordenadas por distancia.
        """
        similar_cases = []
        for case in self.case_base:
            dist = np.linalg.norm(features - case["features"])
            if dist < threshold:
                similar_cases.append((case, dist))
        similar_cases.sort(key=lambda x: x[1])
        return similar_cases
