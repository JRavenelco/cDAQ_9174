
import unittest
import numpy as np
from modelo_resistencia import EstimadorResistencia

class TestEstimadorResistencia(unittest.TestCase):
    def test_inicializacion(self):
        est = EstimadorResistencia(R0=2.72)
        self.assertEqual(est.R, 2.72)
        self.assertEqual(est.R_amb, 2.72)

    def test_calentamiento_puro(self):
        # Desactivar fusión (gain=0) para probar solo modelo térmico
        # Usar alpha menor para no saturar el clamp (R_max = 2.72 * 1.6 = 4.35)
        est = EstimadorResistencia(R0=2.72, alpha=0.1, beta=0.0, dt=1.0, gain_fusion=0.0)
        
        # Aplicar corriente de 1A durante 1s
        # Power = R * i^2 = 2.72 * 1 = 2.72
        # dR = alpha * Power = 0.1 * 2.72 = 0.272
        # R_expected = 2.72 + 0.272 = 2.992
        est.update(u=10, i=1.0)
        self.assertAlmostEqual(est.R, 2.992, places=3)

    def test_enfriamiento_puro(self):
        # Desactivar fusión
        est = EstimadorResistencia(R0=2.72, alpha=0.0, beta=0.5, dt=1.0, gain_fusion=0.0)
        est.R = 4.72 # R > R_amb
        
        # dR = -beta * (R - R_amb) = -0.5 * (4.72 - 2.72) = -1.0
        # R_expected = 4.72 - 1.0 = 3.72
        est.update(u=0, i=0)
        self.assertAlmostEqual(est.R, 3.72, places=2)
    
    def test_fusion_sensor(self):
        # Probar que la corrección de Ohm funciona
        # alpha=0, beta=0 (sin térmica)
        # decimation=1 (corregir en cada paso para el test)
        est = EstimadorResistencia(R0=2.0, alpha=0.0, beta=0.0, dt=0.1, gain_fusion=0.5)
        est.decimation = 1 
        
        # Pre-cargar filtros promedio (para que u_avg/i_avg sean válidos rápido)
        est.u_avg = 3.0
        est.i_avg = 1.0
        
        # Situación: R_real = 3.0 (V=3, I=1), R_est = 2.0
        # Al llamar update, el filtro suavizará un poco, pero si u_avg ya es 3.0 y pasamos 3.0...
        # u_avg_new = (1-alpha)*3.0 + alpha*3.0 = 3.0
        est.update(u=3.0, i=1.0)
        
        # R_meas = 3.0 / 1.0 = 3.0
        # Error = 3.0 - 2.0 = 1.0
        # Corrección = gain * Error = 0.5 * 1.0 = 0.5
        # R_new = 2.0 + 0.5 = 2.5
        
        self.assertAlmostEqual(est.R, 2.5, places=1)

    def test_limites(self):
        est = EstimadorResistencia(R0=10.0)
        est.R = 5.0 
        # Al actualizar, el clamp debe forzar R >= R_amb
        est.update(0, 0)
        self.assertTrue(est.R >= 10.0)

if __name__ == '__main__':
    unittest.main()
