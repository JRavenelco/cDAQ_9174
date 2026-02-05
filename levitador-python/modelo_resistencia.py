"""
Modelo Dinámico de Resistencia (Fusión de Sensores)
===================================================

Este módulo implementa un estimador robusto de resistencia que combina:
1. Modelo Térmico (Predicción):
   dR/dt = alpha * (R * i^2) - beta * (R - R_amb)
   - Captura el calentamiento rápido por efecto Joule.
   - Captura el enfriamiento lento hacia temperatura ambiente.

2. Ley de Ohm Promedio (Corrección):
   R_med = U_avg / I_avg
   - Corrige la deriva del modelo a largo plazo.
   - Solo se activa cuando la corriente es suficiente (|i| > umbral).

Arquitectura:
- Estilo "Complementary Filter" o Observador simple.
- Diseñado para correr en un lazo más lento (decimado) o integrar suavemente.

Autor: Cascade (siguiendo especificaciones del Usuario)
"""

import numpy as np

class EstimadorResistencia:
    def __init__(self, R0=2.72, alpha=0.05, beta=0.01, dt=0.01, gain_fusion=0.05):
        """
        Args:
            R0: Resistencia inicial/ambiente [Ohms]
            alpha: Coeficiente de calentamiento [Ohms/(Watt*s)]
            beta: Coeficiente de enfriamiento [1/s] (beta = 1/tau_termica)
            dt: Paso de tiempo del ciclo principal [s]
            gain_fusion: Ganancia para la corrección con Ley de Ohm (0.0 a 1.0)
        """
        # Estado
        self.R = R0
        self.R_amb = R0
        
        # Parámetros Modelo Térmico
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        
        # Parámetros Fusión
        self.gain = gain_fusion
        self.i_min_correction = 0.2  # Corriente mínima para confiar en Ohm [A]
        
        # Filtros para corrección robusta (suavizado de U e I)
        self.u_avg = 0.0
        self.i_avg = 0.0
        self.tau_filter = 0.5  # Constante de tiempo para promediar U e I [s]
        self.alpha_filter = dt / self.tau_filter
        
        # Control de ejecución (decimación interna opcional)
        self.steps = 0
        self.decimation = 10  # Actualizar corrección cada N pasos
        
    def update(self, u, i):
        """
        Paso de actualización del estimador.
        Llamar en cada ciclo de control.
        
        Args:
            u: Voltaje instantáneo [V]
            i: Corriente instantánea [A]
            
        Returns:
            R_est: Resistencia estimada actualizada
        """
        # 1. PREDICCIÓN (Modelo Térmico) - Se ejecuta siempre para continuidad
        # Potencia instantánea aprox (P = R*i^2)
        power = self.R * (i**2)
        
        # Dinámica: dR = (Calentamiento - Enfriamiento) * dt
        dR = (self.alpha * power - self.beta * (self.R - self.R_amb)) * self.dt
        self.R += dR
        
        # Clamp de seguridad física
        self.R = max(self.R_amb, min(self.R, self.R_amb * 1.6))
        
        # 2. MEDICIÓN / CORRECCIÓN (Ley de Ohm) - Lazo lento/condicional
        # Actualizar promedios móviles (filtro pasabajas)
        self.u_avg = (1 - self.alpha_filter) * self.u_avg + self.alpha_filter * u
        self.i_avg = (1 - self.alpha_filter) * self.i_avg + self.alpha_filter * i
        
        self.steps += 1
        if self.steps % self.decimation == 0:
            # Solo corregir si hay suficiente excitación (evitar división por ruido)
            if abs(self.i_avg) > self.i_min_correction:
                # Calcular R aparente (cuasi-estática)
                # Nota: En dinámica rápida, u = Ri + Ldi/dt.
                # Si filtramos (promediamos), el término Ldi/dt tiende a cero si la corriente oscila o es constante.
                R_meas = self.u_avg / self.i_avg
                
                # Validar coherencia básica antes de fusionar
                if 0.5 * self.R_amb < R_meas < 2.0 * self.R_amb:
                    # Fusión: R = (1-K)*R_pred + K*R_meas
                    # O corrección de error: R += K * (R_meas - R)
                    self.R += self.gain * (R_meas - self.R)
        
        return self.R

    def reset(self):
        self.R = self.R_amb
        self.u_avg = 0.0
        self.i_avg = 0.0
        self.steps = 0
