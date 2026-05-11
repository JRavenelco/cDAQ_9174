# INSTRUCCIONES PARA GEMINI DEEP RESEARCH

## 🎯 OBJETIVO DE LA INVESTIGACIÓN

Necesito encontrar métodos para implementar **control sensorless 100%** de un levitador magnético, eliminando completamente la dependencia del sensor óptico de posición.

---

## 📁 ARCHIVOS QUE DEBES DARLE A GEMINI

### Archivos principales (en orden de importancia):

1. **`REPORTE_ESTADO_REAL_PARA_DEEPRESEARCH.md`**
   - Estado actual honesto del proyecto
   - Problemas identificados
   - Preguntas de investigación

2. **`CODIGO_PARA_DEEPRESEARCH.txt`**
   - Código C++ del control PID (funciona)
   - Código del observador (requiere sensor)
   - Red KAN en Python
   - Datos reales del hardware

3. **`MONIT.txt`** (primeras 100 líneas)
   - Datos reales capturados del hardware
   - Muestra el comportamiento del sistema

---

## 📝 PROMPT PARA GEMINI DEEP RESEARCH

Copia y pega esto en Gemini:

```
Estoy desarrollando un sistema de control sensorless para un levitador magnético (masa 18g). 

ESTADO ACTUAL:
- Tengo un observador basado en integración de flujo magnético (Fórmula de Santana 2023)
- El observador funciona pero REQUIERE el 20% del sensor óptico para:
  1. Inicialización (no sé y₀ inicial)
  2. Corrección de drift del integrador
  3. Fallback cuando corriente es baja

MODELO DEL SISTEMA:
- Inductancia: L(y) = K₀ + K/(1 + y/a), donde K₀=0.0657H, K=0.0393H, a=0.00498m
- Ecuación eléctrica: u = R·i + dφ/dt, donde φ = L(y)·i
- Fuerza magnética: F = (K·i²)/(2a·(1 + y/a)²)
- Equilibrio: m·g = F_mag

FÓRMULA DEL OBSERVADOR (que requiere sensor):
  y = (a·K·i)/(φ - K₀·i) - a
  donde φ = ∫(u - R·i)dt

PROBLEMAS A RESOLVER:
1. ¿Cómo INICIALIZAR el observador sin conocer y₀?
2. ¿Cómo CORREGIR EL DRIFT del integrador sin sensor?
3. ¿Cómo usar PINN/KAN para reemplazar la corrección del sensor?

BUSCA:
- Métodos de inicialización de observadores de flujo sin sensor
- Técnicas de corrección de drift en integradores electromagnéticos
- Physics-Informed Neural Networks para estimación de estados
- Observadores sensorless para levitadores/bearings magnéticos
- Self-sensing en actuadores electromagnéticos
- Sliding Mode Observers o High-Gain Observers para este tipo de sistemas

DAME:
1. Algoritmos específicos con ecuaciones
2. Pseudocódigo implementable en C++
3. Parámetros típicos y cómo ajustarlos
4. Referencias a papers relevantes
5. Ejemplos de implementaciones exitosas

RESTRICCIONES:
- Tiempo real: 10ms por ciclo de control
- Solo mido: corriente i(t) y voltaje u(t)
- NO puedo usar el sensor óptico de posición
- Implementación final en C++ (no Python)
```

---

## 🔍 TÉRMINOS DE BÚSQUEDA SUGERIDOS

Si Gemini pide más contexto, usa estos términos:

### En inglés:
```
- "sensorless magnetic levitation control"
- "flux observer electromagnetic actuator"
- "integrator drift compensation magnetic bearing"
- "self-sensing electromagnetic actuator"
- "physics-informed neural network state estimation"
- "Kolmogorov-Arnold network physical constraints"
- "high-gain observer nonlinear magnetic system"
- "sliding mode observer magnetic levitation"
- "inductance-based position estimation"
- "model-based sensorless control levitation"
```

### En español:
```
- "control sensorless levitador magnético"
- "observador de flujo actuador electromagnético"
- "compensación drift integrador"
- "estimación de posición sin sensor"
- "redes neuronales informadas por física"
```

---

## ❓ PREGUNTAS ESPECÍFICAS PARA DEEP RESEARCH

### Pregunta 1: Inicialización
```
¿Cómo inicializar un observador de flujo magnético para levitación 
sin conocer la posición inicial?

El flujo inicial es φ₀ = L(y₀)·i₀, pero sin sensor no conozco y₀.

Posibles enfoques:
- Usar el punto de equilibrio magnético
- Barrido de corriente para estimar L(y)
- Múltiples hipótesis y convergencia
- Observador adaptativo que estima y₀
```

### Pregunta 2: Drift del integrador
```
¿Cómo corregir el drift acumulativo en φ = ∫(u - R·i)dt sin sensor?

El problema es que errores en R causan drift sistemático en φ.
Sin corrección externa, el observador diverge en segundos.

Posibles enfoques:
- Estimador de bias del integrador
- Reset periódico en puntos de equilibrio conocidos
- Filtro de Kalman con modelo de drift
- PINN con restricción de Kirchhoff
```

### Pregunta 3: KAN-PINN para sensorless
```
¿Cómo diseñar una red KAN/PINN que estime posición sin sensor?

Restricciones físicas disponibles:
1. Kirchhoff: u = R·i + dφ/dt
2. Flujo: φ = L(y)·i
3. Equilibrio: m·g = F_mag(y,i)

La red debe:
- Entrada: solo i(t), u(t) y sus historias
- Salida: y(t) estimada
- Respetar las restricciones físicas
- Funcionar en tiempo real (10ms)
```

### Pregunta 4: Observadores alternativos
```
¿Qué observadores de estado han funcionado para levitadores 
magnéticos completamente sensorless?

Investigar:
- High-Gain Observers
- Sliding Mode Observers (SMO)
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Neural Network Observers
- Hybrid Observers (modelo + datos)
```

---

## 📊 DATOS TÉCNICOS ADICIONALES

### Parámetros del sistema (para que Gemini los tenga):
```
Masa:           m = 0.018 kg
Gravedad:       g = 9.81 m/s²
Inductancia:    L(y) = 0.0657 + 0.0393/(1 + y/0.00498) [H]
Resistencia:    R ≈ 16 Ω (varía con temperatura)
Rango posición: 0.1 - 20 mm
Rango corriente: 0 - 0.83 A
Rango voltaje:  0 - 9.86 V
Período control: Ts = 10 ms
```

### Métricas actuales (con sensor):
```
- MAE del observador: 0.010 mm (con 20% sensor)
- Correlación: 1.0 (con 20% sensor)
- Sin el sensor: DIVERGE en segundos
```

---

## ✅ QUÉ ESPERAR DE GEMINI

El documento de respuesta debería incluir:

1. **Algoritmos de inicialización** sin sensor
   - Ecuaciones matemáticas
   - Pseudocódigo
   - Parámetros a ajustar

2. **Métodos de corrección de drift**
   - Técnicas probadas en literatura
   - Implementación práctica
   - Pros y contras de cada método

3. **Arquitectura PINN/KAN recomendada**
   - Estructura de la red
   - Funciones de pérdida con restricciones físicas
   - Entrenamiento online vs offline

4. **Observadores alternativos**
   - Comparativa de métodos
   - Cuál es más adecuado para este sistema
   - Ejemplo de implementación

5. **Referencias a papers**
   - Implementaciones exitosas de sensorless
   - Estado del arte en el tema

---

## 🔄 DESPUÉS DE LA INVESTIGACIÓN

Una vez que Gemini devuelva el documento:

1. **Compártelo conmigo** (Cascade)
2. **Identificaremos** el método más prometedor
3. **Implementaremos** en C++ para tiempo real
4. **Probaremos** en hardware real
5. **Iteraremos** hasta lograr sensorless 100%

---

**Fecha**: 23 de Diciembre de 2025
**Objetivo**: Sensorless 100% sin dependencia del sensor óptico
