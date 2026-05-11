# REPORTE ESTADO REAL: LEVITADOR MAGNÉTICO - PARA DEEP RESEARCH

## 🎯 OBJETIVO FINAL (NO LOGRADO AÚN)
Control sensorless 100% de un levitador magnético usando SOLO corriente y voltaje, sin sensor óptico de posición.

---

## 📊 ESTADO ACTUAL HONESTO

### ✅ LO QUE FUNCIONA

| Componente | Estado | Descripción |
|------------|--------|-------------|
| `levitador.cpp` | ✅ FUNCIONA | Control PID con sensor óptico (baseline) |
| `levitador_obs_minimal.cpp` | ✅ FUNCIONA | Observador en paralelo (80% obs + 20% sensor) |
| Integrador de Flujo | ✅ FUNCIONA | φ = ∫(u - R·i)dt con método trapecio |
| Estimador R Adaptativo | ✅ FUNCIONA | R_est converge, drift ≈ 0.16 Ω |
| Fórmula 2023 Santana | ⚠️ PARCIAL | y = (a·k·i)/(φ - k₀·i) - a, requiere fusión con sensor |
| Red KAN (Python) | ✅ FUNCIONA | Entrenamiento offline, mejora 20.9% |

### ❌ LO QUE NO FUNCIONA

| Componente | Problema |
|------------|----------|
| **Sensorless 100%** | NO PROBADO - El observador diverge sin fusión con sensor |
| **Fórmula 2023 pura** | DIVERGE - Sin el 20% del sensor, la estimación diverge |
| **KAN en C++** | NO CAPTURA DATOS - Problema de lectura serial |
| **Inicialización** | REQUIERE SENSOR - El observador necesita y_sensor inicial |

### ⚠️ EL PROBLEMA CENTRAL

```
El observador de Santana 2023 tiene DOS problemas no resueltos:

1. INICIALIZACIÓN: Necesita conocer y_sensor inicial para calcular φ₀
   φ₀ = L(y₀)·i₀ = [K₀ + K/(1 + y₀/a)]·i₀
   Sin y₀, no puede arrancar correctamente.

2. DRIFT DEL INTEGRADOR: La integración ∫(u - R·i)dt acumula errores
   - Error en R → error en φ → error en y
   - Sin corrección externa, diverge en segundos
   
SOLUCIÓN ACTUAL (NO ES SENSORLESS):
   - Fusión 80% observador + 20% sensor
   - Fallback a sensor cuando i < 0.05 A
   - Esto FUNCIONA pero NO es sensorless 100%
```

---

## 📐 MODELO FÍSICO DEL SISTEMA

### Parámetros del Levitador
```
Masa:         m = 18 g = 0.018 kg
Gravedad:     g = 9.81 m/s²
Inductancia:  L(y) = K₀ + K/(1 + y/a)
  K₀ = 0.0657 H (inductancia cuando y → ∞)
  K = 0.0393 H (inductancia diferencial)
  a = 0.00498 m (parámetro de geometría)
Resistencia:  R ≈ 16 Ω (varía con temperatura)
```

### Ecuaciones del Sistema
```
Ecuación eléctrica:
  u(t) = R·i(t) + dφ/dt
  φ = L(y)·i

Ecuación mecánica:
  m·ÿ = m·g - F_mag
  F_mag = (1/2)·(dL/dy)·i² = (K·i²)/(2a·(1 + y/a)²)

Equilibrio magnético:
  m·g = (K·i_eq²)/(2a·(1 + y_eq/a)²)
```

### Fórmula del Observador 2023 (Santana)
```
Despejando y del flujo:
  φ = L(y)·i = [K₀ + K/(1 + y/a)]·i

Manipulando:
  φ/i = K₀ + K/(1 + y/a)
  φ/i - K₀ = K/(1 + y/a)
  1 + y/a = K/(φ/i - K₀)
  y = a·[K/(φ/i - K₀) - 1]
  y = a·K·i/(φ - K₀·i) - a

Forma final:
  y = (a·K·i)/(φ - K₀·i) - a
```

---

## 📁 ARCHIVOS CLAVE PARA DEEP RESEARCH

### 1. Código C++ que FUNCIONA (baseline)

**`levitador.cpp`** - Control PID con sensor (ESTABLE)
```cpp
// Loop principal que funciona
while(!_kbhit()) { 
    ReadFile(h,&chRead,1,&dwRead,NULL);
    if(dwRead==1) {        
        dato[flagcom]=chRead;
        if(flagcom==0 && dato[0]!=(char)0xAA){
            flagcom=0;
        }
        else{
            flagcom++;
            if(flagcom==6){
                pv =((int)dato[1]<<8)+(int)dato[2];
                icte=((int)dato[3]<<8)+(int)dato[4];
                
                if(pv<=1023 && icte<=1023){
                    y=esc*pv;        // Posición del SENSOR
                    ie=esci*icte;    // Corriente medida
                    
                    // PID usa SENSOR
                    ef=yd-y;
                    // ... control PID ...
                }
            }
        }
    }
}
```

### 2. Observador que REQUIERE sensor

**`levitador_obs_minimal.cpp`** - Observador con fusión
```cpp
void observador_paso(float i, float u, float y_sensor, float *y_out, float *r_out) {
    // Actualizar R adaptativo
    if (i > 0.05f && fabs(u) > 0.1f) {
        float R_meas = u / i;
        if (R_meas > 8.0f && R_meas < 24.0f) {
            obs_r = 0.98f * obs_r + 0.02f * R_meas;
        }
    }
    
    // PROBLEMA 1: Inicialización requiere sensor
    if (!obs_init && i > 0.03f) {
        obs_init = 1;
        obs_y_est = y_sensor;  // ⚠️ USA SENSOR
        float L_init = K0 + K / (1.0f + obs_y_est / A);
        obs_phi = L_init * i;
    }
    
    if (!obs_init) {
        *y_out = y_sensor;  // ⚠️ USA SENSOR
        *r_out = obs_r;
        return;
    }
    
    // Integrar flujo (trapecio)
    float z_now = u - obs_r * i;
    float z_prev = obs_u_prev - obs_r * obs_i_prev;
    float dphi = 0.5f * (z_now + z_prev) * Ts;
    obs_phi += dphi;
    
    // Fórmula 2023
    float denom = obs_phi - K0 * i;
    if (fabs(denom) > 1e-6f && i > 0.05f) {
        float y_new = (A * K * i) / denom - A;
        if (y_new > 0.0005f && y_new < 0.022f) {
            // PROBLEMA 2: Fusión con sensor para evitar drift
            obs_y_est = 0.8f * y_new + 0.2f * y_sensor;  // ⚠️ USA SENSOR
        }
    } else {
        obs_y_est = y_sensor;  // ⚠️ USA SENSOR
    }
    
    *y_out = obs_y_est;
    *r_out = obs_r;
}
```

### 3. Red KAN entrenada (Python)

**`entrenar_kan_hibrido.py`** - Arquitectura [3, 32, 1]
```python
# Red que aprende a corregir residuos de la Fórmula 2023
# Entrada: [φ, i, y_formula]
# Salida: residuo (corrección)

class KANHibrido:
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=1):
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
    
    def forward(self, X):
        h = X @ self.w1 + self.b1
        h = np.maximum(0, h)  # ReLU
        y = h @ self.w2 + self.b2
        return y

# Resultado: Mejora 20.9% sobre Fórmula 2023 pura
# PERO: Solo funciona offline, no probado en tiempo real
```

---

## 📊 DATOS CAPTURADOS

### Formato de MONIT.txt (8 columnas)
```
t       yd      y_sensor    y_obs   R_est   ied     ie      u
0.0100  0.0050  0.0001      0.0050  16.00   0.0000  0.0067  0.0000
0.0200  0.0050  0.0001      0.0050  16.00   0.0000  0.0067  0.0000
...
```

### Estadísticas de los datos
```
Muestras: 5894 (58.9 segundos)
Rango y_sensor: 0.0001 - 0.0202 m (0.1 - 20.2 mm)
Rango ie: 0.0067 - 0.8176 A
Rango u: 0.0 - 9.86 V
Rango R_est: 16.0 - 16.16 Ω
```

---

## ❓ PREGUNTAS PARA DEEP RESEARCH

### Pregunta 1: Inicialización sin sensor
```
¿Cómo inicializar el observador de flujo magnético sin conocer 
la posición inicial y₀?

Contexto:
- El flujo inicial es φ₀ = L(y₀)·i₀
- Sin y₀, no podemos calcular L(y₀)
- El sistema arranca con el imán en posición desconocida

Posibles enfoques a investigar:
- Estimación de φ₀ desde equilibrio magnético
- Múltiples hipótesis de y₀ y convergencia
- Observador de orden reducido
- Arranque con barrido de corriente
```

### Pregunta 2: Corrección de drift sin sensor
```
¿Cómo corregir el drift del integrador de flujo sin usar 
el sensor de posición?

Contexto:
- φ = ∫(u - R·i)dt acumula errores
- Error en R → error sistemático en φ
- Sin corrección, el observador diverge en segundos

Posibles enfoques a investigar:
- Physics-Informed Neural Networks (PINN) con restricción de Kirchhoff
- Filtro de Kalman Extendido con modelo físico
- Observador de Luenberger con modelo no-lineal
- Reset periódico del integrador en puntos conocidos
- Estimador de bias del integrador
```

### Pregunta 3: KAN-PINN para sensorless
```
¿Cómo usar una red KAN con restricciones físicas (PINN) para 
estimar posición sin sensor?

Contexto:
- Red KAN actual: [3, 32, 1] entrada [φ, i, y_formula]
- Mejora 20.9% sobre Fórmula 2023
- PERO requiere y_formula que depende del sensor

Posibles enfoques a investigar:
- PINN con pérdida de Kirchhoff: L_física = ||u - R·i - dφ/dt||²
- PINN con pérdida de equilibrio: L_eq = ||m·g - F_mag||²
- Entrenamiento online vs offline
- Transferencia de pesos Python → C++
```

### Pregunta 4: Observadores alternativos
```
¿Qué observadores de estado funcionan para levitadores magnéticos 
sin sensor de posición?

Contexto:
- Luenberger requiere modelo linealizado (inestable)
- Kalman Extendido requiere modelo preciso de ruido
- Sliding Mode Observer (SMO) - ¿funciona para este sistema?

Investigar:
- High-Gain Observers
- Sliding Mode Observers
- Neural Network Observers
- Hybrid Observers (modelo + datos)
```

---

## 🔬 RESTRICCIONES FÍSICAS PARA PINN

### Restricción 1: Ley de Kirchhoff
```
L_kirchhoff = ||u - R·i - dφ/dt||²

Donde:
  u = voltaje medido
  R = resistencia (estimada o fija)
  i = corriente medida
  dφ/dt = derivada del flujo (de la red)
```

### Restricción 2: Relación Flujo-Inductancia
```
L_flujo = ||φ - L(y)·i||²

Donde:
  φ = flujo estimado por la red
  L(y) = K₀ + K/(1 + y/a)
  y = posición estimada por la red
  i = corriente medida
```

### Restricción 3: Equilibrio Magnético
```
L_equilibrio = ||m·g - (K·i²)/(2a·(1 + y/a)²)||²

Donde:
  m·g = fuerza gravitacional (constante)
  F_mag = fuerza magnética (función de y, i)
  En régimen permanente, deben ser iguales
```

---

## 📋 RESUMEN PARA GEMINI DEEP RESEARCH

### Dame:
1. **Archivos de código**:
   - `levitador.cpp` (baseline funcional)
   - `levitador_obs_minimal.cpp` (observador con fusión)
   - `entrenar_kan_hibrido.py` (red KAN)

2. **Datos**:
   - `MONIT.txt` (datos reales del hardware)

3. **Este documento**:
   - `REPORTE_ESTADO_REAL_PARA_DEEPRESEARCH.md`

### Busca:
```
1. Métodos de inicialización de observadores de flujo magnético 
   sin sensor de posición

2. Técnicas de corrección de drift en integradores para 
   levitadores magnéticos

3. Physics-Informed Neural Networks (PINN) aplicados a 
   estimación de estados en sistemas electromagnéticos

4. Observadores sensorless para actuadores electromagnéticos 
   (motores, levitadores, bearings magnéticos)

5. KAN (Kolmogorov-Arnold Networks) con restricciones físicas 
   para estimación de estados

6. Métodos de auto-calibración y reset de observadores 
   en sistemas de levitación magnética
```

### Devuélveme:
```
Un documento técnico con:

1. ALGORITMOS específicos para:
   - Inicialización sin sensor
   - Corrección de drift
   - PINN para estimación de posición

2. ECUACIONES y fórmulas listas para implementar

3. PSEUDOCÓDIGO o código de referencia

4. PARÁMETROS típicos y cómo ajustarlos

5. REFERENCIAS a papers y implementaciones existentes
```

---

## 🎯 OBJETIVO DE LA INVESTIGACIÓN

```
ENTRADA ACTUAL:
  - Corriente medida: i(t)
  - Voltaje aplicado: u(t)
  - Sensor óptico: y(t) [QUEREMOS ELIMINAR]

SALIDA DESEADA:
  - Posición estimada: ŷ(t) [SIN SENSOR]
  - Control PID usando SOLO ŷ(t)
  - Sistema que levita de forma estable

RESTRICCIONES:
  - Tiempo real: 10 ms por ciclo
  - Implementación en C++ (no Python)
  - Sin dependencia del sensor óptico
  - Robusto a variaciones de R (temperatura)
```

---

**Fecha**: 23 de Diciembre de 2025
**Estado**: PENDIENTE - Sensorless 100% no logrado
**Siguiente paso**: Deep Research para encontrar solución
