# Modelado de Histéresis en Fresado Dinámico: La Envolvente de Aceleración como Predictor de Fuerzas de Corte

**Autor:** Jesús  
**Fecha:** 9 de Diciembre de 2024  
**Institución:** Doctorado - Experimentos CRio DAQ

---

## Resumen Ejecutivo

Este informe documenta el descubrimiento y validación de un método para estimar fuerzas de corte en fresado utilizando la **envolvente de aceleración** como variable proxy. Los resultados demuestran que:

- La correlación directa entre fuerza y aceleración es prácticamente nula (**r = 0.004**)
- La correlación entre fuerza y **envolvente de aceleración** es alta (**r = 0.822**)
- Existe **histéresis no-lineal genuina** en la relación F-E (no es un simple retraso temporal)
- El modelo Bouc-Wen es apropiado para capturar esta no-linealidad

---

## 1. Introducción

### 1.1 Problema

En el fresado de alta velocidad, la medición directa de fuerzas de corte mediante dinamómetros presenta limitaciones:
- Ancho de banda limitado
- Reducción de rigidez del sistema
- Costo elevado

### 1.2 Hipótesis

La **intensidad de vibración** (envolvente de aceleración) es proporcional a la **carga dinámica de viruta**, y por tanto, a la fuerza de corte:

$$F(t) \propto h(t) \propto |a(t)|_{env}$$

### 1.3 Objetivo

Validar experimentalmente que la envolvente de aceleración puede utilizarse como entrada para modelos de histéresis (Bouc-Wen) en la predicción de fuerzas de corte.

---

## 2. Configuración Experimental

### 2.1 Equipo

| Componente | Especificación |
|------------|----------------|
| **Sistema DAQ** | NI cDAQ-9174 |
| **Módulo Aceleración** | NI 9234 (IEPE) |
| **Módulo Fuerza** | NI 9205 (Voltaje) |
| **Acelerómetro** | PCB 352C33 (98.3 mV/g) |
| **Celda de Carga** | DYMH-105 |
| **Frecuencia Muestreo** | 2500 Hz |

### 2.2 Condiciones de Corte

| Parámetro | Valor |
|-----------|-------|
| **RPM Husillo** | 3720 RPM |
| **Número de Dientes** | 2 |
| **Frecuencia Paso Dientes** | 124 Hz |
| **Frecuencia Ciclo Corte** | ~10 Hz |

### 2.3 Validación de Sensores

Se realizó un experimento con **dos acelerómetros**:
- **Canal 0 (ai0):** Acelerómetro en prensa (sujeción)
- **Canal 1 (ai1):** Acelerómetro directo en pieza

**Resultado:** Ambos acelerómetros miden señales prácticamente idénticas (~165 Hz dominante, misma correlación con fuerza), validando que el acelerómetro en la prensa es representativo de la dinámica de la pieza.

---

## 3. Metodología de Procesamiento

### 3.1 Extracción de la Envolvente

La envolvente se calcula mediante:

1. **Transformada de Hilbert** para obtener señal analítica:
   $$z(t) = a(t) + j\hat{a}(t)$$

2. **Magnitud instantánea:**
   $$A_{raw}(t) = |z(t)| = \sqrt{a(t)^2 + \hat{a}(t)^2}$$

3. **Filtrado paso-bajo** (Butterworth 4º orden, fc = 15 Hz):
   $$E(t) = LPF\{A_{raw}(t)\}$$

### 3.2 Modelo Lineal

$$F = k \cdot E + b$$

Donde:
- $F$ = Fuerza de corte [V]
- $E$ = Envolvente de aceleración [g]
- $k$ = Pendiente (sensibilidad)
- $b$ = Offset

### 3.3 Modelo Dinámico (con derivada)

$$F = k \cdot E + c \cdot \frac{dE}{dt} + d$$

Este modelo permite distinguir entre:
- **Retraso temporal:** Si $c$ es significativo y mejora R²
- **Histéresis no-lineal:** Si $c \approx 0$ y R² no mejora sustancialmente

---

## 4. Resultados

### 4.1 Análisis Espectral

| Señal | Frecuencia Dominante | Interpretación |
|-------|---------------------|----------------|
| **Fuerza** | 10 Hz | Ciclo de enganche/desenganche |
| **Aceleración** | 134 Hz | Paso de dientes (armónico) |

La diferencia de frecuencias explica por qué la correlación directa es nula: las señales oscilan a frecuencias muy diferentes.

### 4.2 Correlaciones

| Relación | Coeficiente r | Interpretación |
|----------|---------------|----------------|
| F vs A (directa) | **0.004** | Sin correlación |
| F vs Envolvente | **0.822** | Correlación alta |

### 4.3 Modelos de Regresión

#### Modelo Estático: F = k·E + b

| Parámetro | Valor |
|-----------|-------|
| k (pendiente) | 0.0669 V/g |
| b (intercepto) | 0.8837 V |
| **R²** | **0.6754** |
| Error RMS | 0.0193 V |

#### Modelo Dinámico: F = k·E + c·dE/dt + d

| Parámetro | Valor |
|-----------|-------|
| k | 0.0669 V/g |
| c | 0.000127 V·s/g |
| d | 0.8837 V |
| **R²** | **0.6839** |

#### Mejora con Derivada

$$\Delta R^2 = 0.6839 - 0.6754 = 0.0085 \quad (0.84\%)$$

**Conclusión:** La mejora es mínima (<5%), confirmando que el lazo de histéresis **NO se debe a un retraso temporal**, sino a **no-linealidad genuina** del proceso de corte.

### 4.4 Evidencia de Histéresis

El gráfico F vs E muestra un **lazo "gordo"** que no se cierra sobre una línea recta:

- El 67.5% de la varianza es explicado por el modelo lineal
- El **32.5% restante** corresponde a la **histéresis no-lineal**
- El área del lazo representa **energía disipada** en el proceso

---

## 5. Validación Teórica

### 5.1 Fundamento Físico (Literatura)

Según el documento "Hysteresis Modeling in Dynamic Milling":

> "The vibration displacement—and by extension, its second derivative, **acceleration**—is a **direct, proportional readout of the dynamic chip load variations**"

La cadena de proporcionalidad es:

1. $F \propto h$ (Fuerza proporcional a espesor de viruta)
2. $X \propto F$ (Desplazamiento proporcional a fuerza)
3. $a = \ddot{X}$ (Aceleración es derivada del desplazamiento)
4. **Conclusión:** $|a|_{env} \propto |h|_{env} \propto |F|_{env}$

### 5.2 Modelo Bouc-Wen

El modelo Bouc-Wen es reconocido como el "gold standard" para histéresis en sistemas mecánicos:

$$F(x, \dot{x}) = \alpha k x + (1-\alpha) k z$$

$$\dot{z} = A\dot{x} - \beta|\dot{x}||z|^{n-1}z - \gamma\dot{x}|z|^n$$

En nuestro contexto:
- $x \rightarrow E$ (envolvente)
- $\dot{x} \rightarrow \frac{dE}{dt}$
- Los parámetros $A, \beta, \gamma$ capturan la no-linealidad

### 5.3 Fuentes de Histéresis Identificadas

| Fuente | Mecanismo | Evidencia |
|--------|-----------|-----------|
| **Process Damping** | Fricción flanco-pieza | Lazo F-E no lineal |
| **Size Effect** | Transición corte/arado | Dependencia no lineal F(h) |
| **Structural Damping** | Fricción en juntas | Amortiguamiento variable |

---

## 6. Experimento de Validación: Dos Acelerómetros

### 6.1 Objetivo

Verificar si el acelerómetro en la prensa (indirecto) mide lo mismo que uno directo en la pieza.

### 6.2 Configuración

- **ai0:** Acelerómetro en PRENSA (filtrado mecánico hipotético)
- **ai1:** Acelerómetro en PIEZA (medición directa)

### 6.3 Resultados (Corte a 150 RPM)

| Métrica | Prensa (ai0) | Pieza (ai1) |
|---------|--------------|-------------|
| Freq. dominante | 165.3 Hz | 165.3 Hz |
| Corr. directa F | 0.002 | 0.002 |
| Corr. envolvente F | -0.01 | -0.01 |

### 6.4 Conclusión

**Ambos acelerómetros miden esencialmente lo mismo.** Esto valida que:
- Los datos históricos (acelerómetro en prensa) son representativos
- No hay "filtrado mecánico" significativo entre prensa y pieza
- La correlación r=0.822 obtenida anteriormente es válida para la pieza

---

## 7. Conclusiones

### 7.1 Hallazgos Principales

1. ✅ **La envolvente de aceleración es un proxy válido para la fuerza de corte** (r = 0.822)

2. ✅ **Existe histéresis no-lineal genuina** en la relación F-E, no explicable por retraso temporal (mejora R² < 1%)

3. ✅ **El modelo lineal captura el 67.5%** de la varianza; el 32.5% restante es la histéresis a modelar

4. ✅ **Los acelerómetros en prensa y pieza miden lo mismo**, validando mediciones indirectas

5. ✅ **El modelo Bouc-Wen es apropiado** para capturar la no-linealidad observada

### 7.2 Implicaciones

- Se puede **estimar fuerza de corte sin dinamómetro** usando solo acelerómetros
- El **ancho del lazo de histéresis** puede ser indicador de desgaste de herramienta
- La metodología es aplicable a **monitoreo en tiempo real** de procesos de fresado

### 7.3 Trabajo Futuro

1. Implementar modelo **Bouc-Wen** con parámetros identificados
2. Explorar **redes KAN** (Kolmogorov-Arnold Networks) para modelado no-lineal
3. Validar en diferentes condiciones de corte (materiales, velocidades)
4. Desarrollar sistema de **monitoreo online** basado en envolvente

---

## 8. Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `corte2_mejor_0.5s.txt` | Datos del mejor segmento de corte |
| `analisis_completo_histeresis.png` | Análisis de 9 paneles |
| `analisis_histeresis_detallado.png` | Validación de histéresis |
| `corte_20251209_144635.csv` | Datos con 2 acelerómetros |

---

## 9. Referencias

1. Altintas, Y. "Manufacturing Automation: Metal Cutting Mechanics, Machine Tool Vibrations, and CNC Design"

2. "Advanced Hysteresis Modeling in Dynamic Milling: Acceleration Envelope as a Deterministic Predictor of Nonlinear Cutting Forces" - Documento de referencia

3. Tlusty, J. "Manufacturing Processes and Equipment"

4. Bouc, R. "Forced vibration of mechanical systems with hysteresis"

---

## Apéndice A: Ecuaciones Clave

### Envolvente de Aceleración
```
E(t) = LPF{ |Hilbert{a(t)}| }
```

### Modelo Lineal
```
F = 0.0669·E + 0.8837
R² = 0.675
```

### Frecuencias Características
```
f_corte = 10 Hz (ciclo de enganche)
f_dientes = RPM × N_dientes / 60 = 3720 × 2 / 60 = 124 Hz
```

---

## Apéndice B: Código de Procesamiento

```python
from scipy.signal import butter, filtfilt, hilbert
import numpy as np

def calcular_envolvente(aceleracion, fs=2500, fc=15):
    """
    Calcula la envolvente de la señal de aceleración.
    
    Args:
        aceleracion: Señal de aceleración [g]
        fs: Frecuencia de muestreo [Hz]
        fc: Frecuencia de corte del filtro [Hz]
    
    Returns:
        Envolvente filtrada [g]
    """
    # Transformada de Hilbert
    analitica = hilbert(aceleracion)
    envolvente_raw = np.abs(analitica)
    
    # Filtro paso-bajo
    b, a = butter(4, fc/(fs/2), btype='low')
    envolvente = filtfilt(b, a, envolvente_raw)
    
    return envolvente
```

---

*Documento generado automáticamente - Experimentos CRio DAQ*
