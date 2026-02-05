# Capítulo 4. Resultados

## 4.1 Introducción

Los resultados presentados en este capítulo se alinean con los objetivos específicos planteados en la investigación. Se documentan los avances parciales obtenidos hasta la fecha, incluyendo el desarrollo de modelos matemáticos para histéresis, la implementación de técnicas de inteligencia artificial, y la validación experimental del sistema de adquisición de datos.

---

## 4.2 Publicaciones y Difusión Científica

### 4.2.1 Artículo en Congreso IFToMM

Se redactó y presentó el artículo titulado **"Detección de histéresis F-v a partir de vibraciones: sensor virtual de fuerza y modelos Dahl/LuGre/Bouc-Wen en fresado de aluminio"** en el congreso de IFToMM.

**Contribuciones principales del artículo:**

1. **Pipeline reproducible** para detectar eventos en aceleración, reconstruir velocidad e inferir fuerza sintética mediante modelos de fricción con memoria (Dahl, LuGre, Bouc-Wen) con selección automática por índice de histéresis.

2. **Cuantificación adimensional** robusta a ganancias desconocidas, mediante área y ancho normalizados del lazo de histéresis:
   $$\tilde{E}_{\text{loop}} = \left|\oint \tilde{F} \, d\tilde{v}\right|, \quad \tilde{W}_{\text{loop}} = \max(\tilde{v}) - \min(\tilde{v})$$
   donde $\tilde{\cdot}$ denota normalización z-score.

3. **Índice de histéresis (H-index)** definido como:
   $$H = \frac{\tilde{E}_{\text{loop}}}{\tilde{W}_{\text{loop}} \cdot \text{ptp}(\tilde{F})}$$

4. **Base metodológica** para incorporar FRF/Kalman (tiempo real) y calibración absoluta de energía (J/ciclo).

**Estado actual:** El artículo recibió retroalimentación detallada y se están implementando las correcciones sugeridas. Se está transcribiendo de LaTeX a Word para facilitar la revisión colaborativa.

### 4.2.2 Reporte Técnico PINN

Se completó el documento técnico **"Modelado de levitador magnético con redes neuronales"** que documenta la metodología y resultados del modelo PINN aplicado a un sistema de levitación magnética como banco de pruebas para la validación de la arquitectura propuesta.

---

## 4.3 Sistema Experimental de Adquisición

### 4.3.1 Configuración del Hardware

Se diseñó e implementó un sistema de adquisición de datos basado en el chasis NI cDAQ-9174 con los siguientes módulos:

| Módulo | Función | Especificaciones |
|--------|---------|------------------|
| NI 9205 | Fuerza (voltaje analógico) | ±10V, diferencial, ai0-ai3, 2.5 kHz |
| NI 9234 | Vibración (IEPE) | 4 canales, 51.2 kS/s, excitación 4mA |
| NI 9219 | Celda de carga (puente) | 24-bit, 100 S/s, excitación configurable |

**Sensores utilizados:**
- **Acelerómetros:** PCB 352C33, sensibilidad 98.3-100 mV/g
- **Celda de carga:** DYMH-105 (500 kg, 1.7 mV/V) con acondicionador INA-4LC-8NTC (G=601)
- **Shaker:** TIRA TV 51144IN con amplificador BAA 1000

### 4.3.2 Interfaz de Caracterización Desarrollada

Se desarrolló una interfaz gráfica completa (`caracterizacion_sensor_fuerza.py`, 1085 líneas) para la caracterización del sensor de fuerza con las siguientes funcionalidades:

**Características implementadas:**
- Adquisición simultánea de fuerza y aceleración (2 canales)
- Visualización en tiempo real de señales, FFT y FRF
- Cálculo de transmisibilidad entre acelerómetros
- Validación F=ma integrada
- Diagramas de Lissajous para análisis de fase
- Guardado automático de experimentos con metadatos

**Resultados de validación de datos:**
```
Archivo: sesion_20250610_191323_fuerza_200_rpm_arana.csv
- Muestras: 41,900 por canal
- NaN/Inf: 0/0
- Tiempos monótonos: ✓
- Outliers (|z|>6): 339 en ai0, 0 en ai1-ai3
```

### 4.3.3 Análisis de Función de Respuesta en Frecuencia (FRF)

Se realizaron experimentos de caracterización a múltiples frecuencias de excitación:

| Frecuencia (Hz) | |H(f)| (g/V) | Fase (°) | Coherencia |
|-----------------|-------------|----------|------------|
| 5 | 12.80 | -114.1 | - |
| 10 | 19.26 | 158.5 | 0.004 |
| 20 | 0.39 | 75.9 | 0.009 |
| 40 | 0.80 | -86.5 | - |
| 50 | 8.38 | -29.2 | 0.026 |
| 70 | 10.06 | 173.9 | 0.009 |
| 80 | 6.76 | 25.4 | 0.012 |
| 120 | 10.92 | -66.9 | 0.047 |

**Observación:** La coherencia baja en algunas frecuencias indica la necesidad de mejorar la sincronización entre canales y aumentar el tiempo de adquisición para promediar más ciclos.

---

## 4.4 Modelo Matemático de Histéresis con PINN

### 4.4.1 Arquitectura del Modelo

Se implementó una Red Neuronal Informada por la Física (PINN) para modelar el comportamiento de histéresis. La arquitectura consta de:

**Modelo Base (MLP):**
- Entrada: tiempo normalizado $t$
- Capas ocultas: 3 capas × 64 neuronas
- Activación: $\tanh$ (diferenciable para AD)
- Salida: posición $\hat{y}$, corriente $\hat{i}$

**Función de pérdida total:**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{phys}}$$

donde:
$$\mathcal{L}_{\text{data}} = \text{MSE}\left(\frac{\hat{y} - y}{\sigma_y}\right) + \text{MSE}\left(\frac{\hat{i} - i}{\sigma_i}\right)$$

$$\mathcal{L}_{\text{phys}} = \text{MSE}(R_{\text{mech}}) + \text{MSE}(R_{\text{elect}})$$

**Residuales físicos:**
- **Mecánico:** $R_{\text{mech}} = m\ddot{y} - \left(\frac{1}{2}\frac{\partial L(y)}{\partial y}i^2 + mg\right)$
- **Eléctrico:** $R_{\text{elect}} = L(y)\dot{i} - \left(u - Ri - \frac{\partial L(y)}{\partial y}\dot{y}i\right)$

### 4.4.2 Resultados Comparativos

Se evaluaron tres configuraciones del modelo:

| Modelo | Variable | RMSE | R² | Mejora vs Base |
|--------|----------|------|-----|----------------|
| **Base (solo datos)** | Posición | 0.0057 m | 0.3432 | - |
| | Corriente | 0.2282 A | 0.3283 | - |
| **PINN (λ=10⁻⁴)** | Posición | 0.0049 m | 0.5105 | +48.7% |
| | Corriente | 0.2010 A | 0.4788 | +45.8% |
| **PINN (λ=10⁻³)** | Posición | 0.0049 m | 0.5058 | +47.4% |
| | Corriente | 0.1968 A | 0.5003 | +52.4% |

**Hallazgo clave:** La incorporación de restricciones físicas mejoró el R² de ~0.33 a ~0.50, demostrando que el modelo se beneficia significativamente de la información física.

### 4.4.3 Identificación de Parámetros Físicos

Se exploró la capacidad del PINN para identificar los parámetros del modelo de inductancia:

$$L(y) = k_0 + \frac{k}{1 + y/a}$$

| Parámetro | Valor Inicial | Valor Identificado | Cambio |
|-----------|---------------|-------------------|--------|
| $k_0$ (H) | 3.63×10⁻² | 9.92×10⁻³ | -72.7% |
| $k$ (H·m) | 3.50×10⁻³ | 5.23×10⁻² | +1394% |
| $a$ (m) | 5.20×10⁻³ | 4.48×10⁻³ | -13.8% |

**Resultado importante:** El modelo con parámetros entrenables mostró peor rendimiento predictivo (R² = 0.2983 vs 0.5058), sugiriendo convergencia a mínimos locales. Este hallazgo demuestra un desafío clave en PINNs: más grados de libertad no garantizan mejor rendimiento y pueden conducir a soluciones no físicas.

### 4.4.4 Arquitectura KAN-PINN Optimizada

Se desarrolló una versión eficiente basada en Kolmogorov-Arnold Networks (KAN):

**Arquitectura:**
```
Input (t) → KANLayer(1→64) → KANLayer(64→64) → KANLayer(64→64) 
          → KANLayer(64→2) → Output (y, i)
```

**Optimizaciones implementadas:**
1. Diferencias finitas para derivadas 2º orden (10x speedup)
2. Curriculum learning (transición gradual datos→física)
3. Adaptive weighting de λ_phys
4. Clipping de parámetros físicos en bounds razonables

**Tiempos de entrenamiento:**
| GPU | Tiempo | Speedup |
|-----|--------|---------|
| A100 | 35-50 min | 10-15x |
| T4 (gratis) | 1-1.5 h | 6-8x |
| GTX 1060 | 8+ h | 1x |

---

## 4.5 Comparativa DOE: PINN Clásicos vs CNN

### 4.5.1 Diseño del Experimento

Se realizó un Diseño de Experimentos (DOE) comprehensivo comparando:
- **Modelos PINN clásicos:** Bouc-Wen, LuGre, Dahl
- **Modelos CNN:** Arquitectura temporal optimizada para GPU

**Datos experimentales:** Señales de fricción a 400 Hz de frecuencia de muestreo.

### 4.5.2 Resultados del DOE

| Enfoque | R² Promedio | Desv. Est. | Tasa Éxito | Tiempo Prom. |
|---------|-------------|------------|------------|--------------|
| **CNN DOE** | **0.9716** | 0.0757 | 100% (11/11) | 37.09 s |
| PINN Clásicos | 0.0256 | 0.0541 | 24.5% | 1879.4 s |
| CNN Sesiones | -0.0496 | - | 100% | 68.1 s |

**Mejor resultado CNN:** R² = 0.999928

### 4.5.3 Análisis de Resultados

**Cambio de paradigma demostrado:**
1. Los enfoques data-driven (CNN) superan dramáticamente a modelos físicos clásicos para datos de alta frecuencia
2. GPU acceleration permite entrenamiento 50-100x más rápido
3. CNN captura patrones temporales complejos que PINN clásicos no pueden modelar

**Implicaciones para tribología:**
- Datos experimentales modernos requieren enfoques modernos de ML
- Modelos clásicos (Bouc-Wen, LuGre, Dahl) válidos solo para regímenes quasi-estáticos
- CNN abre posibilidades para monitoreo de desgaste en tiempo real

### 4.5.4 Optimizaciones GPU Implementadas

Para hardware con memoria limitada (3GB):
- Modelo lightweight: 16→32→16 filtros (vs 64→128→64)
- Datasets reducidos: 10k puntos máximo
- Batching agresivo: batch_size=32
- Fallback automático a CPU si OOM
- Secuencias cortas: 50 puntos (vs 100)

---

## 4.6 Análisis de Histéresis Fuerza-Velocidad

### 4.6.1 Metodología del Sensor Virtual

Se implementó un **sensor virtual de fuerza** que permite detectar y cuantificar histéresis F-v sin necesidad de dinamómetro:

**Pipeline de procesamiento:**
1. Filtrado pasa-altas Butterworth (0.5 Hz, 2º orden)
2. Integración trapezoidal para obtener velocidad
3. Recentrado a media cero
4. Síntesis de fuerza mediante modelos de fricción

### 4.6.2 Modelos de Fricción Implementados

**Modelo Dahl:**
$$\dot{z} = \alpha v \left(1 - \frac{z}{F_c}\text{sign}(v)\right), \quad \hat{F} = \sigma_0 z + \sigma_1 v$$

**Modelo LuGre:**
$$\dot{z} = v - \frac{\sigma_0|v|}{g(v)}z, \quad g(v) = F_c + (F_s - F_c)e^{-(v/v_s)^2}$$
$$\hat{F} = \sigma_0 z + \sigma_1 \dot{z} + \sigma_2 v$$

**Modelo Bouc-Wen:**
$$\dot{z} = Av - \beta|v||z|^{n-1}z - \gamma v|z|^n, \quad \hat{F} = k_v v + \alpha z$$

### 4.6.3 Detección de Eventos Transitorios

Se implementó detección automática de "pelitos" (eventos transitorios) mediante:
- Umbrales adaptativos sobre picos (2.5σ)
- Distancia mínima entre eventos
- Extracción de micro-ventanas (±100 muestras)

**Resultado:** En eventos transitorios, los lazos F-v son pronunciados y multi-valuados. El modelo Dahl resulta ganador con frecuencia (pre-deslizamiento dominante), mientras que LuGre captura efectos Stribeck cuando la velocidad cruza por cero.

### 4.6.4 Método de Cruce por Cero

Se implementó el método de cruce por cero para cálculo preciso de fricción dinámica:

```python
# Zona de cruce: |x| < 10% del stroke
x_threshold = stroke * 0.10

# Separar por dirección de velocidad
F_forward = mean(F[near_zero & (v > 0)])   # Avance
F_backward = mean(F[near_zero & (v < 0)])  # Retroceso

# Fricción dinámica
F_friction_zc = (F_forward - F_backward) / 2
```

**Ventaja:** Más preciso que el método pico a pico para ciclos de histéresis asimétricos.

---

## 4.7 Validación Experimental

### 4.7.1 Calidad de Datos

Se verificó la calidad de los datos experimentales:

| Métrica | Resultado |
|---------|-----------|
| NaN/Inf | 0 en todos los canales |
| Tiempos monótonos | ✓ Verificado |
| Outliers (|z|>6) | <1% de muestras |
| Coherencia temporal | Verificada |

### 4.7.2 Validación F = ma

Se implementó validación en tiempo real de la relación fuerza-aceleración:

$$F_{\text{medida}} \approx m \cdot a_{\text{sensor}}$$

**Métricas calculadas:**
- Error porcentual entre F medida y m×a
- Transmisibilidad entre acelerómetros
- Desfase temporal entre señales

### 4.7.3 Estado de Calibración

| Componente | Estado | Observaciones |
|------------|--------|---------------|
| Acelerómetros PCB 352C33 | ✓ Validado | Sensibilidad verificada |
| Celda de carga DYMH-105 | ⚠ Pendiente | Requiere masas patrón |
| Sincronización canales | ⚠ En proceso | Lag detectado en algunas sesiones |

---

## 4.8 Infraestructura de Software

### 4.8.1 Archivos Principales Desarrollados

| Archivo | Líneas | Descripción |
|---------|--------|-------------|
| `caracterizacion_sensor_fuerza.py` | 1,085 | GUI caracterización con shaker |
| `train_kanpinn_efficient.py` | 878 | Entrenamiento KAN-PINN optimizado |
| `interfaz_DAQ_acel_fuerza.py` | ~800 | Adquisición tiempo real |
| `analisis_frf_sistema.py` | 395 | Análisis FRF offline |
| `articulo_histeresis_fv.tex` | 116 | Artículo LaTeX IFToMM |
| `reporte_parcial.tex` | 300 | Documentación PINN |

### 4.8.2 Documentación Generada

- `RESUMEN_EJECUTIVO.txt` - Guía rápida para Colab
- `MAPA_COMPLETO_VERIFICADO.md` - Auditoría detallada del código
- `README_COLAB_COMPLETO.md` - Instrucciones paso a paso
- Múltiples reportes de validación de datos

---

## 4.9 Resultados Esperados y Progreso

### 4.9.1 Alineación con Objetivos

| Objetivo Específico | Progreso | Estado |
|---------------------|----------|--------|
| Redactar 2 artículos científicos | 1 presentado, 1 en preparación | 50% |
| Analizar muestra de sistemas mecánicos | Sistema DAQ caracterizado | 70% |
| Diseñar modelo matemático de histéresis | PINN + modelos clásicos implementados | 80% |
| Implementar modelo predictivo | KAN-PINN funcional | 60% |
| Simular bajo diversas condiciones | DOE completado | 75% |
| Optimizar algoritmos de IA | Optimizaciones GPU implementadas | 70% |
| Evaluar impacto del modelo | Métricas comparativas obtenidas | 65% |

### 4.9.2 Métricas de Avance

| Indicador | Valor Actual |
|-----------|--------------|
| R² mejor modelo PINN | 0.50 |
| R² mejor modelo CNN | 0.9999 |
| Mejora PINN vs base | +47% |
| Speedup GPU vs CPU | 50-100x |
| Sesiones experimentales | 14+ |
| Archivos de código | 25+ |

---

## 4.10 Conclusiones Parciales

1. **Validación del enfoque PINN:** Se demostró que incorporar restricciones físicas mejora significativamente el rendimiento predictivo (R² de 0.33 a 0.50).

2. **Superioridad de CNN para alta frecuencia:** Los modelos CNN superan dramáticamente a los PINN clásicos para datos experimentales de alta frecuencia (R² = 0.97 vs 0.03).

3. **Identificación de parámetros:** Se identificó un desafío clave: más grados de libertad no garantizan mejor rendimiento y pueden conducir a mínimos locales.

4. **Sensor virtual de fuerza:** Se validó la metodología para detectar histéresis F-v sin dinamómetro, usando modelos de fricción con memoria.

5. **Infraestructura robusta:** Se desarrolló un sistema completo de adquisición, procesamiento y análisis de datos experimentales.

---

## 4.11 Trabajo Futuro

### Corto plazo (próximo mes):
- Completar calibración de celdas de carga con masas patrón
- Finalizar correcciones del artículo IFToMM
- Intensificar recolección de datos experimentales

### Mediano plazo (próximo semestre):
- Desarrollar CNN híbridos que incorporen conocimiento físico
- Implementar incertidumbre bayesiana en predicciones
- Validar modelo en condiciones de corte reales

### Largo plazo (conclusión de tesis):
- Integrar modelo predictivo en sistema de control
- Publicar segundo artículo en revista indexada
- Documentar metodología completa para reproducibilidad

---

*Documento generado: Diciembre 2024*
*Última actualización de datos experimentales: Noviembre 2024*
