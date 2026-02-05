# Reporte Mensual de Avances - Noviembre 2024

**Estudiante:** José de Jesús Santana Ramírez  
**Programa:** Doctorado  
**Fecha:** Diciembre 2024

---

## Resumen Ejecutivo

Este mes de noviembre, el trabajo se centró en dos frentes principales: por un lado, se sigue trabajando en la calibración del sistema experimental para dejarlo listo, y por el otro, avancé significativamente en el desarrollo del modelo computacional. Además, estuve trabajando activamente en las correcciones de mi artículo, gracias a una valiosa retroalimentación que recibí.

---

## 1. Avances en la Tesis: Del Experimento al Modelo

### 1.1 Sistema de Adquisición y Calibración

Siguiendo el plan, este mes continué con la etapa de calibración de las celdas de carga y los acelerómetros. Se realizaron las siguientes actividades:

- **Desarrollo de interfaz de caracterización:** Se creó una nueva interfaz gráfica (`caracterizacion_sensor_fuerza.py`) para la caracterización del sensor de fuerza DYMH-105 con el shaker TIRA TV 51144IN.
  - Configuración completa para celda de carga (500 kg, 1.7 mV/V) con acondicionador INA-4LC-8NTC (G=601)
  - Soporte para 2 acelerómetros PCB 352C33 simultáneos (ai0: bancada, ai1: sobre sensor de fuerza)
  - Cálculo de transmisibilidad en tiempo real entre acelerómetros
  - Validación F=ma integrada en la interfaz

- **Pruebas de validación:** Se confirmó que las celdas funcionan correctamente y que los filtros de ruido operan como se esperaba. El reporte de validación de datos (20251003) mostró:
  - Sin valores NaN/Inf en ningún canal
  - Tiempos monótonos verificados
  - Outliers controlados (|z|>6 detectados y documentados)

- **Estado actual:** La calibración final de las celdas de carga aún no está completa. Se requiere validación con masas patrón para obtener factores de conversión trazables.

### 1.2 Desarrollo del Modelo PINN

El esfuerzo computacional se centró en la arquitectura de las Redes Neuronales Informadas por la Física (PINNs). Los avances más significativos fueron:

#### Modelo Base vs PINN - Resultados Comparativos

| Modelo | Variable | RMSE | R² |
|--------|----------|------|-----|
| Base (solo datos) | Posición (y) | 0.0057 m | 0.3432 |
| Base (solo datos) | Corriente (i) | 0.2282 A | 0.3283 |
| PINN (λ=10⁻⁴) | Posición (y) | 0.0049 m | 0.5105 |
| PINN (λ=10⁻⁴) | Corriente (i) | 0.2010 A | 0.4788 |
| **PINN (λ=10⁻³)** | **Posición (y)** | **0.0049 m** | **0.5058** |
| **PINN (λ=10⁻³)** | **Corriente (i)** | **0.1968 A** | **0.5003** |

**Mejora lograda:** El R² aumentó de ~0.33 (modelo base) a ~0.50 (PINN), demostrando que incorporar las ecuaciones diferenciales del sistema mejora significativamente la capacidad predictiva.

#### Arquitectura KAN-PINN Optimizada

Se desarrolló una versión eficiente del modelo KAN-PINN (`train_kanpinn_efficient.py`) con las siguientes mejoras:

1. **Diferencias finitas para derivadas 2º orden** - 10x speedup esperado
2. **Curriculum learning** - Transición gradual de datos a física
3. **Adaptive weighting** de λ_phys
4. **Arquitectura MLP+Fourier** como alternativa a KAN puro

**Tiempos de entrenamiento verificados:**
- GPU A100: 35-50 minutos
- GPU T4 (gratuita): 1-1.5 horas
- GTX 1060: 8+ horas

#### Identificación de Parámetros Físicos

Se exploró la capacidad del PINN para identificar los parámetros del modelo de inductancia:

$$L(y) = k_0 + \frac{k}{1 + y/a}$$

| Parámetro | Valor Inicial | Valor Identificado |
|-----------|---------------|-------------------|
| k₀ | 3.63×10⁻² H | 9.92×10⁻³ H |
| k | 3.50×10⁻³ H·m | 5.23×10⁻² H·m |
| a | 5.20×10⁻³ m | 4.48×10⁻³ m |

**Hallazgo importante:** El modelo con parámetros entrenables mostró peor rendimiento predictivo, sugiriendo convergencia a mínimos locales. Esto demuestra un desafío clave en el diseño de PINNs: más grados de libertad no garantizan mejor rendimiento.

---

## 2. Trabajo en Artículos y Colaboración

### 2.1 Artículo IFToMM - Histéresis F-v

Dando seguimiento al artículo presentado en el congreso de IFToMM ("Detección de histéresis F-v a partir de vibraciones: sensor virtual de fuerza y modelos Dahl/LuGre/Bouc-Wen en fresado de aluminio"), se recibió retroalimentación detallada.

**Trabajo realizado:**
- Procesamiento de la retroalimentación recibida
- Implementación de correcciones al artículo LaTeX (`articulo_histeresis_fv.tex`)
- Transcripción del artículo de LaTeX a Word para facilitar revisión colaborativa

**Contribuciones principales del artículo:**
1. Pipeline reproducible para detectar eventos en aceleración y reconstruir velocidad
2. Inferencia de fuerza sintética mediante modelos Dahl/LuGre/Bouc-Wen con selección automática
3. Cuantificación adimensional (área/ancho normalizados) robusta a ganancias desconocidas
4. Base metodológica para incorporar FRF/Kalman y calibración absoluta de energía

### 2.2 Reporte Parcial PINN

Se completó el documento `reporte_parcial.tex` que documenta:
- Metodología completa del modelo base y PINN
- Arquitectura con diagramas TikZ
- Resultados comparativos con tablas y figuras
- Análisis de residuales físicos
- Discusión sobre identificación de parámetros

---

## 3. Infraestructura de Software Desarrollada

### 3.1 Archivos Principales Creados/Modificados

| Archivo | Descripción |
|---------|-------------|
| `caracterizacion_sensor_fuerza.py` | GUI para caracterización con shaker (1085 líneas) |
| `train_kanpinn_efficient.py` | Entrenamiento KAN-PINN optimizado (878 líneas) |
| `reporte_parcial.tex` | Documento LaTeX con resultados PINN (300 líneas) |
| `articulo_histeresis_fv.tex` | Artículo para IFToMM (116 líneas) |
| `kan_model.py` | Modelo KAN con B-splines adaptativos |
| `losses_efficient.py` | Funciones de pérdida optimizadas |

### 3.2 Documentación Generada

- `RESUMEN_EJECUTIVO.txt` - Guía rápida para Colab
- `MAPA_COMPLETO_VERIFICADO.md` - Auditoría detallada del código
- `README_COLAB_COMPLETO.md` - Instrucciones paso a paso
- Múltiples reportes de validación de datos

---

## 4. Próximos Pasos (Diciembre)

El plan para el siguiente mes es claro:

1. **Calibración experimental:**
   - Completar calibración de celdas de carga con masas patrón
   - Validar factores de conversión para obtener mediciones trazables
   - Preparar placa de instrumentación final

2. **Recolección de datos:**
   - Intensificar la recolección de datos experimentales bajo diferentes condiciones de corte
   - Documentar parámetros de cada experimento

3. **Modelo PINN:**
   - Refinar el modelo PINN de predicción de desgaste
   - Integrar las correcciones del artículo
   - Incorporar los nuevos datos experimentales

4. **Artículo:**
   - Finalizar correcciones del artículo IFToMM
   - Preparar versión final para reenvío

---

## 5. Métricas de Progreso

| Indicador | Octubre | Noviembre | Cambio |
|-----------|---------|-----------|--------|
| R² modelo PINN | 0.34 (base) | 0.50 (PINN) | +47% |
| Archivos de código | ~15 | ~25 | +67% |
| Documentación (páginas) | ~10 | ~30 | +200% |
| Sesiones experimentales | 4 | 8 | +100% |

---

## 6. Conclusiones

El mes de noviembre representó un avance significativo en el desarrollo del modelo computacional PINN, logrando una mejora del 47% en el coeficiente de determinación respecto al modelo base. La infraestructura de software para adquisición y análisis está prácticamente completa. El trabajo pendiente se concentra en la calibración final del sistema experimental y la integración de datos reales con el modelo PINN.

La retroalimentación recibida sobre el artículo de IFToMM ha sido valiosa para mejorar la calidad del trabajo, y las correcciones están en proceso de implementación.

---

*Documento generado: Diciembre 2024*
