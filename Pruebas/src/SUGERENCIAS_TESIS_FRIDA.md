# Sugerencias de Revisión - Tesis Frida Andrade
## Calcetín Inteligente con Sensores Textiles para Monitoreo Térmico

---

## 📋 RESUMEN EJECUTIVO DE CAMBIOS PRIORITARIOS

| Prioridad | Sección | Problema Principal |
|-----------|---------|-------------------|
| 🔴 Alta | Conclusiones | Demasiado breves, parecen notas |
| 🔴 Alta | Metodología | Falta justificación estadística |
| 🟡 Media | Estado del Arte | Falta síntesis comparativa |
| 🟡 Media | Resultados | Sin análisis cuantitativo formal |
| 🟢 Baja | Redacción | Errores menores de ortografía |

---

## 1. TÍTULO Y PORTADA

**Texto actual:** (No visible en extracción)

**Sugerencias:**
- [ ] Verificar que el título sea específico: incluir "monitoreo térmico" y "prevención de úlceras diabéticas"
- [ ] Ejemplo propuesto: *"Desarrollo de un Calcetín Inteligente con Sensores Textiles de Impedancia para el Monitoreo Térmico en la Prevención de Úlceras del Pie Diabético"*

---

## 2. RESUMEN / ABSTRACT

**Sugerencias:**
- [ ] Incluir: objetivo, metodología, resultados clave (error <2% a 90kHz), conclusión principal
- [ ] Máximo 250 palabras
- [ ] Agregar 5 palabras clave: *textiles inteligentes, impedancia, AD5934, pie diabético, ESP32*

---

## 3. INTRODUCCIÓN

**Texto actual:** Menciona problemática de diabetes y úlceras plantares.

**Sugerencias:**
- [ ] **Agregar estadísticas actualizadas**: 
  - "Según la IDF (2023), México ocupa el 6° lugar mundial en prevalencia de diabetes con 14.1 millones de adultos afectados"
  - "El 15-25% de diabéticos desarrollará úlceras plantares durante su vida (Armstrong et al., 2017)"
- [ ] **Definir claramente el GAP de investigación**: ¿Qué no han logrado los trabajos previos que este trabajo sí aborda?
- [ ] **Agregar objetivo general y específicos** al final de la introducción (si no están)

---

## 4. ESTADO DEL ARTE / ANTECEDENTES

### 4.1 Trabajos Citados

**Texto actual:** Se describen 4-5 trabajos previos de calcetines inteligentes.

**Sugerencias por trabajo citado:**

#### Trabajo 1: Calcetín piezorresistivo (8 sensores, presión)
**Problemas detectados:**
- [ ] Falta la referencia bibliográfica completa (autor, año, revista)
- [ ] Agregar: "Este trabajo se limita a presión, no temperatura"

#### Trabajo 2: Consumer Acceptance (Kent et al.)
**Problemas detectados:**
- [ ] Mejorar redacción: "El prototipo que se diseño" → "El prototipo que se **diseñó**"
- [ ] Agregar año de publicación en el texto

#### Trabajo 3: Continuous temperature-monitoring socks (Neurofabric)
**Problemas detectados:**
- [ ] Excelente referencia, pero falta comparar sus resultados con los propios
- [ ] Agregar: "A diferencia de este trabajo, nuestro enfoque utiliza medición de impedancia en lugar de termistores discretos"

#### Trabajo 4: Smart Socks and plantar pressure (pilot study)
**Problemas detectados:**
- [ ] Falta referencia completa

### 4.2 Síntesis Comparativa (FALTANTE)

**⚠️ AGREGAR TABLA COMPARATIVA:**

```
| Autor (Año) | Tipo Sensor | Variable | Comunicación | Lavable | Limitación |
|-------------|-------------|----------|--------------|---------|------------|
| Kent (2016) | Termistor   | Temp     | BLE          | Sí      | Comodidad  |
| Neurofabric | Termistor   | Temp     | BLE          | Sí      | Costo      |
| [Autor]     | Piezorres.  | Presión  | WiFi         | No      | Consumo    |
| **Este trabajo** | Impedancia | Temp | BLE      | Pendiente | Prototipo |
```

---

## 5. MARCO TEÓRICO

### 5.1 Textil NW170-PI

**Texto actual:** Menciona propiedades pero no las cuantifica.

**Sugerencias:**
- [ ] Agregar tabla con especificaciones del datasheet:
  ```
  | Propiedad | Valor | Unidad |
  |-----------|-------|--------|
  | Resistencia superficial | X | Ω/sq |
  | Rango de temperatura | -40 a 80 | °C |
  | Espesor | X | mm |
  ```
- [ ] Citar el datasheet como referencia

### 5.2 AD5934

**Texto actual:** "es un Convertidor analógico digital (ADC) de gran precisión"

**⚠️ ERROR TÉCNICO:** El AD5934 NO es un ADC, es un **analizador de impedancia** con generador de señal + ADC integrado.

**Corrección sugerida:**
> "El AD5934 es un analizador de impedancia de alta precisión que integra un generador de señal programable (DDS), un convertidor digital-analógico (DAC) y un convertidor analógico-digital (ADC) de 12 bits. Permite medir impedancia compleja (magnitud y fase) mediante la técnica de excitación sinusoidal y análisis de respuesta en frecuencia."

### 5.3 CN0349

**Texto actual:** Lista características sin explicar funcionamiento.

**Sugerencias:**
- [ ] Agregar diagrama de bloques del circuito
- [ ] Explicar el rol de cada componente: amplificador de transimpedancia, referencia de calibración

---

## 6. METODOLOGÍA / DESARROLLO

### 6.1 Selección de Geometría

**Texto actual:** 
> "la resistencia variaba significativamente dependiendo de la orientación del cuadrado, aumentando aproximadamente 2 unidades"

**Sugerencias:**
- [ ] "2 unidades" → Especificar: "2 Ω" o "2%"
- [ ] Agregar justificación matemática del cambio a geometría circular
- [ ] Incluir imagen del proceso de corte

### 6.2 Caracterización Temperatura-Impedancia

**Texto actual:**
> "se utilizó una pistola de calor... tomando mediciones en intervalos de un grado centígrado"

**Sugerencias:**
- [ ] **Agregar modelo de la pistola de calor** (marca, potencia)
- [ ] **Agregar modelo de cámara FLIR** (resolución, precisión ±X°C)
- [ ] **Especificar tiempo de estabilización** entre mediciones
- [ ] **Agregar incertidumbre de medición**: ¿cuál es la resolución del multímetro?

### 6.3 Integración en Calcetín

**Texto actual:**
> "los sensores fueron colocados estratégicamente en puntos específicos del calcetín, determinados previamente en la sección de Marco Teórico"

**Sugerencias:**
- [ ] Especificar los 6 puntos anatómicos exactos (ej: hallux, 1er metatarso, talón medial, etc.)
- [ ] Referenciar estudios que justifiquen estos puntos como críticos para úlceras
- [ ] Agregar diagrama con ubicación numerada de sensores

### 6.4 Calibración del AD5934

**Texto actual:** 
> "fue necesario calibrar en diferentes frecuencias y valores para obtener los errores más bajos"

**⚠️ FALTA DETALLE CRÍTICO:**
- [ ] Describir procedimiento de calibración paso a paso
- [ ] Especificar resistencia de calibración usada (1kΩ mencionada en resultados)
- [ ] Explicar fórmula de corrección: `Z_real = Z_medida * Factor_Calibración`

### 6.5 Procesamiento de Señal (ESP32)

**Texto actual:**
> "Se implementaron algoritmos de filtrado digital... un algoritmo de normalización"

**⚠️ MUY VAGO - AGREGAR:**
- [ ] Tipo de filtro: ¿FIR/IIR? ¿Orden? ¿Frecuencia de corte?
- [ ] Tipo de normalización: ¿Min-Max? ¿Z-score?
- [ ] Frecuencia de muestreo
- [ ] Incluir fragmento de código o pseudocódigo

---

## 7. RESULTADOS

### 7.1 Caracterización de Sensores

**Texto actual:** 6 gráficas de impedancia vs temperatura (28-45°C)

**⚠️ ANÁLISIS FALTANTE:**
- [ ] **Calcular y reportar para cada sensor:**
  - Sensibilidad: ΔZ/ΔT [Ω/°C]
  - Linealidad: R² del ajuste lineal
  - Repetibilidad: desviación estándar entre las 3 pruebas
  - Histéresis: diferencia entre calentamiento y enfriamiento

- [ ] **Agregar tabla resumen:**
```
| Sensor | Sensibilidad [Ω/°C] | R² | σ (repetibilidad) |
|--------|--------------------|----|-------------------|
| 1      | X.XX               | 0.XX | ±X.X Ω          |
| 2      | X.XX               | 0.XX | ±X.X Ω          |
| ...    | ...                | ... | ...              |
```

- [ ] **Modelo matemático:** Proponer ecuación Z(T) = aT + b para cada sensor

### 7.2 Selección de Frecuencia (90 kHz)

**Texto actual:** Análisis cualitativo del error por frecuencia.

**Sugerencias:**
- [ ] Excelente análisis, pero falta:
  - Gráfica de error vs frecuencia (no solo la tabla)
  - Justificación del rango 100Ω-3.3kΩ vs rango real de los sensores

### 7.3 Medición de Impedancia en Sistema

**Texto actual:** Tabla de errores en 3 frecuencias.

**Sugerencias:**
- [ ] Agregar columna de incertidumbre expandida (k=2)
- [ ] Comparar con especificaciones del fabricante del AD5934

### 7.4 Aplicación Móvil (INCOMPLETO)

**Texto actual:** Solo título, sin contenido.

**⚠️ AGREGAR:**
- [ ] Capturas de pantalla de la app
- [ ] Descripción de funcionalidades
- [ ] Protocolo BLE (UUID de servicios/características)
- [ ] Tasa de transmisión de datos

---

## 8. DISCUSIÓN (FALTANTE)

**⚠️ SECCIÓN COMPLETA FALTANTE - AGREGAR:**

- [ ] Comparación con trabajos previos (tabla del estado del arte)
- [ ] Limitaciones del estudio:
  - Pruebas solo en laboratorio, no en pacientes reales
  - Falta validación de lavabilidad
  - Muestra pequeña (6 sensores)
- [ ] Fuentes de error:
  - Contacto sensor-piel
  - Deriva térmica del ESP32
  - Interferencia electromagnética

---

## 9. CONCLUSIONES

**Texto actual:**
> "Puedo concluir que es un prototipo, es necesario elegir otros componentes electrónicos..."
> - Es necesario buscar la forma mas eficiente...
> - Es necesario hacer un diseño para guardar la electrónica
> - Hacer pruebas de la impedancia lavable
> - diseñar una PCB

**⚠️ PROBLEMAS GRAVES:**
1. Escrito en primera persona informal
2. Son notas/pendientes, no conclusiones
3. No responden a los objetivos
4. Lista incompleta y desordenada

**REESCRIBIR COMPLETAMENTE:**

> **Conclusiones:**
> 
> 1. Se desarrolló exitosamente un sistema de monitoreo térmico basado en sensores textiles de impedancia, capaz de detectar variaciones de temperatura en el rango clínicamente relevante de 28°C a 45°C.
> 
> 2. La caracterización del textil NW170-PI demostró una sensibilidad promedio de X.X Ω/°C con una linealidad R² > 0.XX, validando su aplicabilidad como sensor de temperatura.
> 
> 3. La frecuencia de excitación óptima para el AD5934 es 90 kHz, con errores menores al 2% en el rango de impedancia de los sensores (100Ω-3.3kΩ).
> 
> 4. El sistema embebido basado en ESP32 con comunicación BLE permite la transmisión inalámbrica de datos a una aplicación móvil, cumpliendo con los requerimientos de portabilidad.
> 
> 5. El prototipo desarrollado sienta las bases para un dispositivo de prevención de úlceras diabéticas, requiriendo validación clínica en trabajos futuros.

---

## 10. TRABAJO FUTURO (FALTANTE O MEJORAR)

**Convertir las notas actuales en trabajo futuro estructurado:**

> **Trabajo Futuro:**
> 
> - Validación clínica con pacientes diabéticos (n≥30)
> - Pruebas de durabilidad: ciclos de lavado (≥50 ciclos a 40°C)
> - Diseño de PCB miniaturizada para integración en el calcetín
> - Optimización del consumo energético para autonomía >24h
> - Desarrollo de algoritmos de alerta temprana basados en umbrales de temperatura

---

## 11. BIBLIOGRAFÍA

**Sugerencias:**
- [ ] Verificar formato consistente (IEEE o APA)
- [ ] Mínimo 20-30 referencias para tesis
- [ ] Incluir referencias clave faltantes:
  - Armstrong, D. G., et al. (2017). Diabetic foot ulcers and their recurrence. NEJM.
  - Lavery, L. A., et al. (2007). Preventing diabetic foot ulcer recurrence in high-risk patients. Diabetes Care.
  - Bus, S. A., et al. (2020). Guidelines on offloading foot ulcers in persons with diabetes. Diabetes/Metabolism Research and Reviews.

---

## 12. ERRORES DE REDACCIÓN Y ORTOGRAFÍA

| Ubicación | Error | Corrección |
|-----------|-------|------------|
| Metodología | "se diseño" | "se diseñó" |
| Metodología | "tambien" | "también" |
| Resultados | "agrego" | "agregó" |
| Conclusiones | "mas eficiente" | "más eficiente" |
| Conclusiones | "rehusables" | "reutilizables" |
| Varias | Espacios dobles | Eliminar |
| Gráficas | Sin numeración | Agregar "Gráfica 1:", "Gráfica 2:", etc. |
| Figuras | Sin numeración | Agregar "Figura 1:", "Figura 2:", etc. |

---

## 13. FORMATO Y PRESENTACIÓN

- [ ] Verificar márgenes consistentes
- [ ] Numerar todas las figuras y tablas
- [ ] Agregar lista de figuras y lista de tablas
- [ ] Verificar pie de página con número de página
- [ ] Agregar encabezado con título corto del capítulo

---

## ✅ CHECKLIST FINAL

### Antes de entregar:
- [ ] Revisar ortografía con corrector automático
- [ ] Verificar que todas las figuras tengan título y fuente
- [ ] Verificar que todas las tablas tengan título
- [ ] Verificar referencias cruzadas (Figura X, Tabla Y)
- [ ] Revisar que la bibliografía esté completa y en formato correcto
- [ ] Pedir a alguien más que lea el documento

---

*Documento generado el 13 de diciembre de 2025*
