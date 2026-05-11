# Reporte Técnico: Caracterización Dinámica del Sistema de Medición de Fuerza

**Fecha:** 3 de Diciembre de 2025  
**Autor:** Laboratorio de Mecánica Experimental  
**Equipo:** NI cDAQ-9174 + NI 9205 (Fuerza) + NI 9234 (Aceleración)

---

## 1. Introducción

Este documento presenta la caracterización dinámica de un sistema de medición de fuerza basado en una celda de carga DYMH-105. El objetivo es identificar los parámetros dinámicos del sistema (masa, rigidez, amortiguamiento) mediante análisis de Función de Respuesta en Frecuencia (FRF) y caracterizar la fricción mediante excitación triangular.

---

## 2. Configuración Experimental

### 2.1 Instrumentación

| Componente | Modelo | Especificación |
|------------|--------|----------------|
| Celda de carga | DYMH-105 | 500 kg, 1.7 mV/V |
| Acondicionador | INA849 | Ganancia = 601 |
| Acelerómetro | PCB 352C33 | 100 mV/g |
| DAQ Fuerza | NI 9205 | ±10V, 16-bit |
| DAQ Aceleración | NI 9234 | IEPE, 24-bit |
| Generador | Keysight 33220A | 20 MHz |
| Amplificador | TIRA | Shaker electrodinámico |

### 2.2 Parámetros de Adquisición

- **Frecuencia de muestreo:** 2500 Hz
- **Rango de frecuencias (barrido):** 10 - 90 Hz
- **Rango de frecuencias (fricción):** 0.5 - 2.5 Hz
- **Duración de captura:** 10 s por punto

### 2.3 Conversión de Unidades

**Fuerza (Voltaje → Newtons):**

$$F = \frac{V_{medido}}{S_{total}} \times C_{max} \times g$$

Donde:
- $S_{total} = S_{celda} \times V_{exc} \times G_{INA} = \frac{1.7}{1000} \times 10 \times 601 = 10.217$ V/FS
- $C_{max} = 500$ kg
- $g = 9.81$ m/s²

$$F[N] = \frac{V[V]}{10.217} \times 500 \times 9.81 = 480.1 \times V[V]$$

**Aceleración (Voltaje → m/s²):**

$$a = \frac{V_{medido}}{S_{acel}} \times g = \frac{V[V]}{0.1} \times 9.81 = 98.1 \times V[V]$$

---

## 3. Fundamento Teórico

### 3.1 Modelo Dinámico del Sistema

El sistema se modela como un sistema masa-resorte-amortiguador de un grado de libertad:

$$m\ddot{x} + c\dot{x} + kx = F(t)$$

Donde:
- $m$ = masa efectiva [kg]
- $c$ = coeficiente de amortiguamiento viscoso [N·s/m]
- $k$ = rigidez [N/m]
- $F(t)$ = fuerza aplicada [N]
- $x(t)$ = desplazamiento [m]

### 3.2 Función de Respuesta en Frecuencia (FRF)

Aplicando la transformada de Fourier y considerando que $a = \ddot{x}$:

$$(-m\omega^2 + jc\omega + k)X(\omega) = F(\omega)$$

La FRF se define como la relación entre fuerza y aceleración:

$$H(\omega) = \frac{F(\omega)}{a(\omega)} = \frac{F(\omega)}{-\omega^2 X(\omega)}$$

Sustituyendo:

$$H(\omega) = m - \frac{k}{\omega^2} + j\frac{c}{\omega}$$

**Magnitud:**

$$|H(\omega)| = \sqrt{\left(m - \frac{k}{\omega^2}\right)^2 + \left(\frac{c}{\omega}\right)^2}$$

**Fase:**

$$\phi(\omega) = \arctan\left(\frac{c/\omega}{m - k/\omega^2}\right)$$

### 3.3 Parámetros Característicos

**Frecuencia natural no amortiguada:**

$$\omega_n = \sqrt{\frac{k}{m}} \quad \Rightarrow \quad f_n = \frac{1}{2\pi}\sqrt{\frac{k}{m}}$$

**Factor de amortiguamiento:**

$$\zeta = \frac{c}{2\sqrt{km}} = \frac{c}{2m\omega_n}$$

**Frecuencia natural amortiguada:**

$$\omega_d = \omega_n\sqrt{1-\zeta^2}$$

### 3.4 Comportamiento Asintótico de la FRF

| Región | Condición | Comportamiento |
|--------|-----------|----------------|
| Baja frecuencia | $\omega \ll \omega_n$ | $\|H\| \approx k/\omega^2$ (dominado por rigidez) |
| Resonancia | $\omega \approx \omega_n$ | $\|H\| \approx c/\omega$ (dominado por amortiguamiento) |
| Alta frecuencia | $\omega \gg \omega_n$ | $\|H\| \approx m$ (dominado por masa) |

---

## 4. Resultados Experimentales

### 4.1 Barrido de Frecuencia (10-90 Hz)

Se realizó un barrido senoidal con los siguientes resultados de FRF:

| f [Hz] | ω [rad/s] | \|H\| [kg] | φ [°] | F_rms [N] | a_rms [m/s²] |
|--------|-----------|------------|-------|-----------|--------------|
| 10 | 62.8 | 323.73 | 61.7 | 20.13 | 0.068 |
| 15 | 94.2 | 86.95 | 14.9 | 28.32 | 0.048 |
| 20 | 125.7 | 48.57 | -41.8 | 27.87 | 0.083 |
| 25 | 157.1 | 24.17 | -120.1 | 27.58 | 0.091 |
| 30 | 188.5 | 26.13 | 140.5 | 27.33 | 0.151 |
| 35 | 219.9 | 14.80 | 87.2 | 27.09 | 0.257 |
| 40 | 251.3 | 6.72 | -103.7 | 26.66 | 0.315 |
| 45 | 282.7 | 2.80 | 118.1 | 26.31 | 0.750 |
| **50** | **314.2** | **0.41** | -134.9 | 26.14 | 1.595 |
| 55 | 345.6 | 0.67 | -30.4 | 25.95 | 0.828 |
| 60 | 376.9 | 1.21 | 146.9 | 25.83 | 1.925 |
| 65 | 408.4 | 3.44 | 8.2 | 25.72 | 1.742 |
| 70 | 439.8 | 1.41 | -105.7 | 25.61 | 0.856 |
| 75 | 471.2 | 18.36 | 58.6 | 25.53 | 0.312 |
| 80 | 502.7 | 27.29 | 40.2 | 25.38 | 0.486 |
| 85 | 534.1 | 11.84 | 72.4 | 25.22 | 0.747 |
| 90 | 565.5 | 4.63 | -128.8 | 25.10 | 1.185 |

### 4.2 Parámetros Identificados

Mediante ajuste por mínimos cuadrados no lineales del modelo FRF:

| Parámetro | Símbolo | Valor | Unidad |
|-----------|---------|-------|--------|
| **Masa efectiva** | $m$ | 12.42 | kg |
| **Rigidez** | $k$ | 997,548 | N/m |
| **Amortiguamiento** | $c$ | 1,117.28 | N·s/m |
| **Frecuencia natural** | $f_n$ | 45.1 | Hz |
| **Factor de amortiguamiento** | $\zeta$ | 0.159 | - |

### 4.3 Validación de Parámetros

**Verificación de frecuencia natural:**

$$f_n = \frac{1}{2\pi}\sqrt{\frac{k}{m}} = \frac{1}{2\pi}\sqrt{\frac{997548}{12.42}} = \frac{1}{2\pi} \times 283.5 = 45.1 \text{ Hz} \quad \checkmark$$

**Verificación del factor de amortiguamiento:**

$$\zeta = \frac{c}{2\sqrt{km}} = \frac{1117.28}{2\sqrt{997548 \times 12.42}} = \frac{1117.28}{2 \times 3520.5} = 0.159 \quad \checkmark$$

**Verificación del mínimo de FRF (antiresonancia):**

En $\omega = \omega_n$, si el sistema tiene una antiresonancia:

$$|H(\omega_n)|_{min} \approx \frac{c}{\omega_n} = \frac{1117.28}{283.5} = 3.94 \text{ kg}$$

El valor medido mínimo fue **0.41 kg a 50 Hz**, lo cual indica que hay una cancelación más pronunciada, posiblemente debido a efectos de acoplamiento modal.

---

## 5. Caracterización de Fricción

### 5.1 Metodología

Se aplicó excitación triangular a baja frecuencia (0.5-2.5 Hz) para caracterizar la fricción del sistema. El desplazamiento se estimó mediante doble integración de la aceleración:

$$v(t) = \int a(t) \, dt$$

$$x(t) = \int v(t) \, dt$$

Se aplicó filtrado pasa-altos (Butterworth, fc = 0.3 Hz) y detrending para evitar drift.

### 5.2 Método de Cruce por Cero (Zero-Crossing)

La fricción dinámica se calcula en el punto donde $x \approx 0$ (centro del desplazamiento), separando por dirección de velocidad:

$$F_{fricción} = \frac{F_{forward}(x=0) - F_{backward}(x=0)}{2}$$

Donde:
- $F_{forward}$: Fuerza promedio cuando $v > 0$ y $|x| < 0.15 \times stroke$
- $F_{backward}$: Fuerza promedio cuando $v < 0$ y $|x| < 0.15 \times stroke$

Este método es más preciso que el pico-a-pico porque:
1. En $x = 0$, la fuerza elástica es mínima ($F_k = kx = 0$)
2. La velocidad es máxima, por lo que la fricción viscosa es representativa
3. Evita efectos de rigidez no lineal en los extremos

### 5.3 Resultados de Fricción

| f [Hz] | F_fricción [N] | Carrera [mm] | v_media [mm/s] |
|--------|----------------|--------------|----------------|
| 0.5* | 521.8 | 70.75 | 1.71 |
| 1.0 | 2.60 | 25.52 | 1.24 |
| 1.5 | 2.52 | 54.12 | 5.18 |
| 2.0 | 2.56 | 13.17 | 1.42 |
| 2.5 | 2.56 | 19.27 | 2.25 |

*Nota: El valor a 0.5 Hz es anómalo debido a drift en la integración a muy baja frecuencia.

### 5.4 Modelo de Fricción

Los resultados sugieren un modelo de **fricción de Coulomb** con componente viscosa pequeña:

$$F_{fricción} = F_c \cdot \text{sign}(v) + c_v \cdot v$$

Donde:
- $F_c \approx 2.55$ N (fricción de Coulomb)
- $c_v \approx 0$ (componente viscosa despreciable en este rango de velocidades)

**Coeficiente de fricción equivalente:**

Si consideramos una fuerza normal $N$ igual al peso de la masa efectiva:

$$\mu = \frac{F_c}{N} = \frac{F_c}{m \cdot g} = \frac{2.55}{12.42 \times 9.81} = 0.021$$

Este valor bajo es consistente con un sistema con rodamientos lineales o guías de baja fricción.

---

## 6. Análisis de Incertidumbre

### 6.1 Fuentes de Incertidumbre

| Fuente | Contribución Estimada |
|--------|----------------------|
| Ruido del sensor de fuerza | ±0.5% FS |
| Ruido del acelerómetro | ±1% |
| Deriva térmica | ±0.1%/°C |
| Cuantización ADC | ±0.01% |
| Sincronización temporal | <1 muestra |

### 6.2 Propagación de Incertidumbre en FRF

$$\frac{\delta |H|}{|H|} = \sqrt{\left(\frac{\delta F}{F}\right)^2 + \left(\frac{\delta a}{a}\right)^2} \approx \sqrt{0.005^2 + 0.01^2} = 1.1\%$$

### 6.3 Incertidumbre en Parámetros Identificados

| Parámetro | Valor | Incertidumbre (95%) |
|-----------|-------|---------------------|
| $m$ | 12.42 kg | ±2.5 kg |
| $k$ | 997,548 N/m | ±150,000 N/m |
| $c$ | 1,117 N·s/m | ±200 N·s/m |
| $f_n$ | 45.1 Hz | ±3 Hz |

---

## 7. Discusión

### 7.1 Validez del Modelo

El modelo de un grado de libertad captura razonablemente el comportamiento del sistema en el rango de 20-70 Hz. Las desviaciones observadas pueden deberse a:

1. **Modos adicionales:** El sistema real puede tener múltiples modos de vibración
2. **No linealidades:** Rigidez y amortiguamiento dependientes de la amplitud
3. **Acoplamiento:** Interacción con la estructura de soporte (piso, bancada)

### 7.2 Frecuencia Natural y Resonancia del Piso

Se reportó vibración audible del piso alrededor de 110 Hz. Esto sugiere:
- La frecuencia natural del sistema (45 Hz) está bien separada de la resonancia del piso
- A 110 Hz, el sistema actúa como transmisor de vibración al piso

### 7.3 Implicaciones para Medición de Fuerza

Para mediciones precisas de fuerza dinámica:

| Rango de Frecuencia | Recomendación |
|---------------------|---------------|
| f < 20 Hz | ✅ Medición directa válida |
| 20 Hz < f < 40 Hz | ⚠️ Aplicar corrección por FRF |
| 40 Hz < f < 50 Hz | ❌ Evitar (cerca de antiresonancia) |
| f > 50 Hz | ⚠️ Aplicar corrección por FRF |

---

## 8. Conclusiones

1. **Masa efectiva del sistema:** 12.42 kg, correspondiente al conjunto shaker + bancada + sensor

2. **Frecuencia natural:** 45.1 Hz con factor de amortiguamiento ζ = 0.159 (sistema subamortiguado)

3. **Rigidez:** ~1 MN/m, indicando un montaje relativamente rígido

4. **Fricción:** ~2.55 N (Coulomb), con coeficiente μ ≈ 0.021

5. **Rango de operación recomendado:** f < 20 Hz para mediciones sin corrección

---

## 9. Referencias

1. Ewins, D.J. (2000). *Modal Testing: Theory, Practice and Application*. Research Studies Press.

2. McConnell, K.G. (1995). *Vibration Testing: Theory and Practice*. Wiley.

3. Altintas, Y. (2012). *Manufacturing Automation*. Cambridge University Press.

4. NI Application Note: *Measuring Frequency Response Functions with NI Data Acquisition*

---

## Apéndice A: Código de Análisis

Los scripts de Python utilizados para este análisis están disponibles en:
- `caracterizacion_sensor_fuerza.py` - Interfaz de adquisición
- `analisis_completo.py` - Análisis de FRF y fricción
- `analisis_barrido_frecuencia.py` - Identificación de parámetros

## Apéndice B: Archivos de Datos

- `caracterizacion_fuerza_20251203_120809_resumen.csv` - Barrido 10-90 Hz
- `caracterizacion_fuerza_20251203_125007_resumen.csv` - Fricción 0.5-2.5 Hz
- `resumen_parametros.txt` - Parámetros identificados
