# PROMPT PARA GEMINI - Análisis de Histéresis en Corte CNC

## CONTEXTO DEL PROYECTO

Soy estudiante de doctorado trabajando en la **caracterización de histéresis en fresado CNC de aluminio**. Mi objetivo es desarrollar un modelo que capture la relación no lineal entre fuerza de corte y vibración usando redes KAN (Kolmogorov-Arnold Networks) informadas por física.

## SETUP EXPERIMENTAL

```
        Husillo (Z vertical)
           │
           ▼ RPM ~3720
    ═══════════════►  X (dirección de corte/avance)
           │
    ┌──────┴──────┐
    │    PIEZA    │ ← Acelerómetro PCB 352C33 (mide X)
    │  (Aluminio) │
    └─────────────┘
           │
    ┌─────────────┐
    │   MORDAZA   │
    └─────────────┘
           │
    ┌─────────────┐
    │ CELDA CARGA │ ← DYMH-105 (mide fuerza X)
    └─────────────┘
```

### Especificaciones
- **Fresa**: 2 dientes
- **RPM estimado**: 3,720 (basado en frecuencia de paso de dientes 124 Hz)
- **Celda de carga**: DYMH-105, NI 9205, 2500 Hz
- **Acelerómetro**: PCB 352C33 (98.3 mV/g), NI 9234 IEPE, 2500 Hz
- **Material**: Aluminio 6061-T6

## DATOS DISPONIBLES

### Archivo: `corte2_mejor_0.5s.txt`
- **Formato**: TSV (tiempo_s, fuerza_V, aceleracion_sensor_g)
- **Muestras**: 1,250
- **Duración**: 0.5 segundos
- **Fs**: 2,500 Hz

### Estadísticas
| Variable | Media | Std | Rango |
|----------|-------|-----|-------|
| Fuerza | 0.9245 V | 0.034 V | 0.86-0.98 V |
| Aceleración | 0.004 g | 0.59 g | -4 a +2.5 g |

## ANÁLISIS REALIZADO

### 1. Análisis Espectral (FFT)
| Señal | Frecuencia Dominante | Interpretación |
|-------|---------------------|----------------|
| **Fuerza** | ~10 Hz | Ciclo de enganche/desenganche |
| **Aceleración** | ~124 Hz | Paso de dientes (2 × RPM/60) |

**Problema**: Las señales están en bandas de frecuencia completamente diferentes.

### 2. Correlación Directa
```
Correlación F vs a (directa) = 0.004  ❌ (prácticamente cero)
```

### 3. Descubrimiento Clave: ENVOLVENTE
La **masa de la mordaza actúa como filtro pasa-bajas mecánico**:
- Acelerómetro: captura impactos instantáneos de cada diente (124 Hz)
- Celda de carga: mide fuerza promediada por la inercia (10 Hz)

**Solución**: Usar la ENVOLVENTE de la aceleración (intensidad de vibraciones):
```python
acel_rect = np.abs(acel)  # Rectificar
b, a = signal.butter(2, 15/(fs/2), btype='low')  # Filtro 15 Hz
acel_env = signal.filtfilt(b, a, acel_rect)
```

### 4. Resultados con Envolvente
| Método | Correlación |
|--------|-------------|
| F vs Envolvente | **0.813** ✅ |
| F vs RMS (20ms) | **0.764** ✅ |

### 5. Modelo Lineal
```
F = 0.1148 × Envolvente - 0.0436
R² = 0.661
RMSE = 19.7 mV
```

### 6. EVIDENCIA DE HISTÉRESIS
El modelo lineal tiene R² = 0.66, NO 1.0. El 34% restante corresponde a:
1. **Desfase temporal** (~10-20 ms) entre envolvente y fuerza
2. **Lazo de histéresis** visible en el scatter F vs Envolvente
3. **Memoria del sistema** (la fuerza depende del historial, no solo del estado actual)

## MODELO BOUC-WEN INTENTADO

Se intentó ajustar un modelo Bouc-Wen clásico:
```
F = α·k·x + (1-α)·k·z + c·v

dz/dt = A·v - β|v||z|^(n-1)·z - γ·v|z|^n
```

**Resultado**: No funcionó con aceleración directa (correlación -0.03) porque:
- La velocidad integrada de 124 Hz no corresponde al movimiento de 10 Hz
- El modelo necesita la envolvente como entrada, no la señal cruda

## LO QUE NECESITO DE TI

### Opción A: Modelo Bouc-Wen con Envolvente
Proponer cómo modificar el modelo Bouc-Wen para usar:
- **Entrada**: Envolvente de aceleración (o su derivada)
- **Salida**: Fuerza de corte
- **Capturar**: El desfase temporal y el lazo de histéresis

### Opción B: Red KAN para Histéresis
Diseñar una arquitectura KAN (Kolmogorov-Arnold Network) que:
- Tome como entrada: envolvente(t), d/dt[envolvente(t)], historial temporal
- Prediga: fuerza(t)
- Capture la no-linealidad tipo histéresis

### Opción C: Modelo Físico Híbrido
Combinar:
- Modelo mecanístico: F = Kt × h × b (espesor viruta × ancho)
- Modelo de histéresis: Para el componente de fricción/amortiguamiento
- La envolvente como proxy de "intensidad de corte"

## PREGUNTAS ESPECÍFICAS

1. ¿Es correcto interpretar el R² = 0.66 como evidencia de histéresis?
2. ¿Cómo incorporar el desfase temporal en un modelo de histéresis?
3. ¿Qué arquitectura KAN recomiendas para este problema?
4. ¿Debería usar la envolvente directamente o transformarla (log, derivada, etc.)?

## ARCHIVOS ADJUNTOS

Por favor analiza los siguientes archivos que te adjunto:

1. **corte2_mejor_0.5s.txt** - Datos crudos
2. **analisis_completo.png** - Gráfica resumen de 9 paneles
3. **corte2_envolvente.png** - Correlación F vs Envolvente
4. **bouc_wen_corte2.png** - Intento fallido con Bouc-Wen
5. **RESUMEN_ANALISIS_CORTE_CNC.md** - Documentación completa

## CÓDIGO BASE

```python
import pandas as pd
import numpy as np
from scipy import signal

# Cargar datos
df = pd.read_csv('corte2_mejor_0.5s.txt', sep='\t')
t = df['tiempo_s'].values
fuerza = df['fuerza_V'].values - np.mean(df['fuerza_V'].values)
acel = df['aceleracion_sensor_g'].values

# Calcular envolvente
fs = 2500
acel_env = signal.filtfilt(*signal.butter(2, 15/(fs/2), 'low'), np.abs(acel))

# Correlación
print(f"Correlación F vs Envolvente: {np.corrcoef(fuerza, acel_env)[0,1]:.4f}")
# Output: 0.8129
```

## OBJETIVO FINAL

Desarrollar un modelo que:
1. **Prediga fuerza de corte** a partir de aceleración (monitoreo indirecto)
2. **Capture la histéresis** del proceso de fresado
3. **Sea interpretable físicamente** (no caja negra)
4. **Sirva para mi tesis doctoral** sobre caracterización de fricción no lineal en CNC

---

*Generado el 8 de diciembre de 2025*
*Datos capturados en fresadora CNC vertical*
