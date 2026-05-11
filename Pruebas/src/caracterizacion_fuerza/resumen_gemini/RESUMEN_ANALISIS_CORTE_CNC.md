# Análisis de Datos de Corte CNC - Fresado con 2 Dientes

## 1. Configuración Experimental

### Hardware
- **Fresadora CNC vertical**
- **Cortador**: Fresa de 2 dientes
- **RPM estimado**: ~3,720 RPM (basado en frecuencia de paso de dientes)

### Sensores
| Sensor | Modelo | Ubicación | Dirección | DAQ |
|--------|--------|-----------|-----------|-----|
| **Celda de carga** | DYMH-105 | Base de mordaza | X (avance) | NI 9205 ai0 |
| **Acelerómetro** | PCB 352C33 (98.3 mV/g) | Mordaza/sujetador | X (avance) | NI 9234 ai0 (IEPE) |

### Configuración de adquisición
- **Sample Rate**: 2,500 Hz
- **Modo**: Datos crudos (Fuerza en V, Aceleración en g)

### Setup físico
```
        Husillo (Z vertical)
           │
           ▼
    ═══════════════►  X (dirección de corte y avance)
           │
    ┌──────┴──────┐
    │    PIEZA    │ ← Acelerómetro (mide X)
    └─────────────┘
           │
    ┌─────────────┐
    │   MORDAZA   │
    └─────────────┘
           │
    ┌─────────────┐
    │    CELDA    │ ← Mide fuerza en X
    └─────────────┘
```

---

## 2. Datos Capturados

### Archivo principal: `corte2_mejor_0.5s.txt`
- **Duración**: 0.5 segundos
- **Muestras**: 1,250
- **Formato**: TSV (tiempo_s, fuerza_V, aceleracion_sensor_g)

### Estadísticas del segmento
| Variable | Media | Std | Rango |
|----------|-------|-----|-------|
| **Fuerza** | 0.9245 V | 0.034 V | 0.86 - 0.98 V |
| **Aceleración** | 0.004 g | 0.59 g | -4 a +2.5 g |

---

## 3. Análisis Espectral (FFT)

### Frecuencias dominantes
| Señal | Frecuencia | Interpretación |
|-------|-----------|----------------|
| **Fuerza** | ~10 Hz | Ciclo de enganche/desenganche de corte |
| **Aceleración** | ~124 Hz | Frecuencia de paso de dientes |

### Cálculo de RPM
- Frecuencia de dientes: 124 Hz
- Número de dientes: 2
- **RPM = 124 × 60 / 2 = 3,720 RPM**

### Gráfica: `corte2_mejor_segmento.png`
- Muestra señales temporales y FFT de ambos canales

---

## 4. Problema Inicial: Baja Correlación

### Correlación directa F vs a
| Método | Correlación |
|--------|-------------|
| F vs a (cruda) | **0.001** ❌ |
| F vs velocidad integrada | **0.05** ❌ |

### Causa identificada
Las señales están en **diferentes bandas de frecuencia**:
- Fuerza: 10 Hz (fuerza promedio filtrada por masa)
- Aceleración: 124 Hz (vibraciones instantáneas de dientes)

La **masa de la mordaza + pieza actúa como filtro pasa-bajas mecánico**.

### Gráficas relacionadas
- `analisis_corte_reciente.png` - Comparación de dos cortes
- `corte2_filtrado_10Hz.png` - Intento de filtrar aceleración a 10 Hz
- `corte2_alta_frecuencia.png` - Análisis en banda 100-150 Hz

---

## 5. Modelo Bouc-Wen Clásico

### Intento inicial
Se intentó ajustar un modelo Bouc-Wen clásico:
```
F = α·k·x + (1-α)·k·z + c·v

donde z sigue la ecuación diferencial:
dz/dt = A·v - β|v||z|^(n-1)·z - γ·v|z|^n
```

### Parámetros optimizados
| Parámetro | Valor |
|-----------|-------|
| k (rigidez) | 90.87 |
| c (amortiguamiento) | 0.001 |
| α (ratio elástico) | 0.01 |
| A | 9.34 |
| β | 9.99 |
| γ | -4.99 |
| n | 1.0 |

### Resultado
- **Correlación**: -0.03 ❌
- **El modelo no ajusta** porque la velocidad integrada no correlaciona con la fuerza

### Gráfica: `bouc_wen_corte2.png`

---

## 6. SOLUCIÓN: Usar Envolvente de Aceleración

### Concepto físico
La **envolvente** de la aceleración representa la **intensidad promedio de los impactos** de los dientes, que es lo que realmente mide la celda de carga (filtrada por la masa).

### Método
1. Rectificar aceleración: `|a(t)|`
2. Filtro pasa-bajas a 15 Hz
3. Resultado: envolvente que sigue el patrón de la fuerza

### Resultados de correlación
| Método | Correlación |
|--------|-------------|
| **F vs Envolvente** | **0.813** ✅ |
| **F vs RMS (20ms)** | **0.764** ✅ |

### Observaciones clave
1. Fuerza y envolvente siguen el **mismo patrón temporal**
2. Existe un **pequeño desfase** (envolvente adelantada ~10-20ms)
3. Relación **casi lineal**: `F ≈ k × Envolvente + offset`

### Gráfica principal: `corte2_envolvente.png`

---

## 7. Interpretación Física

### Modelo conceptual
```
Impactos de dientes (124 Hz)
        │
        ▼
    ┌───────────┐
    │   MASA    │  ← Mordaza + Pieza (filtro pasa-bajas)
    └───────────┘
        │
        ▼
Fuerza promediada (10 Hz) → Celda de carga
```

### Relación matemática propuesta
```
F(t) ≈ k × Envolvente[a(t-τ)] + c × d/dt{Envolvente[a(t)]}

donde:
- Envolvente[a] = LPF(|a|, fc=15Hz)
- τ = desfase temporal (~10-20 ms)
- k, c = constantes a identificar
```

---

## 8. Próximos Pasos Sugeridos

### Para modelo de histéresis
1. **Usar envolvente como entrada** en lugar de aceleración cruda
2. **Compensar desfase temporal** entre envolvente y fuerza
3. **Entrenar modelo Bouc-Wen** o KAN con estas variables corregidas

### Para mejora experimental
1. Capturar datos con **diferentes RPM** para validar relación
2. Medir **fuerza y aceleración en misma ubicación** (más cerca)
3. Usar **acelerómetro triaxial** para capturar todas las direcciones

### Modelo alternativo sugerido
Red neuronal KAN con entradas:
- Envolvente de aceleración
- Derivada de envolvente
- Historial temporal (ventana de 50-100 ms)

Salida:
- Fuerza de corte predicha

---

## 9. Archivos Incluidos

| Archivo | Descripción |
|---------|-------------|
| `corte2_mejor_0.5s.txt` | Datos crudos del mejor segmento |
| `corte2_mejor_segmento.png` | Señales temporales y FFT |
| `analisis_corte_reciente.png` | Comparación de dos capturas |
| `bouc_wen_corte2.png` | Resultado del modelo Bouc-Wen clásico |
| `corte2_filtrado_10Hz.png` | Análisis con aceleración filtrada |
| `corte2_alta_frecuencia.png` | Análisis en banda 100-150 Hz |
| `corte2_envolvente.png` | **Correlación F vs Envolvente (r=0.81)** |

---

## 10. Código de Análisis

### Cálculo de envolvente
```python
import numpy as np
from scipy import signal

# Calcular envolvente de aceleración
acel_rect = np.abs(acel)  # Rectificar
b, a = signal.butter(2, 15/(fs/2), btype='low')  # Filtro 15 Hz
acel_env = signal.filtfilt(b, a, acel_rect)

# Correlación
corr = np.corrcoef(fuerza, acel_env)[0,1]
print(f'Correlación: {corr:.4f}')  # → 0.8129
```

### Cálculo de RMS móvil
```python
import pandas as pd

window = int(0.02 * fs)  # ventana 20ms
acel_rms = np.sqrt(pd.Series(acel**2).rolling(window, center=True).mean())
```

---

## Conclusión

Los datos de corte CNC muestran una **clara relación física** entre fuerza y aceleración cuando se usa la **envolvente** como variable intermedia. Esto abre la puerta a:

1. **Estimación de fuerza a partir de aceleración** (monitoreo indirecto)
2. **Modelo de histéresis** basado en envolvente + desfase temporal
3. **Detección de anomalías** comparando envolvente esperada vs medida

La correlación de **0.81** es suficientemente alta para entrenar modelos predictivos.
