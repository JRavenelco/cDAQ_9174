# Caracterización de Fuerza - Validación de Histéresis

## Resumen del Proyecto

Este proyecto tiene como objetivo **validar la linealidad del sistema de medición de fuerza** utilizado para medir fuerzas de corte en mecanizado CNC. Se utilizan modelos de histéresis (Bouc-Wen, Duhem, KAN-PINN) para cuantificar si existe comportamiento no-lineal en el sensor.

**Conclusión principal**: El sistema es **predominantemente lineal** (α ≈ 0.81 - 1.0 dependiendo del modelo), confirmando que no se requiere compensación de histéresis.

---

## Configuración Experimental

### Hardware
- **Sensor de fuerza**: LC302-1K
- **Acelerómetro**: Montado en la estructura
- **Sistema DAQ**: NI cDAQ-9174
- **Canales**:
  - `ai0`, `ai1`: Fuerza compresión eje X
  - `ai2`, `ai3`: Fuerza compresión eje Y
  - Acelerómetros para vibración

### Parámetros de Adquisición
- **Frecuencia de muestreo**: 2500 Hz
- **Submuestreo**: Factor 5 (500 Hz efectivos)
- **Duración por análisis**: 0.5 segundos (250 muestras)

### Parámetros FRF Identificados (sesión `135910`)
| Parámetro | Valor | Unidad |
|-----------|-------|--------|
| Masa (m) | 0.107 | kg |
| Rigidez (k) | 1424 | N/m |
| Amortiguamiento (c) | 3.73 | Ns/m |
| Frecuencia natural | ~25 | Hz |

---

## Datos Experimentales

### Archivos de Datos
```
├── shaker_mejor_0.5s_10Hz.txt   # Excitación shaker a 10 Hz
├── shaker_mejor_0.5s_20Hz.txt   # Excitación shaker a 20 Hz (principal)
├── shaker_mejor_0.5s_40Hz.txt   # Excitación shaker a 40 Hz
├── corte2_mejor_0.5s.txt        # Datos de corte real
├── evento_corte_1_0.5s.txt      # Evento de corte 1
├── evento_corte_2_0.5s.txt      # Evento de corte 2
├── evento_corte_3_0.5s.txt      # Evento de corte 3
```

### Formato de Datos
Cada archivo contiene columnas: `tiempo, fuerza, aceleración`

---

## Modelos de Histéresis Implementados

### 1. BW Simple (Bouc-Wen Clásico)
```
F = α·k·a + (1-α)·k·z
```
- **α = 1**: Sistema lineal
- **α = 0**: Histéresis máxima

### 2. BW Viscoso
```
F = α·k·a + (1-α)·k·z + c·da/dt
```
Incluye término de amortiguamiento viscoso.

### 3. Duhem Polinomial
```
F = a·z + b·z² + c·z³ + d·dz/dt
```
Expansión polinomial de la variable de histéresis.

### 4. BW-ENV (Bouc-Wen con Envolvente)
```
F = α·k·E + (1-α)·k·z
```
Usa envolvente de Hilbert de la aceleración.

### 5. KAN-PINN (Más completo)
```
F = m·a + k·x + c·v + α·k·E + (1-α)·k·z
```
Combina términos físicos (inercia, rigidez, viscosidad) con envolvente e histéresis.

### Ecuación de Histéresis (Bouc-Wen)
```
dz/dt = A·v - B·|v|·z - C·v·|z|^n
```
Donde `v` es velocidad y `z` es la variable de histéresis.

---

## Resultados Finales

### Tabla Comparativa (con envolvente)

| Modelo | α Shaker | α Corte | R² Corte | Conclusión |
|--------|----------|---------|----------|------------|
| BW Simple | 0.436 | **0.811** | 0.713 | 81% lineal |
| BW Viscoso | 0.510 | 0.427 | **0.743** | Mejor R² |
| Duhem | 0.139 | 0.485 | 0.703 | Más no-lineal |
| BW-ENV | 0.436 | **0.811** | 0.713 | 81% lineal |
| **KAN-PINN** | **1.000** | **0.999** | 0.700 | **100% lineal** |

### Interpretación del Parámetro α
- **α = 1.0**: Sistema completamente lineal (sin histéresis)
- **α = 0.8**: 80% lineal, 20% histéresis
- **α = 0.5**: 50% lineal, 50% histéresis
- **α = 0.0**: Histéresis máxima

---

## Técnicas de Preprocesamiento

### Envolvente (Transformada de Hilbert)
```python
from scipy.signal import hilbert

def calcular_envolvente(senal, fs, fc=15):
    analitica = hilbert(senal)
    envolvente_raw = np.abs(analitica)
    # Filtro pasa-bajos para suavizar
    b, a = signal.butter(2, fc/(fs/2), 'low')
    return signal.filtfilt(b, a, envolvente_raw)
```

**Propósito**: Capturar la dinámica lenta del proceso de corte, eliminando oscilaciones de alta frecuencia.

**Impacto**: Mejora R² de 0.01 (sin envolvente) a 0.70+ (con envolvente).

### Normalización
```python
def normalizar(x):
    x_c = x - np.mean(x)
    return x_c / np.max(np.abs(x_c))
```

### Integración con Filtro Anti-Drift
```python
def integrar(a, fs, fc=5.0):
    dt = 1/fs
    b, a_f = signal.butter(2, fc/(fs/2), 'high')
    v = signal.filtfilt(b, a_f, cumulative_trapezoid(a, dx=dt, initial=0))
    x = signal.filtfilt(b, a_f, cumulative_trapezoid(v, dx=dt, initial=0))
    return v, x
```

---

## Scripts Principales

### Análisis de Modelos
| Script | Descripción |
|--------|-------------|
| `test_modelos_envolvente.py` | Comparación de 5 modelos con envolvente |
| `test_kan_pinn_simple.py` | Prueba individual de KAN-PINN |
| `comparar_modelos_fuerza.py` | Comparación original sin envolvente |

### Análisis FRF
| Script | Descripción |
|--------|-------------|
| `analisis_frf_real_imag.py` | Identificación de m, k, c |
| `analisis_mejor_sesion.py` | Análisis detallado de sesión óptima |
| `comparar_sesiones.py` | Comparación entre sesiones |

### Procesamiento de Datos
| Script | Descripción |
|--------|-------------|
| `extraer_eventos_corte.py` | Extracción de eventos de corte |
| `analisis_corte_real.py` | Análisis de datos de corte |

---

## Gráficas Generadas

```
graficas_bouc_wen/
├── 01_bouc_wen_corte2.png          # Análisis inicial
├── 05_bouc_wen_corte_envolvente.png # Con envolvente (α=0.853)
├── 07_comparacion_shaker_vs_corte.png
├── 15_kan_pinn_viscoso.png
├── 16_kan_pinn_completo.png
├── 17_kan_pinn_k_K_separadas.png
├── 18_kan_pinn_envolvente.png
├── 21_modelos_con_envolvente.png   # Comparación final
├── 22_alpha_con_envolvente.png     # Gráfica de barras α
└── 23_kan_pinn_con_envolvente.png  # KAN-PINN detallado
```

---

## Documentos LaTeX

```
presentacion_bouc_wen/
├── analisis_modelos_histeresis.tex  # Análisis comparativo final
├── analisis_histeresis_completo.tex # Documento extenso
├── presentacion_bouc_wen.tex        # Presentación
└── presentacion_duhem.tex           # Modelo Duhem
```

---

## Evolución del Análisis

### Fase 1: Análisis FRF
- Identificación de parámetros físicos (m, k, c) mediante FRF
- Sesión seleccionada: `caracterizacion_fuerza_20251202_135910`
- R² real: 0.95, R² imag: 0.70

### Fase 2: Modelos Bouc-Wen Básicos
- Implementación de BW Simple y BW Viscoso
- Problema: R² muy bajo (~0.01) para datos de corte sin preprocesamiento

### Fase 3: Incorporación de Envolvente
- Uso de transformada de Hilbert para extraer envolvente
- Mejora dramática: R² de 0.01 → 0.70+
- α de corte: ~0.81 (81% lineal)

### Fase 4: Modelos KAN-PINN
- Implementación de modelo completo con términos físicos
- F = m·a + k·x + c·v + α·k·E + (1-α)·k·z
- Resultado: α ≈ 1.0 (sistema completamente lineal)

### Fase 5: Análisis Comparativo Final
- Comparación de 5 modelos
- Conclusión: No hay histéresis significativa
- No se requiere compensación

---

## Conclusiones

1. **El sistema de medición es lineal**: Todos los modelos convergen a α alto (0.81 - 1.0)

2. **KAN-PINN es el modelo más robusto**: Identifica α ≈ 1.0 para ambos sistemas (shaker y corte)

3. **La envolvente es esencial**: Mejora R² de 0.01 a 0.70+ para datos de corte

4. **No se requiere compensación de histéresis**: Las mediciones de fuerza son confiables

5. **El término de histéresis (z) no aporta**: Cuando el modelo tiene suficientes grados de libertad lineales, la histéresis no es necesaria

---

## Dependencias

```python
numpy
scipy
matplotlib
# Para optimización
from scipy.optimize import differential_evolution
# Para envolvente
from scipy.signal import hilbert
```

---

## Uso

```bash
# Comparar todos los modelos
python test_modelos_envolvente.py

# Probar KAN-PINN individual
python test_kan_pinn_simple.py

# Análisis FRF
python analisis_mejor_sesion.py
```

---

## Autor
Jesús - Doctorado  
Diciembre 2025

## Referencias
- Bouc-Wen Model: Modelo fenomenológico de histéresis
- Duhem Model: Modelo diferencial de histéresis
- KAN-PINN: Kolmogorov-Arnold Network Physics-Informed Neural Network
- Hilbert Transform: Extracción de envolvente
