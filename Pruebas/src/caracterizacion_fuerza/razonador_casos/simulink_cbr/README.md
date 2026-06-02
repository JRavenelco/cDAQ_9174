# Razonador CBR en Simulink

Traducción del Razonador Basado en Casos (`razonador_local.py`) a MATLAB/Simulink
para diagnóstico de histéresis en fresado. Replica **exactamente** la lógica de
retrieval de Python:

- 7 features con pesos
- Normalización robusta: `(x - mediana) / IQR`
- Distancia euclidiana ponderada: `dist = norm((q-c).*w) / sqrt(7)`
- Similitud: `score = 1 / (1 + dist)`

## Features (orden FIJO)

| # | Feature | Peso |
|---|---------|------|
| 1 | `force_rms` | 1.0 |
| 2 | `force_peak_abs` | 0.8 |
| 3 | `input_rms` | 1.0 |
| 4 | `input_peak_abs` | 0.8 |
| 5 | `corr_force_input` | 0.7 |
| 6 | `loop_area_norm` | 1.4 |
| 7 | `duration_s` | 0.3 |

## Archivos

| Archivo | Parte | Qué hace |
|---------|-------|----------|
| `exportar_base_casos.m` | 1 | Lee `casos_historicos.csv` → genera `datos/base_de_casos_cbr.mat` |
| `cbr_reasoner.m` | 2 | Bloque **MATLAB Function** (codegen, tiempo real / HIL) |
| `CBRHysteresisSystem.m` | 3 | Bloque **MATLAB System** (threshold tunable en caliente) |
| `probar_cbr.m` | 4a | Valida la traducción fuera de Simulink (1-vs-resto) |
| `crear_modelo_simulink.m` | 4b | Arma el modelo Simulink de demo automáticamente |

## Flujo de uso (4 pasos)

### 1. Generar la base de casos
```matlab
>> cd Pruebas\src\caracterizacion_fuerza\razonador_casos\simulink_cbr
>> exportar_base_casos
```
Crea `datos/base_de_casos_cbr.mat` con la matriz de casos, centros, escalas,
pesos, etiquetas de histéresis y IDs.

### 2. Validar la traducción (sin Simulink)
```matlab
>> probar_cbr
```
Recorre los 19 casos y muestra el vecino más cercano de cada uno. Compáralo con
el razonador de Python para confirmar que coinciden.

### 3. Armar el modelo Simulink (automático)
```matlab
>> crear_modelo_simulink
```
Genera y abre `modelo_cbr_hysteresis.slx`:
`Constant (features) → CBR Reasoner → Displays`. Pulsa **Run**.

### 4. (Manual) Usar el bloque en tu propio modelo
- Arrastra **MATLAB System** desde la librería de Simulink.
- En "System name" escribe: `CBRHysteresisSystem`
- En **Simulate using** elige **`Interpreted execution`** (ver nota abajo).
- Conecta una señal `[1x7]` (las 7 features) a la entrada `features`.
- Ajusta `Threshold` desde el diálogo del bloque (ej. `1.0`; usa `1e6` para no rechazar).

> **IMPORTANTE — Simulate using:**
> `CBRHysteresisSystem` usa `load()` de un `.mat`, que produce tamaños variables y
> NO es compatible con generación de código. Por eso este bloque debe correr en
> **`Interpreted execution`** (el builder ya lo configura así).
>
> Si necesitas **generación de código C / tiempo real / HIL**, usa el bloque
> **MATLAB Function** con `cbr_reasoner.m` (Parte 2), que sí es `%#codegen` gracias
> a `coder.load`.

## Salidas del bloque

| Salida | Significado |
|--------|-------------|
| `class_id` | 1=lineal, 2=moderada, 3=marcada, 0=rechazado (dist ≥ threshold) |
| `min_dist` | distancia ponderada al caso más cercano |
| `best_score` | similitud `1/(1+min_dist)` |
| `best_idx` | índice del caso recuperado en la base (0 si rechazado) |

> Para mapear `best_idx` → `case_id`, carga `base_de_casos_cbr.mat` y usa
> `case_ids(best_idx)`.

## ¿MATLAB Function o MATLAB System?

| | MATLAB Function (`cbr_reasoner.m`) | MATLAB System (`CBRHysteresisSystem.m`) |
|---|---|---|
| Complejidad | Baja (imperativo) | Media (OOP) |
| Tiempo real / Coder | Óptimo (C/C++ nativo) | Soportado |
| Threshold ajustable en caliente | Requiere entrada extra | Trivial (propiedad tunable) |
| Ideal para | HIL, embebido | Lógica con estado / UI |

**Recomendación:** usa `CBRHysteresisSystem` para experimentar en Simulink (puedes
mover el `Threshold` con un slider), y `cbr_reasoner` cuando vayas a generar código C
para tiempo real.

## Etiqueta de histéresis (derivada de `loop_area_norm`)

- `< 0.3` → **lineal** (clase 1)
- `0.3 – 1.0` → **moderada** (clase 2)
- `> 1.0` → **marcada** (clase 3)

Ajusta estos umbrales en `exportar_base_casos.m` si tu criterio físico cambia.

## Nota sobre la normalización

En Python las estadísticas (mediana/IQR) se recalculan incluyendo el caso query.
Aquí se **precomputan offline** desde la librería (19 casos) y se embeben en el
`.mat`, porque en tiempo real no puedes recalcular el dataset completo en cada paso.
El efecto de 1 caso extra sobre 19 es despreciable; los vecinos recuperados coinciden.
