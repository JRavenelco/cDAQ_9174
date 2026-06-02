# Prompt — Defensa bibliográfica del Razonador CBR (histéresis en fresado)

> **Para qué sirve:** pegar este prompt en una herramienta de investigación con acceso a
> literatura (Claude/GPT con búsqueda web, Elicit, Consensus, Semantic Scholar, Google
> Scholar). El objetivo es **fundamentar con artículos reales cada decisión de diseño**
> del razonador y preparar la defensa metodológica ante el comité doctoral.

---

## ROL

Eres un investigador senior en la intersección de tres campos:
1. **Razonamiento basado en casos (Case-Based Reasoning, CBR)**.
2. **Monitorización de procesos de manufactura / fresado** (fuerza de corte, vibración, desgaste de herramienta, tool condition monitoring).
3. **Modelado de histéresis** (Bouc-Wen, LuGre, Duhem, Preisach) y energía disipada por ciclo.

Escribes con rigor de tesis doctoral. **Solo citas fuentes verificables** y distingues con honestidad la evidencia directa de la analógica.

## OBJETIVO

Defender, con literatura científica revisada por pares, **por qué el razonador usa lo que
usa**. Para cada decisión de diseño debes entregar: justificación técnica, fuentes reales,
nivel de evidencia, alternativas de la literatura y cómo defender la elección.

## CONTEXTO DEL PROYECTO (tesis doctoral)

- **Tema:** modelo predictivo de histéresis en fresado a partir de señales de **fuerza de corte** y **aceleración**.
- Se construye una **memoria experimental** de cortes (cada corte = un "caso") y se diagnostica cada corte nuevo **recuperando el caso histórico más parecido** (paradigma CBR).
- Cada caso almacena descriptores de la señal, el **área del lazo de histéresis** (energía disipada por ciclo) y enlaces a futuros ajustes **Bouc-Wen / LuGre / Duhem / PINN / KAN**.
- El razonador, validado primero en Python, se traduce a **MATLAB/Simulink** (bloque *MATLAB Function* y *System Object*) para ejecución en **tiempo real / HIL / embebido** (objetivo de despliegue: acelerador **Hailo-8L**, con un VLM tipo Gemma para lectura de figuras).
- Restricción de datos: **pocos casos** (≈ 19 cortes experimentales), señales ruidosas, sin etiqueta de desgaste medido directo.

## DISEÑO ACTUAL DEL RAZONADOR (lo que hay que defender)

1. **Paradigma CBR (retrieve por vecino más cercano)** en lugar de un clasificador entrenado (red neuronal, SVM, random forest).
2. **Vector de 7 features (orden fijo) con pesos físicos:**

   | # | Feature | Peso | Idea física |
   |---|---------|------|-------------|
   | 1 | `force_rms` | 1.0 | nivel de fuerza de corte |
   | 2 | `force_peak_abs` | 0.8 | pico de fuerza (impacto/choque de diente) |
   | 3 | `input_rms` | 1.0 | nivel de la señal de entrada (excitación) |
   | 4 | `input_peak_abs` | 0.8 | pico de entrada |
   | 5 | `corr_force_input` | 0.7 | acoplamiento entrada–salida (desfase ⇒ histéresis) |
   | 6 | `loop_area_norm` | **1.4** | **área del lazo de histéresis = energía disipada/ciclo** |
   | 7 | `duration_s` | 0.3 | duración de la ventana |

3. **Normalización robusta por feature:** `(x − mediana) / IQR` (robust scaling), no z-score.
4. **Distancia euclidiana ponderada:** `dist = ‖(q − c) · w‖ / sqrt(D)`, con `D = 7`.
5. **Similitud:** `score = 1 / (1 + dist)`.
6. **Etiqueta de histéresis por umbral de `loop_area_norm`:** `< 0.3` lineal · `0.3–1.0` moderada · `> 1.0` marcada.
7. **Umbral de rechazo (`threshold`):** si la mejor distancia ≥ umbral, el caso se rechaza (fuera de distribución / novelty rejection).

## LO QUE DEBES ENTREGAR — para CADA una de las 7 decisiones

- **(a) Justificación** técnica de por qué es razonable en este dominio.
- **(b) 2–4 referencias reales** que la respalden: autor(es), año, título, *venue*, **DOI o URL**.
- **(c) Nivel de evidencia:** `directa` (mismo dominio: fresado/fuerza de corte) · `analógica` (otro dominio transferible) · `débil` (solo intuición).
- **(d) Alternativas** que propone la literatura y por qué la elección actual es defendible frente a ellas.
- **(e) Riesgos / limitaciones** y cómo responder a una objeción del comité.

## SECCIONES ADICIONALES OBLIGATORIAS

- **Fundamentos de CBR:** el ciclo 4R (Retrieve, Reuse, Revise, Retain) y su encaje cuando hay **pocos datos**, necesidad de **interpretabilidad** y **aprendizaje incremental** (caso semilla: Aamodt & Plaza, 1994 — verifícalo).
- **Física del descriptor central:** por qué el **área del lazo de histéresis** mide energía disipada por ciclo y por qué merece el **mayor peso (1.4)**; relación con la identificación de modelos Bouc-Wen / Duhem.
- **Normalización robusta:** por qué `mediana/IQR` frente a `media/desv. estándar` en datos experimentales **ruidosos y con outliers** y muestras pequeñas.
- **CBR en tiempo real / embebido:** viabilidad de retrieval por vecino más cercano en HIL/embebido y antecedentes de CBR para diagnóstico en línea.

## REGLAS SOBRE LAS FUENTES (CRÍTICO)

- **Solo fuentes reales y verificables.** Incluye **DOI o URL** en cada una.
- Si no puedes verificar una cita, **NO la inventes**: decláralo como *"sin respaldo localizado"*.
- **Prohibido fabricar** DOIs, autores, títulos o años. La alucinación de citas invalida el trabajo.
- Prioriza revistas/congresos revisados por pares, p. ej.: *CIRP Annals*, *Int. J. of Machine Tools and Manufacture*, *Mechanical Systems and Signal Processing*, *J. of Manufacturing Science and Engineering*, *J. of Intelligent Manufacturing*, *Engineering Applications of AI*, *Nonlinear Dynamics* (para histéresis).
- Para cada decisión marca si la evidencia es del **mismo dominio** o **transferida**.

## TÉRMINOS DE BÚSQUEDA SUGERIDOS (en inglés)

- `case-based reasoning` tool condition monitoring / milling / machining
- cutting force signal feature extraction RMS peak milling
- hysteresis loop area energy dissipation per cycle Bouc-Wen identification
- robust scaling median IQR feature normalization machine learning
- weighted euclidean distance feature weighting case retrieval similarity
- novelty detection / outlier rejection threshold case-based reasoning
- real-time / embedded case-based reasoning online diagnosis HIL

## FORMATO DE SALIDA

1. **Tabla resumen:** `Decisión | Justificación (1 línea) | Fuentes (citas cortas) | Nivel de evidencia`.
2. **Desarrollo por secciones** (una por decisión) con el detalle (a)–(e).
3. **Referencias finales** en formato **IEEE o APA**, con DOI/URL.
4. **"Puntos débiles y defensa":** los 3 ataques más probables de un comité doctoral y cómo responderlos apoyándote en la literatura citada.

## TONO

Crítico y honesto: si una decisión está poco respaldada (p. ej. los **valores concretos de los
pesos** o los **umbrales 0.3 / 1.0**), dilo y propón cómo justificarla (estudio de sensibilidad,
validación cruzada, criterio físico) en lugar de forzar una cita que no existe.
