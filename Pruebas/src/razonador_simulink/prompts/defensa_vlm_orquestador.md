# Prompt — Defensa bibliográfica del VLM-orquestador del ciclo 4R (CBR + visión)

> **Para qué sirve:** pegar este prompt en una herramienta de investigación con acceso a
> literatura (Claude/GPT con búsqueda web, Elicit, Consensus, Semantic Scholar, Google
> Scholar). El objetivo es **fundamentar con artículos reales** por qué un **VLM orquesta
> el ciclo 4R de un razonador CBR** (selecciona el caso, valida con visión y decide en OOD),
> y preparar la defensa metodológica ante el comité doctoral.
>
> **Artefacto de salida esperado:** `compass_artifact_vlm_*.md` (mismas reglas que el
> artefacto del razonador CBR base).

---

## ROL

Eres un investigador senior en la intersección de cuatro campos:
1. **Razonamiento basado en casos (Case-Based Reasoning, CBR)** y su ciclo 4R.
2. **Modelos de lenguaje/visión (LLM/VLM) como agentes orquestadores con uso de herramientas (tool-use)**.
3. **Aprendizaje con opción de rechazo / predicción selectiva / deferral** (cascadas de modelos).
4. **Inspección y diagnóstico visual en manufactura** (lectura de curvas, lazos, imágenes de proceso).

Escribes con rigor de tesis doctoral. **Solo citas fuentes verificables** y distingues con
honestidad la evidencia directa de la analógica.

## OBJETIVO

Defender, con literatura científica revisada por pares, **por qué es metodológicamente
sólido** que un VLM orqueste el ciclo 4R de un CBR determinista, actuando **por excepción**
(System 1 / System 2). Para cada decisión de diseño debes entregar: justificación técnica,
fuentes reales, nivel de evidencia, alternativas de la literatura y cómo defender la elección.

## CONTEXTO DEL PROYECTO (tesis doctoral)

- **Tema:** modelo predictivo de histéresis en fresado a partir de **fuerza de corte** y
  **aceleración**. Cada corte = un "caso"; se diagnostica recuperando el caso histórico más
  parecido (paradigma CBR, ciclo 4R: Retrieve → Reuse → Revise → Retain).
- **Capa reactiva (System 1):** CBR determinista en MATLAB/Simulink (kNN ponderado sobre 7
  features + clasificación por umbral de `loop_area_norm`), embebible en **Hailo-8L**, ms.
- **Capa deliberativa (System 2):** un **VLM** (qwen2.5-VL, dual Ollama local ↔ OpenRouter)
  que recibe **la imagen del lazo F–x normalizado + las 7 features + los top-K casos** y
  orquesta el 4R. Se dispara **solo por excepción** (baja confianza / OOD).
- **Restricción de datos:** **pocos casos** (≈ 19 cortes), señales ruidosas, sin etiqueta de
  desgaste medido directo.

## DISEÑO A DEFENDER (autoridad HÍBRIDA del VLM)

1. **Disparo por excepción** (no en cada corte): si el CBR tiene **alta confianza**, decide
   el CBR directamente; el VLM solo entra cuando hay duda. (Patrón cascada / coste-aware.)
2. **Alta confianza → CBR directo** (`fuente = CBR`). Reproducible y determinista.
3. **Baja confianza, NO OOD → VLM AUDITOR:** la clase oficial sigue siendo la del CBR; si el
   VLM discrepa, se marca `flag_discrepancia = true` y se guarda su explicación para revisión
   humana (`fuente = CBR+audit`). El resultado sigue siendo reproducible (manda el CBR).
4. **OOD (`min_dist ≥ threshold`, novedad) → VLM DECISOR:** la clase final la da el VLM
   (`fuente = VLM`), porque por diseño el CBR ya no es fiable fuera de su distribución.
5. **Entrada multimodal al VLM:** imagen del **lazo F–x** (representación visual de la
   histéresis) + vector de 7 features + top-K casos recuperados.
6. **Cierre del ciclo (Retain):** un caso OOD validado puede añadirse a la base (aprendizaje
   incremental).
7. **Gobernanza / reproducibilidad:** temperatura baja, y **registro de prompt + respuesta +
   modelo + versión** en cada llamada al VLM (trazabilidad doctoral).

## HONESTIDAD METODOLÓGICA (OBLIGATORIO declararlo)

El VLM aporta **redundancia de representación**: mira la **misma señal** en otra
representación (imagen del lazo F–x vs. vector de 7 features). **NO es evidencia
independiente.** Su valor es **atrapar la fragilidad de la representación escalar** (los
umbrales 0.3/1.0 y los pesos sin cita), no "confirmar" el diagnóstico de forma independiente.
Esto es **distinto** de la **validez convergente** del DOE (Ra = rugosidad/superficie ⟂
loop_area = señal dinámica, canales físicamente independientes). Ante el comité, **mantener
separados** ambos argumentos y no sobrevender el VLM.

## LO QUE DEBES ENTREGAR — para CADA una de las 7 decisiones

- **(a) Justificación** técnica de por qué es razonable en este dominio.
- **(b) 2–4 referencias reales** que la respalden: autor(es), año, título, *venue*, **DOI o URL**.
- **(c) Nivel de evidencia:** `directa` (mismo dominio: CBR/manufactura/diagnóstico) ·
  `analógica` (otro dominio transferible: NLP, visión general, ML teórico) · `débil` (intuición).
- **(d) Alternativas** que propone la literatura y por qué la elección es defendible.
- **(e) Riesgos / limitaciones** y cómo responder a una objeción del comité.

## SECCIONES ADICIONALES OBLIGATORIAS

- **VLM/LLM como agente orquestador con tool-use** sobre un sistema simbólico (CBR): patrón
  *razonar+actuar* y llamada a herramientas. Hilos a rastrear (DOI por verificar, **sin
  inventar**): **ReAct** (Yao et al., 2022/2023); **Toolformer** (Schick et al., 2023).
- **Escalado en cascada / learning-to-defer / reject option:** modelo barato determinista +
  escalado a modelo capaz en casos difíciles/OOD. Hilos: **reject option** (Chow, 1970);
  **selective prediction** (El-Yaniv & Wiener, 2010; Geifman & El-Yaniv, 2017);
  **learning-to-defer** (Madras, Pitassi & Zemel, 2018). Conectar con **novelty rejection en
  CBR** (Perner, 2008) ya citado en el razonador base.
- **Foundation / vision-language models para inspección y diagnóstico en manufactura**
  (leer curvas/lazos, detección de defectos, monitoreo de proceso). Buscar evidencia
  `directa` reciente (2023–2025) y marcar honestamente si es escasa.
- **Human/AI-on-the-loop y supervisión:** por qué el VLM **no rompe la interpretabilidad**
  (explica su veredicto + cita el caso recuperado; el CBR sigue trazable).
- **Reproducibilidad y gobernanza de LLMs en ciencia:** no-determinismo, temperatura, versión
  de modelo, registro de prompts; cómo el rol híbrido **acota** el no-determinismo a los casos OOD.
- **Dual-process (System 1 / System 2)** como encuadre: CBR rápido/automático vs. VLM
  lento/deliberativo (Kahneman, 2011 como marco conceptual; verificar usos en ML/IA híbrida).

## REGLAS SOBRE LAS FUENTES (CRÍTICO)

- **Solo fuentes reales y verificables.** Incluye **DOI o URL** en cada una.
- Si no puedes verificar una cita, **NO la inventes**: decláralo como *"sin respaldo localizado"*.
- **Prohibido fabricar** DOIs, autores, títulos o años. La alucinación de citas invalida el trabajo.
- Prioriza venues revisados por pares y de prestigio: *NeurIPS, ICML, ICLR, ACL, EMNLP, TPAMI,
  JMLR* (ML/NLP); *CIRP Annals, Int. J. Machine Tools & Manufacture, Mechanical Systems and
  Signal Processing, J. Intelligent Manufacturing, Engineering Applications of AI* (manufactura).
- Para cada decisión marca si la evidencia es del **mismo dominio** o **transferida**.
- **Distingue preprints (arXiv) de versiones revisadas por pares** y dilo explícitamente.

## TÉRMINOS DE BÚSQUEDA SUGERIDOS (en inglés)

- `LLM agent tool use orchestration` / `ReAct reasoning acting language models`
- `Toolformer language models use tools`
- `learning to defer` / `learning to reject` / `reject option classification`
- `selective prediction neural networks` / `classification with a reject option Chow`
- `model cascade cheap expensive classifier escalation cost-aware inference`
- `vision language model industrial inspection defect detection manufacturing`
- `foundation models manufacturing process monitoring diagnosis`
- `case-based reasoning novelty detection out-of-distribution rejection`
- `human-on-the-loop AI oversight interpretability symbolic neural`
- `reproducibility large language models science temperature determinism governance`
- `dual process system 1 system 2 hybrid AI neuro-symbolic`

## FORMATO DE SALIDA (artefacto `compass_artifact_vlm_*.md`)

1. **Tabla resumen:** `Decisión | Justificación (1 línea) | Fuentes (citas cortas) | Nivel de evidencia`.
2. **Desarrollo por secciones** (una por decisión) con el detalle (a)–(e).
3. **Sección de honestidad:** declaración explícita de *redundancia de representación* vs.
   *validez convergente* (ver arriba), para blindar el argumento ante el comité.
4. **Referencias finales** en formato **IEEE o APA**, con DOI/URL, separando peer-reviewed de preprints.
5. **"Puntos débiles y defensa":** los 3 ataques más probables de un comité doctoral
   (p. ej. *"el VLM alucina"*, *"no es reproducible"*, *"no aporta información independiente"*)
   y cómo responderlos apoyándote en la literatura citada y en el diseño híbrido.

## TONO

Crítico y honesto: si una decisión está poco respaldada por evidencia `directa` (p. ej. VLM
para lectura de lazos de histéresis en fresado es probablemente escaso), **dilo** y apóyate en
evidencia `analógica` (inspección visual industrial, lectura de gráficas científicas) marcándola
como tal. No fuerces citas que no existen.
