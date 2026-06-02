# Prompt — Defensa bibliográfica del VLM‑orquestador del razonador CBR

> **Para qué sirve:** pegar en una herramienta de investigación con acceso a literatura
> (Claude/GPT con búsqueda web, Elicit, Consensus, Semantic Scholar, Google Scholar).
> Objetivo: **fundamentar con artículos reales** por qué el razonador usa un **VLM como
> orquestador (System 2)** sobre el CBR determinista (System 1), y preparar la defensa
> ante el comité doctoral. Es hermano de `defensa_bibliografica_razonador.md` (que ya
> defendió el CBR) y se enlaza con él por el **rechazo por novedad de Perner (2008)**.

---

## ROL

Eres un investigador senior en la intersección de:
1. **Selective prediction / reject option / learning‑to‑defer** (cuándo un clasificador debe abstenerse o derivar a un experto).
2. **Agentes LLM/VLM con *tool‑use*** (un modelo de lenguaje que orquesta herramientas/sistemas simbólicos).
3. **Foundation / vision‑language models para inspección y diagnóstico en manufactura** (lectura de curvas, imágenes de proceso).
4. **Reproducibilidad y gobernanza de LLMs** en investigación científica.

Escribes con rigor de tesis doctoral. **Solo citas fuentes verificables** y distingues evidencia directa de analógica.

## OBJETIVO

Defender **por qué el razonador incorpora un VLM como orquestador** del ciclo CBR, con estas características concretas, y entregar para cada una: justificación, fuentes reales, nivel de evidencia, alternativas y riesgos.

## CONTEXTO DEL PROYECTO (tesis doctoral)

- Sistema de diagnóstico de **histéresis en fresado**. Núcleo: un **CBR determinista** (recuperación por vecino más cercano sobre 7 features, ciclo 4R de Aamodt & Plaza 1994) ya defendido bibliográficamente.
- Se añade un **VLM orquestador** (`qwen2.5-VL`, dual Ollama local ↔ OpenRouter) que **selecciona** (qué caso top‑K aplica / rechazo) y **valida** (fase *Revise* del 4R) mirando la **imagen del lazo fuerza–desplazamiento**.
- **Escalado por excepción:** el CBR resuelve los cortes claros en ms (embebible, Hailo‑8L); el VLM entra **solo cuando la confianza es baja**.
- **Rol híbrido del VLM:** *auditor* en baja confianza no‑OOD (el CBR mantiene la clase oficial; el VLM marca discrepancias), y *decisor* **solo en casos OOD** (fuera de distribución), donde el CBR ya no es fiable por diseño.
- Encuadre: **CBR = System 1** (rápido, determinista), **VLM = System 2** (lento, deliberativo, por excepción).

## DECISIONES A DEFENDER (con fuentes)

1. **Usar un VLM/LLM como orquestador con *tool‑use*** sobre un sistema simbólico (el CBR), en lugar de un único modelo monolítico.
2. **Escalado por excepción** como un problema clásico de **reject option / selective prediction / learning‑to‑defer**: un modelo barato y fiable decide por defecto y deriva los casos difíciles a un razonador más capaz.
3. **Rol híbrido (decisor solo en OOD):** acotar la autoridad del VLM a los casos de novedad (enlaza con el rechazo OOD de Perner 2008 ya citado en la defensa del CBR).
4. **VLM con visión** que lee la **forma del lazo F–x** (no solo el área escalar): foundation/vision‑language models para inspección y diagnóstico visual en manufactura.
5. **El VLM NO rompe la interpretabilidad** que defiende el CBR: explica en lenguaje natural y **cita el caso recuperado**; el CBR sigue siendo la base trazable.
6. **HONESTIDAD — redundancia de representación, NO evidencia independiente:** el VLM observa la **misma señal** en otra representación (imagen del lazo vs vector de 7 features). Su valor es **atrapar la fragilidad de la representación escalar** (los umbrales 0.3/1.0), no aportar un canal de evidencia independiente. Esto debe distinguirse explícitamente de la **validez convergente** (p. ej. rugosidad Ra ⟂ área de lazo, que sí son canales físicamente independientes).
7. **Reproducibilidad y gobernanza del VLM** en un contexto doctoral: temperatura baja, versión de modelo fijada, registro de prompt+respuesta.

## LO QUE DEBES ENTREGAR — para CADA decisión
- **(a) Justificación** técnica de por qué es razonable.
- **(b) 2–4 referencias reales** con autor, año, título, *venue* y **DOI/URL**.
- **(c) Nivel de evidencia:** `directa` (mismo problema) · `analógica` (transferible) · `débil`.
- **(d) Alternativas** de la literatura y por qué la elección es defendible.
- **(e) Riesgos / limitaciones** y cómo responder a una objeción del comité.

## SECCIONES ADICIONALES OBLIGATORIAS
- **System 1 / System 2** como marco (modelo rápido por defecto + modelo deliberativo por excepción) — encuadrar, sin sobre‑atribuir.
- **El argumento de honestidad (decisión 6)** desarrollado: por qué redundancia de representación ≠ validez convergente, y por qué aun así es valiosa (robustez frente al modo de fallo de la representación escalar).
- **Continuidad con la defensa del CBR:** cómo Perner (2008) / rechazo por novedad es la bisagra hacia reject option (Chow 1970) y learning‑to‑defer.

## REGLAS SOBRE LAS FUENTES (CRÍTICO)
- **Solo fuentes reales y verificables.** Incluye **DOI o URL**.
- Si no puedes verificar una cita, **NO la inventes**: decláralo *"sin respaldo localizado"*.
- **Prohibido fabricar** DOIs, autores, títulos o años. La alucinación de citas invalida el trabajo.
- Prioriza venues revisados: *JMLR, NeurIPS, ICLR, IEEE T‑*, ICML*; para manufactura *J. Intelligent Manufacturing, J. Manufacturing Systems, CIRP, Engineering Applications of AI*.

## TÉRMINOS DE BÚSQUEDA SUGERIDOS (inglés)
- reject option classification `Chow 1970`; classification with a reject option
- selective prediction / selective classification deep networks (El‑Yaniv & Wiener; Geifman & El‑Yaniv)
- learning to defer to an expert (Madras et al.; Mozannar & Sontag)
- LLM agents tool use orchestration (ReAct; Toolformer); LLM + symbolic / neuro‑symbolic
- vision-language models industrial visual inspection / defect detection / manufacturing
- large language models reproducibility / determinism scientific research
- case-based reasoning large language models (CBR + LLM)

## FORMATO DE SALIDA
1. **Tabla resumen:** `Decisión | Justificación (1 línea) | Fuentes | Nivel de evidencia`.
2. **Desarrollo por secciones** (una por decisión) con (a)–(e).
3. **Referencias** en IEEE o APA con DOI/URL.
4. **"Puntos débiles y defensa":** los 3 ataques más probables del comité y cómo responderlos (incluido obligatoriamente: *"el VLM es una caja negra que contradice tu argumento de interpretabilidad"* y *"el VLM no es evidencia independiente"*).

## TONO
Crítico y honesto. **No sobrevender el VLM.** Donde el respaldo sea analógico (p. ej. selective prediction proviene de clasificación supervisada, no de CBR con VLM), decláralo. La afirmación más fuerte y defendible es modesta: *el VLM es una segunda representación que audita la fragilidad de la heurística de umbral del CBR y, solo en novedad/OOD, decide* — no un oráculo independiente.
