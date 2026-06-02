# Defensa metodológica del VLM como orquestador del razonador CBR (histéresis en fresado)

> Hermano de la defensa del CBR (`compass_artifact_wf-*.md`). Mientras aquella defiende el
> núcleo determinista (ciclo 4R, retrieval), ésta defiende la **capa deliberativa**: un VLM
> que **selecciona y valida** por excepción, con autoridad **híbrida** (auditor en baja
> confianza, decisor solo en OOD). Todas las citas con DOI/URL fueron verificadas por
> búsqueda; lo no verificado se marca explícitamente.

## TL;DR
- El **escalado por excepción** del CBR al VLM NO es ad‑hoc: es el problema clásico de **reject option** (Chow 1970), **selective prediction** (Geifman & El‑Yaniv 2017) y **learning‑to‑defer a un experto** (Madras et al. 2018; Mozannar & Sontag 2020). Respaldo **directo y formal**.
- Que un LLM/VLM **orqueste herramientas/sistemas simbólicos** (aquí, el CBR) está respaldado por **ReAct** (Yao et al. 2023) y **Toolformer** (Schick et al. 2023); y **CBR+LLM** es un frente de investigación reconocido (review arXiv 2504.06943, 2025) — el enfoque es actual, no marginal.
- El uso de **VLM con visión para inspección/diagnóstico en manufactura** tiene literatura emergente (2025), pero es un campo joven: evidencia **directa pero incipiente**.
- **Flanco honesto #1:** el VLM es **redundancia de representación**, NO evidencia independiente — mira la misma señal como imagen del lazo en vez de 7 escalares. Es **argumento de diseño**, sin cita que lo "demuestre"; no debe confundirse con la **validez convergente** del DOE (Ra ⟂ área de lazo).
- **Flanco honesto #2:** los LLMs son **no deterministas incluso con `temperature=0`** (floating‑point, kernels; Ouyang et al. 2308.02828). Exige **registro y gobernanza**; el rol **híbrido (decisor solo en OOD)** acota el no‑determinismo a los casos en que el CBR ya no es fiable por diseño.

## Key Findings
1. **El escalado tiene marco formal sólido y citable.** Reject option (Chow 1970) define el óptimo error‑rechazo; selective prediction lo lleva a redes profundas (Geifman & El‑Yaniv 2017); learning‑to‑defer formaliza "derivar al experto" con estimadores consistentes (Mozannar & Sontag 2020). Aquí el "experto" al que se difiere es el VLM.
2. **La orquestación LLM↔herramienta está establecida** (ReAct, Toolformer), y su unión con CBR está documentada como dirección de investigación activa (2024–2025).
3. **El VLM en visión industrial es prometedor pero joven**: revisiones 2025 lo señalan como tendencia (con XAI), reconociendo que es más lento que CNN/ViT en el edge — coherente con usarlo **por excepción**, no en el lazo rápido.
4. **Los puntos atacables son honestos y manejables**: redundancia de representación (no sobrevender) y no‑determinismo (gobernar y acotar a OOD).

## Tabla resumen

| Decisión | Justificación (1 línea) | Fuentes | Nivel de evidencia |
|---|---|---|---|
| 1. VLM/LLM orquesta el CBR (tool‑use) | Un LLM puede dirigir herramientas/sistemas simbólicos e interfaces externas | Yao et al. (2023, ReAct); Schick et al. (2023, Toolformer); review CBR+LLM (2025) | Analógica (agentes) + Directa‑emergente (CBR+LLM) |
| 2. Escalado por excepción (baja confianza) | Es reject option / selective prediction / learning‑to‑defer | Chow (1970); Geifman & El‑Yaniv (2017); Madras et al. (2018); Mozannar & Sontag (2020) | **Directa** (marco formal) |
| 3. VLM decisor SOLO en OOD; auditor si no | Rechazar/derivar en novedad preserva la competencia del CBR | Perner (2008, ya citado); Chow (1970); Madras et al. (2018) | Analógica / Directa |
| 4. VLM con visión lee el lazo F–x | Foundation/VLM para inspección y diagnóstico visual en manufactura | VLM+ICL inspección (2025, arXiv 2502.09057); review anomalías industriales (2025) | Directa pero **emergente** |
| 5. No rompe la interpretabilidad del CBR | El VLM explica y cita el caso; el CBR sigue trazable | Yao et al. (2023, trazas de razonamiento) + argumento | Débil / argumentativo |
| 6. Redundancia de representación (no evidencia independiente) | El VLM ve la MISMA señal en otra forma; audita la fragilidad del escalar | Razonamiento de diseño (sin cita) | **Débil (diseño)** |
| 7. Reproducibilidad / gobernanza | LLMs no deterministas aun con T=0 → registrar y acotar a OOD | Ouyang et al. (arXiv 2308.02828); Thinking Machines Lab | Directa (no‑determinismo) |

---

### Decisión 1 — Un VLM/LLM orquesta el CBR (tool‑use), no un modelo monolítico
**(a)** En lugar de pedir a un único modelo que lo resuelva todo, el VLM actúa como **orquestador**: invoca el CBR (recuperación simbólica), interpreta su salida y la imagen del lazo, y decide. Es el patrón **razonar+actuar** y **uso de herramientas**.
**(b)** Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", ICLR 2023, arXiv:2210.03629. Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools", NeurIPS 2023, arXiv:2302.04761. Marco emergente CBR+LLM: "Review of Case‑Based Reasoning for LLM Agents", 2025, arXiv:2504.06943.
**(c)** Analógica (ReAct/Toolformer son de NLP/agentes) reforzada por la línea **directa‑emergente** CBR+LLM.
**(d) Alternativas:** un único modelo end‑to‑end (caja negra, no trazable, exige datos) o reglas fijas (frágiles). La orquestación mantiene el CBR simbólico como herramienta auditable.
**(e) Riesgo:** ReAct/Toolformer no son de manufactura → declararlo analógico; el respaldo de dominio es el campo CBR+LLM, aún joven.

### Decisión 2 — Escalado por excepción = reject option / selective prediction / learning‑to‑defer
**(a)** Que un modelo barato y fiable resuelva por defecto y **derive los casos difíciles** a un razonador más capaz es exactamente el problema de la opción de rechazo y la predicción selectiva. Es el respaldo **más fuerte** de toda la arquitectura.
**(b)** C. K. Chow, "On optimum recognition error and reject tradeoff", IEEE Trans. Information Theory 16(1):41–46, 1970, DOI 10.1109/TIT.1970.1054406. Y. Geifman, R. El‑Yaniv, "Selective Classification for Deep Neural Networks", NeurIPS 2017, arXiv:1705.08500. D. Madras, T. Pitassi, R. Zemel, "Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer", NeurIPS 2018, arXiv:1711.06664. H. Mozannar, D. Sontag, "Consistent Estimators for Learning to Defer to an Expert", ICML 2020 (PMLR v119), arXiv:2006.01862.
**(c) Directa** (es literalmente el marco formal del escalado/deferral).
**(d) Alternativas:** umbral de confianza simple (Chow) vs deferral aprendido (Madras/Mozannar). Con pocos datos, el umbral es defendible; el deferral aprendido es la evolución natural.
**(e) Riesgo:** estas obras asumen un clasificador entrenado y, a veces, un experto humano; aquí el "clasificador" es CBR y el "experto" es un VLM → transferencia conceptual, declararla.

### Decisión 3 — VLM decisor SOLO en OOD; auditor en el resto (híbrido)
**(a)** Acotar la autoridad del VLM a la **novedad/OOD** (donde el CBR extrapola y no es fiable) es coherente con el rechazo por novedad ya defendido y con derivar al experto solo cuando el modelo base debe abstenerse.
**(b)** P. Perner (2008), novelty detection en CBR (ya citado en la defensa del CBR, DOI 10.1007/978-3-540-73435-2_3); Chow (1970); Madras et al. (2018).
**(c)** Analógica/Directa.
**(d) Alternativas:** VLM siempre decisor (máximo no‑determinismo) o nunca decisor (VLM inútil en OOD). El híbrido es el punto medio defendible.
**(e) Riesgo:** definir OOD con `min_dist ≥ threshold` hereda la discusión del umbral de rechazo del CBR (calibrar por percentil de distancias LOOCV).

### Decisión 4 — VLM con visión: lee la forma del lazo F–x
**(a)** El VLM no recibe solo 7 escalares: ve la **imagen del lazo fuerza–desplazamiento**, lo que le permite auditar la clasificación cruda (umbral de área) mirando la **forma** (apertura, asimetría, saturación), no solo el área.
**(b)** Inspección visual con VLM + in‑context learning (2025), arXiv:2502.09057; revisión de detección de anomalías industriales que señala VLM + XAI como tendencia (Int. J. Computer Integrated Manufacturing, 2025, DOI 10.1080/0951192X.2025.2599548).
**(c) Directa (dominio manufactura) pero emergente.**
**(d) Alternativas:** features visuales hechas a mano del lazo (curvatura, momentos) + clasificador. Más interpretables y baratas; el VLM aporta flexibilidad sin reentrenar.
**(e) Riesgo:** "leer un lazo F–x" no es exactamente "detección de defectos en imagen"; la analogía es de **lectura visual de patrón de proceso**. Declararlo; y validar que el VLM realmente usa la forma (ablación: misma consulta sin imagen).

### Decisión 5 — El VLM no rompe el argumento de interpretabilidad del CBR
**(a)** La defensa del CBR presume "interpretable vs caja negra". Se sostiene porque: el **diagnóstico base sigue siendo el caso recuperado trazable**; el VLM **explica en lenguaje natural y cita el caso/score**; y sus trazas de razonamiento son inspeccionables (estilo ReAct).
**(b)** Yao et al. (2023) — trazas de razonamiento explícitas e inspeccionables.
**(c) Débil/argumentativo.**
**(d) Alternativas:** restringir al VLM a salida estructurada + justificación citada (lo que hace `orquestador_vlm.md`).
**(e) Riesgo:** una explicación en lenguaje natural puede ser *post‑hoc* y no fiel al cómputo interno. Respuesta: el VLM solo decide en OOD; en el resto audita, y la autoridad queda en el CBR trazable.

### Decisión 6 — Honestidad: redundancia de representación, NO evidencia independiente
**(a)** Punto metodológico central. El VLM y el CBR observan **la misma señal física** del mismo corte, solo en **representaciones distintas**: el CBR un vector de 7 features; el VLM la imagen del lazo. Por tanto el VLM **no es un canal de evidencia independiente**: su valor es **atrapar los modos de fallo de la representación escalar** (la fragilidad de los umbrales 0.3/1.0), no "confirmar" el diagnóstico desde otra fuente.
**(b)** Es **razonamiento de diseño**, sin cita que lo demuestre. La distinción con **validez convergente** (medir el mismo constructo con métodos independientes) se apoya conceptualmente en la tradición multimétodo (Campbell & Fiske 1959 — *no verificado aquí*, citar solo si se confirma).
**(c) Débil (diseño).**
**(d)** Contraparte: el **DOE run‑to‑failure** sí da convergencia genuina (Ra de superficie ⟂ área de lazo de señal = canales físicamente independientes). Mantener separados ambos argumentos ante el comité.
**(e) Riesgo:** sobrevender el VLM como "segunda opinión independiente". Respuesta: declararlo redundancia de representación; el aporte es robustez frente al fallo de la heurística escalar.

### Decisión 7 — Reproducibilidad y gobernanza del VLM
**(a)** Para rigor doctoral hay que tratar el no‑determinismo: **fijar `temperature` baja, versión de modelo, y registrar prompt+imagen+respuesta** de cada llamada.
**(b)** Z. Ouyang et al., "An Empirical Study of the Non‑determinism of ChatGPT in Code Generation", arXiv:2308.02828 (muestra variabilidad entre ejecuciones). Análisis de ingeniería: Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).
**(c) Directa** sobre el hecho del no‑determinismo.
**(d) Alternativas:** modelo local con seed fijo y kernels deterministas (mejor reproducibilidad, menor capacidad) vs cloud potente (qwen2.5‑VL‑72b, menos control). El fallback dual cubre ambos; declarar cuál se usó en cada resultado.
**(e) Riesgo (clave):** `temperature=0` **no** garantiza determinismo (floating‑point, kernels de atención, empates de tokens). Respuesta: por eso el VLM **no decide salvo en OOD**, se **registra todo** para trazabilidad, y los resultados se reportan reconociendo la variabilidad.

---

## Referencias (IEEE, con DOI/URL verificados salvo indicación)
1. C. K. Chow, "On optimum recognition error and reject tradeoff," *IEEE Trans. Inf. Theory*, vol. 16, no. 1, pp. 41–46, 1970. DOI: 10.1109/TIT.1970.1054406.
2. Y. Geifman and R. El‑Yaniv, "Selective Classification for Deep Neural Networks," *NeurIPS 30*, 2017. arXiv:1705.08500.
3. D. Madras, T. Pitassi, and R. Zemel, "Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer," *NeurIPS 31*, 2018. arXiv:1711.06664.
4. H. Mozannar and D. Sontag, "Consistent Estimators for Learning to Defer to an Expert," *ICML 2020*, PMLR vol. 119. arXiv:2006.01862.
5. S. Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," *ICLR 2023*. arXiv:2210.03629.
6. T. Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools," *NeurIPS 36*, 2023. arXiv:2302.04761.
7. "Vision‑Language In‑Context Learning Driven Few‑Shot Visual Inspection Model," 2025. arXiv:2502.09057.
8. Review, "Anomaly detection for industrial applications: challenges, solutions, and future directions," *Int. J. Computer Integrated Manufacturing*, 2025. DOI: 10.1080/0951192X.2025.2599548 *(confirmar paginación final)*.
9. Z. Ouyang et al., "An Empirical Study of the Non‑determinism of ChatGPT in Code Generation," arXiv:2308.02828 *(versión journal en ACM TOSEM por confirmar)*.
10. "Review of Case‑Based Reasoning for LLM Agents: Theoretical Foundations, Architectural Components, and Cognitive Integration," 2025. arXiv:2504.06943.
11. P. Perner, "Concepts for novelty detection... case‑based reasoning," 2008 (ver defensa del CBR; LNCS DOI 10.1007/978-3-540-73435-2_3).

*Pendiente de confirmar DOI/venue final antes de imprenta: entradas 7, 8, 9, 10 (varias son preprints/versiones recientes). Campbell & Fiske (1959, validez convergente) NO verificado en esta búsqueda — citar solo tras confirmar.*

## Puntos débiles y defensa (los 3 ataques más probables del comité)
**Ataque 1: "El VLM es una caja negra y contradice tu argumento de interpretabilidad del CBR."**
Defensa: el VLM **no sustituye** al CBR; lo **orquesta**. El diagnóstico base sigue siendo el caso recuperado trazable; el VLM solo **audita** (y decide únicamente en OOD), explica en lenguaje natural y **cita el caso**. La autoridad reproducible permanece en el CBR (Decisiones 3 y 5).

**Ataque 2: "Añadir un VLM es meter un segundo modelo no determinista al diagnóstico."**
Defensa: cierto, y se gobierna — registro de prompt/respuesta/modelo/versión, temperatura baja, y **autoridad acotada a OOD** (Decisión 7). El no‑determinismo existe aun a T=0 (Ouyang et al.); por eso el VLM no decide en el régimen donde el CBR sí es fiable.

**Ataque 3: "El VLM no aporta nada que el área del lazo no diga ya."**
Defensa: aporta **robustez frente al modo de fallo de la representación escalar** — los umbrales 0.3/1.0 que la propia defensa del CBR marca como débiles. El VLM audita la **forma** del lazo, no solo el área. Pero **es redundancia de representación, no evidencia independiente** (Decisión 6): no se vende como segunda opinión independiente — eso lo aporta el DOE.

## Recommendations
1. **Implementar el rol híbrido tal cual** (auditor / decisor‑solo‑OOD): es el punto defendible entre protagonismo del VLM y rigor.
2. **Gobernanza desde el día 1**: registrar cada llamada (prompt, imagen, modelo, versión, respuesta) en `salidas/razonamientos/`; reportar variabilidad.
3. **Ablación de visión**: comparar veredicto del VLM con y sin la imagen del lazo, para evidenciar que la forma aporta (Decisión 4).
4. **Encuadre System 1 / System 2** en la presentación, manteniendo separado el argumento de **redundancia de representación** (VLM) del de **validez convergente** (DOE).
5. **Confirmar DOIs** de las entradas emergentes (7–10) antes de la versión final; donde no haya respaldo, declarar "sin respaldo localizado".

## Caveats
- El respaldo **más fuerte** es el del **escalado/deferral** (Chow; selective prediction; learning‑to‑defer): directo y formal. El de **VLM‑en‑visión‑industrial** es **emergente**; preséntalo como tendencia, no como práctica consolidada.
- La afirmación honesta y defendible es **modesta**: el VLM es una **segunda representación** que audita la fragilidad de la heurística de umbral y, **solo en OOD**, decide. No es un oráculo ni evidencia independiente.
- El no‑determinismo de LLMs es real **incluso a `temperature=0`**; sin gobernanza, compromete la reproducibilidad doctoral.
