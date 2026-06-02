# razonador_simulink — Razonador CBR (+ VLM orquestador) en Simulink

Carpeta **standalone** (autocontenida) del Razonador Basado en Casos para
diagnóstico de histéresis en fresado, ejecutándose en **MATLAB/Simulink**, sin
dependencia de Python en runtime.

Implementa el **ciclo 4R de CBR** (Aamodt & Plaza, 1994):
**Retrieve → Reuse → Revise → Retain**, con un **VLM como orquestador por
excepción** (System 1 / System 2):

- **CBR determinista** (rápido, embebible en Hailo-8L) = *System 1*.
- **VLM** (qwen2.5-VL, dual Ollama ↔ OpenRouter) = *System 2*, disparado **solo
  por baja confianza / OOD**. Mira la **imagen del lazo F–x** + 7 features + top-K.

> **Honestidad metodológica:** el VLM aporta *redundancia de representación*
> (misma señal, otra representación), NO evidencia independiente. No sustituye la
> validez convergente del DOE (Ra ⟂ loop_area).

## Estado actual

**Fase: base CBR funcional copiada y aislada.** Los componentes del orquestador
4R+VLM se construyen paso a paso (ver plan).

### Capa reactiva (determinista, ya presente)

| Archivo | Rol |
|---------|-----|
| `exportar_base_casos.m` | Lee `datos/casos_historicos.csv` → `datos/base_de_casos_cbr.mat` |
| `cbr_reasoner.m` | Bloque MATLAB Function (`%#codegen`) — retrieval ponderado |
| `CBRHysteresisSystem.m` | Bloque MATLAB System (threshold tunable, Interpreted) |
| `probar_cbr.m` | Validación 1-vs-resto fuera de Simulink |
| `crear_modelo_simulink.m` | Builder del modelo `.slx` de demo |

### Componentes del orquestador 4R+VLM (en construcción)

| Archivo | Rol | Estado |
|---------|-----|--------|
| `prompts/defensa_vlm_orquestador.md` | Fase 0: defensa bibliográfica del VLM | pendiente |
| `cbr_retrieve_topk.m` | Retrieve top-K `[idx, dist, score, clase]` | pendiente |
| `cbr_confianza.m` | Estado {alta, baja_no_OOD, OOD} | pendiente |
| `vlm_cliente.m` | Cliente VLM dual (`webwrite`) + fallback | pendiente |
| `prompts/orquestador_vlm.md` | System prompt (responde SOLO JSON) | pendiente |
| `orquestador_cbr_vlm.m` | Lógica híbrida 4R (alta/audita/decide) | pendiente |
| `cbr_retain.m` | Retain: append de caso validado al `.mat` | pendiente |
| `crear_modelo_orquestado.m` | Builder `modelo_cbr_vlm_orquestado.slx` | pendiente |

## Arranque rápido (base CBR)

```matlab
>> cd Pruebas\src\razonador_simulink
>> exportar_base_casos      % genera datos/base_de_casos_cbr.mat
>> probar_cbr               % valida retrieval vs Python
>> crear_modelo_simulink    % arma y abre el modelo de demo
```

## Features (orden FIJO) y pesos

| # | Feature | Peso |
|---|---------|------|
| 1 | `force_rms` | 1.0 |
| 2 | `force_peak_abs` | 0.8 |
| 3 | `input_rms` | 1.0 |
| 4 | `input_peak_abs` | 0.8 |
| 5 | `corr_force_input` | 0.7 |
| 6 | `loop_area_norm` | 1.4 |
| 7 | `duration_s` | 0.3 |

Normalización robusta `(x - mediana)/IQR` precomputada offline; distancia
euclidiana ponderada `norm((q-c).*w)/sqrt(7)`; similitud `1/(1+dist)`.

## Datos

- `datos/casos_historicos.csv` — dataset histórico (copia local, versionado).
- `datos/base_de_casos_cbr.mat` — base compilada (regenerable; ignorada por git).

## Dependencias

- MATLAB + Simulink.
- Para el VLM: Ollama local y/o `OPENROUTER_API_KEY` en el `.env` de la raíz del repo.
- **No** depende de `serve_razonador.py` ni `razonador_local.py` (que quedan intactos
  para n8n/web).
