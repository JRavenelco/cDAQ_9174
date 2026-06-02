# Razonador CBR orquestado por VLM (ciclo 4R) — MATLAB/Simulink

Capa deliberativa (System 2) sobre el CBR determinista (System 1). Completa el
**ciclo 4R de CBR** (Aamodt & Plaza, 1994) con un **VLM que orquesta por excepción**:
recupera (Retrieve), reutiliza (Reuse), **revisa con visión** (Revise) y aprende
(Retain). El VLM solo entra en **baja confianza / OOD**.

> **Honestidad metodológica:** el VLM aporta *redundancia de representación* (misma
> señal, otra representación: imagen del lazo F–x vs. vector de 7 features), **NO
> evidencia independiente**. Distinto de la validez convergente del DOE (Ra ⟂ loop_area).

## Arquitectura (dos capas, todo en MATLAB/Simulink)

```
CAPA REACTIVA (determinista, ms, codegen/Hailo-8L)
  señal (t, fuerza, entrada)
    → extraer_features_ventana → cbr_retrieve_topk (kNN ponderado) → cbr_confianza
        → ALTA           → Reuse directo (manda CBR)            fuente=CBR
        → BAJA (no OOD)  → VLM AUDITA (manda CBR, flag_discrep) fuente=CBR+audit
        → OOD            → VLM DECIDE (manda VLM)               fuente=VLM
CAPA DELIBERATIVA (VLM asíncrono, s; qwen2.5-VL dual Ollama↔OpenRouter)
  orquestador_cbr_vlm → vlm_cliente (webwrite + IMAGEN lazo F-x b64) → veredicto JSON
    → (si retain) cbr_retain: base N → N+1   (cierra 4R)
```

## Componentes

| Archivo | Capa | codegen | Rol |
|---------|------|---------|-----|
| `cbr_retrieve_topk.m` | reactiva | `%#codegen` | Retrieve top-K `[idx,dist,score,clase]` |
| `cbr_confianza.m` | reactiva | `%#codegen` | Estado {alta, baja_no_OOD, OOD} |
| `extraer_features_ventana.m` | reactiva | interpretado | Señal cruda → 7 features (port de `construir_casos.py`) |
| `generar_lazo_png.m` | deliberativa | extrinsic | Lazo F–x → PNG → base64 |
| `vlm_cliente.m` | deliberativa | extrinsic | Cliente VLM dual + fallback (`webwrite`) |
| `orquestador_cbr_vlm.m` | orquestación | extrinsic | Lógica híbrida 4R |
| `cbr_retain.m` | retain | extrinsic | Append de caso validado al `.mat` |
| `crear_modelo_orquestado.m` | builder | — | `modelo_cbr_vlm_orquestado.slx` |
| `probar_orquestador.m` | test | — | Valida los 3 caminos sin Simulink |
| `prompts/orquestador_vlm.md` | prompt | — | System prompt (responde SOLO JSON) |
| `prompts/defensa_vlm_orquestador.md` | Fase 0 | — | Defensa bibliográfica del VLM |

## Flujo de uso

```matlab
>> cd Pruebas\src\razonador_simulink
>> exportar_base_casos        % 1) genera datos/base_de_casos_cbr.mat
>> probar_orquestador         % 2) valida los 3 caminos (VLM off por defecto)
>> crear_modelo_orquestado    % 3) arma y abre el modelo Simulink orquestado
```

Para probar el VLM real (requiere Ollama local y/o `OPENROUTER_API_KEY` en `.env`):
```matlab
>> p = struct('K',3,'threshold',0.01,'tau_score',0.5,'tau_margen',0.05, ...
              'modelo','cloud','usar_vlm',true,'img_b64','');
>> out = orquestador_cbr_vlm([50 80 5 9 0.9 25 0.5], p)   % caso OOD → VLM decide
```

## Autoridad híbrida del VLM (resumen)

| Estado | Disparo | Quién manda | `fuente` | Reproducible |
|--------|---------|-------------|----------|--------------|
| ALTA | `score≥tau` y margen/clases OK | CBR | `CBR` | sí (determinista) |
| BAJA, no OOD | `score<tau` / margen estrecho / clases divergentes | CBR (VLM audita) | `CBR+audit` | sí (manda CBR) |
| OOD | `min_dist≥threshold` | VLM | `VLM` | acotado a OOD |
| VLM caído | fallo de red en baja/OOD | CBR (degradado) | `CBR(degradado)` | sí |

## Mapeo a la investigación (para el comité)

- **Ciclo 4R de CBR:** Aamodt & Plaza (1994).
- **Rechazo / OOD (novelty):** Perner (2008) — bisagra hacia *reject option* (Chow, 1970),
  *selective prediction* (El-Yaniv & Wiener; Geifman & El-Yaniv) y *learning-to-defer*
  (Madras et al., 2018).
- **VLM como orquestador con tool-use:** ReAct (Yao et al.); Toolformer (Schick et al.).
- **Lazo → energía disipada/ciclo:** Charalampakis & Koumousis (2008) (Bouc-Wen).
- **Defensa bibliográfica completa:** `prompts/defensa_vlm_orquestador.md` (Fase 0) →
  artefacto `compass_artifact_vlm_*.md`.

> Las citas anteriores deben **verificarse** (DOI) en la Fase 0 antes de la defensa.

## Reproducibilidad / gobernanza

- `temperature = 0.2` en el VLM; el rol híbrido limita el no-determinismo a casos OOD.
- `vlm_cliente` devuelve `.raw` y `.fuente_modelo` para **registrar prompt+respuesta+modelo**.
- La base se versiona vía CSV; el `.mat`/`.slx` se regeneran (ignorados por git).

## Limitaciones

- Llamada al VLM **bloqueante** (s) en Interpreted execution → demo/HIL suave, no tiempo
  real duro. Evolución: **MQTT** (Mosquitto del cerebro digital) para asíncrono.
- Requiere Ollama local con un VLM (p.ej. `llava`) y/o `OPENROUTER_API_KEY`.
- `cbr_retrieve_topk` y `cbr_confianza` permanecen `%#codegen`; el resto (load/webwrite/
  gráficos) en Interpreted execution.
