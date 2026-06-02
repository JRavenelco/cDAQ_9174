# n8n Local + Razonador CBR (Windows)

Orquestador **n8n** corriendo en Docker en tu PC, conectado a tu Razonador CBR
y a Ollama (ambos nativos en Windows).

## Arquitectura

```
┌─────────────────────────────────────────────────────┐
│  Tu PC (Windows)                                      │
│                                                       │
│  ┌──────────────┐                                    │
│  │ n8n (Docker) │  http://localhost:5678             │
│  │  puerto 5678 │                                    │
│  └──────┬───────┘                                    │
│         │ host.docker.internal:8100                  │
│         ▼                                             │
│  ┌──────────────────┐    ┌──────────────────┐       │
│  │ Razonador CBR    │───►│ Ollama (qwen3:8b)│       │
│  │ (FastAPI :8100)  │    │   puerto 11434   │       │
│  └──────────────────┘    └──────────────────┘       │
└─────────────────────────────────────────────────────┘
```

## Arranque (3 pasos)

### 1. Levantar el Razonador CBR (terminal 1)

```powershell
cd Pruebas\src\caracterizacion_fuerza\razonador_casos
python serve_razonador.py --port 8100 --model qwen3:8b
```

Verifica: http://localhost:8100/docs

### 2. Levantar n8n (terminal 2)

```powershell
cd Pruebas\src\caracterizacion_fuerza\razonador_casos\n8n_local
docker compose up -d
```

Espera ~30s y abre: **http://localhost:5678**

- Usuario: `cidesi`
- Password: `cerebro2026`

### 3. Importar el workflow

1. En n8n: menu (arriba derecha) → **Import from File**
2. Selecciona `workflow_razonador_cbr.json`
3. Click en **Execute Workflow** (botón abajo)

## URLs importantes

| Servicio | Desde tu navegador | Desde dentro de n8n |
|----------|-------------------|---------------------|
| n8n UI | `http://localhost:5678` | — |
| Razonador CBR | `http://localhost:8100` | `http://host.docker.internal:8100` |
| Ollama | `http://localhost:11434` | `http://host.docker.internal:11434` |

> **Importante:** Dentro de n8n (Docker) NO uses `localhost` para llamar a tus
> servicios del host. Usa `host.docker.internal`.

## Endpoints del Razonador (para nodos HTTP Request)

### POST /reason  (retrieval + LLM)

**Modelo LOCAL (Ollama):**
```json
{
  "case_id": "corte_20260311_131921_600_40",
  "top_k": 3,
  "model": "qwen3:8b"
}
```

**Modelo NUBE (OpenRouter — usa OPENROUTER_MODEL del .env):**
```json
{
  "case_id": "corte_20260311_131921_600_40",
  "top_k": 3,
  "model": "cloud"
}
```

> El campo `model` enruta automaticamente:
> - `qwen3:8b`, `gemma4`, `llama3.2:3b` → Ollama local
> - `cloud` o `qwen/qwen2.5-vl-72b-instruct` (con `/`) → OpenRouter nube
>
> La nube responde mas rapido y con mayor calidad, pero requiere
> `OPENROUTER_API_KEY` configurada en el `.env` de la raiz del repo.

### POST /retrieve  (solo similares, instantaneo)
```json
{
  "case_id": "corte_20260311_131921_600_40",
  "top_k": 5
}
```

### POST /reason  con features directas (caso nuevo sin ID)
```json
{
  "features": {
    "force_rms": 0.89,
    "force_peak_abs": 1.20,
    "input_rms": 0.017,
    "input_peak_abs": 0.044,
    "corr_force_input": -0.003,
    "loop_area_norm": 2.08,
    "duration_s": 31.0
  },
  "top_k": 3
}
```

### GET /cases  (lista los 19 casos)
### GET /health (status)

## Comandos utiles

```powershell
# Ver logs de n8n
docker compose logs -f n8n

# Detener n8n
docker compose down

# Reiniciar n8n
docker compose restart

# Ver estado
docker compose ps
```

## Notas

- El timeout del nodo HTTP Request esta en **300000 ms (5 min)** porque
  `qwen3:8b` puede tardar 1-2 min en la primera llamada (carga del modelo).
- Para respuestas mas rapidas usa `llama3.2:3b` en el campo `model`.
- Los resultados tambien se guardan en `salidas/razonamientos/` en tu PC.
