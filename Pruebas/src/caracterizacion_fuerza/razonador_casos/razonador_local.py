#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
razonador_local.py — Razonador Basado en Casos (CBR) con LLM local.

Pipeline completo:
  1. Carga base de casos históricos (JSONL)
  2. Recibe caso consulta (por ID, CSV nuevo, o features directas)
  3. Retrieval: top-K casos similares (distancia ponderada)
  4. Reasoning: envía contexto a Ollama local (qwen3:8b / gemma4)
  5. Devuelve diagnóstico estructurado

Uso:
    # Razonar sobre un caso existente de la base:
    python razonador_local.py --case-id corte_20260311_131921_600_40

    # Razonar sobre un CSV nuevo (captura reciente de la GUI):
    python razonador_local.py --csv ../corte_20260506_151429.csv

    # Razonar sobre todos los casos (batch):
    python razonador_local.py --batch --top-k 3

    # Usar modelo diferente:
    python razonador_local.py --case-id corte_20260311_131921_600_40 --model gemma4

    # Solo retrieval (sin LLM):
    python razonador_local.py --case-id corte_20260311_131921_600_40 --no-llm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

try:
    import requests
    _HAVE_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    _HAVE_REQUESTS = False

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CASES_JSONL = SCRIPT_DIR / "salidas" / "casos_dataset" / "casos_historicos.jsonl"
PROMPT_FILE = SCRIPT_DIR / "prompts" / "razonador_final.md"
OUTPUT_DIR = SCRIPT_DIR / "salidas" / "razonamientos"

# Buscar TODOS los .env del árbol (de la raíz hacia abajo), combinándolos.
# Los .env más cercanos al script tienen prioridad sobre los de la raíz,
# pero se rellenan claves faltantes (p.ej. OPENROUTER_* suele estar en la raíz).
def _find_env_files() -> list[Path]:
    found: list[Path] = []
    for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            found.append(candidate)
    return found


def _parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if path and path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


def _load_env() -> dict[str, str]:
    """Combina todos los .env del árbol. Closer-to-script gana, raíz rellena."""
    env: dict[str, str] = {}
    # Procesar de la raíz hacia el script: así los más cercanos sobreescriben
    for path in reversed(_find_env_files()):
        env.update(_parse_env_file(path))
    # Las variables de entorno del sistema tienen prioridad final
    for k in ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "OPENROUTER_MODEL"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


_ENV = _load_env()

# ── Features y pesos para retrieval ────────────────────────────────────────
RETRIEVAL_FEATURES = [
    "force_rms",
    "force_peak_abs",
    "input_rms",
    "input_peak_abs",
    "corr_force_input",
    "loop_area_norm",
    "duration_s",
]

FEATURE_WEIGHTS = {
    "force_rms": 1.0,
    "force_peak_abs": 0.8,
    "input_rms": 1.0,
    "input_peak_abs": 0.8,
    "corr_force_input": 0.7,
    "loop_area_norm": 1.4,   # El área de lazo es clave para histéresis
    "duration_s": 0.3,
}

# ── Ollama config (local) ───────────────────────────────────────────────────
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = "qwen3:8b"

# ── OpenRouter config (cloud) ─────────────────────────────────────────────────
OPENROUTER_API_KEY = _ENV.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = _ENV.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = _ENV.get("OPENROUTER_MODEL", "qwen/qwen2.5-vl-72b-instruct")


def resolve_provider(model: str) -> str:
    """Decide el proveedor según el nombre del modelo.

    - 'cloud' / 'openrouter' / contiene '/'  → OpenRouter
    - cualquier otro (qwen3:8b, gemma4, ...) → Ollama local
    """
    m = (model or "").strip().lower()
    if m in ("cloud", "openrouter"):
        return "openrouter"
    if "/" in m:  # ej. qwen/qwen2.5-vl-72b-instruct
        return "openrouter"
    return "ollama"


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_cases(path: Path) -> list[dict[str, Any]]:
    """Carga casos históricos desde JSONL."""
    if not path.exists():
        print(f"ERROR: No se encontró {path}")
        print(f"  Ejecuta primero: python construir_casos.py")
        sys.exit(1)
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_system_prompt(path: Path) -> str:
    """Carga el prompt del sistema."""
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return (
        "Eres un razonador basado en casos para diagnóstico de histéresis "
        "y desgaste en corte de fresado. Analiza el caso actual comparándolo "
        "con los casos históricos más similares."
    )


# ===========================================================================
# FEATURE EXTRACTION (para CSVs nuevos)
# ===========================================================================

def extract_features_from_csv(csv_path: Path) -> dict[str, Any]:
    """Extrae features de un CSV de corte nuevo (formato GUI)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    # Detectar columnas
    time_col = next((c for c in cols if "tiempo" in c.lower() or "time" in c.lower()), cols[0])
    force_col = next((c for c in cols if "fuerza" in c.lower() or "force" in c.lower()), None)
    accel_col = next((c for c in cols if "aceler" in c.lower() or "accel" in c.lower()), None)

    if not force_col:
        # Intentar por posición (col 1 = fuerza, col 2 = accel en formato GUI)
        if len(cols) >= 3:
            force_col = cols[1]
            accel_col = cols[2]
        else:
            raise ValueError(f"No se detectó columna de fuerza en {csv_path}")

    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    force = pd.to_numeric(df[force_col], errors="coerce").to_numpy(dtype=float)

    if accel_col:
        accel = pd.to_numeric(df[accel_col], errors="coerce").to_numpy(dtype=float)
    else:
        accel = np.zeros_like(force)

    # Limpiar NaN
    valid = np.isfinite(t) & np.isfinite(force) & np.isfinite(accel)
    t, force, accel = t[valid], force[valid], accel[valid]

    if len(t) < 16:
        raise ValueError(f"Muy pocas muestras válidas: {len(t)}")

    t = t - t[0]
    dt = np.diff(t)
    dt = dt[dt > 0]
    fs = 1.0 / np.median(dt) if len(dt) > 0 else 2500.0

    # Normalizar para loop area
    force_centered = force - np.mean(force)
    scale_f = np.max(np.abs(force_centered))
    force_norm = force_centered / scale_f if scale_f > 1e-12 else np.zeros_like(force_centered)

    accel_centered = accel - np.mean(accel)
    scale_a = np.max(np.abs(accel_centered))
    accel_norm = accel_centered / scale_a if scale_a > 1e-12 else np.zeros_like(accel_centered)

    # Loop area
    dx = np.gradient(accel_norm)
    area_raw = abs(float(np.nansum(force_norm * dx)))
    rect = (np.max(accel_norm) - np.min(accel_norm)) * (np.max(force_norm) - np.min(force_norm))
    loop_area = area_raw / rect if rect > 1e-12 else 0.0

    # Correlación
    corr = float(np.corrcoef(force_norm, accel_norm)[0, 1])
    if not np.isfinite(corr):
        corr = 0.0

    return {
        "case_id": csv_path.stem,
        "source_kind": "csv_nuevo",
        "source_file": csv_path.name,
        "rpm_estimada": "",
        "fs_hz": float(fs),
        "duration_s": float(t[-1] - t[0]),
        "n_samples": int(len(t)),
        "force_col": force_col,
        "input_col": accel_col or "N/A",
        "force_mean": float(np.mean(force)),
        "force_std": float(np.std(force)),
        "force_rms": float(np.sqrt(np.mean(force**2))),
        "force_peak_abs": float(np.max(np.abs(force))),
        "input_mean": float(np.mean(accel)),
        "input_std": float(np.std(accel)),
        "input_rms": float(np.sqrt(np.mean(accel**2))),
        "input_peak_abs": float(np.max(np.abs(accel))),
        "corr_force_input": corr,
        "loop_area_norm": loop_area,
    }


# ===========================================================================
# RETRIEVAL ENGINE
# ===========================================================================

def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key, default)
        if v == "" or v is None:
            return default
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def compute_stats(cases: list[dict], features: list[str]) -> dict[str, tuple[float, float]]:
    """Estadísticas robustas para normalización."""
    stats = {}
    for feat in features:
        values = sorted(as_float(c, feat) for c in cases)
        if not values:
            stats[feat] = (0.0, 1.0)
            continue
        center = float(median(values))
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        if not math.isfinite(iqr) or iqr <= 1e-12:
            iqr = float(np.std(values))
        if not math.isfinite(iqr) or iqr <= 1e-12:
            iqr = 1.0
        stats[feat] = (center, iqr)
    return stats


def vectorize(case: dict, features: list[str], stats: dict) -> np.ndarray:
    vals = []
    for feat in features:
        center, scale = stats[feat]
        vals.append((as_float(case, feat) - center) / scale)
    return np.array(vals)


@dataclass
class RetrievalResult:
    case: dict
    score: float
    distance: float


def retrieve_similar(
    query: dict,
    library: list[dict],
    features: list[str],
    weights: dict[str, float],
    top_k: int = 5,
    exclude_self: bool = True,
) -> list[RetrievalResult]:
    """Encuentra los K casos más similares al query."""
    all_cases = library + [query]
    stats = compute_stats(all_cases, features)
    weight_vec = np.array([weights.get(f, 1.0) for f in features])

    q_vec = vectorize(query, features, stats)
    query_id = query.get("case_id", "")

    results = []
    for case in library:
        if exclude_self and case.get("case_id") == query_id:
            continue
        c_vec = vectorize(case, features, stats)
        diff = (q_vec - c_vec) * weight_vec
        distance = float(np.linalg.norm(diff) / math.sqrt(len(features)))
        score = 1.0 / (1.0 + distance)
        results.append(RetrievalResult(case=case, score=score, distance=distance))

    results.sort(key=lambda r: (-r.score, r.distance))
    return results[:top_k]


# ===========================================================================
# LLM REASONING (Ollama local / OpenRouter cloud)
# ===========================================================================

def _http_post_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 300) -> dict:
    """POST JSON con requests o urllib (fallback). Devuelve el JSON parseado."""
    headers = headers or {"Content-Type": "application/json"}
    if _HAVE_REQUESTS:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    else:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


def call_ollama(prompt: str, system: str, model: str, temperature: float = 0.3) -> str:
    """Llama a Ollama API (local) y devuelve la respuesta."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 2048,
        },
    }
    url = f"{OLLAMA_URL}/api/generate"
    try:
        return _http_post_json(url, payload, timeout=300).get("response", "")
    except Exception as e:
        if "Connection" in type(e).__name__ or "URLError" in type(e).__name__:
            return "[ERROR] No se pudo conectar a Ollama. ¿Está corriendo? (ollama serve)"
        return f"[ERROR] Ollama: {e}"


def call_openrouter(prompt: str, system: str, model: str, temperature: float = 0.3) -> str:
    """Llama a OpenRouter (cloud, API compatible con OpenAI) y devuelve la respuesta."""
    if not OPENROUTER_API_KEY:
        return ("[ERROR] OPENROUTER_API_KEY no configurada. "
                "Agrégala a tu .env (OPENROUTER_API_KEY=...).")

    # Si pidieron el modelo cloud genérico, usar el del .env
    actual_model = OPENROUTER_MODEL if model.lower() in ("cloud", "openrouter") else model

    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/JRavenelco/cDAQ_9174",
        "X-Title": "Razonador CBR - Histeresis",
    }
    payload = {
        "model": actual_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 2048,
    }
    try:
        data = _http_post_json(url, payload, headers=headers, timeout=180)
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return f"[ERROR] OpenRouter respuesta sin choices: {data}"
    except Exception as e:
        return f"[ERROR] OpenRouter ({actual_model}): {e}"


def call_llm(prompt: str, system: str, model: str, temperature: float = 0.3) -> str:
    """Enruta la llamada al proveedor correcto según el modelo.

    - Modelos con '/' o 'cloud'/'openrouter' → OpenRouter (nube)
    - Resto (qwen3:8b, gemma4, llama3.2:3b ...) → Ollama (local)
    """
    if resolve_provider(model) == "openrouter":
        return call_openrouter(prompt, system, model, temperature)
    return call_ollama(prompt, system, model, temperature)


def format_case_summary(case: dict) -> str:
    """Formatea un caso para el prompt del LLM."""
    lines = []
    lines.append(f"  ID: {case.get('case_id', '?')}")
    lines.append(f"  Tipo: {case.get('source_kind', '?')}")
    rpm = case.get('rpm_estimada', '')
    if rpm:
        lines.append(f"  RPM estimada: {rpm}")
    rpm_nom = case.get('rpm_nominal_filename', '')
    if rpm_nom:
        lines.append(f"  RPM nominal: {rpm_nom}")
    paso = case.get('paso_filename', '')
    if paso:
        lines.append(f"  Avance (paso): {paso}")
    lines.append(f"  Duración: {as_float(case, 'duration_s'):.2f} s ({case.get('n_samples', '?')} muestras)")
    lines.append(f"  Fs: {as_float(case, 'fs_hz'):.1f} Hz")
    lines.append(f"  Fuerza RMS: {as_float(case, 'force_rms'):.4f} V")
    lines.append(f"  Fuerza pico: {as_float(case, 'force_peak_abs'):.4f} V")
    lines.append(f"  Fuerza σ: {as_float(case, 'force_std'):.4f}")
    lines.append(f"  Entrada RMS: {as_float(case, 'input_rms'):.4f}")
    lines.append(f"  Entrada pico: {as_float(case, 'input_peak_abs'):.4f}")
    lines.append(f"  Correlación fuerza-entrada: {as_float(case, 'corr_force_input'):.4f}")
    lines.append(f"  Área de lazo normalizada: {as_float(case, 'loop_area_norm'):.4f}")
    return "\n".join(lines)


def build_reasoning_prompt(
    query: dict,
    matches: list[RetrievalResult],
) -> str:
    """Construye el prompt completo para el LLM."""
    sections = []

    # Caso actual
    sections.append("## CASO ACTUAL (consulta)")
    sections.append(format_case_summary(query))
    sections.append("")

    # Clasificación rápida del lazo
    loop_area = as_float(query, "loop_area_norm")
    if loop_area >= 1.0:
        hyst_hint = "HISTÉRESIS MARCADA (área > 1.0)"
    elif loop_area >= 0.25:
        hyst_hint = "histéresis moderada (0.25 < área < 1.0)"
    elif loop_area >= 0.05:
        hyst_hint = "cuasi-lineal (0.05 < área < 0.25)"
    else:
        hyst_hint = "prácticamente lineal (área < 0.05)"
    sections.append(f"Indicador de histéresis: {hyst_hint}")
    sections.append("")

    # Casos similares
    sections.append(f"## CASOS HISTÓRICOS MÁS SIMILARES (top-{len(matches)})")
    for i, m in enumerate(matches, 1):
        sections.append(f"\n### Match #{i} (similitud: {m.score:.4f}, distancia: {m.distance:.4f})")
        sections.append(format_case_summary(m.case))
        m_area = as_float(m.case, "loop_area_norm")
        sections.append(f"  [Δ área lazo: {abs(loop_area - m_area):.4f}]")

    sections.append("")
    sections.append("## INSTRUCCIONES")
    sections.append("Basándote en los datos anteriores:")
    sections.append("1. ¿El caso actual es lineal, cuasi-lineal o histéretico? Justifica con números.")
    sections.append("2. ¿Qué casos históricos son más cercanos y por qué?")
    sections.append("3. ¿Se recomienda ajuste Bouc-Wen? ¿Hay evidencia suficiente?")
    sections.append("4. Decisión experimental: ¿seguir, repetir, ajustar parámetros, o revisar herramienta?")
    sections.append("5. Conclusión breve para tesis doctoral.")

    return "\n".join(sections)


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def reason_about_case(
    query: dict,
    library: list[dict],
    model: str = DEFAULT_MODEL,
    top_k: int = 5,
    use_llm: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Pipeline completo: retrieval + reasoning."""
    t0 = time.time()

    # 1. Retrieval
    matches = retrieve_similar(
        query=query,
        library=library,
        features=RETRIEVAL_FEATURES,
        weights=FEATURE_WEIGHTS,
        top_k=top_k,
    )

    retrieval_time = time.time() - t0

    if verbose:
        print(f"\n{'='*70}")
        print(f"  CASO: {query.get('case_id', '?')}")
        print(f"  Área lazo: {as_float(query, 'loop_area_norm'):.4f}")
        print(f"  Retrieval: {len(matches)} matches en {retrieval_time*1000:.1f} ms")
        print(f"{'='*70}")
        for i, m in enumerate(matches, 1):
            print(f"  #{i} {m.case.get('case_id', '?'):40s}  "
                  f"score={m.score:.4f}  area={as_float(m.case, 'loop_area_norm'):.4f}")

    # 2. Reasoning
    reasoning_response = ""
    reasoning_time = 0.0

    if use_llm:
        system_prompt = load_system_prompt(PROMPT_FILE)
        user_prompt = build_reasoning_prompt(query, matches)

        if verbose:
            provider = resolve_provider(model)
            print(f"\n  Llamando a {provider} ({model})...")

        t1 = time.time()
        reasoning_response = call_llm(user_prompt, system_prompt, model)
        reasoning_time = time.time() - t1

        if verbose:
            print(f"  Respuesta en {reasoning_time:.1f} s")
            print(f"\n{'─'*70}")
            print(reasoning_response)
            print(f"{'─'*70}")

    total_time = time.time() - t0

    # 3. Resultado estructurado
    result = {
        "case_id": query.get("case_id", ""),
        "query_features": {f: as_float(query, f) for f in RETRIEVAL_FEATURES},
        "top_matches": [
            {
                "case_id": m.case.get("case_id", ""),
                "score": round(m.score, 6),
                "distance": round(m.distance, 6),
                "loop_area_norm": as_float(m.case, "loop_area_norm"),
                "rpm_estimada": m.case.get("rpm_estimada", ""),
                "source_kind": m.case.get("source_kind", ""),
            }
            for m in matches
        ],
        "reasoning": reasoning_response,
        "model": model if use_llm else "none",
        "retrieval_time_ms": round(retrieval_time * 1000, 1),
        "reasoning_time_s": round(reasoning_time, 2),
        "total_time_s": round(total_time, 2),
    }

    return result


def save_result(result: dict, output_dir: Path) -> Path:
    """Guarda el resultado del razonamiento."""
    output_dir.mkdir(parents=True, exist_ok=True)
    case_id = result.get("case_id", "unknown")
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"razonamiento_{case_id}_{ts}.json"
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Razonador Basado en Casos (CBR) con LLM local (Ollama) o nube (OpenRouter)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Local (Ollama):
  python razonador_local.py --case-id corte_20260311_131921_600_40
  python razonador_local.py --case-id ... --model qwen3:8b

  # Nube (OpenRouter, usa OPENROUTER_MODEL del .env):
  python razonador_local.py --case-id ... --model cloud
  python razonador_local.py --case-id ... --model qwen/qwen2.5-vl-72b-instruct

  python razonador_local.py --csv ../corte_20260506_151429.csv
  python razonador_local.py --batch --top-k 3 --no-llm
  python razonador_local.py --case-id corte_20260312_134351_400_20 --model gemma4
        """,
    )

    # Fuente del caso
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--case-id", type=str, help="ID de caso existente en la base")
    source.add_argument("--csv", type=Path, help="CSV nuevo (captura de GUI)")
    source.add_argument("--batch", action="store_true", help="Procesar todos los casos")

    # Parámetros
    parser.add_argument("--top-k", type=int, default=5, help="Número de matches (default: 5)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Modelo Ollama (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-llm", action="store_true", help="Solo retrieval, sin LLM")
    parser.add_argument("--cases-file", type=Path, default=CASES_JSONL, help="JSONL de casos")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directorio de salida")
    parser.add_argument("--quiet", action="store_true", help="Menos output")

    args = parser.parse_args()

    # Cargar base de casos
    library = load_cases(args.cases_file)
    print(f"Base de casos: {len(library)} casos cargados de {args.cases_file.name}")

    # Determinar caso(s) a procesar
    queries: list[dict] = []

    if args.csv:
        if not args.csv.exists():
            print(f"ERROR: {args.csv} no existe")
            sys.exit(1)
        print(f"Extrayendo features de: {args.csv}")
        features = extract_features_from_csv(args.csv)
        queries.append(features)

    elif args.case_id:
        found = [c for c in library if c.get("case_id") == args.case_id]
        if not found:
            print(f"ERROR: case_id '{args.case_id}' no encontrado.")
            print(f"  Disponibles: {[c['case_id'] for c in library]}")
            sys.exit(1)
        queries.append(found[0])

    elif args.batch:
        queries = list(library)
        print(f"Modo batch: procesando {len(queries)} casos")

    else:
        # Default: mostrar lista de casos y salir
        print("\nCasos disponibles:")
        print(f"{'ID':<50s} {'Área lazo':<12s} {'RPM':<10s} {'Tipo'}")
        print("-" * 90)
        for c in library:
            print(f"{c['case_id']:<50s} "
                  f"{as_float(c, 'loop_area_norm'):<12.4f} "
                  f"{str(c.get('rpm_estimada', '')):<10s} "
                  f"{c.get('source_kind', '')}")
        print(f"\nUsa --case-id <ID> para razonar sobre un caso.")
        return

    # Procesar
    results = []
    for i, query in enumerate(queries, 1):
        if len(queries) > 1:
            print(f"\n[{i}/{len(queries)}] {query.get('case_id', '?')}")

        result = reason_about_case(
            query=query,
            library=library,
            model=args.model,
            top_k=args.top_k,
            use_llm=not args.no_llm,
            verbose=not args.quiet,
        )
        results.append(result)

        # Guardar resultado individual
        path = save_result(result, args.output_dir)
        if not args.quiet:
            print(f"  Guardado: {path}")

    # Resumen batch
    if len(results) > 1:
        summary_path = args.output_dir / f"batch_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResumen batch: {summary_path}")

    print(f"\n✅ Completado: {len(results)} caso(s) procesados.")


if __name__ == "__main__":
    main()
