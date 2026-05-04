#!/usr/bin/env python3
"""Compare virtual UDP cases against historical cutting cases.

This script closes the first reasoning layer:
virtual UDP -> case record -> nearest historical cases -> Bouc-Wen/VLM handoff.
It is intentionally dependency-light for Jetson use: standard library + numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
CAR_ROOT = SRC_ROOT / "caracterizacion_fuerza"
DEFAULT_FEATURES = [
    "force_rms",
    "force_peak_abs",
    "input_rms",
    "input_peak_abs",
    "corr_force_input",
    "loop_area_norm",
    "duration_s",
]
DEFAULT_WEIGHTS = {
    "force_rms": 1.0,
    "force_peak_abs": 0.8,
    "input_rms": 1.0,
    "input_peak_abs": 0.8,
    "corr_force_input": 0.7,
    "loop_area_norm": 1.4,
    "duration_s": 0.4,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"No se encontro JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        flat = flatten_for_csv(row)
        for key in flat:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_for_csv(row))


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = value
    return flat


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value == "" or value is None:
            return default
        value_f = float(value)
        return value_f if math.isfinite(value_f) else default
    except Exception:
        return default


def robust_stats(rows: list[dict[str, Any]], features: list[str]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for feature in features:
        values = sorted(as_float(row, feature) for row in rows)
        if not values:
            stats[feature] = (0.0, 1.0)
            continue
        center = float(median(values))
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        if not math.isfinite(iqr) or iqr <= 1e-12:
            iqr = float(np.std(values))
        if not math.isfinite(iqr) or iqr <= 1e-12:
            iqr = 1.0
        stats[feature] = (center, iqr)
    return stats


def vectorize(row: dict[str, Any], features: list[str], stats: dict[str, tuple[float, float]]) -> np.ndarray:
    values = []
    for feature in features:
        center, scale = stats[feature]
        values.append((as_float(row, feature) - center) / scale)
    return np.asarray(values, dtype=float)


def similarity_label(score: float, top_distance: float, case: dict[str, Any]) -> str:
    loop_area = as_float(case, "loop_area_norm")
    source_kind = str(case.get("source_kind", ""))
    if score >= 0.78 and top_distance <= 1.2:
        closeness = "alta"
    elif score >= 0.55:
        closeness = "media"
    else:
        closeness = "baja"
    if loop_area >= 1.0:
        hysteresis = "histeresis_marcada"
    elif loop_area >= 0.25:
        hysteresis = "histeresis_moderada"
    else:
        hysteresis = "baja_histeresis"
    if "chatter" in source_kind:
        mode = "virtual_chatter"
    elif "wear" in source_kind:
        mode = "virtual_wear"
    elif "impact" in source_kind:
        mode = "virtual_impact"
    else:
        mode = "virtual_normal"
    return f"{mode}_{hysteresis}_similitud_{closeness}"


def should_run_boucwen(score: float, case: dict[str, Any], match: dict[str, Any]) -> bool:
    has_window = bool(case.get("window_relpath"))
    has_reference = bool(match.get("window_relpath"))
    loop_area = as_float(case, "loop_area_norm")
    return has_window and has_reference and (score >= 0.45 or loop_area >= 0.20)


def make_llm_summary(case: dict[str, Any], matches: list[dict[str, Any]], score: float, run_boucwen: bool) -> str:
    if not matches:
        return "No hay casos historicos disponibles para comparar."
    best = matches[0]
    return (
        f"El caso virtual {case.get('case_id')} ({case.get('source_kind')}) se parece mas a "
        f"{best.get('case_id')} con similitud {score:.3f}. "
        f"El lazo virtual tiene area {as_float(case, 'loop_area_norm'):.4f} y correlacion fuerza-entrada "
        f"{as_float(case, 'corr_force_input'):.4f}. "
        f"Bouc-Wen {'si' if run_boucwen else 'no'} se recomienda como siguiente paso."
    )


def compare_cases(
    virtual_cases: list[dict[str, Any]],
    historical_cases: list[dict[str, Any]],
    features: list[str],
    weights: dict[str, float],
    top_k: int,
) -> list[dict[str, Any]]:
    if not historical_cases:
        raise SystemExit("No hay casos historicos para comparar")
    stats = robust_stats(historical_cases + virtual_cases, features)
    weight_vec = np.asarray([weights.get(feature, 1.0) for feature in features], dtype=float)
    hist_vectors = [(case, vectorize(case, features, stats)) for case in historical_cases]
    results: list[dict[str, Any]] = []

    for case in virtual_cases:
        v = vectorize(case, features, stats)
        ranked = []
        for hist, hv in hist_vectors:
            diff = (v - hv) * weight_vec
            distance = float(np.linalg.norm(diff) / math.sqrt(len(features)))
            score = float(1.0 / (1.0 + distance))
            ranked.append((score, distance, hist))
        ranked.sort(key=lambda item: (-item[0], item[1], str(item[2].get("case_id", ""))))
        top = ranked[:top_k]
        top_matches = [
            {
                "rank": idx + 1,
                "case_id": hist.get("case_id", ""),
                "source_kind": hist.get("source_kind", ""),
                "similarity_score": round(score, 6),
                "distance": round(distance, 6),
                "loop_area_norm": as_float(hist, "loop_area_norm"),
                "corr_force_input": as_float(hist, "corr_force_input"),
                "window_relpath": hist.get("window_relpath", ""),
                "suggested_boucwen_command": hist.get("suggested_boucwen_command", ""),
            }
            for idx, (score, distance, hist) in enumerate(top)
        ]
        best_score, best_distance, best_hist = top[0]
        run_boucwen = should_run_boucwen(best_score, case, best_hist)
        result = dict(case)
        result.update(
            {
                "matched_case_id": best_hist.get("case_id", ""),
                "matched_source_kind": best_hist.get("source_kind", ""),
                "similarity_score": round(best_score, 6),
                "similarity_distance": round(best_distance, 6),
                "top_matches": top_matches,
                "recommended_boucwen_reference": best_hist.get("window_relpath", ""),
                "recommended_boucwen_reference_command": best_hist.get("suggested_boucwen_command", ""),
                "run_boucwen_next": run_boucwen,
                "diagnostic_label_seed": similarity_label(best_score, best_distance, case),
                "llm_summary_seed": make_llm_summary(case, top_matches, best_score, run_boucwen),
                "comparison_features": features,
            }
        )
        results.append(result)
    return results


def parse_weights(text: str | None) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if not text:
        return weights
    for chunk in text.split(","):
        if not chunk.strip():
            continue
        key, value = chunk.split("=", 1)
        weights[key.strip()] = float(value)
    return weights


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank virtual UDP cases against historical cases")
    parser.add_argument(
        "--virtual-jsonl",
        type=Path,
        default=CAR_ROOT / "razonador_casos" / "salidas" / "casos_virtual_udp" / "casos_virtuales.jsonl",
    )
    parser.add_argument(
        "--historical-jsonl",
        type=Path,
        default=CAR_ROOT / "razonador_casos" / "salidas" / "casos_dataset" / "casos_historicos.jsonl",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=CAR_ROOT / "razonador_casos" / "salidas" / "casos_virtual_udp" / "casos_virtuales_ranked.jsonl",
    )
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--weights", default=None, help="Comma list like loop_area_norm=2,input_rms=1")
    args = parser.parse_args()

    features = [part.strip() for part in args.features.split(",") if part.strip()]
    weights = parse_weights(args.weights)
    virtual_cases = load_jsonl(args.virtual_jsonl)
    historical_cases = load_jsonl(args.historical_jsonl)
    ranked = compare_cases(virtual_cases, historical_cases, features, weights, max(1, args.top_k))
    out_csv = args.out_csv or args.out_jsonl.with_suffix(".csv")
    write_jsonl(args.out_jsonl, ranked)
    write_csv(out_csv, ranked)

    print(f"Casos virtuales: {len(virtual_cases)}")
    print(f"Casos historicos: {len(historical_cases)}")
    print(f"Salida JSONL: {args.out_jsonl}")
    print(f"Salida CSV: {out_csv}")
    if ranked:
        first = ranked[0]
        print(
            f"Primer match: {first.get('case_id')} -> {first.get('matched_case_id')} "
            f"score={first.get('similarity_score')} run_boucwen={first.get('run_boucwen_next')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

