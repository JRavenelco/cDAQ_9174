#!/usr/bin/env python3
"""
Build a static 3D memory view from the existing case-reasoner outputs.

This adapter intentionally does not replace the case builder. It reads the
JSONL index and feature CSVs produced by razonador_casos/construir_casos.py,
then derives entropy-aware similarity, nearest-neighbor links and reproducible
3D positions for the React viewer.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
VIEW_ROOT = HERE.parents[1]
PROJECT_ROOT = VIEW_ROOT.parents[1]
DATASET_ROOT = PROJECT_ROOT / "razonador_casos" / "salidas" / "casos_dataset"
GRAPHICS_ROOT = PROJECT_ROOT / "graficas_bouc_wen"
CASES_JSONL = DATASET_ROOT / "casos_historicos.jsonl"
OUT_JSON = VIEW_ROOT / "public" / "memory_cases.json"
PUBLIC_ARTIFACTS = VIEW_ROOT / "public" / "artifacts"

CEREBRO_URL = "http://192.168.137.164:8765"
CEREBRO_TIMEOUT = 5  # segundos

FEATURE_COLUMNS = [
    "force_mean",
    "force_std",
    "force_rms",
    "force_peak_abs",
    "input_mean",
    "input_std",
    "input_rms",
    "input_peak_abs",
    "corr_force_input",
    "loop_area_norm",
    "duration_s",
    "rpm_estimada",
    "paso_filename",
]

ARTIFACT_DESCRIPTIONS = {
    "01_bouc_wen_corte2": {
        "title": "Modelo Bouc-Wen - Corte 2 dientes",
        "family": "Bouc-Wen",
        "alpha": 0.729,
        "r2_corte": 0.135,
        "rmse": 0.0486,
        "corr_corte": -0.032,
        "interpretation": "Bouc-Wen clasico aplicado al corte; muestra que la dinamica no queda capturada por desplazamiento directo.",
    },
    "02_validacion_linealidad": {
        "title": "Validacion de linealidad del sensor",
        "family": "validacion",
        "alpha": 0.991,
        "interpretation": "El sensor se comporta casi lineal; la histeresis relevante viene del proceso de corte.",
    },
    "03_confirmacion_histeresis": {
        "title": "Confirmacion de histeresis genuina",
        "family": "validacion",
        "r2_static": 0.661,
        "r2_dynamic": 0.676,
        "improvement": 0.015,
        "interpretation": "Agregar dE/dt solo mejora 1.5%, señal de histeresis no-lineal genuina.",
    },
    "04_validacion_alpha_0.75": {
        "title": "Validacion Bouc-Wen alpha 0.75",
        "family": "Bouc-Wen",
        "alpha": 0.751,
        "z_rms_um": 0.17,
        "fn_hz": 30.4,
    },
    "05_bouc_wen_corte_envolvente": {
        "title": "Bouc-Wen con envolvente de corte",
        "family": "Bouc-Wen envolvente",
        "r2_corte": 0.700,
    },
    "06_bouc_wen_corte_modelo_completo": {
        "title": "Bouc-Wen corte modelo completo",
        "family": "Bouc-Wen",
        "alpha": 0.858,
        "z_rms_um": 0.45,
    },
    "07_comparacion_bouc_wen_shaker_vs_corte": {
        "title": "Comparacion Bouc-Wen shaker vs corte",
        "family": "comparacion",
    },
    "08_comparacion_modelos_fuerza": {
        "title": "Comparacion de modelos de fuerza",
        "family": "comparacion",
    },
    "13_comparacion_modelos_corregido": {
        "title": "Comparacion de modelos corregido",
        "family": "comparacion",
    },
    "21_modelos_con_envolvente": {
        "title": "Modelos con envolvente de aceleracion",
        "family": "comparacion envolvente",
    },
    "23_kan_pinn_con_envolvente": {
        "title": "KAN-PINN con envolvente",
        "family": "KAN-PINN envolvente",
        "r2_corte": 0.283,
    },
}


def safe_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def finite_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def median(values: list[float]) -> float:
    finite = sorted(finite_values(values))
    if not finite:
        return 0.0
    mid = len(finite) // 2
    if len(finite) % 2:
        return finite[mid]
    return (finite[mid - 1] + finite[mid]) / 2


def percentile(values: list[float], q: float) -> float:
    finite = sorted(finite_values(values))
    if not finite:
        return 0.0
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return finite[lo]
    weight = position - lo
    return finite[lo] * (1 - weight) + finite[hi] * weight


def discrete_entropy(values: list[float], bins: int = 12) -> float:
    finite = finite_values(values)
    if len(finite) <= 1:
        return 0.0
    min_value = min(finite)
    max_value = max(finite)
    width = max_value - min_value
    if width <= 1e-12:
        return 0.0
    counts = [0] * bins
    for value in finite:
        index = min(bins - 1, int((value - min_value) / width * bins))
        counts[index] += 1
    total = sum(counts)
    entropy = 0.0
    for count in counts:
        if count:
            prob = count / total
            entropy -= prob * math.log2(prob)
    return entropy / math.log2(bins)


def read_cases() -> list[dict[str, Any]]:
    if not CASES_JSONL.exists():
        raise FileNotFoundError(f"No existe {CASES_JSONL}")
    records = []
    with CASES_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_metric_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def model_metrics_by_name() -> dict[str, dict[str, float | str]]:
    metrics: dict[str, dict[str, float | str]] = {}
    for file_name in ("comparacion_modelos_v2.csv", "comparacion_modelos_fuerza.csv"):
        for row in read_metric_rows(GRAPHICS_ROOT / file_name):
            model = row.get("Modelo", "")
            if not model:
                continue
            key = normalize_name(model)
            current = metrics.setdefault(key, {"model": model, "equation": row.get("Ecuacion", "")})
            for column, value in row.items():
                numeric = safe_number(value)
                if math.isfinite(numeric):
                    current[column.lower()] = numeric
                elif value:
                    current[column.lower()] = value
    for row in read_metric_rows(GRAPHICS_ROOT / "resultados_bouc_wen_comparativo.csv"):
        parameter = row.get("Parametro", "").lower()
        if not parameter:
            continue
        key = "boucwencomparativo"
        current = metrics.setdefault(key, {"model": "Bouc-Wen comparativo"})
        for side in ("Shaker", "Corte"):
            value = safe_number(row.get(side))
            if math.isfinite(value):
                current[f"{parameter}_{side.lower()}"] = value
    return metrics


def infer_artifact_family(stem: str) -> str:
    lowered = stem.lower()
    if "kan" in lowered and "envolvente" in lowered:
        return "KAN-PINN envolvente"
    if "kan" in lowered:
        return "KAN-PINN"
    if "duhem" in lowered:
        return "Duhem"
    if "viscoso" in lowered:
        return "Bouc-Wen viscoso"
    if "bouc" in lowered or "bw" in lowered:
        return "Bouc-Wen"
    if "linealidad" in lowered or "histeresis" in lowered or "alpha" in lowered:
        return "validacion"
    if "comparacion" in lowered or "modelos" in lowered:
        return "comparacion"
    return "artefacto"


def metrics_for_artifact(stem: str, metrics_lookup: dict[str, dict[str, float | str]]) -> dict[str, float | str]:
    known = dict(ARTIFACT_DESCRIPTIONS.get(stem, {}))
    normalized = normalize_name(stem)
    aliases = {
        "bw_simple": "bwsimple",
        "bw_viscoso": "bwviscoso",
        "duhem": "duhem",
        "kanvisc": "kanvisc",
        "kanpinn": "kanpinn",
    }
    for token, key in aliases.items():
        if token.replace("_", "") in normalized or key in normalized:
            known.update(metrics_lookup.get(key, {}))
    if "comparacion_bouc_wen_shaker_vs_corte" in stem:
        known.update(metrics_lookup.get("boucwencomparativo", {}))
    alpha_match = re.search(r"alpha[_-](\d+(?:[._]\d+)?)", stem.lower())
    if alpha_match and "alpha" not in known:
        known["alpha"] = safe_number(alpha_match.group(1).replace("_", "."))
    return known


def build_artifact_cases(experimental_ids: list[str]) -> list[dict[str, Any]]:
    if not GRAPHICS_ROOT.exists():
        return []
    PUBLIC_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    metrics_lookup = model_metrics_by_name()
    artifacts = []
    pngs = sorted(GRAPHICS_ROOT.glob("*.png"))
    for index, path in enumerate(pngs):
        public_path = PUBLIC_ARTIFACTS / path.name
        if not public_path.exists() or public_path.stat().st_mtime < path.stat().st_mtime:
            shutil.copy2(path, public_path)
        stem = path.stem
        metrics = metrics_for_artifact(stem, metrics_lookup)
        family = str(metrics.get("family") or infer_artifact_family(stem))
        title = str(metrics.get("title") or stem.replace("_", " "))
        related_experimental = []
        lowered = stem.lower()
        if "corte2" in lowered:
            related_experimental = [case_id for case_id in experimental_ids if "corte2" in case_id][:2]
        elif "corte" in lowered or "kan" in lowered or "bouc" in lowered or "duhem" in lowered:
            related_experimental = experimental_ids[:4]
        if "shaker" in lowered:
            related_experimental.extend([case_id for case_id in experimental_ids if "evento" in case_id][:2])
        related_experimental = list(dict.fromkeys(related_experimental))

        feature_vector = {
            "artifact_index": float(index),
            "alpha": safe_number(metrics.get("alpha", metrics.get("alpha_corte", ""))),
            "r2_corte": safe_number(metrics.get("r2_corte", "")),
            "r2_shaker": safe_number(metrics.get("r2_shaker", "")),
            "corr_corte": safe_number(metrics.get("corr_corte", "")),
            "corr_shaker": safe_number(metrics.get("corr_shaker", "")),
            "rmse": safe_number(metrics.get("rmse", metrics.get("rmse_corte", ""))),
            "z_rms_um": safe_number(metrics.get("z_rms_um", "")),
            "improvement": safe_number(metrics.get("improvement", "")),
        }
        feature_vector = {key: value for key, value in feature_vector.items() if math.isfinite(value)}
        artifacts.append(
            {
                "id": f"artifact_{stem}",
                "position": [0.0, 0.0, 0.0],
                "metadata": {
                    "case_id": f"artifact_{stem}",
                    "memory_kind": "artifact",
                    "artifact_role": family,
                    "thesis_role": family,
                    "title": title,
                    "source_file": path.name,
                    "source_relpath": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "artifact_url": f"/artifacts/{path.name}",
                    "artifact_relpath": str(public_path.relative_to(VIEW_ROOT / "public")).replace("\\", "/"),
                    "file_size_bytes": path.stat().st_size,
                    "related_experimental_cases": related_experimental,
                    "interpretation": metrics.get("interpretation", ""),
                    "equation": metrics.get("equation", ""),
                },
                "entropy": {
                    "force_entropy": 0.0,
                    "input_entropy": 0.0,
                    "energy_entropy": safe_number(feature_vector.get("r2_corte", 0.0)) if feature_vector else 0.0,
                    "hysteresis_entropy": safe_number(feature_vector.get("alpha", 0.0)) if feature_vector else 0.0,
                },
                "feature_vector": feature_vector,
                "neighbors": [],
            }
        )
    return artifacts


def signal_signature(case: dict[str, Any]) -> dict[str, float]:
    path = PROJECT_ROOT / case["features_relpath"]
    if not path.exists():
        return {
            "force_entropy": 0.0,
            "input_entropy": 0.0,
            "energy_entropy": 0.0,
            "hysteresis_entropy": 0.0,
        }
    force: list[float] = []
    input_signal: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            force.append(safe_number(row.get("fuerza_norm")))
            input_signal.append(safe_number(row.get("entrada_norm")))
    energy = [f * f + u * u for f, u in zip(force, input_signal) if math.isfinite(f) and math.isfinite(u)]
    hysteresis = []
    for index in range(1, min(len(force), len(input_signal)) - 1):
        f = force[index]
        du = (input_signal[index + 1] - input_signal[index - 1]) / 2
        if math.isfinite(f) and math.isfinite(du):
            hysteresis.append(f * du)
    return {
        "force_entropy": discrete_entropy(force),
        "input_entropy": discrete_entropy(input_signal),
        "energy_entropy": discrete_entropy(energy),
        "hysteresis_entropy": discrete_entropy(hysteresis),
    }


def robust_matrix(cases: list[dict[str, Any]]) -> tuple[list[list[float]], list[dict[str, float]]]:
    rows = []
    signatures = []
    for case in cases:
        signature = signal_signature(case)
        signatures.append(signature)
        scalar_features = [safe_number(case.get(column)) for column in FEATURE_COLUMNS]
        entropy_features = [signature[key] for key in sorted(signature)]
        rows.append(scalar_features + entropy_features)

    columns = list(zip(*rows))
    centers = [median(list(column)) for column in columns]
    q25 = [percentile(list(column), 0.25) for column in columns]
    q75 = [percentile(list(column), 0.75) for column in columns]
    scales = []
    for index, column in enumerate(columns):
        spread = q75[index] - q25[index]
        if spread <= 1e-12:
            centered = [value - centers[index] for value in finite_values(list(column))]
            variance = sum(value * value for value in centered) / max(1, len(centered))
            spread = math.sqrt(variance)
        scales.append(spread if spread > 1e-12 else 1.0)

    normalized = []
    for row in rows:
        normalized.append(
            [
                ((value if math.isfinite(value) else centers[index]) - centers[index]) / scales[index]
                for index, value in enumerate(row)
            ]
        )
    return normalized, signatures


def distance(row_a: list[float], row_b: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(row_a, row_b)) / max(1, len(row_a)))


def force_layout(distances: list[list[float]], similarities: list[list[float]]) -> list[list[float]]:
    n = len(distances)
    if n == 0:
        return []
    positions = []
    golden = math.pi * (3 - math.sqrt(5))
    for index in range(n):
        z = 1 - (2 * index + 1) / n
        radius = math.sqrt(max(0.0, 1 - z * z))
        theta = index * golden
        positions.append([radius * math.cos(theta) * 5, z * 4, radius * math.sin(theta) * 5])

    max_distance = max((value for row in distances for value in row), default=1.0) or 1.0
    for _ in range(180):
        forces = [[0.0, 0.0, 0.0] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dx = [positions[j][axis] - positions[i][axis] for axis in range(3)]
                current = math.sqrt(sum(value * value for value in dx)) or 1e-6
                target = 1.2 + 7.0 * distances[i][j] / max_distance
                strength = 0.015 + similarities[i][j] * 0.025
                delta = (current - target) * strength
                unit = [value / current for value in dx]
                for axis in range(3):
                    forces[i][axis] += unit[axis] * delta
                    forces[j][axis] -= unit[axis] * delta
        for i in range(n):
            for axis in range(3):
                positions[i][axis] += forces[i][axis]

    max_abs = max((abs(value) for point in positions for value in point), default=1.0) or 1.0
    return [[value / max_abs * 6 for value in point] for point in positions]


# ══════════════════════════════════════════════════════════════════
# Neuronas VLM — casos remotos desde el cerebro en la Jetson
# ══════════════════════════════════════════════════════════════════

def fetch_remote_cases(cerebro_url: str = CEREBRO_URL) -> list[dict[str, Any]]:
    """Obtiene los casos aprobados por el VLM desde la Jetson."""
    try:
        url = f"{cerebro_url}/api/casos"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=CEREBRO_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("casos", [])
            return []
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError) as e:
        print(f"  [cerebro] Sin conexion con Jetson ({cerebro_url}): {e}")
        return []


def build_neuron_cases(
    remote_casos: list[dict[str, Any]],
    experimental_ids: list[str],
) -> list[dict[str, Any]]:
    """Crea nodos de 'neurona VLM' desde los casos remotos."""
    if not remote_casos:
        return []

    neuronas = []
    cond_colors = {"Nuevo": "#2ecc71", "Medio uso": "#f39c12", "Desgastado": "#e74c3c"}

    golden = math.pi * (3 - math.sqrt(5))
    for i, caso in enumerate(remote_casos):
        meta = caso.get("metadata", {}) if isinstance(caso.get("metadata"), dict) else {}
        vlm = caso.get("vlm", {}) if isinstance(caso.get("vlm"), dict) else {}
        cond = meta.get("cutter_condition") or vlm.get("cutter") or "N/A"
        rpm = meta.get("rpm_husillo") or vlm.get("RPM") or 0
        q = vlm.get("quality") or "N/A"
        cutting = vlm.get("cutting_state", False)
        timestamp = caso.get("timestamp", caso.get("_file", f"neurona_{i}"))
        area = caso.get("loop_area_norm") or vlm.get("loop_area") or 1.0
        confidence = float(cutting) if cutting else 0.5

        theta = i * golden
        neuron_id = caso.get("case_id") or f"neurona_vlm_{i}"

        neuronas.append({
            "id": neuron_id,
            "position": [
                round(math.cos(theta) * 8.0 + (i % 3 - 1) * 1.5, 5),
                round(4.5 - (i % 5) * 1.1, 5),
                round(math.sin(theta) * 8.0 + (i // 3) * 1.2, 5),
            ],
            "metadata": {
                "case_id": neuron_id,
                "memory_kind": "neurona_vlm",
                "artifact_role": "VLM experto",
                "thesis_role": "razonamiento en tiempo real",
                "title": f"VLM: {cond} @ {rpm}rpm ({q})",
                "cutter_condition": cond,
                "rpm": rpm,
                "quality": q,
                "cutting_state": cutting,
                "confidence": confidence,
                "vlm_model": vlm.get("_vlm_model", "moondream"),
                "timestamp": timestamp,
                "area_hysteresis": area,
                "notes": vlm.get("notes", vlm.get("_vlm_description", "")),
                "anomaly": vlm.get("anomaly"),
                "color": cond_colors.get(cond, "#3498db"),
                "should_store": vlm.get("should_store", True),
                "source": "cerebro_jetson",
                "source_url": f"{CEREBRO_URL}/api/casos",
            },
            "entropy": {
                "force_entropy": round(area, 5),
                "input_entropy": round(confidence, 5),
                "energy_entropy": round(float(cutting) * area, 5),
                "hysteresis_entropy": round(area * confidence, 5),
            },
            "feature_vector": {
                "rpm": float(rpm) if rpm else 0.0,
                "area_hysteresis": float(area),
                "confidence": float(confidence),
                "cutting_flag": 1.0 if cutting else 0.0,
            },
            "neighbors": [],
        })

    # Enlazar con casos experimentales por RPM cercano
    for neurona in neuronas:
        neurona_rpm = neurona["metadata"].get("rpm", 0)
        neurona_cond = neurona["metadata"].get("cutter_condition", "")
        for exp_id in experimental_ids[:8]:
            neurona["neighbors"].append({
                "case_id": exp_id,
                "distance": 0.55,
                "similarity": 0.82,
                "reasons": [
                    f"neurona VLM etiquetada como {neurona_cond}",
                    f"RPM de referencia: {neurona_rpm}",
                ],
            })

    return neuronas


def build_payload() -> dict[str, Any]:
    cases = read_cases()
    matrix, signatures = robust_matrix(cases)
    n = len(cases)
    distances = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = distance(matrix[i], matrix[j])
            distances[i][j] = distances[j][i] = d

    positive = [value for row in distances for value in row if value > 0]
    sigma = median(positive) if positive else 1.0
    sigma = sigma if sigma > 1e-12 else 1.0
    similarities = [
        [math.exp(-(distances[i][j] ** 2) / (2 * sigma**2)) for j in range(n)]
        for i in range(n)
    ]
    positions = force_layout(distances, similarities)

    visual_cases = []
    links = []
    for i, case in enumerate(cases):
        neighbor_order = [idx for idx, _ in sorted(enumerate(distances[i]), key=lambda item: item[1]) if idx != i]
        neighbors = []
        for idx in neighbor_order[:5]:
            reasons = []
            if case.get("duplicate_group") == cases[idx].get("duplicate_group"):
                reasons.append("mismo grupo experimental")
            if abs(safe_number(case.get("loop_area_norm")) - safe_number(cases[idx].get("loop_area_norm"))) < 0.25:
                reasons.append("area de histeresis cercana")
            if abs(signatures[i]["energy_entropy"] - signatures[idx]["energy_entropy"]) < 0.08:
                reasons.append("entropia de energia cercana")
            if abs(safe_number(case.get("corr_force_input")) - safe_number(cases[idx].get("corr_force_input"))) < 0.01:
                reasons.append("correlacion fuerza-entrada parecida")
            neighbors.append(
                {
                    "case_id": cases[idx]["case_id"],
                    "distance": round(float(distances[i][idx]), 5),
                    "similarity": round(float(similarities[i][idx]), 5),
                    "reasons": reasons or ["perfil multifeature cercano"],
                }
            )
        visual_cases.append(
            {
                "id": case["case_id"],
                "position": [round(float(x), 5) for x in positions[i]],
                "metadata": case,
                "entropy": {key: round(float(value), 5) for key, value in signatures[i].items()},
                "feature_vector": {
                    key: safe_number(case.get(key))
                    for key in FEATURE_COLUMNS
                    if math.isfinite(safe_number(case.get(key)))
                },
                "neighbors": neighbors,
            }
        )

    seen = set()
    for i in range(n):
        for neighbor in visual_cases[i]["neighbors"][:3]:
            j = next(index for index, item in enumerate(cases) if item["case_id"] == neighbor["case_id"])
            key = tuple(sorted((cases[i]["case_id"], cases[j]["case_id"])))
            if key in seen:
                continue
            seen.add(key)
            links.append(
                {
                    "source": key[0],
                    "target": key[1],
                    "similarity": neighbor["similarity"],
                    "distance": neighbor["distance"],
                }
            )

    artifact_cases = build_artifact_cases([case["case_id"] for case in cases])
    if artifact_cases:
        golden = math.pi * (3 - math.sqrt(5))
        for index, artifact in enumerate(artifact_cases):
            theta = index * golden
            radius = 7.4 + (index % 4) * 0.52
            artifact["position"] = [
                round(math.cos(theta) * radius, 5),
                round(-3.6 + (index % 7) * 1.15, 5),
                round(math.sin(theta) * radius, 5),
            ]

        artifacts_by_family: dict[str, list[dict[str, Any]]] = {}
        for artifact in artifact_cases:
            family = str(artifact["metadata"].get("artifact_role", "artefacto"))
            artifacts_by_family.setdefault(family, []).append(artifact)

        experimental_by_id = {case["id"]: case for case in visual_cases}
        for artifact in artifact_cases:
            neighbors = []
            family = str(artifact["metadata"].get("artifact_role", "artefacto"))
            for peer in artifacts_by_family.get(family, [])[:6]:
                if peer["id"] == artifact["id"]:
                    continue
                neighbors.append(
                    {
                        "case_id": peer["id"],
                        "distance": 0.35,
                        "similarity": 0.88,
                        "reasons": [f"misma familia de modelo: {family}"],
                    }
                )
            for case_id in artifact["metadata"].get("related_experimental_cases", [])[:3]:
                if case_id not in experimental_by_id:
                    continue
                neighbors.append(
                    {
                        "case_id": case_id,
                        "distance": 0.65,
                        "similarity": 0.78,
                        "reasons": ["artefacto generado para interpretar este corte/senal"],
                    }
                )
            artifact["neighbors"] = neighbors[:5]

        all_known_ids = {case["id"] for case in visual_cases + artifact_cases}
        for artifact in artifact_cases:
            for neighbor in artifact["neighbors"][:4]:
                if neighbor["case_id"] not in all_known_ids:
                    continue
                key = tuple(sorted((artifact["id"], neighbor["case_id"])))
                if key in seen:
                    continue
                seen.add(key)
                links.append(
                    {
                        "source": key[0],
                        "target": key[1],
                        "similarity": neighbor["similarity"],
                        "distance": neighbor["distance"],
                    }
                )
        visual_cases.extend(artifact_cases)

    # ── Neuronas VLM desde la Jetson ──
    remote_casos = fetch_remote_cases()
    neuron_cases = build_neuron_cases(remote_casos, [case["case_id"] for case in cases])

    if neuron_cases:
        # Enlazar neuronas entre sí (misma condición → cluster)
        for i, ni in enumerate(neuron_cases):
            for j, nj in enumerate(neuron_cases):
                if i >= j:
                    continue
                if ni["metadata"].get("cutter_condition") == nj["metadata"].get("cutter_condition"):
                    key = tuple(sorted((ni["id"], nj["id"])))
                    if key not in seen:
                        seen.add(key)
                        links.append({
                            "source": key[0],
                            "target": key[1],
                            "similarity": 0.92,
                            "distance": 0.25,
                        })

        for neurona in neuron_cases:
            for neighbor in neurona["neighbors"][:3]:
                if neighbor["case_id"] not in all_known_ids:
                    # Vincular con artefactos de modelo como fallback
                    for art in artifact_cases:
                        neighbor["case_id"] = art["id"]
                        break
                    else:
                        continue
                key = tuple(sorted((neurona["id"], neighbor["case_id"])))
                if key in seen:
                    continue
                seen.add(key)
                links.append({
                    "source": key[0],
                    "target": key[1],
                    "similarity": neighbor["similarity"],
                    "distance": neighbor["distance"],
                })

        visual_cases.extend(neuron_cases)

    return {
        "generated_from": str(CASES_JSONL.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "case_count": len(visual_cases),
        "experimental_case_count": n,
        "artifact_case_count": len(artifact_cases),
        "neurona_vlm_count": len(neuron_cases),
        "cerebro_url": CEREBRO_URL,
        "feature_columns": FEATURE_COLUMNS,
        "equivalence_model": {
            "name": "entropia_discreta_multifeature",
            "description": "Normaliza features escalares del caso, suma entropias discretas de fuerza/entrada/energia/histeresis y agrega artefactos Bouc-Wen/KAN/PINN como recuerdos de validacion y neuronas VLM desde la Jetson en tiempo real.",
            "similarity": "exp(-(distancia^2)/(2*mediana_distancias^2))",
            "positioning": "layout 3D determinista con atraccion por similitud y separacion por distancia",
        },
        "cases": visual_cases,
        "links": links,
    }


def main() -> int:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Vista 3D generada: {OUT_JSON}")
    print(f"Casos: {payload['case_count']}, relaciones: {len(payload['links'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
