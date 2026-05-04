from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CaseRecord:
    case_id: str
    source_kind: str
    source_file: str
    source_relpath: str
    duplicate_group: str
    is_duplicate_candidate: bool
    rpm_nominal_filename: float | str
    paso_filename: float | str
    rpm_estimada: float | str
    fs_hz: float
    duration_s: float
    n_samples: int
    force_col: str
    input_col: str
    force_mean: float
    force_std: float
    force_rms: float
    force_peak_abs: float
    input_mean: float
    input_std: float
    input_rms: float
    input_peak_abs: float
    corr_force_input: float
    loop_area_norm: float
    window_start_s: float
    window_end_s: float
    boucwen_ready_relpath: str
    window_relpath: str
    features_relpath: str
    suggested_boucwen_out_relpath: str
    suggested_boucwen_command: str
    thesis_role: str


def car_root() -> Path:
    return SRC_ROOT / "caracterizacion_fuerza"


def normalize_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    centered = x - np.nanmean(x)
    scale = np.nanmax(np.abs(centered))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(centered)
    return centered / scale


def rms(x: np.ndarray | list[float]) -> float:
    arr = np.asarray(x, dtype=float)
    if len(arr) == 0:
        return 0.0
    return float(np.sqrt(np.nanmean(arr ** 2)))


def loop_area(input_norm: np.ndarray, force_norm: np.ndarray) -> float:
    if len(input_norm) < 3:
        return 0.0
    dx = np.gradient(input_norm)
    area = abs(float(np.nansum(force_norm * dx)))
    rect = (np.nanmax(input_norm) - np.nanmin(input_norm)) * (np.nanmax(force_norm) - np.nanmin(force_norm))
    if not np.isfinite(rect) or rect <= 1e-12:
        return 0.0
    value = area / rect
    return float(value) if np.isfinite(value) else 0.0


def std_from_mean_rms(mean: float, rms_value: float) -> float:
    variance = max(float(rms_value) ** 2 - float(mean) ** 2, 0.0)
    return float(np.sqrt(variance))


def rel_any(path: Path, root: Path) -> str:
    return os.path.relpath(str(path.resolve()), str(root.resolve())).replace("\\", "/")


def sanitize_case_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def discover_latest_pair(logs_dir: Path) -> tuple[Path | None, Path | None]:
    case_files = sorted(logs_dir.glob("virtual_instrument_*_cases.jsonl"), key=lambda p: p.stat().st_mtime)
    if not case_files:
        csv_files = sorted(logs_dir.glob("virtual_instrument_*.csv"), key=lambda p: p.stat().st_mtime)
        return (csv_files[-1], None) if csv_files else (None, None)
    cases_path = case_files[-1]
    csv_path = cases_path.with_name(cases_path.name.replace("_cases.jsonl", ".csv"))
    return (csv_path if csv_path.exists() else None, cases_path)


def infer_pair(csv_path: Path | None, cases_path: Path | None) -> tuple[Path | None, Path | None]:
    if csv_path is None and cases_path is None:
        return None, None
    if csv_path is None and cases_path is not None:
        candidate = cases_path.with_name(cases_path.name.replace("_cases.jsonl", ".csv"))
        return (candidate if candidate.exists() else None, cases_path)
    if cases_path is None and csv_path is not None:
        candidate = csv_path.with_name(csv_path.stem + "_cases.jsonl")
        return (csv_path, candidate if candidate.exists() else None)
    return csv_path, cases_path


def load_cases_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_raw_csv(path: Path | None) -> list[dict] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    needed = {"seq", "t_sample_s", "fs_hz", "mode", "force_v", "accel_g"}
    if rows and not needed.issubset(rows[0].keys()):
        raise ValueError(f"CSV sin columnas esperadas: {path}")
    return rows


def f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except Exception:
        return default


def rows_for_feature(csv_rows: list[dict] | None, seq: int, fs: float, window_seconds: float) -> list[dict] | None:
    if not csv_rows:
        return None
    n_window = max(16, int(round(window_seconds * fs)))
    eligible = [row for row in csv_rows if int(float(row.get("seq", 0))) <= seq]
    return eligible[-n_window:] if eligible else None


def write_rows(path: Path, rows: list[dict], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    return f"{float(value):.9g}"


def build_signal_artifacts(
    case_id: str,
    case_rows: list[dict] | None,
    out_root: Path,
    window_seconds: float,
) -> tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    if not case_rows:
        return "", "", "", np.array([]), np.array([]), np.array([]), 0.0, 0.0, 0

    t_abs = np.asarray([f(row, "t_sample_s") for row in case_rows], dtype=float)
    t = t_abs - t_abs[0]
    force = np.asarray([f(row, "force_v") for row in case_rows], dtype=float)
    accel = np.asarray([f(row, "accel_g") for row in case_rows], dtype=float)
    force_norm = normalize_signal(force)
    input_norm = normalize_signal(accel)

    feature_rows = []
    ready_rows = []
    window_rows = []
    for i in range(len(t)):
        feature_rows.append(
            {
                "tiempo_s": format_float(t[i]),
                "fuerza_original": format_float(force[i]),
                "entrada_original": format_float(accel[i]),
                "fuerza_norm": format_float(force_norm[i]),
                "entrada_norm": format_float(input_norm[i]),
            }
        )
        ready_rows.append(
            {
                "tiempo_s": format_float(t[i]),
                "fuerza_norm": format_float(force_norm[i]),
                "entrada_norm": format_float(input_norm[i]),
            }
        )
        window_rows.append(
            {
                "tiempo_s": format_float(t[i]),
                "fuerza_V": format_float(force[i]),
                "entrada_g": format_float(accel[i]),
                "fuerza_norm": format_float(force_norm[i]),
                "entrada_norm": format_float(input_norm[i]),
            }
        )

    features_path = out_root / "features" / f"{case_id}_features.csv"
    ready_path = out_root / "boucwen_ready" / f"{case_id}_boucwen_ready.txt"
    window_path = out_root / "ventanas" / f"{case_id}_ventana_{window_seconds:g}s.txt"
    write_rows(features_path, feature_rows)
    write_rows(ready_path, ready_rows, delimiter="\t")
    write_rows(window_path, window_rows, delimiter="\t")
    return (
        rel_any(features_path, car_root()),
        rel_any(ready_path, car_root()),
        rel_any(window_path, car_root()),
        t_abs,
        force,
        accel,
        float(t_abs[0]),
        float(t_abs[-1]),
        int(len(t_abs)),
    )


def record_from_feature(
    feature: dict,
    csv_rows: list[dict] | None,
    csv_path: Path | None,
    out_root: Path,
    default_shaker: Path,
    fallback_window_seconds: float,
) -> CaseRecord:
    seq = int(feature.get("seq", 0))
    fs = float(feature.get("fs_hz", 2500.0))
    mode = str(feature.get("mode", "unknown"))
    base_case_id = str(feature.get("case_id") or f"virtual_udp_{mode}_{seq:06d}")
    case_id = sanitize_case_id(base_case_id)
    window_seconds = float(feature.get("window_seconds", fallback_window_seconds))
    case_rows = rows_for_feature(csv_rows, seq, fs, window_seconds)

    features_relpath, ready_relpath, window_relpath, t_abs, force, accel, w_start, w_end, n_samples = build_signal_artifacts(
        case_id=case_id,
        case_rows=case_rows,
        out_root=out_root,
        window_seconds=window_seconds,
    )

    if n_samples > 0:
        duration_s = float(t_abs[-1] - t_abs[0]) if len(t_abs) else 0.0
        force_mean = float(np.nanmean(force))
        force_std = float(np.nanstd(force))
        force_rms = rms(force)
        force_peak_abs = float(np.nanmax(np.abs(force)))
        input_mean = float(np.nanmean(accel))
        input_std = float(np.nanstd(accel))
        input_rms = rms(accel)
        input_peak_abs = float(np.nanmax(np.abs(accel)))
        force_norm = normalize_signal(force)
        input_norm = normalize_signal(accel)
        corr = float(np.corrcoef(force_norm, input_norm)[0, 1]) if len(force_norm) > 2 else 0.0
        corr = corr if np.isfinite(corr) else 0.0
        loop_norm = loop_area(input_norm, force_norm)
    else:
        duration_s = window_seconds
        force_mean = float(feature.get("force_mean", 0.0))
        force_rms = float(feature.get("force_rms", 0.0))
        force_std = std_from_mean_rms(force_mean, force_rms)
        force_peak_abs = max(abs(float(feature.get("force_min", 0.0))), abs(float(feature.get("force_max", 0.0))))
        input_mean = float(feature.get("accel_mean", 0.0))
        input_rms = float(feature.get("accel_rms", 0.0))
        input_std = std_from_mean_rms(input_mean, input_rms)
        input_peak_abs = max(abs(float(feature.get("accel_min", 0.0))), abs(float(feature.get("accel_max", 0.0))))
        corr = float(feature.get("corr_force_accel", 0.0))
        loop_norm = 0.0
        w_start = 0.0
        w_end = window_seconds
        n_samples = int(round(window_seconds * fs))

    source_file = csv_path.name if csv_path is not None else ""
    source_relpath = rel_any(csv_path, car_root()) if csv_path is not None and csv_path.exists() else ""
    duplicate_group = sanitize_case_id(f"virtual_udp_{mode}")
    boucwen_out = out_root / "boucwen_resultados" / case_id
    command = ""
    if window_relpath:
        shaker_relpath = rel_any(default_shaker, car_root()) if default_shaker.exists() else "datos_txt_ventanas/shaker_mejor_0.5s_20Hz.txt"
        command = (
            'python scripts_modelos/comparar_bouc_wen_shaker_corte.py '
            f'--shaker-file "{shaker_relpath}" '
            f'--corte-file "{window_relpath}" '
            f'--out-dir "{rel_any(boucwen_out, car_root())}" --no-show'
        )

    return CaseRecord(
        case_id=case_id,
        source_kind=f"virtual_udp_{mode}",
        source_file=source_file,
        source_relpath=source_relpath,
        duplicate_group=duplicate_group,
        is_duplicate_candidate=False,
        rpm_nominal_filename="",
        paso_filename="",
        rpm_estimada="",
        fs_hz=fs,
        duration_s=duration_s,
        n_samples=n_samples,
        force_col="force_v",
        input_col="accel_g",
        force_mean=force_mean,
        force_std=force_std,
        force_rms=force_rms,
        force_peak_abs=force_peak_abs,
        input_mean=input_mean,
        input_std=input_std,
        input_rms=input_rms,
        input_peak_abs=input_peak_abs,
        corr_force_input=corr,
        loop_area_norm=loop_norm,
        window_start_s=w_start,
        window_end_s=w_end,
        boucwen_ready_relpath=ready_relpath,
        window_relpath=window_relpath,
        features_relpath=features_relpath,
        suggested_boucwen_out_relpath=rel_any(boucwen_out, car_root()),
        suggested_boucwen_command=command,
        thesis_role="simulacion_virtual_udp",
    )


def load_historic_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_dict_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--historicos-jsonl", type=Path, default=None)
    args = parser.parse_args()

    csv_path, cases_path = infer_pair(args.csv, args.cases)
    if csv_path is None and cases_path is None:
        csv_path, cases_path = discover_latest_pair(args.logs_dir.resolve())
    if cases_path is None or not cases_path.exists():
        raise SystemExit("No se encontro *_cases.jsonl para convertir")

    out_root = (args.out_dir or (car_root() / "razonador_casos" / "salidas" / "casos_virtual_udp")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    csv_rows = load_raw_csv(csv_path)
    feature_rows = load_cases_jsonl(cases_path)
    if args.limit and args.limit > 0:
        feature_rows = feature_rows[: args.limit]

    default_shaker = car_root() / "datos_txt_ventanas" / "shaker_mejor_0.5s_20Hz.txt"
    records = [
        record_from_feature(
            feature=row,
            csv_rows=csv_rows,
            csv_path=csv_path,
            out_root=out_root,
            default_shaker=default_shaker,
            fallback_window_seconds=args.window_seconds,
        )
        for row in feature_rows
    ]

    record_dicts = [asdict(r) for r in records]
    csv_out = out_root / "casos_virtuales.csv"
    jsonl_out = out_root / "casos_virtuales.jsonl"
    write_dict_csv(csv_out, record_dicts)
    write_jsonl(jsonl_out, record_dicts)

    historicos_path = args.historicos_jsonl or (car_root() / "razonador_casos" / "salidas" / "casos_dataset" / "casos_historicos.jsonl")
    historicos = load_historic_jsonl(historicos_path)
    if historicos:
        merged = historicos + record_dicts
        write_jsonl(out_root / "casos_comparables_con_historicos.jsonl", merged)
        write_dict_csv(out_root / "casos_comparables_con_historicos.csv", merged)

    print(f"CSV fuente: {csv_path if csv_path else 'N/A'}")
    print(f"Cases fuente: {cases_path}")
    print(f"Casos virtuales: {len(records)}")
    print(f"Salida: {out_root}")
    print(f"JSONL virtual: {jsonl_out}")
    if historicos:
        print(f"JSONL combinado: {out_root / 'casos_comparables_con_historicos.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
