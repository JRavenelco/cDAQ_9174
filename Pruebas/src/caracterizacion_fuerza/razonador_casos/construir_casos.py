#!/usr/bin/env python3
"""
Build a case base from force/cutting experiments.

The script prepares every cut/window as reusable data for later Bouc-Wen,
LuGre, Hailo and VLM stages. It does not fit Bouc-Wen parameters; it creates a
clean, traceable case dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ACCEL_PRIORITY = (
    "aceleracion_pieza_g",
    "aceleracion_sensor_g",
    "aceleracion_prensa_g",
)


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


def repo_feature_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    centered = x - np.nanmean(x)
    scale = np.nanmax(np.abs(centered))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(centered)
    return centered / scale


def infer_fs(t: np.ndarray) -> float:
    if len(t) < 3:
        return float("nan")
    dt = np.diff(t.astype(float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return float("nan")
    return float(1.0 / np.median(dt))


def safe_float(value: object) -> float | str:
    try:
        f = float(value)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return ""


def parse_filename_metadata(path: Path) -> tuple[str, float | str, float | str]:
    stem = path.stem
    m = re.search(r"(corte_\d{8}_\d{6})(?:_(\d+))?(?:_(\d+))?", stem)
    if not m:
        return stem, "", ""
    duplicate_group = m.group(1)
    rpm = safe_float(m.group(2)) if m.group(2) else ""
    paso = safe_float(m.group(3)) if m.group(3) else ""
    return duplicate_group, rpm, paso


def read_table(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=r"\s+", engine="python")


def choose_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    columns = list(df.columns)
    time_col = "tiempo_s" if "tiempo_s" in columns else columns[0]

    force_candidates = [c for c in columns if "fuerza" in c.lower()]
    if not force_candidates:
        raise ValueError("No force column found")
    force_col = force_candidates[0]

    input_col = ""
    for candidate in ACCEL_PRIORITY:
        if candidate in columns:
            input_col = candidate
            break
    if not input_col:
        accel_candidates = [c for c in columns if "aceler" in c.lower()]
        if accel_candidates:
            input_col = accel_candidates[0]
    if not input_col:
        numeric = [c for c in columns if c not in (time_col, force_col)]
        if not numeric:
            raise ValueError("No input/acceleration column found")
        input_col = numeric[0]

    return time_col, force_col, input_col


def discover_sources(root: Path, include_shaker: bool) -> list[tuple[Path, str]]:
    sources: list[tuple[Path, str]] = []
    source_dirs = [
        (root / "matlab_cortes_marzo_2026", "marzo_corte_largo"),
        (root / "datos_txt_ventanas", "ventana_exportada"),
    ]
    for directory, kind in source_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.txt")):
            name = path.name.lower()
            if name.startswith("shaker"):
                if include_shaker:
                    sources.append((path, "referencia_shaker"))
                continue
            if name.startswith("corte") or name.startswith("evento"):
                sources.append((path, kind))
    return sources


def load_rpm_lookup(root: Path) -> dict[str, float]:
    lookup: dict[str, float] = {}
    rpm_file = root / "matlab_cortes_marzo_2026" / "rpm_estimada.csv"
    if not rpm_file.exists():
        return lookup
    df = pd.read_csv(rpm_file)
    if "archivo" not in df.columns or "rpm_estimada" not in df.columns:
        return lookup
    for _, row in df.iterrows():
        stem = Path(str(row["archivo"])).stem
        value = float(row["rpm_estimada"])
        lookup[stem] = value
        m = re.search(r"(corte_\d{8}_\d{6})", stem)
        if m and m.group(1) not in lookup:
            lookup[m.group(1)] = value
    return lookup


def rolling_energy_window(force_norm: np.ndarray, fs: float, seconds: float) -> tuple[int, int]:
    n = len(force_norm)
    if n == 0:
        return 0, 0
    if not np.isfinite(fs) or fs <= 0:
        fs = 2500.0
    window = max(16, int(round(seconds * fs)))
    if n <= window:
        return 0, n
    energy = np.convolve(force_norm ** 2, np.ones(window) / window, mode="valid")
    start = int(np.nanargmax(energy))
    return start, start + window


def loop_area(input_norm: np.ndarray, force_norm: np.ndarray) -> float:
    if len(input_norm) < 3:
        return float("nan")
    dx = np.gradient(input_norm)
    area = abs(float(np.nansum(force_norm * dx)))
    rect = (np.nanmax(input_norm) - np.nanmin(input_norm)) * (np.nanmax(force_norm) - np.nanmin(force_norm))
    if not np.isfinite(rect) or rect <= 1e-12:
        return 0.0
    return float(area / rect)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2)))


def rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def write_txt(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, float_format="%.9g")


def build_case(
    root: Path,
    out_root: Path,
    source: Path,
    source_kind: str,
    rpm_lookup: dict[str, float],
    duplicate_counts: dict[str, int],
    window_seconds: float,
    default_shaker: Path,
) -> CaseRecord:
    df = read_table(source)
    time_col, force_col, input_col = choose_columns(df)

    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    force = pd.to_numeric(df[force_col], errors="coerce").to_numpy(dtype=float)
    input_signal = pd.to_numeric(df[input_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(force) & np.isfinite(input_signal)
    t = t[valid]
    force = force[valid]
    input_signal = input_signal[valid]
    if len(t) < 16:
        raise ValueError("Too few valid samples")

    t = t - t[0]
    fs = infer_fs(t)
    force_norm = normalize_signal(force)
    input_norm = normalize_signal(input_signal)

    duplicate_group, rpm_nominal, paso = parse_filename_metadata(source)
    case_id = re.sub(r"[^A-Za-z0-9_]+", "_", source.stem)
    rpm_estimada = rpm_lookup.get(source.stem, rpm_lookup.get(duplicate_group, ""))

    start, end = rolling_energy_window(force_norm, fs, window_seconds)
    w_t = t[start:end] - t[start]
    w_force = force[start:end]
    w_input = input_signal[start:end]
    w_force_norm = force_norm[start:end]
    w_input_norm = input_norm[start:end]

    features_df = pd.DataFrame(
        {
            "tiempo_s": t,
            "fuerza_original": force,
            "entrada_original": input_signal,
            "fuerza_norm": force_norm,
            "entrada_norm": input_norm,
        }
    )
    ready_df = pd.DataFrame(
        {
            "tiempo_s": t,
            "fuerza_norm": force_norm,
            "entrada_norm": input_norm,
        }
    )
    window_df = pd.DataFrame(
        {
            "tiempo_s": w_t,
            "fuerza_V": w_force,
            "entrada_g": w_input,
            "fuerza_norm": w_force_norm,
            "entrada_norm": w_input_norm,
        }
    )

    features_path = out_root / "features" / f"{case_id}_features.csv"
    ready_path = out_root / "boucwen_ready" / f"{case_id}_boucwen_ready.txt"
    window_path = out_root / "ventanas" / f"{case_id}_ventana_{window_seconds:g}s.txt"
    boucwen_out = out_root / "boucwen_resultados" / case_id

    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(features_path, index=False, float_format="%.9g")
    write_txt(ready_path, ready_df)
    write_txt(window_path, window_df)

    corr = float(np.corrcoef(force_norm, input_norm)[0, 1]) if len(force_norm) > 2 else float("nan")
    if not np.isfinite(corr):
        corr = 0.0

    thesis_role = "caso_de_corte"
    if source_kind == "referencia_shaker":
        thesis_role = "referencia_excitacion_controlada"
    elif duplicate_counts.get(duplicate_group, 0) > 1:
        thesis_role = "caso_con_repeticion_o_duplicado"

    command = (
        "python scripts_modelos/comparar_bouc_wen_shaker_corte.py "
        f"--shaker-file \"{rel(default_shaker, root)}\" "
        f"--corte-file \"{rel(window_path, root)}\" "
        f"--out-dir \"{rel(boucwen_out, root)}\" --no-show"
    )

    return CaseRecord(
        case_id=case_id,
        source_kind=source_kind,
        source_file=source.name,
        source_relpath=rel(source, root),
        duplicate_group=duplicate_group,
        is_duplicate_candidate=duplicate_counts.get(duplicate_group, 0) > 1,
        rpm_nominal_filename=rpm_nominal,
        paso_filename=paso,
        rpm_estimada=safe_float(rpm_estimada),
        fs_hz=float(fs),
        duration_s=float(t[-1] - t[0]) if len(t) else 0.0,
        n_samples=int(len(t)),
        force_col=force_col,
        input_col=input_col,
        force_mean=float(np.nanmean(force)),
        force_std=float(np.nanstd(force)),
        force_rms=rms(force),
        force_peak_abs=float(np.nanmax(np.abs(force))),
        input_mean=float(np.nanmean(input_signal)),
        input_std=float(np.nanstd(input_signal)),
        input_rms=rms(input_signal),
        input_peak_abs=float(np.nanmax(np.abs(input_signal))),
        corr_force_input=corr,
        loop_area_norm=loop_area(input_norm, force_norm),
        window_start_s=float(t[start]),
        window_end_s=float(t[end - 1]) if end > start else float(t[start]),
        boucwen_ready_relpath=rel(ready_path, root),
        window_relpath=rel(window_path, root),
        features_relpath=rel(features_path, root),
        suggested_boucwen_out_relpath=rel(boucwen_out, root),
        suggested_boucwen_command=command,
        thesis_role=thesis_role,
    )


def write_summary(out_root: Path, records: list[CaseRecord]) -> None:
    by_kind = pd.Series([r.source_kind for r in records]).value_counts().to_dict() if records else {}
    duplicates = sorted({r.duplicate_group for r in records if r.is_duplicate_candidate})
    top_area = sorted(records, key=lambda r: r.loop_area_norm, reverse=True)[:8]

    lines = [
        "# Resumen de casos historicos",
        "",
        "Base preparada para curvas Bouc-Wen y razonamiento basado en casos.",
        "",
        "## Conteo por fuente",
    ]
    for kind, count in by_kind.items():
        lines.append(f"- {kind}: {count}")
    lines.extend(["", "## Grupos con repeticion/duplicado"])
    if duplicates:
        lines.extend([f"- {d}" for d in duplicates])
    else:
        lines.append("- Sin duplicados detectados por nombre base.")
    lines.extend(["", "## Casos con mayor area normalizada de lazo"])
    for r in top_area:
        lines.append(
            f"- {r.case_id}: area={r.loop_area_norm:.4f}, rpm_estimada={r.rpm_estimada}, ventana={r.window_start_s:.3f}-{r.window_end_s:.3f}s"
        )
    lines.extend(
        [
            "",
            "## Uso doctoral",
            "Estos casos forman la memoria experimental para comparar condiciones actuales contra historicos,",
            "ajustar curvas Bouc-Wen/LuGre y alimentar posteriormente una CNN ligera exportable a Hailo-8L.",
        ]
    )
    (out_root / "resumen_casos.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=repo_feature_root())
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--window-seconds", type=float, default=0.5)
    parser.add_argument("--include-shaker", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    out_root = (args.out_dir or root / "razonador_casos" / "salidas" / "casos_dataset").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(root, include_shaker=args.include_shaker)
    rpm_lookup = load_rpm_lookup(root)
    groups = [parse_filename_metadata(path)[0] for path, _ in sources]
    duplicate_counts = {g: groups.count(g) for g in set(groups)}
    default_shaker = root / "datos_txt_ventanas" / "shaker_mejor_0.5s_20Hz.txt"

    records: list[CaseRecord] = []
    errors: list[dict[str, str]] = []
    for source, kind in sources:
        try:
            record = build_case(
                root=root,
                out_root=out_root,
                source=source,
                source_kind=kind,
                rpm_lookup=rpm_lookup,
                duplicate_counts=duplicate_counts,
                window_seconds=args.window_seconds,
                default_shaker=default_shaker,
            )
            records.append(record)
            print(f"OK {record.case_id}")
        except Exception as exc:
            errors.append({"file": str(source), "error": str(exc)})
            print(f"ERROR {source.name}: {exc}")

    records_df = pd.DataFrame([asdict(r) for r in records])
    records_df.to_csv(out_root / "casos_historicos.csv", index=False)
    with (out_root / "casos_historicos.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    pd.DataFrame(errors).to_csv(out_root / "errores_conversion.csv", index=False)
    write_summary(out_root, records)

    print("\nResumen")
    print(f"  Casos convertidos: {len(records)}")
    print(f"  Errores: {len(errors)}")
    print(f"  Salida: {out_root}")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())

