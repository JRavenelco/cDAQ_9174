#!/usr/bin/env python3
"""
Puente UDP-session → razonador_casos.

Lee un CSV de sesion del receiver (cdaq_remote_TIMESTAMP.csv), su JSONL LLM
(_cases.jsonl) y el sidecar de metadatos (_meta.json), extrae el mejor
segmento de 0.5s por corte, lo etiqueta con cutter_condition y lo indexa en
casos_historicos.jsonl.

Sidecar _meta.json esperado (generado por el sender en Windows):
  {
    "cutter_condition": "Nuevo" | "Medio uso" | "Desgastado",
    "rpm_husillo": 600,
    "rpm_avance_x": 40,
    "fs_hz": 2500,
    "timestamp": "2026-05-07T10:00:00",
    "notas": "..."
  }

Modos de uso
------------
  python3 udp_sesion_a_casos.py logs/cdaq_remote_TIMESTAMP.csv
  python3 udp_sesion_a_casos.py --watch --update-npz
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ── Rutas ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_CARACT = _HERE.parent / "caracterizacion_fuerza"
_RAZONADOR = _CARACT / "razonador_casos"
_CASOS_JSONL = _RAZONADOR / "salidas" / "casos_dataset" / "casos_historicos.jsonl"
_EXPORTAR = _RAZONADOR / "exportar_segmentos_hailo.py"
_PARTIR = _RAZONADOR / "partir_splits_hailo.py"
_LOGS_DIR = _HERE / "logs"

# ── Constantes ────────────────────────────────────────────────────────────────
MIN_SAMPLES = 500
MIN_LOOP_AREA = 0.02
WINDOW_SECONDS = 0.5        # duracion del mejor segmento a extraer
IDLE_LABELS = {"idle", "test_idle", "reposo", "sin_corte", ""}
LLM_CUTTING_KEY = "cutting_state"

CUTTER_CLASS_MAP: dict[str, int] = {
    "nuevo": 0,
    "medio uso": 1,
    "medio_uso": 1,
    "mediouso": 1,
    "desgastado": 2,
    "worn": 2,
    "new": 0,
    "used": 1,
}


def cutter_label_to_class(label: str) -> int:
    """Convierte 'Nuevo'/'Medio uso'/'Desgastado' a 0/1/2, o -1 si no reconoce."""
    return CUTTER_CLASS_MAP.get(label.lower().strip(), -1)


# ── Parseo del CSV del receiver ───────────────────────────────────────────────

def parse_receiver_csv(path: Path) -> tuple[list[dict[str, str]], dict[int, dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    conditions: dict[int, dict[str, Any]] = {}
    current_cond: dict[str, Any] = {}

    with path.open("r", encoding="utf-8", newline="") as fh:
        header: list[str] = []
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.startswith("#"):
                _parse_condition_comment(line, current_cond)
                if "cut_id" in current_cond:
                    conditions[int(float(current_cond["cut_id"]))] = dict(current_cond)
                continue
            if not header:
                header = [c.strip() for c in line.split(",")]
                continue
            if header and line.strip():
                parts = line.split(",")
                if len(parts) >= len(header):
                    rows.append({header[i]: parts[i].strip() for i in range(len(header))})

    return rows, conditions


def _parse_condition_comment(line: str, cond: dict[str, Any]) -> None:
    body = line.lstrip("# ").strip()
    for token in body.replace(",", " ").split():
        if "=" in token:
            k, _, v = token.partition("=")
            k, v = k.strip(), v.strip().rstrip("²")
            try:
                cond[k] = float(v)
            except ValueError:
                cond[k] = v


# ── Sidecar _meta.json ────────────────────────────────────────────────────────

def load_meta_json(csv_path: Path) -> dict[str, Any]:
    """
    Busca CSVNAME_meta.json (o CSVNAME.json) junto al CSV.
    Devuelve dict vacio si no existe.
    """
    candidates = [
        csv_path.with_name(csv_path.stem + "_meta.json"),
        csv_path.with_suffix(".json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass
    return {}


# ── Parseo del JSONL LLM ──────────────────────────────────────────────────────

def load_cases_jsonl(path: Path) -> dict[int, list[dict[str, Any]]]:
    by_seq: dict[int, list[dict[str, Any]]] = {}
    if not path.exists():
        return by_seq
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                seq = int(obj.get("seq", -1))
                by_seq.setdefault(seq, []).append(obj)
            except (json.JSONDecodeError, ValueError):
                continue
    return by_seq


# ── Agrupación por cut_id ─────────────────────────────────────────────────────

def group_by_cut(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        try:
            cid = int(row.get("cut_id", 0))
        except (ValueError, TypeError):
            cid = 0
        groups.setdefault(cid, []).append(row)
    return groups


def is_active_cut(
    cut_id: int,
    conditions: dict[int, dict[str, Any]],
    llm_cases: dict[int, list[dict[str, Any]]],
) -> bool:
    cond = conditions.get(cut_id, {})
    label = str(cond.get("label", "")).strip().lower()
    if label not in IDLE_LABELS:
        return True
    for case_list in llm_cases.values():
        for case in case_list:
            if int(case.get("meta", {}).get("cut_id", -1)) == cut_id:
                if (case.get("llm") or {}).get(LLM_CUTTING_KEY, False):
                    return True
    return False


# ── Señal desde filas CSV ─────────────────────────────────────────────────────

def rows_to_signal(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruye tiempo_s desde t_sender_s (timestamp de bloque) + sample_idx.
    Devuelve (tiempo_s, force_v, accel_g) como float32 ordenados por tiempo.
    """
    ts_blk, sidx, f_raw, a_raw = [], [], [], []
    for row in rows:
        try:
            tb = float(row.get("t_sender_s") or row.get("t_local_s") or "nan")
            si = int(row.get("sample_idx", 0))
            fv = float(row.get("force_v", "nan"))
            ag = float(row.get("accel_g", "nan"))
        except (ValueError, TypeError):
            continue
        if not (math.isfinite(tb) and math.isfinite(fv) and math.isfinite(ag)):
            continue
        ts_blk.append(tb)
        sidx.append(si)
        f_raw.append(fv)
        a_raw.append(ag)

    if not ts_blk:
        return np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)

    tb_arr = np.asarray(ts_blk, np.float64)
    si_arr = np.asarray(sidx, np.int32)

    unique_tb = np.unique(tb_arr)
    if unique_tb.size >= 2:
        blk_dt = float(np.median(np.diff(unique_tb)))
        spr = int(si_arr.max()) + 1
        fs_est = spr / blk_dt if blk_dt > 0 else 2500.0
    else:
        fs_est = 2500.0

    t0 = float(tb_arr[0])
    t = (tb_arr - t0 + si_arr / fs_est).astype(np.float32)

    # Ordenar por tiempo reconstruido
    order = np.argsort(t)
    return t[order], np.asarray(f_raw, np.float32)[order], np.asarray(a_raw, np.float32)[order]


def infer_fs(t: np.ndarray) -> float:
    if t.size < 3:
        return 2500.0
    dt = np.diff(t.astype(np.float64))
    dt = dt[(dt > 0) & np.isfinite(dt)]
    return float(1.0 / np.median(dt)) if dt.size else 2500.0


# ── Extracción del mejor segmento de 0.5s ────────────────────────────────────

def extraer_mejor_segmento_0_5s(
    t: np.ndarray,
    force: np.ndarray,
    accel: np.ndarray,
    fs: float,
    seconds: float = WINDOW_SECONDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve la ventana de `seconds` segundos con mayor energia en fuerza
    (señal centrada y normalizada por RMS).  Criterio: energia media de la
    fuerza centrada al cuadrado por ventana deslizante.
    """
    n = len(force)
    win = max(16, int(round(seconds * fs)))
    if n <= win:
        t0 = t - t[0]
        return t0, force.copy(), accel.copy()

    force_c = force - float(np.mean(force))
    kernel = np.ones(win, dtype=np.float64) / win
    energy = np.convolve(force_c.astype(np.float64) ** 2, kernel, mode="valid")
    start = int(np.argmax(energy))
    end = start + win

    t_win = (t[start:end] - t[start]).astype(np.float32)
    return t_win, force[start:end], accel[start:end]


# ── Métricas ─────────────────────────────────────────────────────────────────

def loop_area_norm(force: np.ndarray, accel: np.ndarray) -> float:
    if force.size < 4:
        return 0.0
    f = force.astype(np.float64) - force.mean()
    a = accel.astype(np.float64) - accel.mean()
    _trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = float(abs(_trap(f, a)))
    bbox = (f.max() - f.min()) * (a.max() - a.min())
    return float(np.clip(area / bbox, 0.0, 1.0)) if bbox > 1e-15 else 0.0


# ── TXT compatible con exportar_segmentos_hailo.py ────────────────────────────

def write_segment_txt(path: Path, t: np.ndarray, force: np.ndarray, accel: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("tiempo_s\tfuerza_V\taceleracion_sensor_g\n")
        for i in range(len(t)):
            fh.write(f"{float(t[i]):.9g}\t{float(force[i]):.9g}\t{float(accel[i]):.9g}\n")


# ── casos_historicos.jsonl ────────────────────────────────────────────────────

def load_existing_case_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    ids.add(str(json.loads(line).get("case_id", "")))
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def append_case(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Procesar sesion ───────────────────────────────────────────────────────────

def process_session(
    csv_path: Path,
    update_npz: bool = False,
    dry_run: bool = False,
    min_loop_area: float = MIN_LOOP_AREA,
) -> int:
    cases_path = Path(str(csv_path).replace(".csv", "_cases.jsonl"))
    meta = load_meta_json(csv_path)

    print(f"\n=== Sesion: {csv_path.name} ===")
    if meta:
        print(f"  Meta: cutter={meta.get('cutter_condition','?')}  "
              f"rpm_husillo={meta.get('rpm_husillo','?')}  "
              f"rpm_avance={meta.get('rpm_avance_x','?')}")

    rows, conditions = parse_receiver_csv(csv_path)
    llm_cases = load_cases_jsonl(cases_path)
    groups = group_by_cut(rows)

    existing_ids = load_existing_case_ids(_CASOS_JSONL)
    timestamp = csv_path.stem.replace("cdaq_remote_", "")
    added = 0

    for cut_id, cut_rows in sorted(groups.items()):
        cond = conditions.get(cut_id, {})
        label = str(cond.get("label", "")).strip()

        if not is_active_cut(cut_id, conditions, llm_cases):
            print(f"  SKIP cut_id={cut_id} label={label!r} (idle)")
            continue

        t_full, force_full, accel_full = rows_to_signal(cut_rows)
        if t_full.size < MIN_SAMPLES:
            print(f"  SKIP cut_id={cut_id} solo {t_full.size} muestras (min={MIN_SAMPLES})")
            continue

        fs = infer_fs(t_full)

        # Extraer mejor ventana de 0.5s
        t_win, force_win, accel_win = extraer_mejor_segmento_0_5s(t_full, force_full, accel_full, fs)

        la = loop_area_norm(force_win, accel_win)
        if la < min_loop_area:
            print(f"  SKIP cut_id={cut_id} loop_area={la:.4f} < {min_loop_area} (probable idle)")
            continue

        case_id = f"udp_{timestamp}_cut{cut_id}"
        if case_id in existing_ids:
            print(f"  SKIP {case_id} ya existe en historicos")
            continue

        # Prioridad: _meta.json > condiciones del receiver > fallback
        cutter_condition = str(meta.get("cutter_condition", "")).strip()
        rpm = float(meta.get("rpm_husillo", 0.0) or cond.get("rpm_spindle", 0.0) or 0.0)
        rpm_avance = float(meta.get("rpm_avance_x", 0.0) or cond.get("rpm_feed", 0.0) or 0.0)
        cutter_class = cutter_label_to_class(cutter_condition)

        if dry_run:
            cls_str = f"clase={cutter_class}" if cutter_condition else "sin_etiqueta"
            print(f"  [DRY] {case_id}  N_win={t_win.size}  fs={fs:.0f}  "
                  f"rpm={rpm}  loop_area={la:.4f}  cutter={cutter_condition!r}  {cls_str}")
            added += 1
            continue

        txt_name = f"{case_id}.txt"
        txt_path = _CARACT / txt_name
        write_segment_txt(txt_path, t_win, force_win, accel_win)

        cls_str = f"clase={cutter_class}" if cutter_condition else "sin_etiqueta"
        print(f"  OK  {case_id}  N={t_win.size}  fs={fs:.0f}Hz  "
              f"rpm={rpm}  loop_area={la:.4f}  cutter={cutter_condition!r}  {cls_str}")
        print(f"      TXT -> {txt_path}")

        record: dict[str, Any] = {
            "case_id": case_id,
            "source_file": txt_name,
            "source_relpath": txt_name,
            "window_relpath": "",
            "source_kind": "udp_live",
            "window_start_s": float(t_win[0]),
            "window_end_s": float(t_win[-1]),
            "fs_hz": fs,
            "rpm_estimada": rpm if rpm > 0 else "",
            "rpm_avance": rpm_avance if rpm_avance > 0 else "",
            "rpm_nominal_filename": "",
            "paso_filename": "",
            "loop_area_norm": la,
            "thesis_role": "caso_de_corte",
            "cutter_condition": cutter_condition,
            "cutter_class": cutter_class,
            "ap_mm": float(meta.get("ap_mm", cond.get("ap_mm", 0.0)) or 0.0),
            "tool_diam_mm": float(cond.get("tool_diam_mm", 25.4)),
            "n_flutes": int(cond.get("n_flutes", 2)),
            "cut_label": label,
            "notas": str(meta.get("notas", "")),
            "udp_session": csv_path.name,
            "cut_id_in_session": cut_id,
        }
        append_case(_CASOS_JSONL, record)
        added += 1

    print(f"  Casos nuevos: {added}")
    if added > 0 and update_npz and not dry_run:
        _run_update_pipeline()

    return added


def _run_update_pipeline() -> None:
    print("\n--- Actualizando NPZ de entrenamiento ---")
    for script in [_EXPORTAR, _PARTIR]:
        if not script.exists():
            print(f"  [WARN] No encontrado: {script}")
            continue
        print(f"  Ejecutando {script.name} ...")
        result = subprocess.run([sys.executable, str(script)], text=True)
        if result.returncode != 0:
            print(f"  [ERROR] {script.name} retorno {result.returncode}")


# ── Modo watcher ──────────────────────────────────────────────────────────────

def watch_logs(update_npz: bool, dry_run: bool, poll_s: float = 5.0) -> None:
    print(f"Vigilando {_LOGS_DIR} (cada {poll_s:.0f}s) — Ctrl+C para detener")
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    processed: set[str] = set()

    def _raw_csvs() -> list[Path]:
        return sorted(
            p for p in _LOGS_DIR.glob("cdaq_remote_*.csv")
            if "_inference" not in p.name
        )

    for p in _raw_csvs():
        processed.add(p.name)
    print(f"  {len(processed)} sesiones previas ignoradas")

    try:
        while True:
            for csv_path in _raw_csvs():
                if csv_path.name in processed:
                    continue
                try:
                    age_s = time.time() - csv_path.stat().st_mtime
                except OSError:
                    continue
                if age_s < 10.0:
                    continue
                processed.add(csv_path.name)
                try:
                    process_session(csv_path, update_npz=update_npz, dry_run=dry_run)
                except Exception as exc:
                    print(f"  [ERROR] {csv_path.name}: {exc}")
            time.sleep(poll_s)
    except KeyboardInterrupt:
        print("\nWatcher detenido.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Puente UDP-sesion → casos_historicos.jsonl")
    ap.add_argument("csv", nargs="?", type=Path,
                    help="CSV de sesion (omitir para --watch)")
    ap.add_argument("--watch", action="store_true",
                    help="Vigilar logs/ y procesar sesiones nuevas")
    ap.add_argument("--update-npz", action="store_true",
                    help="Regenerar NPZ tras cada sesion procesada")
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--min-loop-area", type=float, default=MIN_LOOP_AREA)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.watch:
        watch_logs(update_npz=args.update_npz, dry_run=args.dry_run, poll_s=args.poll_s)
        return 0

    if args.csv is None:
        ap.print_help()
        return 1

    csv_path = args.csv.resolve()
    if "_inference" in csv_path.name:
        print("ERROR: pasa el CSV raw, no el _inference.csv")
        return 1
    if not csv_path.exists():
        print(f"ERROR: {csv_path}")
        return 1

    process_session(csv_path, update_npz=args.update_npz, dry_run=args.dry_run,
                    min_loop_area=args.min_loop_area)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
