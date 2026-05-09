#!/usr/bin/env python3
"""
Acumula casos aprobados por el experto VLM en un NPZ de entrenamiento.

Lee todos los JSON en casos_experto/ (o el directorio indicado), extrae el
mejor segmento de 0.5 s (máxima energía de fuerza), mapea cutter_condition
a clase numérica (0=Nuevo, 1=Medio uso, 2=Desgastado) y apila en un NPZ:

  X_raw[N, T, 2]     — canales [fuerza_V, accel_g]
  y_condition[N]     — clase int32  (-1 = sin etiqueta)

Salida por defecto: casos_experto/dataset_hailo/hysteresis_dataset.npz

Uso:
  python3 acumular_dataset_hailo.py
  python3 acumular_dataset_hailo.py --casos-dir /ruta/casos_experto
  python3 acumular_dataset_hailo.py --out dataset_hailo/hysteresis_dataset.npz
  python3 acumular_dataset_hailo.py --window-s 0.5 --min-loop-area 0.02
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

CUTTER_CLASS_MAP: dict[str, int] = {
    "nuevo": 0,
    "new": 0,
    "0": 0,
    "medio uso": 1,
    "medio_uso": 1,
    "mediouso": 1,
    "used": 1,
    "semi": 1,
    "1": 1,
    "desgastado": 2,
    "worn": 2,
    "wear": 2,
    "2": 2,
}
CUTTER_CLASS_NAMES = ["Nuevo", "Medio uso", "Desgastado"]
DEFAULT_WINDOW_S = 0.5
DEFAULT_MIN_LOOP_AREA = 0.02


def _here() -> Path:
    return Path(__file__).resolve().parent


def default_casos_dir() -> Path:
    return _here() / "casos_experto"


def default_out_npz(casos_dir: Path) -> Path:
    return casos_dir / "dataset_hailo" / "hysteresis_dataset.npz"


def cutter_label_to_class(label: str) -> int:
    return CUTTER_CLASS_MAP.get(label.lower().strip(), -1)


def extraer_mejor_segmento(
    t: np.ndarray,
    force: np.ndarray,
    accel: np.ndarray,
    fs_hz: float,
    seconds: float = DEFAULT_WINDOW_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    win = max(16, int(round(seconds * fs_hz)))
    force_c = (force - float(np.mean(force))).astype(np.float32)
    if len(force_c) <= win:
        pad = win - len(force_c)
        force_c = np.pad(force_c, (0, pad))
        force = np.pad(force.astype(np.float32), (0, pad))
        accel = np.pad(accel.astype(np.float32), (0, pad))
        t = np.pad(t.astype(np.float32), (0, pad))
        return t[:win] - t[0], force[:win], accel[:win]
    kernel = np.ones(win, dtype=np.float32) / win
    energy = np.convolve(force_c**2, kernel, mode="valid")
    start = int(np.argmax(energy))
    t_seg = t[start : start + win].astype(np.float32)
    return t_seg - t_seg[0], force[start : start + win], accel[start : start + win]


def load_caso(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        print(f"  SKIP {path.name}: {e}")
        return None


def process_caso(
    data: dict[str, Any],
    window_s: float,
    min_loop_area: float,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    meta = data.get("metadata") or data.get("meta") or {}
    datos = data.get("datos") or {}

    loop_area = float(data.get("loop_area_norm") or 0.0)
    if loop_area < min_loop_area:
        return None, None, -1

    cutter_cond = str(
        meta.get("cutter_condition")
        or meta.get("cut_label")
        or data.get("cutter_condition")
        or ""
    )
    y_class = cutter_label_to_class(cutter_cond)

    t_raw = datos.get("t") or []
    force_raw = datos.get("fuerza_V") or datos.get("force") or []
    accel_raw = datos.get("accel_g") or datos.get("accel") or []

    if not force_raw or not accel_raw:
        return None, None, -1

    t = np.asarray(t_raw, dtype=np.float32)
    force = np.asarray(force_raw, dtype=np.float32)
    accel = np.asarray(accel_raw, dtype=np.float32)

    min_len = min(len(t), len(force), len(accel))
    if min_len < 16:
        return None, None, -1
    t, force, accel = t[:min_len], force[:min_len], accel[:min_len]

    fs_hz = float(meta.get("fs_hz") or datos.get("fs_hz") or 2500.0)
    if fs_hz <= 1.0:
        fs_hz = 2500.0

    _, f_seg, a_seg = extraer_mejor_segmento(t, force, accel, fs_hz, window_s)
    X = np.stack([f_seg, a_seg], axis=-1).astype(np.float32)
    return X, cutter_cond, y_class


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Acumula casos_experto/*.json en un NPZ de entrenamiento Hailo"
    )
    ap.add_argument(
        "--casos-dir",
        type=Path,
        default=default_casos_dir(),
        help="Directorio con los JSON aprobados por el experto VLM",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Ruta NPZ de salida (por defecto: <casos-dir>/dataset_hailo/hysteresis_dataset.npz)",
    )
    ap.add_argument(
        "--window-s",
        type=float,
        default=DEFAULT_WINDOW_S,
        help="Duración en segundos del segmento extraído (default: 0.5)",
    )
    ap.add_argument(
        "--min-loop-area",
        type=float,
        default=DEFAULT_MIN_LOOP_AREA,
        help="Área mínima de lazo de histéresis (normalizada) para incluir caso",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar detalles de cada caso procesado",
    )
    args = ap.parse_args()

    casos_dir = args.casos_dir.resolve()
    if not casos_dir.exists():
        raise SystemExit(f"Directorio no encontrado: {casos_dir}")

    out_path = args.out.resolve() if args.out else default_out_npz(casos_dir)

    json_files = sorted(casos_dir.glob("**/*.json"))
    if not json_files:
        raise SystemExit(f"No se encontraron JSON en {casos_dir}")

    print(f"Escaneando {len(json_files)} JSON en {casos_dir}")

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    cutter_list: list[str] = []
    skipped = 0

    for path in json_files:
        data = load_caso(path)
        if data is None:
            skipped += 1
            continue
        X, cutter_cond, y_class = process_caso(data, args.window_s, args.min_loop_area)
        if X is None:
            skipped += 1
            if args.verbose:
                print(f"  SKIP {path.name} (area baja o datos insuficientes)")
            continue
        X_list.append(X)
        y_list.append(y_class)
        cutter_list.append(str(cutter_cond))
        if args.verbose:
            cls_name = CUTTER_CLASS_NAMES[y_class] if 0 <= y_class <= 2 else "?"
            print(f"  OK  {path.name}  cutter={cutter_cond!r}  clase={cls_name}")

    total = len(X_list)
    if total == 0:
        raise SystemExit("Sin muestras válidas. Verifica que los JSON tengan datos y loop_area_norm >= min-loop-area.")

    # Pad to common window length
    win_len = max(x.shape[0] for x in X_list)
    X_padded = np.zeros((total, win_len, 2), dtype=np.float32)
    for i, x in enumerate(X_list):
        n = min(x.shape[0], win_len)
        X_padded[i, :n, :] = x[:n]

    y_arr = np.asarray(y_list, dtype=np.int32)
    cutter_arr = np.asarray(cutter_list, dtype=object)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X_raw=X_padded, y_condition=y_arr, cutter_conditions=cutter_arr)

    # Summary
    labeled = Counter(int(v) for v in y_arr if v >= 0)
    unlabeled = int(np.sum(y_arr < 0))
    print(f"\nDataset: {total} casos  ({skipped} descartados)")
    for cls_idx, name in enumerate(CUTTER_CLASS_NAMES):
        n = labeled.get(cls_idx, 0)
        print(f"  {name:12s}: {n}")
    if unlabeled:
        print(f"  Sin etiqueta : {unlabeled}")
    print(f"  Forma X      : {X_padded.shape}")
    print(f"\nGuardado en: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
