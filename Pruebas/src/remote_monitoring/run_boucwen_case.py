#!/usr/bin/env python3
"""Run Bouc-Wen model on a boucwen_ready file and produce metrics + output.

This is the lightweight edge-friendly runner for the Bouc-Wen pipeline.
It takes a pre-normalised boucwen_ready TSV (columns: tiempo_s, fuerza_norm,
entrada_norm) and runs the identified Bouc-Wen model forward, producing:
  - R², RMSE, loop area, correlation, hysteresis energy %
  - Output TSV with model prediction columns
  - Summary JSONL record for downstream LLM/VLM consumption

Dependencies: numpy only (no scipy, pandas, matplotlib, torch).
Designed to run on Jetson Orin NX without heavy packages.

Identified physical parameters (sesion 135910, KAN-PINN):
  mass = 0.107 kg, stiffness = 1424 N/m, damping = 3.73 Ns/m
  alpha = 0.811, A_bw = 1.0, B_bw = 0.5, C_bw = 0.5, n_bw = 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════
# Physical parameters (defaults from cdaq_inference_engine.py)
# ═══════════════════════════════════════════════════════════════

# --- Classic Bouc-Wen (for normalised signals [-1,1]) ---
# These are reasonable starting defaults; the optimizer in
# comparar_bouc_wen_shaker_corte.py refines them per dataset.
DEFAULT_A_BW = 1.0
DEFAULT_B_BW = 0.5
DEFAULT_C_BW = 0.5
DEFAULT_N_BW = 1.0
DEFAULT_K_BW = 1.0          # rigidez (normalised scale)
DEFAULT_ALPHA = 0.811       # ratio lineal

# --- KAN-PINN (for physical units) ---
DEFAULT_MASS = 0.107        # kg
DEFAULT_STIFFNESS = 1424.0  # N/m
DEFAULT_DAMPING = 3.73      # Ns/m

DEFAULT_FS = 2500.0         # Hz


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_boucwen_ready(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load a boucwen_ready TSV file.

    Returns (t, force_norm, entrada_norm, fs_estimated).
    """
    t_list: list[float] = []
    force_list: list[float] = []
    entrada_list: list[float] = []

    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            t_list.append(float(row["tiempo_s"]))
            force_list.append(float(row["fuerza_norm"]))
            entrada_list.append(float(row["entrada_norm"]))

    t = np.asarray(t_list, dtype=np.float64)
    force = np.asarray(force_list, dtype=np.float64)
    entrada = np.asarray(entrada_list, dtype=np.float64)

    # Estimate fs from time vector
    if len(t) > 1:
        dt_median = float(np.median(np.diff(t)))
        fs = 1.0 / dt_median if dt_median > 0 else DEFAULT_FS
    else:
        fs = DEFAULT_FS

    return t, force, entrada, fs


# ═══════════════════════════════════════════════════════════════
# Bouc-Wen forward model (RK4, same as cdaq_inference_engine.py)
# ═══════════════════════════════════════════════════════════════

def integrate_trapezoid(signal: np.ndarray, dt: float) -> np.ndarray:
    """Cumulative trapezoidal integration."""
    n = len(signal)
    out = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        out[i] = out[i - 1] + 0.5 * (signal[i - 1] + signal[i]) * dt
    return out


def highpass_simple(signal: np.ndarray, fc: float, fs: float) -> np.ndarray:
    """Very simple first-order IIR high-pass filter (no scipy needed)."""
    if fs <= 0 or fc <= 0 or fc >= fs / 2:
        return signal
    rc = 1.0 / (2.0 * math.pi * fc)
    dt = 1.0 / fs
    alpha = rc / (rc + dt)
    out = np.zeros_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = alpha * (out[i - 1] + signal[i] - signal[i - 1])
    return out


def run_bouc_wen_clasico(
    u_norm: np.ndarray,
    fs: float,
    A_bw: float = DEFAULT_A_BW,
    B_bw: float = DEFAULT_B_BW,
    C_bw: float = DEFAULT_C_BW,
    n_bw: float = DEFAULT_N_BW,
    k_bw: float = DEFAULT_K_BW,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classic Bouc-Wen on normalised signals (same as comparar_bouc_wen_shaker_corte.py).

    Equations:
        dz = A*du - B*|du|*z - C*du*|z|^n
        F  = alpha*k*u + (1-alpha)*k*z

    For LINEAR system: alpha -> 1 => F = k*u
    For HYSTERESIS:    alpha < 1 => F depends on z (memory)

    Returns (F_model, z_arr, du).
    """
    dt = 1.0 / fs
    n = len(u_norm)

    # du/dt
    du = np.zeros(n, dtype=np.float64)
    du[1:] = (u_norm[1:] - u_norm[:-1]) / dt

    # Solve ODE for z (hysteresis variable)
    z_arr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        du_i = du[i]
        z_prev = z_arr[i - 1]
        dz = A_bw * du_i - B_bw * abs(du_i) * z_prev - C_bw * du_i * (abs(z_prev) ** n_bw)
        z_arr[i] = z_prev + dz * dt
        # Clamp for stability
        z_arr[i] = max(-5.0, min(5.0, z_arr[i]))

    # Force: F = alpha*k*u + (1-alpha)*k*z
    F_model = alpha * k_bw * u_norm + (1.0 - alpha) * k_bw * z_arr

    return F_model, z_arr, du


def run_bouc_wen_kanpinn(
    entrada_norm: np.ndarray,
    fs: float,
    mass: float = DEFAULT_MASS,
    stiffness: float = DEFAULT_STIFFNESS,
    damping: float = DEFAULT_DAMPING,
    alpha: float = DEFAULT_ALPHA,
    A_bw: float = DEFAULT_A_BW,
    B_bw: float = DEFAULT_B_BW,
    C_bw: float = DEFAULT_C_BW,
    n_bw: float = DEFAULT_N_BW,
    drift_fc: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """KAN-PINN Bouc-Wen on physical-unit signals.

    F = m*a + k*x + c*v + alpha*k*E + (1-alpha)*k*z

    Returns (F_model_norm, z_arr, vel, disp, envelope).
    """
    dt = 1.0 / fs
    n = len(entrada_norm)
    accel = entrada_norm.copy()

    vel_raw = integrate_trapezoid(accel, dt)
    vel = highpass_simple(vel_raw, drift_fc, fs)
    disp_raw = integrate_trapezoid(vel, dt)
    disp = highpass_simple(disp_raw, drift_fc, fs)
    envelope = np.abs(accel)

    z_arr = np.zeros(n, dtype=np.float64)
    z = 0.0
    for i in range(n):
        v = float(vel[i])

        def f_dz(z_in: float, _v=v) -> float:
            abs_z = abs(z_in) + 1e-8
            sign_vz = 1.0 if _v * z_in >= 0 else -1.0
            return _v * (A_bw - (abs_z ** n_bw) * (B_bw * sign_vz + C_bw))

        k1 = f_dz(z)
        k2 = f_dz(z + 0.5 * dt * k1)
        k3 = f_dz(z + 0.5 * dt * k2)
        k4 = f_dz(z + dt * k3)
        z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        z = max(-1.0, min(1.0, z))
        z_arr[i] = z

    F_model_raw = (mass * accel
                   + stiffness * disp
                   + damping * vel
                   + alpha * stiffness * envelope
                   + (1.0 - alpha) * stiffness * z_arr)

    bw_mean = float(np.mean(F_model_raw))
    bw_std = float(np.std(F_model_raw)) + 1e-10
    F_model_norm = (F_model_raw - bw_mean) / bw_std
    F_model_norm = np.clip(F_model_norm, -3.0, 3.0)

    return F_model_norm, z_arr, vel, disp, envelope


# ═══════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════

@dataclass
class BoucWenMetrics:
    """Bouc-Wen model evaluation metrics."""
    case_id: str
    input_file: str
    model_mode: str
    n_samples: int
    fs_hz: float
    duration_s: float

    # Bouc-Wen parameters used
    alpha_bw: float
    k_bw: float
    A_bw: float
    B_bw: float
    C_bw: float
    n_bw: float
    mass: float
    stiffness: float
    damping: float

    # Fit metrics
    rmse: float
    r_squared: float
    correlation: float
    mae: float

    # Hysteresis metrics
    loop_area_shoelace: float
    loop_area_model: float
    loop_area_error_pct: float
    z_mean: float
    z_abs_max: float
    z_last: float
    hyst_energy_pct: float

    # Linearity assessment
    linearity_label: str


def compute_metrics(
    case_id: str,
    input_file: str,
    t: np.ndarray,
    force_norm: np.ndarray,
    entrada_norm: np.ndarray,
    F_model_norm: np.ndarray,
    z_arr: np.ndarray,
    fs: float,
    params: dict,
) -> BoucWenMetrics:
    """Compute all Bouc-Wen evaluation metrics."""
    n = len(t)
    duration = float(t[-1] - t[0]) if n > 1 else 0.0

    # Basic fit metrics
    residuals = force_norm - F_model_norm
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((force_norm - np.mean(force_norm)) ** 2)) + 1e-10
    r2 = max(-1.0, min(1.0, 1.0 - ss_res / ss_tot))
    mae = float(np.mean(np.abs(residuals)))

    # Correlation
    if n > 2:
        corr_matrix = np.corrcoef(force_norm, F_model_norm)
        corr = float(corr_matrix[0, 1])
        if not math.isfinite(corr):
            corr = 0.0
    else:
        corr = 0.0

    # Loop area (shoelace) for real data
    loop_area_real = _shoelace_area(entrada_norm, force_norm)

    # Loop area for model
    loop_area_model = _shoelace_area(entrada_norm, F_model_norm)

    # Loop area error
    if loop_area_real > 1e-8:
        loop_area_error = abs(loop_area_model - loop_area_real) / loop_area_real * 100.0
    else:
        loop_area_error = 0.0

    # z statistics
    z_mean = float(np.mean(z_arr))
    z_abs_max = float(np.max(np.abs(z_arr)))
    z_last = float(z_arr[-1]) if n > 0 else 0.0

    # Hysteresis energy percentage
    total_energy = float(np.sum(np.abs(force_norm))) + 1e-10
    hyst_energy_pct = min(100.0, float(loop_area_model / total_energy * 100.0))

    # Linearity label
    alpha = params.get("alpha", DEFAULT_ALPHA)
    if alpha > 0.95:
        linearity = "LINEAL"
    elif alpha > 0.80:
        linearity = "CUASI-LINEAL"
    else:
        linearity = "HISTERESIS"

    return BoucWenMetrics(
        case_id=case_id,
        input_file=str(input_file),
        model_mode=params.get("mode", "clasico"),
        n_samples=n,
        fs_hz=fs,
        duration_s=duration,
        alpha_bw=alpha,
        k_bw=params.get("k_bw", DEFAULT_K_BW),
        A_bw=params.get("A_bw", DEFAULT_A_BW),
        B_bw=params.get("B_bw", DEFAULT_B_BW),
        C_bw=params.get("C_bw", DEFAULT_C_BW),
        n_bw=params.get("n_bw", DEFAULT_N_BW),
        mass=params.get("mass", DEFAULT_MASS),
        stiffness=params.get("stiffness", DEFAULT_STIFFNESS),
        damping=params.get("damping", DEFAULT_DAMPING),
        rmse=rmse,
        r_squared=r2,
        correlation=corr,
        mae=mae,
        loop_area_shoelace=loop_area_real,
        loop_area_model=loop_area_model,
        loop_area_error_pct=loop_area_error,
        z_mean=z_mean,
        z_abs_max=z_abs_max,
        z_last=z_last,
        hyst_energy_pct=hyst_energy_pct,
        linearity_label=linearity,
    )


def _shoelace_area(x: np.ndarray, y: np.ndarray) -> float:
    """Shoelace formula for loop area, normalised by bounding rectangle."""
    n = len(x)
    if n < 3:
        return 0.0
    dx = np.gradient(x)
    area = abs(float(np.sum(y * dx)))
    rect = (float(np.max(x)) - float(np.min(x))) * (float(np.max(y)) - float(np.min(y)))
    if not math.isfinite(rect) or rect <= 1e-12:
        return 0.0
    val = area / rect
    return float(val) if math.isfinite(val) else 0.0


# ═══════════════════════════════════════════════════════════════
# Output writers
# ═══════════════════════════════════════════════════════════════

def write_output_tsv(
    path: Path,
    t: np.ndarray,
    force_norm: np.ndarray,
    entrada_norm: np.ndarray,
    F_model_norm: np.ndarray,
    z_arr: np.ndarray,
    du_or_vel: np.ndarray,
    disp: np.ndarray | None = None,
    mode: str = "clasico",
) -> None:
    """Write detailed output TSV with all signals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        if mode == "kanpinn" and disp is not None:
            header = ["tiempo_s", "fuerza_norm", "entrada_norm",
                      "F_model_norm", "z_boucwen", "vel_int", "disp_int", "residual"]
        else:
            header = ["tiempo_s", "fuerza_norm", "entrada_norm",
                      "F_model_norm", "z_boucwen", "du_dt", "residual"]
        writer.writerow(header)
        for i in range(len(t)):
            residual = force_norm[i] - F_model_norm[i]
            if mode == "kanpinn" and disp is not None:
                row = [f"{t[i]:.9g}", f"{force_norm[i]:.9g}", f"{entrada_norm[i]:.9g}",
                       f"{F_model_norm[i]:.9g}", f"{z_arr[i]:.9g}",
                       f"{du_or_vel[i]:.9g}", f"{disp[i]:.9g}", f"{residual:.9g}"]
            else:
                row = [f"{t[i]:.9g}", f"{force_norm[i]:.9g}", f"{entrada_norm[i]:.9g}",
                       f"{F_model_norm[i]:.9g}", f"{z_arr[i]:.9g}",
                       f"{du_or_vel[i]:.9g}", f"{residual:.9g}"]
            writer.writerow(row)


def write_metrics_jsonl(path: Path, metrics: BoucWenMetrics) -> None:
    """Append metrics record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(metrics), ensure_ascii=False) + "\n")


def print_summary(metrics: BoucWenMetrics) -> None:
    """Print a human-readable summary to stdout."""
    print("=" * 72)
    print(f"BOUC-WEN FORWARD MODEL ({metrics.model_mode}): {metrics.case_id}")
    print("=" * 72)
    print(f"  Input:          {metrics.input_file}")
    print(f"  Samples:        {metrics.n_samples}  fs={metrics.fs_hz:.0f} Hz  dur={metrics.duration_s:.4f} s")
    print()
    print(f"  Parameters ({metrics.model_mode}):")
    if metrics.model_mode == "kanpinn":
        print(f"    m={metrics.mass} kg  k={metrics.stiffness} N/m  c={metrics.damping} Ns/m")
    else:
        print(f"    k_bw={metrics.k_bw}  (normalised-scale rigidity)")
    print(f"    alpha={metrics.alpha_bw}  A={metrics.A_bw}  B={metrics.B_bw}  C={metrics.C_bw}  n={metrics.n_bw}")
    print()
    print(f"  Fit metrics:")
    print(f"    R^2        = {metrics.r_squared:.6f}")
    print(f"    RMSE       = {metrics.rmse:.6f}")
    print(f"    MAE        = {metrics.mae:.6f}")
    print(f"    Correlation= {metrics.correlation:.6f}")
    print()
    print(f"  Hysteresis:")
    print(f"    Loop area (real)  = {metrics.loop_area_shoelace:.6f}")
    print(f"    Loop area (model) = {metrics.loop_area_model:.6f}")
    print(f"    Area error        = {metrics.loop_area_error_pct:.2f} %")
    print(f"    z_mean={metrics.z_mean:.6f}  z_abs_max={metrics.z_abs_max:.6f}  z_last={metrics.z_last:.6f}")
    print(f"    Hyst energy       = {metrics.hyst_energy_pct:.2f} %")
    print()
    print(f"  Linearity:  {metrics.linearity_label} (alpha={metrics.alpha_bw:.3f})")
    print("=" * 72)


# ═══════════════════════════════════════════════════════════════
# Batch mode: process all ranked cases with run_boucwen_next=true
# ═══════════════════════════════════════════════════════════════

def resolve_boucwen_ready(case: dict, car_root: Path) -> Path | None:
    """Resolve the boucwen_ready path from a ranked case record."""
    rel = case.get("boucwen_ready_relpath", "")
    if not rel:
        return None
    candidate = car_root / rel
    if candidate.exists():
        return candidate
    # Try relative to SRC_ROOT
    src_root = car_root.parent
    candidate2 = src_root / rel
    if candidate2.exists():
        return candidate2
    return None


def run_batch_from_ranked(
    ranked_jsonl: Path,
    car_root: Path,
    out_dir: Path,
    params: dict,
    mode: str = "clasico",
    drift_fc: float = 5.0,
) -> list[BoucWenMetrics]:
    """Process all cases in a ranked JSONL where run_boucwen_next=true."""
    with ranked_jsonl.open("r", encoding="utf-8") as fh:
        cases = [json.loads(line) for line in fh if line.strip()]

    all_metrics: list[BoucWenMetrics] = []
    processed = 0
    skipped = 0

    for case in cases:
        if not case.get("run_boucwen_next", False):
            skipped += 1
            continue

        ready_path = resolve_boucwen_ready(case, car_root)
        if ready_path is None:
            print(f"  [SKIP] {case.get('case_id')}: boucwen_ready not found")
            skipped += 1
            continue

        case_id = case.get("case_id", "unknown")
        print(f"\n--- Processing {case_id} ---")

        metrics = run_single_case(
            ready_path=ready_path,
            case_id=case_id,
            out_dir=out_dir / case_id,
            params=params,
            mode=mode,
            drift_fc=drift_fc,
        )
        all_metrics.append(metrics)
        processed += 1

    print(f"\nBatch complete: {processed} processed, {skipped} skipped")
    return all_metrics


def run_single_case(
    ready_path: Path,
    case_id: str,
    out_dir: Path,
    params: dict,
    mode: str = "clasico",
    drift_fc: float = 5.0,
) -> BoucWenMetrics:
    """Run the Bouc-Wen model on a single boucwen_ready file."""
    t, force_norm, entrada_norm, fs = load_boucwen_ready(ready_path)
    params["mode"] = mode

    if mode == "kanpinn":
        F_model_norm, z_arr, vel, disp, envelope = run_bouc_wen_kanpinn(
            entrada_norm=entrada_norm,
            fs=fs,
            mass=params.get("mass", DEFAULT_MASS),
            stiffness=params.get("stiffness", DEFAULT_STIFFNESS),
            damping=params.get("damping", DEFAULT_DAMPING),
            alpha=params.get("alpha", DEFAULT_ALPHA),
            A_bw=params.get("A_bw", DEFAULT_A_BW),
            B_bw=params.get("B_bw", DEFAULT_B_BW),
            C_bw=params.get("C_bw", DEFAULT_C_BW),
            n_bw=params.get("n_bw", DEFAULT_N_BW),
            drift_fc=drift_fc,
        )
        du_or_vel = vel
        disp_out = disp
    else:
        # Classic Bouc-Wen on normalised signals (default)
        F_model_norm, z_arr, du = run_bouc_wen_clasico(
            u_norm=entrada_norm,
            fs=fs,
            A_bw=params.get("A_bw", DEFAULT_A_BW),
            B_bw=params.get("B_bw", DEFAULT_B_BW),
            C_bw=params.get("C_bw", DEFAULT_C_BW),
            n_bw=params.get("n_bw", DEFAULT_N_BW),
            k_bw=params.get("k_bw", DEFAULT_K_BW),
            alpha=params.get("alpha", DEFAULT_ALPHA),
        )
        du_or_vel = du
        disp_out = None

    metrics = compute_metrics(
        case_id=case_id,
        input_file=str(ready_path),
        t=t,
        force_norm=force_norm,
        entrada_norm=entrada_norm,
        F_model_norm=F_model_norm,
        z_arr=z_arr,
        fs=fs,
        params=params,
    )

    print_summary(metrics)

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    output_tsv = out_dir / f"{case_id}_boucwen_output.tsv"
    write_output_tsv(output_tsv, t, force_norm, entrada_norm, F_model_norm,
                     z_arr, du_or_vel, disp_out, mode=mode)
    print(f"  Output TSV: {output_tsv}")

    metrics_jsonl = out_dir / f"{case_id}_boucwen_metrics.jsonl"
    write_metrics_jsonl(metrics_jsonl, metrics)
    print(f"  Metrics JSONL: {metrics_jsonl}")

    return metrics


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def build_params(args: argparse.Namespace) -> dict:
    return {
        "mode": args.mode,
        "mass": args.mass,
        "stiffness": args.stiffness,
        "damping": args.damping,
        "alpha": args.alpha,
        "k_bw": args.k_bw,
        "A_bw": args.A_bw,
        "B_bw": args.B_bw,
        "C_bw": args.C_bw,
        "n_bw": args.n_bw,
    }


def main() -> int:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    CAR_ROOT = SRC_ROOT / "caracterizacion_fuerza"
    DEFAULT_OUT = CAR_ROOT / "razonador_casos" / "salidas" / "casos_virtual_udp" / "boucwen_resultados"

    parser = argparse.ArgumentParser(
        description="Run Bouc-Wen forward model on boucwen_ready files (edge-friendly, numpy only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python run_boucwen_case.py --input path/to/boucwen_ready.txt

  # Batch: all ranked cases with run_boucwen_next=true
  python run_boucwen_case.py --ranked-jsonl path/to/casos_virtuales_ranked.jsonl

  # Override parameters
  python run_boucwen_case.py --input ready.txt --alpha 0.95 --k-bw 1.2

  # KAN-PINN mode (physical units, not for normalised boucwen_ready)
  python run_boucwen_case.py --input ready.txt --mode kanpinn
""",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path, help="Single boucwen_ready TSV file")
    input_group.add_argument("--ranked-jsonl", type=Path, help="Ranked JSONL for batch processing")

    parser.add_argument("--case-id", default=None, help="Case ID (auto-derived from filename if omitted)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--mode", choices=["clasico", "kanpinn"], default="clasico",
                        help="Model mode: clasico (normalised, default) or kanpinn (physical units)")

    # Classic Bouc-Wen parameters
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Ratio lineal (1=linear, 0=full hysteresis)")
    parser.add_argument("--k-bw", type=float, default=DEFAULT_K_BW, help="Rigidity in normalised scale (clasico mode)")
    parser.add_argument("--A-bw", type=float, default=DEFAULT_A_BW)
    parser.add_argument("--B-bw", type=float, default=DEFAULT_B_BW)
    parser.add_argument("--C-bw", type=float, default=DEFAULT_C_BW)
    parser.add_argument("--n-bw", type=float, default=DEFAULT_N_BW)

    # KAN-PINN parameters (only used in kanpinn mode)
    parser.add_argument("--mass", type=float, default=DEFAULT_MASS, help="Mass (kg, kanpinn mode)")
    parser.add_argument("--stiffness", type=float, default=DEFAULT_STIFFNESS, help="Stiffness (N/m, kanpinn mode)")
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING, help="Damping (Ns/m, kanpinn mode)")
    parser.add_argument("--drift-fc", type=float, default=5.0, help="HP cutoff for anti-drift (Hz, kanpinn mode)")

    args = parser.parse_args()
    params = build_params(args)

    if args.ranked_jsonl:
        # Batch mode
        all_metrics = run_batch_from_ranked(
            ranked_jsonl=args.ranked_jsonl,
            car_root=CAR_ROOT,
            out_dir=args.out_dir,
            params=params,
            mode=args.mode,
            drift_fc=args.drift_fc,
        )
        # Write combined summary
        if all_metrics:
            summary_path = args.out_dir / "boucwen_batch_summary.jsonl"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as fh:
                for m in all_metrics:
                    fh.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
            print(f"\nBatch summary: {summary_path}")
        return 0

    # Single mode
    input_path = args.input
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    case_id = args.case_id
    if case_id is None:
        # Derive from filename: udp_xxx_boucwen_ready.txt -> udp_xxx
        stem = input_path.stem
        if stem.endswith("_boucwen_ready"):
            case_id = stem[: -len("_boucwen_ready")]
        else:
            case_id = stem

    run_single_case(
        ready_path=input_path,
        case_id=case_id,
        out_dir=args.out_dir / case_id,
        params=params,
        mode=args.mode,
        drift_fc=args.drift_fc,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
