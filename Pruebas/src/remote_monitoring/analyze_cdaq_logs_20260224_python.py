#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:
    raise SystemExit("Este script requiere pandas. Instala con: pip install pandas") from e

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit("Este script requiere matplotlib. Instala con: pip install matplotlib") from e

_HAVE_SCIPY = False
try:
    from scipy.signal import hilbert as scipy_hilbert
    from scipy.signal import butter, filtfilt, detrend as scipy_detrend
    _HAVE_SCIPY = True
except Exception:
    scipy_hilbert = None
    butter = None
    filtfilt = None
    scipy_detrend = None
    _HAVE_SCIPY = False


K_ENV = 0.0669
B_ENV = 0.8837
FC_ENV_HZ = 15.0
FC_HP_HZ = 0.5
FMIN_F0_HZ = 5.0


@dataclass
class ThdResult:
    thd_pct: float
    f0_hz: float
    freqs_hz: np.ndarray
    mag: np.ndarray


def list_timestamps(logs_dir: str, prefix: str = "20260224_") -> List[str]:
    ts = set()
    for name in os.listdir(logs_dir):
        if not name.startswith("cdaq_remote_"):
            continue
        if prefix not in name:
            continue
        if name.endswith(".csv") and ("_inference.csv" not in name):
            base = name[len("cdaq_remote_") : -len(".csv")]
            if base.startswith(prefix):
                ts.add(base)
        if name.endswith("_inference.csv"):
            base = name[len("cdaq_remote_") : -len("_inference.csv")]
            if base.startswith(prefix):
                ts.add(base)
        if name.endswith("_cases.jsonl"):
            base = name[len("cdaq_remote_") : -len("_cases.jsonl")]
            if base.startswith(prefix):
                ts.add(base)
    return sorted(ts)


def read_csv_with_comments(path: str) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def read_cases_jsonl(path: str) -> Tuple[int, int]:
    n_lines = 0
    n_bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                n_bad += 1
                continue
            llm = obj.get("llm")
            if isinstance(llm, dict):
                resp = llm.get("response")
                if isinstance(resp, str):
                    s = resp.strip()
                    if s:
                        try:
                            json.loads(s)
                        except Exception:
                            pass
    return n_lines, n_bad


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if _HAVE_SCIPY:
        return np.abs(scipy_hilbert(x))
    n = x.size
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
    xa = np.fft.ifft(X * h)
    return np.abs(xa)


def lowpass(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if fs <= 0 or fc <= 0 or fc >= fs / 2:
        return x
    if _HAVE_SCIPY:
        sos_b, sos_a = butter(2, fc / (fs / 2), btype="low")
        return filtfilt(sos_b, sos_a, x)
    win = max(3, int(round(fs / max(fc, 1e-6))))
    win = min(win, max(3, x.size // 10))
    if win % 2 == 0:
        win += 1
    k = np.ones(win) / win
    return np.convolve(x, k, mode="same")


def highpass(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if fs <= 0 or fc <= 0 or fc >= fs / 2:
        return x - np.nanmean(x)
    if _HAVE_SCIPY:
        b, a = butter(2, fc / (fs / 2), btype="high")
        return filtfilt(b, a, x - np.nanmean(x))
    win = max(3, int(round(fs / max(fc, 1e-6))))
    win = min(max(win, 101), x.size - 1)
    if win % 2 == 0:
        win += 1
    k = np.ones(win) / win
    trend = np.convolve(x, k, mode="same")
    return (x - trend) - np.nanmean(x - trend)


def detrend_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if _HAVE_SCIPY:
        return scipy_detrend(x)
    n = x.size
    if n < 2:
        return x
    t = np.arange(n)
    p = np.polyfit(t, x, 1)
    return x - (p[0] * t + p[1])


def thd_fft(x: np.ndarray, fs: float, n_harmonics: int = 5, fmin_hz: float = 0.0) -> ThdResult:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8 or fs <= 0:
        return ThdResult(0.0, 0.0, np.array([0.0]), np.array([0.0]))

    n_fft = min(x.size, 4096)
    xx = x[-n_fft:].copy()
    xx = xx - np.mean(xx)

    w = np.hanning(n_fft)
    xx = xx * w

    X = np.fft.rfft(xx)
    mag = np.abs(X) * 2.0 / n_fft
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)

    if mag.size <= 2:
        return ThdResult(0.0, 0.0, freqs, mag)

    mag_nodc = mag[1:]
    freqs_nodc = freqs[1:]
    valid = freqs_nodc >= float(max(0.0, fmin_hz))
    if np.any(valid):
        idxs = np.flatnonzero(valid)
        pk_local = int(np.argmax(mag_nodc[idxs]))
        pk = int(idxs[pk_local])
    else:
        pk = int(np.argmax(mag_nodc))

    f0 = float(freqs_nodc[pk])
    fund_mag = float(mag_nodc[pk])

    if fund_mag < 1e-10 or f0 <= 0:
        return ThdResult(0.0, f0, freqs, mag)

    harm_sum_sq = 0.0
    df = freqs_nodc[1] - freqs_nodc[0] if freqs_nodc.size > 1 else 0.0

    for h in range(2, n_harmonics + 2):
        fh = f0 * h
        if fh >= fs / 2:
            break
        idx = int(round(fh / max(df, 1e-12)))
        idx = max(0, min(idx, mag_nodc.size - 1))
        lo = max(0, idx - 2)
        hi = min(mag_nodc.size, idx + 3)
        harm = float(np.max(mag_nodc[lo:hi]))
        harm_sum_sq += harm * harm

    thd_pct = float(min(999.0, math.sqrt(harm_sum_sq) / fund_mag * 100.0))
    return ThdResult(thd_pct, f0, freqs, mag)


def parse_boolish(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"", "none", "null", "n/a", "na"}:
            return False
        if s in {"false", "0", "no", "off"}:
            return False
        if s in {"true", "1", "yes", "on"}:
            return True
        return True
    return False


def parse_llm_payload(llm_obj) -> Dict:
    if not isinstance(llm_obj, dict):
        return {}

    if "response" in llm_obj:
        resp = llm_obj.get("response")
        if isinstance(resp, dict):
            return resp
        if isinstance(resp, str):
            s = resp.strip()
            if not s:
                return {}
            try:
                parsed = json.loads(s)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
    return llm_obj


def read_cases_llm_summary(path: str) -> Dict[str, float]:
    n_lines = 0
    n_bad = 0
    n_store = 0
    n_anom = 0
    n_cut_state = 0
    labels = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                n_bad += 1
                continue

            payload = parse_llm_payload(obj.get("llm"))
            if not payload:
                continue

            should_store = parse_boolish(payload.get("should_store_case"))
            anomaly = parse_boolish(payload.get("anomaly"))

            cs = payload.get("cutting_state")
            cutting_state = False
            if isinstance(cs, str):
                cutting_state = cs.strip().lower() in {"running", "cutting", "true", "on", "active"}
            else:
                cutting_state = parse_boolish(cs)

            if should_store:
                n_store += 1
            if anomaly:
                n_anom += 1
            if cutting_state:
                n_cut_state += 1

            lab = payload.get("label")
            if isinstance(lab, str) and lab.strip():
                labels[lab.strip()] += 1

    store_frac = (n_store / n_lines) if n_lines else float("nan")
    anom_frac = (n_anom / n_lines) if n_lines else float("nan")
    cut_state_frac = (n_cut_state / n_lines) if n_lines else float("nan")
    top_label = labels.most_common(1)[0][0] if labels else ""
    top_label_frac = (labels[top_label] / n_lines) if (labels and n_lines) else float("nan")

    return {
        "cases_lines": float(n_lines),
        "cases_bad_lines": float(n_bad),
        "llm_store_frac": float(store_frac),
        "llm_anomaly_frac": float(anom_frac),
        "llm_cutting_state_frac": float(cut_state_frac),
        "llm_top_label": top_label,
        "llm_top_label_frac": float(top_label_frac),
    }


def summarize_inference(infer: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    cols = infer.columns

    def q(series, qq):
        try:
            return float(series.quantile(qq))
        except Exception:
            return float("nan")

    if "envelope_rms_g" in cols:
        s = pd.to_numeric(infer["envelope_rms_g"], errors="coerce")
        out["infer_env_med_g"] = float(s.median())
        out["infer_env_p95_g"] = q(s, 0.95)

    if "THD_accel_pct" in cols:
        s = pd.to_numeric(infer["THD_accel_pct"], errors="coerce")
        out["infer_thd_accel_med_pct"] = float(s.median())
        out["infer_thd_accel_p95_pct"] = q(s, 0.95)

    if "freq_dom_Hz" in cols:
        s = pd.to_numeric(infer["freq_dom_Hz"], errors="coerce")
        out["infer_fdom_med_hz"] = float(s.median())
        out["infer_fdom_p95_hz"] = q(s, 0.95)

    if "cutting_flag" in cols:
        s = pd.to_numeric(infer["cutting_flag"], errors="coerce")
        out["infer_cutting_frac"] = float(s.mean())

    if "hyst_area" in cols:
        s = pd.to_numeric(infer["hyst_area"], errors="coerce")
        out["infer_hyst_area_med"] = float(s.median())
        out["infer_hyst_area_p95"] = q(s, 0.95)

    return out


def read_csv_auto_sep(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.readline()
    sep = "\t" if "\t" in head else ","
    return pd.read_csv(path, sep=sep)


def infer_fs_from_time(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return float("nan")
    return float(1.0 / np.median(dt))


def pick_first_present(columns: Iterable[str], candidates: List[str]) -> str:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    for c in columns:
        if isinstance(c, str) and "aceleracion" in c:
            return c
    return ""


def list_december_files(carac_dir: str) -> List[Tuple[str, str]]:
    out = []
    for name in os.listdir(carac_dir):
        if not name.lower().endswith(".csv"):
            continue
        if name.startswith("corte_"):
            kind = "cut_20251209" if name.startswith("corte_20251209_") else "cut_style_csv"
            out.append((kind, os.path.join(carac_dir, name)))
            continue
        if name.startswith("caracterizacion_fuerza_20251205_") and ("_exp" in name) and ("resumen" not in name):
            out.append(("carac_20251205", os.path.join(carac_dir, name)))
            continue
    return sorted(out, key=lambda x: x[1])


def list_december_style_csvs(csv_dir: str, prefix: str = "") -> List[str]:
    out: List[str] = []
    for name in os.listdir(csv_dir):
        if not name.lower().endswith(".csv"):
            continue
        if prefix and not name.startswith(prefix):
            continue
        out.append(os.path.join(csv_dir, name))
    return sorted(out)


def extract_timestamp_from_basename(base: str) -> str:
    m = re.search(r"(\d{8}_\d{6})", base)
    return m.group(1) if m else ""


def process_december_style_dir(csv_dir: str, out_dir: str, prefix: str = "", all_accel: bool = False) -> pd.DataFrame:
    rows = []
    files = list_december_style_csvs(csv_dir, prefix=prefix)
    base_out = os.path.join(out_dir, "december_style_processed")
    os.makedirs(base_out, exist_ok=True)

    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        out_ts_dir = os.path.join(base_out, base)
        os.makedirs(out_ts_dir, exist_ok=True)

        df = read_csv_auto_sep(path)
        if "tiempo_s" not in df.columns or "fuerza_V" not in df.columns:
            continue

        t = pd.to_numeric(df["tiempo_s"], errors="coerce").to_numpy(dtype=float)
        force = pd.to_numeric(df["fuerza_V"], errors="coerce").to_numpy(dtype=float)

        accel_candidates = [
            "aceleracion_sensor_g",
            "aceleracion_pieza_g",
            "aceleracion_prensa_g",
            "aceleracion_bancada_g",
        ]
        accel_cols: List[str] = []
        if all_accel:
            for c in accel_candidates:
                if c in df.columns:
                    accel_cols.append(c)
            if not accel_cols:
                accel_cols = [c for c in df.columns if isinstance(c, str) and "aceleracion" in c]
            accel_cols = [c for c in accel_cols if c]
            accel_cols = list(dict.fromkeys(accel_cols))
        else:
            accel_col = pick_first_present(df.columns, accel_candidates)
            accel_cols = [accel_col] if accel_col else []

        if not accel_cols:
            continue

        fs = infer_fs_from_time(t)
        if not np.isfinite(fs) or fs <= 0:
            fs = 2500.0

        t_rel = t - np.nanmin(t)
        ts_guess = extract_timestamp_from_basename(base)
        period = ts_guess[:6] if ts_guess else ""

        for accel_col in accel_cols:
            accel_g = pd.to_numeric(df[accel_col], errors="coerce").to_numpy(dtype=float)
            tag = accel_col.replace("aceleracion_", "").replace("_g", "")
            run_id = f"{base}__{tag}" if all_accel and len(accel_cols) > 1 else base
            out_run_dir = os.path.join(out_ts_dir, run_id)
            os.makedirs(out_run_dir, exist_ok=True)

            extra = {}
            extra.update(plot_processing(run_id, out_run_dir, t_rel, force, accel_g, fs))
            extra.update(plot_hysteresis(run_id, out_run_dir, force, accel_g, fs))

            rows.append(
                {
                    "group": "december_style_replay",
                    "dataset_kind": "december_style_csv",
                    "dataset": base,
                    "run_id": run_id,
                    "timestamp_guess": ts_guess,
                    "period": period,
                    "path": path,
                    "fs_hz": float(fs),
                    "duration_s": float(np.nanmax(t_rel) - np.nanmin(t_rel)) if np.isfinite(np.nanmax(t_rel)) else float("nan"),
                    "accel_col": accel_col,
                    "rows": int(len(df)),
                    **extra,
                }
            )

    return pd.DataFrame(rows)


def process_december_files(carac_dir: str, out_dir: str) -> pd.DataFrame:
    rows = []
    files = list_december_files(carac_dir)
    base_out = os.path.join(out_dir, "dec_2025")
    os.makedirs(base_out, exist_ok=True)

    for kind, path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        out_ts_dir = os.path.join(base_out, base)
        os.makedirs(out_ts_dir, exist_ok=True)

        df = read_csv_auto_sep(path)
        if "tiempo_s" not in df.columns or "fuerza_V" not in df.columns:
            continue

        t = pd.to_numeric(df["tiempo_s"], errors="coerce").to_numpy(dtype=float)
        force = pd.to_numeric(df["fuerza_V"], errors="coerce").to_numpy(dtype=float)

        accel_col = pick_first_present(
            df.columns,
            [
                "aceleracion_sensor_g",
                "aceleracion_pieza_g",
                "aceleracion_prensa_g",
                "aceleracion_bancada_g",
            ],
        )
        if not accel_col:
            continue

        accel_g = pd.to_numeric(df[accel_col], errors="coerce").to_numpy(dtype=float)
        fs = infer_fs_from_time(t)
        if not np.isfinite(fs) or fs <= 0:
            fs = 2500.0

        t_rel = t - np.nanmin(t)

        extra = {}
        extra.update(plot_processing(base, out_ts_dir, t_rel, force, accel_g, fs))
        extra.update(plot_hysteresis(base, out_ts_dir, force, accel_g, fs))

        rows.append(
            {
                "group": "december_2025",
                "dataset_kind": kind,
                "dataset": base,
                "path": path,
                "fs_hz": float(fs),
                "duration_s": float(np.nanmax(t_rel) - np.nanmin(t_rel)) if np.isfinite(np.nanmax(t_rel)) else float("nan"),
                "accel_col": accel_col,
                "rows": int(len(df)),
                **extra,
            }
        )

    return pd.DataFrame(rows)


def reconstruct_time(raw: pd.DataFrame, infer: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    infer = infer.copy()
    infer = infer.sort_values("seq")

    seq = infer["seq"].to_numpy(dtype=int)
    fs = infer["fs_hz"].to_numpy(dtype=float)

    if "t_local_s" in infer.columns:
        t0 = infer["t_local_s"].to_numpy(dtype=float)
    else:
        t0 = infer["t_sender_s"].to_numpy(dtype=float)

    fs_map: Dict[int, float] = {int(s): float(f) for s, f in zip(seq, fs)}
    t0_map: Dict[int, float] = {int(s): float(t) for s, t in zip(seq, t0)}

    raw_seq = raw["seq"].to_numpy(dtype=int)
    samp = raw["sample_idx"].to_numpy(dtype=float)

    fs_vec = np.array([fs_map.get(int(s), np.nan) for s in raw_seq], dtype=float)
    t0_vec = np.array([t0_map.get(int(s), np.nan) for s in raw_seq], dtype=float)

    t = t0_vec + samp / fs_vec

    t0_first = np.nanmin(t)
    t_rel = t - t0_first

    fs_med = float(np.nanmedian(fs_vec[np.isfinite(fs_vec) & (fs_vec > 0)])) if np.any(np.isfinite(fs_vec)) else float("nan")

    force = raw["force_v"].to_numpy(dtype=float)
    accel = raw["accel_g"].to_numpy(dtype=float)

    return t_rel, force, accel, fs_med


def plot_processing(ts: str, out_ts_dir: str, t: np.ndarray, force: np.ndarray, accel_g: np.ndarray, fs: float) -> Dict[str, float]:
    os.makedirs(out_ts_dir, exist_ok=True)

    mask = np.isfinite(t) & np.isfinite(force) & np.isfinite(accel_g)
    t = t[mask]
    force = force[mask]
    accel_g = accel_g[mask]

    if t.size < 16:
        return {}

    order = np.argsort(t)
    t = t[order]
    force = force[order]
    accel_g = accel_g[order]

    if not np.isfinite(fs) or fs <= 0:
        fs = 2000.0

    env_raw = hilbert_envelope(accel_g)
    env = lowpass(env_raw, fs, FC_ENV_HZ)
    f_est = K_ENV * env + B_ENV

    thd_f = thd_fft(force, fs, fmin_hz=FMIN_F0_HZ)
    thd_a = thd_fft(accel_g, fs, fmin_hz=FMIN_F0_HZ)

    max_pts = 250000
    stride = max(1, int(t.size // max_pts))
    tt = t[::stride]
    ff = force[::stride]
    aa = accel_g[::stride]
    ee = env[::stride]
    fe = f_est[::stride]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.subplots(3, 2)

    ax[0, 0].plot(tt, ff, lw=0.6, color="k")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].set_title("Fuerza (raw, decimado)")
    ax[0, 0].set_xlabel("t (s, relativo)")
    ax[0, 0].set_ylabel("force_v")

    ax[0, 1].plot(tt, aa, lw=0.6, color="b")
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].set_title("Aceleración (raw, decimado)")
    ax[0, 1].set_xlabel("t (s, relativo)")
    ax[0, 1].set_ylabel("accel_g")

    ax[1, 0].plot(tt, ee, lw=0.8, color="m")
    ax[1, 0].grid(True, alpha=0.3)
    ax[1, 0].set_title(f"Envolvente (Hilbert + LP {FC_ENV_HZ:.1f} Hz)")
    ax[1, 0].set_xlabel("t (s, relativo)")
    ax[1, 0].set_ylabel("env_g")

    ax[1, 1].plot(tt, fe, lw=0.8, color="r")
    ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].set_title(f"F_est = {K_ENV:.4f}*env + {B_ENV:.4f}")
    ax[1, 1].set_xlabel("t (s, relativo)")
    ax[1, 1].set_ylabel("F_est (V)")

    ax[2, 0].semilogy(thd_f.freqs_hz, thd_f.mag, color="k", lw=0.9)
    ax[2, 0].grid(True, alpha=0.3)
    ax[2, 0].set_xlim(0, min(500, fs / 2))
    ax[2, 0].set_title(f"FFT fuerza: f0={thd_f.f0_hz:.1f} Hz, THD={thd_f.thd_pct:.1f}%")
    ax[2, 0].set_xlabel("f (Hz)")
    ax[2, 0].set_ylabel("|FFT|")

    ax[2, 1].semilogy(thd_a.freqs_hz, thd_a.mag, color="b", lw=0.9)
    ax[2, 1].grid(True, alpha=0.3)
    ax[2, 1].set_xlim(0, min(500, fs / 2))
    ax[2, 1].set_title(f"FFT acel: f0={thd_a.f0_hz:.1f} Hz, THD={thd_a.thd_pct:.1f}%")
    ax[2, 1].set_xlabel("f (Hz)")
    ax[2, 1].set_ylabel("|FFT|")

    fig.suptitle(f"{ts} - procesamiento dic-2025", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.93, hspace=0.35, wspace=0.25)
    fig.savefig(os.path.join(out_ts_dir, "raw_processing_20251210_py.png"), dpi=200)
    plt.close(fig)

    return {
        "env_rms": float(np.sqrt(np.mean(env * env))),
        "thd_force_pct": float(thd_f.thd_pct),
        "f0_force_hz": float(thd_f.f0_hz),
        "thd_accel_pct": float(thd_a.thd_pct),
        "f0_accel_hz": float(thd_a.f0_hz),
    }


def plot_hysteresis(ts: str, out_ts_dir: str, force_v: np.ndarray, accel_g: np.ndarray, fs: float) -> Dict[str, float]:
    mask = np.isfinite(force_v) & np.isfinite(accel_g)
    f = force_v[mask].astype(float)
    a_g = accel_g[mask].astype(float)

    if f.size < 256:
        return {}

    if not np.isfinite(fs) or fs <= 0:
        fs = 2000.0

    a_ms2 = a_g * 9.81
    a_hp = highpass(a_ms2, fs, FC_HP_HZ)

    dt = 1.0 / fs
    v = np.cumsum(a_hp) * dt
    v = detrend_linear(v)

    x = np.cumsum(v) * dt
    x = detrend_linear(x)

    f0 = f - np.mean(f)

    area = 0.5 * abs(np.sum(x[:-1] * f0[1:] - x[1:] * f0[:-1]))

    disp_range = float(np.max(x) - np.min(x))
    force_range = float(np.max(f0) - np.min(f0))

    alpha_est = 1.0
    k_app = 0.0
    if disp_range > 1e-12:
        k_app = force_range / disp_range
        energia_elastica = 0.5 * k_app * (disp_range / 2.0) ** 2
        if energia_elastica > 0:
            ratio = area / energia_elastica
            alpha_est = float(max(0.0, min(1.0, 1.0 - ratio / 4.0)))

    thd_f = thd_fft(f0, fs, fmin_hz=FMIN_F0_HZ)
    thd_a = thd_fft(a_hp, fs, fmin_hz=FMIN_F0_HZ)

    segN = min(5000, f.size)
    f_use = max(thd_f.f0_hz, thd_a.f0_hz, 0.0)
    n_ciclos = 5000
    if f_use > 0.1:
        n_ciclos = int(round(5 * fs / f_use))
    n_ciclos = min(n_ciclos, f.size)

    t = np.arange(f.size) / fs

    fig = plt.figure(figsize=(14, 9))
    ax = fig.subplots(2, 3)

    ax[0, 0].plot(t[:segN], f0[:segN], color="r", lw=0.6)
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].set_title("Fuerza vs tiempo (demean)")
    ax[0, 0].set_xlabel("t (s)")
    ax[0, 0].set_ylabel("force (V)")

    ax[0, 1].semilogy(thd_f.freqs_hz, thd_f.mag, color="r", lw=0.9)
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].set_xlim(0, min(500, fs / 2))
    ax[0, 1].set_title(f"FFT fuerza (THD={thd_f.thd_pct:.1f}%)")
    ax[0, 1].set_xlabel("f (Hz)")
    ax[0, 1].set_ylabel("|FFT|")

    ax[0, 2].semilogy(thd_a.freqs_hz, thd_a.mag, color="b", lw=0.9)
    ax[0, 2].grid(True, alpha=0.3)
    ax[0, 2].set_xlim(0, min(500, fs / 2))
    ax[0, 2].set_title(f"FFT acel HP (THD={thd_a.thd_pct:.1f}%)")
    ax[0, 2].set_xlabel("f (Hz)")
    ax[0, 2].set_ylabel("|FFT|")

    ax[1, 0].plot(x[:n_ciclos] * 1000.0, f0[:n_ciclos], color="k", lw=0.6)
    ax[1, 0].grid(True, alpha=0.3)
    ax[1, 0].set_title(f"Histéresis F-x (alpha≈{alpha_est:.2f})")
    ax[1, 0].set_xlabel("x (mm, rel)")
    ax[1, 0].set_ylabel("force (V)")

    ax[1, 1].plot(a_hp[:n_ciclos], f0[:n_ciclos], color="g", lw=0.6)
    ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].set_title("Lissajous F-a")
    ax[1, 1].set_xlabel("a (m/s^2)")
    ax[1, 1].set_ylabel("force (V)")

    ax[1, 2].plot(t[:segN], a_hp[:segN], color="b", lw=0.6)
    ax[1, 2].grid(True, alpha=0.3)
    ax[1, 2].set_title(f"Aceleración HP (area={area:.3g})")
    ax[1, 2].set_xlabel("t (s)")
    ax[1, 2].set_ylabel("a (m/s^2)")

    fig.suptitle(f"{ts} - histéresis tipo dic-2025", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.93, hspace=0.35, wspace=0.25)
    fig.savefig(os.path.join(out_ts_dir, "raw_hysteresis_20251205_py.png"), dpi=200)
    plt.close(fig)

    return {
        "hyst_area": float(area),
        "alpha_est": float(alpha_est),
    }


def export_december_style_csv(ts: str, export_dir: str, t_s: np.ndarray, force_v: np.ndarray, accel_g: np.ndarray) -> str:
    os.makedirs(export_dir, exist_ok=True)

    mask = np.isfinite(t_s) & np.isfinite(force_v) & np.isfinite(accel_g)
    t = np.asarray(t_s, dtype=float)[mask]
    f = np.asarray(force_v, dtype=float)[mask]
    a = np.asarray(accel_g, dtype=float)[mask]

    if t.size:
        order = np.argsort(t)
        t = t[order]
        f = f[order]
        a = a[order]
        t = t - np.nanmin(t)

    out_path = os.path.join(export_dir, f"corte_{ts}.csv")
    pd.DataFrame(
        {
            "tiempo_s": t,
            "fuerza_V": f,
            "aceleracion_sensor_g": a,
        }
    ).to_csv(out_path, index=False)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--prefix", default="20260224_")
    ap.add_argument("--compare-b", action="store_true")
    ap.add_argument("--carac-dir", default=None)
    ap.add_argument("--fmin-hz", type=float, default=5.0)
    ap.add_argument("--export-december-style", action="store_true")
    ap.add_argument("--export-dir", default=None)
    ap.add_argument("--process-december-style-dir", default=None)
    ap.add_argument("--process-december-style-prefix", default="")
    ap.add_argument("--process-december-style-all-accel", action="store_true")
    args = ap.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = args.logs_dir or os.path.join(this_dir, "logs")
    out_dir = args.out_dir or os.path.join(this_dir, "python_out")
    export_dir = args.export_dir or os.path.join(out_dir, "december_style")

    global FMIN_F0_HZ
    FMIN_F0_HZ = float(max(0.0, args.fmin_hz))

    if not os.path.isdir(logs_dir):
        raise SystemExit(f"No existe logs_dir: {logs_dir}")

    os.makedirs(out_dir, exist_ok=True)

    if args.process_december_style_dir:
        csv_dir = args.process_december_style_dir
        if not os.path.isdir(csv_dir):
            raise SystemExit(f"No existe process_december_style_dir: {csv_dir}")
        df_style = process_december_style_dir(
            csv_dir,
            out_dir,
            prefix=args.process_december_style_prefix,
            all_accel=bool(args.process_december_style_all_accel),
        )
        out_path = os.path.join(out_dir, "december_style_summary.csv")
        df_style.to_csv(out_path, index=False)
        print(f"OK: {out_path}")

        if len(df_style):
            metric_cols = [
                c
                for c in [
                    "env_rms",
                    "thd_force_pct",
                    "f0_force_hz",
                    "thd_accel_pct",
                    "f0_accel_hz",
                    "hyst_area",
                    "alpha_est",
                    "duration_s",
                ]
                if c in df_style.columns
            ]
            key_cols = [c for c in ["period", "accel_col"] if c in df_style.columns]
            if key_cols and metric_cols:
                df_cmp = df_style.groupby(key_cols, dropna=False)[metric_cols].agg(["count", "median", "mean"]).reset_index()
                df_cmp.columns = [
                    (a if b == "" else f"{a}__{b}" if a in key_cols else f"{a}__{b}")
                    for (a, b) in df_cmp.columns.to_flat_index()
                ]
                cmp_path = os.path.join(out_dir, "december_style_comparison.csv")
                df_cmp.to_csv(cmp_path, index=False)
                print(f"OK: {cmp_path}")

        print(f"\nSalida en: {out_dir}")
        return 0

    timestamps = list_timestamps(logs_dir, prefix=args.prefix)
    print(f"Timestamps ({len(timestamps)}):")
    for ts in timestamps:
        print(f"  {ts}")

    rows = []
    for ts in timestamps:
        raw_path = os.path.join(logs_dir, f"cdaq_remote_{ts}.csv")
        inf_path = os.path.join(logs_dir, f"cdaq_remote_{ts}_inference.csv")
        cases_path = os.path.join(logs_dir, f"cdaq_remote_{ts}_cases.jsonl")

        if not os.path.isfile(raw_path) or not os.path.isfile(inf_path):
            print(f"[WARN] {ts}: falta raw o inference; se omite")
            continue

        print(f"\n=== {ts} ===")
        out_ts_dir = os.path.join(out_dir, ts)
        os.makedirs(out_ts_dir, exist_ok=True)

        raw = read_csv_with_comments(raw_path)
        infer = read_csv_with_comments(inf_path)

        t_s, force_v, accel_g, fs_med = reconstruct_time(raw, infer)

        exported_path = ""
        if args.export_december_style:
            exported_path = export_december_style_csv(ts, export_dir, t_s, force_v, accel_g)

        duration_s = float(np.nanmax(t_s) - np.nanmin(t_s)) if np.isfinite(np.nanmax(t_s)) else float("nan")

        extra = {}
        extra.update(plot_processing(ts, out_ts_dir, t_s, force_v, accel_g, fs_med))
        extra.update(plot_hysteresis(ts, out_ts_dir, force_v, accel_g, fs_med))

        infer_extra = summarize_inference(infer)

        llm_extra: Dict[str, float] = {}
        cases_lines = float("nan")
        cases_bad = float("nan")
        if os.path.isfile(cases_path):
            llm_extra = read_cases_llm_summary(cases_path)
            cases_lines = llm_extra.get("cases_lines", float("nan"))
            cases_bad = llm_extra.get("cases_bad_lines", float("nan"))

        rows.append({
            "timestamp": ts,
            "raw_rows": int(len(raw)),
            "inference_rows": int(len(infer)),
            "duration_s": duration_s,
            "fs_hz_median": fs_med,
            "raw_bytes": os.path.getsize(raw_path) if os.path.isfile(raw_path) else float("nan"),
            "inference_bytes": os.path.getsize(inf_path) if os.path.isfile(inf_path) else float("nan"),
            "cases_bytes": os.path.getsize(cases_path) if os.path.isfile(cases_path) else float("nan"),
            "cases_lines": cases_lines,
            "cases_bad_lines": cases_bad,
            "export_december_style_csv": exported_path,
            **extra,
            **infer_extra,
            **llm_extra,
        })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
        print(f"\nOK: {os.path.join(out_dir, 'summary.csv')}")

        if args.compare_b:
            carac_dir = args.carac_dir or os.path.abspath(os.path.join(this_dir, "..", "caracterizacion_fuerza"))
            if os.path.isdir(carac_dir):
                dec_df = process_december_files(carac_dir, out_dir)
                if not dec_df.empty:
                    dec_df.to_csv(os.path.join(out_dir, "december_2025_summary.csv"), index=False)

                compare_df = pd.concat(
                    [
                        df.assign(group="new_20260224", dataset_kind="remote_monitoring", dataset=df["timestamp"], path=""),
                        dec_df,
                    ],
                    ignore_index=True,
                    sort=False,
                )
                compare_path = os.path.join(out_dir, "compare_summary.csv")
                compare_df.to_csv(compare_path, index=False)

                compare_df = compare_df.copy()
                compare_df["plot_group"] = compare_df["group"].astype(str)
                m_dec = compare_df["group"].astype(str).eq("december_2025")
                if "dataset_kind" in compare_df.columns:
                    compare_df.loc[m_dec, "plot_group"] = compare_df.loc[m_dec, "dataset_kind"].astype(str)

                fig = plt.figure(figsize=(12, 6))
                ax = fig.subplots(1, 2)
                for i, col in enumerate(["env_rms", "f0_accel_hz"]):
                    sub = compare_df[["plot_group", col]].copy()
                    sub[col] = pd.to_numeric(sub[col], errors="coerce")
                    preferred = ["carac_20251205", "cut_20251209", "new_20260224"]
                    groups = [g for g in preferred if g in set(sub["plot_group"]) ]
                    data = [sub.loc[sub["plot_group"] == g, col].dropna().to_numpy() for g in groups]
                    try:
                        ax[i].boxplot(data, tick_labels=groups, showfliers=False)
                    except TypeError:
                        ax[i].boxplot(data, labels=groups, showfliers=False)
                    ax[i].set_title(col)
                    ax[i].grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, "compare_boxplots.png"), dpi=200)
                plt.close(fig)

                print(f"\nOK: {compare_path}")
            else:
                print(f"\n[WARN] --compare-b: no existe carac_dir: {carac_dir}")

    if args.export_december_style:
        print(f"CSV estilo diciembre en: {export_dir}")

    print(f"\nSalida en: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
