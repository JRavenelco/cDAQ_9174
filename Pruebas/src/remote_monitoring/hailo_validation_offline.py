#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


@dataclass
class NormSpec:
    in_mean: np.ndarray
    in_std: np.ndarray
    out_mean: Optional[np.ndarray] = None
    out_std: Optional[np.ndarray] = None


def _load_norm_spec(path: str, n_in: int, n_out: int) -> NormSpec:
    """Load normalization spec from JSON.

    Expected schema (flexible):
      {
        "in_mean": [..], "in_std": [..],
        "out_mean": [..], "out_std": [..]
      }

    out_* are optional.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    in_mean = np.asarray(obj.get("in_mean", [0.0] * n_in), dtype=np.float32).reshape((1, 1, n_in))
    in_std = np.asarray(obj.get("in_std", [1.0] * n_in), dtype=np.float32).reshape((1, 1, n_in))
    in_std = np.maximum(in_std, 1e-12)

    out_mean_raw = obj.get("out_mean", None)
    out_std_raw = obj.get("out_std", None)
    out_mean = None
    out_std = None
    if out_mean_raw is not None:
        out_mean = np.asarray(out_mean_raw, dtype=np.float32).reshape((1, 1, n_out))
    if out_std_raw is not None:
        out_std = np.asarray(out_std_raw, dtype=np.float32).reshape((1, 1, n_out))
        out_std = np.maximum(out_std, 1e-12)

    return NormSpec(in_mean=in_mean, in_std=in_std, out_mean=out_mean, out_std=out_std)


def _iter_samples_from_raw_csv(path: str) -> Iterator[Tuple[int, int, float, float]]:
    """Yield (seq, sample_idx, force_v, accel_g) from raw CSV, skipping metadata."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                return
            if not line.startswith("#"):
                f.seek(pos)
                break
        reader = csv.DictReader(f)
        for row in reader:
            try:
                seq = int(row.get("seq", "0"))
                idx = int(row.get("sample_idx", "0"))
                force_v = float(row.get("force_v", "0"))
                accel_g = float(row.get("accel_g", "0"))
            except Exception:
                continue
            yield seq, idx, force_v, accel_g


def load_timeseries_from_raw_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct continuous force/accel arrays from the raw CSV.

    This sorts by (seq, sample_idx) implicitly by streaming order and pads gaps
    per block.
    """
    force_all = []
    accel_all = []

    cur_seq: Optional[int] = None
    cur_force = []
    cur_accel = []

    def flush_block():
        nonlocal cur_seq, cur_force, cur_accel
        if cur_seq is None:
            return
        n = max(len(cur_force), len(cur_accel))
        if n <= 0:
            cur_seq = None
            cur_force = []
            cur_accel = []
            return
        f = np.asarray(cur_force + [np.nan] * (n - len(cur_force)), dtype=np.float32)
        a = np.asarray(cur_accel + [np.nan] * (n - len(cur_accel)), dtype=np.float32)
        f[~np.isfinite(f)] = 0.0
        a[~np.isfinite(a)] = 0.0
        force_all.append(f)
        accel_all.append(a)
        cur_seq = None
        cur_force = []
        cur_accel = []

    for seq, idx, force_v, accel_g in _iter_samples_from_raw_csv(path):
        if cur_seq is None:
            cur_seq = seq
        if seq != cur_seq:
            flush_block()
            cur_seq = seq

        if idx < 0:
            idx = 0
        if idx >= len(cur_force):
            cur_force.extend([np.nan] * (idx + 1 - len(cur_force)))
            cur_accel.extend([np.nan] * (idx + 1 - len(cur_accel)))
        cur_force[idx] = force_v
        cur_accel[idx] = accel_g

    flush_block()

    if not force_all:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(force_all), np.concatenate(accel_all)


def _sliding_windows(x: np.ndarray, win: int, stride: int) -> Iterator[Tuple[int, np.ndarray]]:
    n = int(x.size)
    win = int(win)
    stride = int(stride)
    if win <= 0 or stride <= 0:
        return
    for start in range(0, n - win + 1, stride):
        yield start, x[start:start + win]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = np.asarray(y_pred, dtype=np.float32) - np.asarray(y_true, dtype=np.float32)
    return float(np.sqrt(np.mean(e ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float32)
    p = np.asarray(y_pred, dtype=np.float32)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def highpass_ema(x: np.ndarray, fs: float, cutoff_hz: float = 1.0) -> np.ndarray:
    """Simple 1st-order high-pass: x - lowpass(EMA).

    cutoff is approximate. Good enough to remove drift for loop visualization.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    cutoff_hz = float(cutoff_hz)
    fs = float(fs)
    if cutoff_hz <= 0.0 or fs <= 0.0:
        return x - float(np.mean(x))

    # EMA alpha from RC low-pass: alpha = exp(-2*pi*fc/fs)
    alpha = float(np.exp(-2.0 * np.pi * cutoff_hz / fs))
    lp = np.zeros_like(x)
    lp0 = float(x[0])
    for i in range(x.size):
        lp0 = alpha * lp0 + (1.0 - alpha) * float(x[i])
        lp[i] = lp0
    return x - lp


def integrate_trapezoid(a: np.ndarray, fs: float) -> np.ndarray:
    """Cumulative integral with trapezoid rule (no scipy)."""
    a = np.asarray(a, dtype=np.float32)
    if a.size == 0:
        return a
    dt = 1.0 / float(fs)
    out = np.zeros_like(a)
    acc = 0.0
    prev = float(a[0])
    for i in range(1, a.size):
        cur = float(a[i])
        acc += 0.5 * (prev + cur) * dt
        out[i] = acc
        prev = cur
    return out


def integrate_simpson(a: np.ndarray, fs: float) -> np.ndarray:
    """Cumulative integral using Simpson's rule on equispaced samples.

    Falls back to trapezoid on the last sample when the length is even.
    """
    a = np.asarray(a, dtype=np.float32)
    if a.size == 0:
        return a
    dt = 1.0 / float(fs)
    n = int(a.size)
    out = np.zeros((n,), dtype=np.float32)
    if n == 1:
        return out

    out[1] = 0.5 * (float(a[0]) + float(a[1])) * dt
    i = 2
    while i < n:
        if i + 1 < n:
            inc = (dt / 3.0) * (float(a[i - 2]) + 4.0 * float(a[i - 1]) + float(a[i]))
            out[i] = float(out[i - 2]) + float(inc)
            out[i - 1] = 0.5 * (float(out[i - 2]) + float(out[i]))
            i += 2
        else:
            out[i] = float(out[i - 1]) + 0.5 * (float(a[i - 1]) + float(a[i])) * dt
            i += 1

    for k in range(2, n):
        if out[k] == 0.0:
            out[k] = out[k - 1]
    return out


def loop_area_shoelace(x: np.ndarray, y: np.ndarray) -> float:
    """Area of closed curve using the shoelace formula."""
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    n = min(x.size, y.size)
    if n < 3:
        return 0.0
    x = x[:n]
    y = y[:n]
    return float(0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1]))


def loop_area_green_simpson(x: np.ndarray, y: np.ndarray, fs: float) -> float:
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    n = min(x.size, y.size)
    if n < 3:
        return 0.0
    x = x[:n]
    y = y[:n]

    dt = 1.0 / float(fs)
    dx = np.zeros((n,), dtype=np.float32)
    dy = np.zeros((n,), dtype=np.float32)
    dx[1:-1] = (x[2:] - x[:-2]) * 0.5
    dy[1:-1] = (y[2:] - y[:-2]) * 0.5
    dx[0] = x[1] - x[0]
    dy[0] = y[1] - y[0]
    dx[-1] = x[-1] - x[-2]
    dy[-1] = y[-1] - y[-2]

    integrand = x * dy - y * dx
    m = int(integrand.size)
    if m < 3:
        return 0.0
    if (m % 2) == 0:
        integrand = integrand[:-1]
        m -= 1
    s_odd = float(np.sum(integrand[1:m - 1:2]))
    s_even = float(np.sum(integrand[2:m - 1:2]))
    area = 0.5 * (dt / 3.0) * (float(integrand[0]) + float(integrand[m - 1]) + 4.0 * s_odd + 2.0 * s_even)
    return float(abs(area))


def _resample_polyline(x: np.ndarray, y: np.ndarray, n_pts: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    n = int(min(x.size, y.size))
    if n <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n_pts <= 1 or n <= 1:
        return np.stack([x[:1], y[:1]], axis=-1)
    n_pts = int(min(n_pts, n))
    idx = np.linspace(0, n - 1, n_pts)
    i0 = np.floor(idx).astype(np.int32)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (idx - i0.astype(np.float32)).astype(np.float32)
    xs = (1.0 - w) * x[i0] + w * x[i1]
    ys = (1.0 - w) * y[i0] + w * y[i1]
    return np.stack([xs, ys], axis=-1).astype(np.float32)


def frechet_distance_discrete(P: np.ndarray, Q: np.ndarray) -> float:
    P = np.asarray(P, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)
    if P.ndim != 2 or Q.ndim != 2 or P.shape[1] != 2 or Q.shape[1] != 2:
        return 0.0
    n = int(P.shape[0])
    m = int(Q.shape[0])
    if n == 0 or m == 0:
        return 0.0

    ca = np.full((n, m), -1.0, dtype=np.float32)

    def dist(i: int, j: int) -> float:
        dx = float(P[i, 0] - Q[j, 0])
        dy = float(P[i, 1] - Q[j, 1])
        return float(np.hypot(dx, dy))

    for i in range(n):
        for j in range(m):
            d = dist(i, j)
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i > 0 and j == 0:
                ca[i, j] = max(float(ca[i - 1, 0]), d)
            elif i == 0 and j > 0:
                ca[i, j] = max(float(ca[0, j - 1]), d)
            else:
                ca[i, j] = max(min(float(ca[i - 1, j]), float(ca[i - 1, j - 1]), float(ca[i, j - 1])), d)
    return float(ca[n - 1, m - 1])


def residual_stats(res: np.ndarray) -> Dict[str, float]:
    res = np.asarray(res, dtype=np.float32).ravel()
    n = int(res.size)
    if n <= 0:
        return {"n": 0.0, "mean": 0.0, "std": 0.0, "rmse": 0.0, "skew": 0.0, "kurt": 0.0}
    mu = float(np.mean(res))
    sigma = float(np.std(res))
    rmse_v = float(np.sqrt(np.mean(res ** 2)))
    if sigma < 1e-12:
        skew = 0.0
        kurt = 0.0
    else:
        z = (res - mu) / sigma
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4))
    return {"n": float(n), "mean": float(mu), "std": float(sigma), "rmse": float(rmse_v), "skew": float(skew), "kurt": float(kurt)}


def _draw_axes(draw, rect: Tuple[int, int, int, int], fg: Tuple[int, int, int]):
    x0, y0, x1, y1 = rect
    draw.rectangle([x0, y0, x1, y1], outline=fg, width=1)


def _polyline_to_pixels(x: np.ndarray, y: np.ndarray, rect: Tuple[int, int, int, int]) -> Optional[list]:
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    n = int(min(x.size, y.size))
    if n < 2:
        return None
    x = x[:n]
    y = y[:n]
    x0, y0, x1, y1 = rect
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    if abs(xmax - xmin) < 1e-12:
        xmax = xmin + 1.0
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1.0
    px = x0 + (x - xmin) * float(x1 - x0) / float(xmax - xmin)
    py = y1 - (y - ymin) * float(y1 - y0) / float(ymax - ymin)
    return [(int(px[i]), int(py[i])) for i in range(n)]


def render_loop_page(title: str, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, size: Tuple[int, int] = (1240, 1754)):
    if Image is None:
        return None
    w, h = int(size[0]), int(size[1])
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default() if ImageFont is not None else None
    draw.text((40, 30), str(title), fill=(0, 0, 0), font=font)
    rect = (80, 120, w - 80, h - 120)
    _draw_axes(draw, rect, fg=(0, 0, 0))
    pts_t = _polyline_to_pixels(x, y_true, rect)
    pts_p = _polyline_to_pixels(x, y_pred, rect)
    if pts_t:
        draw.line(pts_t, fill=(30, 90, 200), width=2)
    if pts_p:
        draw.line(pts_p, fill=(200, 60, 40), width=2)
    draw.text((90, 90), "true", fill=(30, 90, 200), font=font)
    draw.text((160, 90), "pred", fill=(200, 60, 40), font=font)
    return img


def render_text_page(title: str, lines: list, size: Tuple[int, int] = (1240, 1754)):
    if Image is None:
        return None
    w, h = int(size[0]), int(size[1])
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default() if ImageFont is not None else None
    draw.text((40, 30), str(title), fill=(0, 0, 0), font=font)
    y = 80
    for ln in lines:
        draw.text((40, y), str(ln), fill=(0, 0, 0), font=font)
        y += 18
        if y > (h - 40):
            break
    return img


def render_hist_page(title: str, bins: np.ndarray, counts: np.ndarray, stats: Dict[str, float], size: Tuple[int, int] = (1240, 1754)):
    if Image is None:
        return None
    w, h = int(size[0]), int(size[1])
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default() if ImageFont is not None else None
    draw.text((40, 30), str(title), fill=(0, 0, 0), font=font)
    draw.text((40, 60), f"n={int(stats.get('n', 0))}  mean={stats.get('mean', 0.0):.6g}  std={stats.get('std', 0.0):.6g}  rmse={stats.get('rmse', 0.0):.6g}", fill=(0, 0, 0), font=font)
    draw.text((40, 80), f"skew={stats.get('skew', 0.0):.6g}  kurt={stats.get('kurt', 0.0):.6g}", fill=(0, 0, 0), font=font)
    rect = (80, 140, w - 80, h - 120)
    _draw_axes(draw, rect, fg=(0, 0, 0))
    bins = np.asarray(bins, dtype=np.float32).ravel()
    counts = np.asarray(counts, dtype=np.float32).ravel()
    if bins.size >= 2 and counts.size >= 1:
        nb = int(min(counts.size, bins.size - 1))
        cmax = float(np.max(counts[:nb])) if nb > 0 else 1.0
        if cmax <= 0.0:
            cmax = 1.0
        x0, y0, x1, y1 = rect
        for i in range(nb):
            frac0 = float(i) / float(nb)
            frac1 = float(i + 1) / float(nb)
            bx0 = int(x0 + frac0 * float(x1 - x0))
            bx1 = int(x0 + frac1 * float(x1 - x0))
            bh = int(float(y1 - y0) * (float(counts[i]) / cmax))
            draw.rectangle([bx0, y1 - bh, bx1 - 1, y1], fill=(120, 120, 160), outline=None)
    return img


def write_pdf(pages: list, out_pdf: str) -> bool:
    if Image is None:
        return False
    pages = [p for p in pages if p is not None]
    if not pages:
        return False
    first = pages[0]
    rest = pages[1:]
    first.save(out_pdf, "PDF", resolution=150.0, save_all=True, append_images=rest)
    return True


class ModelRunner:
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MockRunner(ModelRunner):
    def __init__(self, mode: str = "copy_force", gain: float = 1.0, bias: float = 0.0):
        self.mode = str(mode)
        self.gain = float(gain)
        self.bias = float(bias)

    def predict(self, x: np.ndarray) -> np.ndarray:
        # x: [B, T, C] where channels: force, accel (or user defined)
        if self.mode == "zero":
            y = np.zeros((x.shape[0], x.shape[1], 1), dtype=np.float32)
        elif self.mode == "copy_force":
            y = x[..., :1].astype(np.float32)
        elif self.mode == "copy_accel":
            y = x[..., 1:2].astype(np.float32)
        else:
            y = x[..., :1].astype(np.float32)
        return y * self.gain + self.bias


def build_runner(args) -> ModelRunner:
    # Placeholder: real Hailo runner will be enabled when hailort/hailo_platform are installed.
    return MockRunner(mode=args.mock_mode, gain=args.mock_gain, bias=args.mock_bias)


def main():
    ap = argparse.ArgumentParser(description="Offline validator for Hailo hysteresis models (with mock runner fallback)")
    ap.add_argument("--raw-csv", required=True, help="Raw log CSV (cdaq_remote_*.csv)")
    ap.add_argument("--fs", type=float, default=2500.0, help="Sample rate (Hz)")
    ap.add_argument("--window", type=int, default=500, help="Window length in samples")
    ap.add_argument("--stride", type=int, default=250, help="Stride in samples")
    ap.add_argument("--target-shift", type=int, default=0, help="Shift ground truth by N samples (prediction horizon)")
    ap.add_argument("--norm-json", default="", help="Normalization JSON path (optional)")
    ap.add_argument("--channel-order", default="force,accel", help="Order in model input: 'force,accel' or 'accel,force'")

    ap.add_argument("--hef", default="", help="Path to .hef (optional; requires hailort/hailo_platform)")

    ap.add_argument("--mock-mode", default="copy_force", choices=["copy_force", "copy_accel", "zero"],
                    help="Mock runner mode when Hailo runtime is unavailable")
    ap.add_argument("--mock-gain", type=float, default=1.0)
    ap.add_argument("--mock-bias", type=float, default=0.0)

    ap.add_argument("--loop-x", default="vel", choices=["accel", "vel", "disp"],
                    help="X-axis variable for loop area")
    ap.add_argument("--hp-cutoff-hz", type=float, default=1.0, help="High-pass cutoff (Hz) for integrated signals")
    ap.add_argument("--integrator", default="simpson", choices=["simpson", "trapezoid"],
                    help="Integrator for velocity/displacement reconstruction")

    ap.add_argument("--frechet-points", type=int, default=64,
                    help="Points for discrete Frechet distance (downsample per window)")
    ap.add_argument("--residual-max-samples", type=int, default=200000,
                    help="Max residual samples stored for histogram/stats")
    ap.add_argument("--residual-bins", type=int, default=120,
                    help="Histogram bins for residual analysis")

    ap.add_argument("--out-csv", default="", help="Output metrics CSV (auto if empty)")
    ap.add_argument("--out-npz", default="", help="Output NPZ with some loops (auto if empty)")
    ap.add_argument("--out-pdf", default="", help="Output PDF report (auto if empty; requires Pillow)")
    ap.add_argument("--no-pdf", action="store_true", help="Disable PDF generation")
    ap.add_argument("--pdf-n-loops", type=int, default=12, help="How many loop pages to include in PDF")
    ap.add_argument("--save-n-loops", type=int, default=50, help="Number of windows to store in NPZ")

    args = ap.parse_args()

    raw_csv = args.raw_csv
    if not os.path.isfile(raw_csv):
        raise SystemExit(f"Raw CSV not found: {raw_csv}")

    fs = float(args.fs)
    win = int(args.window)
    stride = int(args.stride)
    shift = int(args.target_shift)

    base = os.path.splitext(os.path.basename(raw_csv))[0]
    out_dir = os.path.dirname(os.path.abspath(raw_csv))
    out_csv = args.out_csv or os.path.join(out_dir, f"{base}_hailo_validate.csv")
    out_npz = args.out_npz or os.path.join(out_dir, f"{base}_hailo_loops.npz")
    out_pdf = args.out_pdf or os.path.join(out_dir, f"{base}_hailo_report.pdf")

    force, accel = load_timeseries_from_raw_csv(raw_csv)
    if force.size == 0:
        raise SystemExit("No samples loaded from raw CSV")

    # Build model input [N,2]
    order = [s.strip().lower() for s in str(args.channel_order).split(",") if s.strip()]
    if order not in (["force", "accel"], ["accel", "force"]):
        raise SystemExit("--channel-order must be 'force,accel' or 'accel,force'")

    if order == ["force", "accel"]:
        x_full = np.stack([force, accel], axis=-1)
    else:
        x_full = np.stack([accel, force], axis=-1)

    n_in = 2
    n_out = 1
    norm = None
    if args.norm_json:
        norm = _load_norm_spec(args.norm_json, n_in=n_in, n_out=n_out)

    runner = build_runner(args)

    rows = []
    loops_x = []
    loops_y_true = []
    loops_y_pred = []
    loop_starts = []
    residual_samples = []
    residual_seen = 0

    # Precompute loop variable from accel if needed
    if args.loop_x == "accel":
        loop_x_full = accel.astype(np.float32)
    else:
        integ = integrate_simpson if str(args.integrator).lower() == "simpson" else integrate_trapezoid
        v = integ(accel, fs=fs)
        v = highpass_ema(v, fs=fs, cutoff_hz=float(args.hp_cutoff_hz))
        if args.loop_x == "vel":
            loop_x_full = v
        else:
            d = integ(v, fs=fs)
            d = highpass_ema(d, fs=fs, cutoff_hz=float(args.hp_cutoff_hz))
            loop_x_full = d

    # Window loop
    for start, x_win in _sliding_windows(x_full, win=win, stride=stride):
        # Target force is always the measured force channel (not re-ordered)
        y_true = force[start:start + win]
        if shift != 0:
            s0 = start + shift
            s1 = s0 + win
            if s0 < 0 or s1 > force.size:
                continue
            y_true = force[s0:s1]

        x_in = x_win.astype(np.float32)[None, :, :]  # [1,T,2]
        if norm is not None:
            x_in = (x_in - norm.in_mean) / norm.in_std

        y_pred = runner.predict(x_in)  # [1,T,1] (mock)
        if norm is not None and (norm.out_mean is not None) and (norm.out_std is not None):
            y_pred = y_pred * norm.out_std + norm.out_mean

        y_pred_1d = np.asarray(y_pred[0, :, 0], dtype=np.float32)

        m_rmse = rmse(y_true, y_pred_1d)
        m_r2 = r2_score(y_true, y_pred_1d)

        # Residual sampling (bounded memory)
        res = (y_pred_1d - np.asarray(y_true, dtype=np.float32)).astype(np.float32)
        max_res = int(args.residual_max_samples)
        if max_res > 0:
            for rv in res.ravel():
                residual_seen += 1
                if len(residual_samples) < max_res:
                    residual_samples.append(float(rv))
                else:
                    j = random.randrange(residual_seen)
                    if j < max_res:
                        residual_samples[j] = float(rv)

        x_loop = loop_x_full[start:start + win]
        if shift != 0:
            s0 = start + shift
            s1 = s0 + win
            if s0 < 0 or s1 > loop_x_full.size:
                continue
            x_loop = loop_x_full[s0:s1]

        area_true = loop_area_shoelace(x_loop, y_true)
        area_pred = loop_area_shoelace(x_loop, y_pred_1d)
        area_err = 0.0
        if abs(area_true) > 1e-12:
            area_err = float((area_pred - area_true) / area_true)

        area_true_simpson = loop_area_green_simpson(x_loop, y_true, fs=fs)
        area_pred_simpson = loop_area_green_simpson(x_loop, y_pred_1d, fs=fs)
        area_err_simpson = 0.0
        if abs(area_true_simpson) > 1e-12:
            area_err_simpson = float((area_pred_simpson - area_true_simpson) / area_true_simpson)

        n_f = int(args.frechet_points)
        if n_f > 1:
            P = _resample_polyline(x_loop, y_true, n_pts=n_f)
            Q = _resample_polyline(x_loop, y_pred_1d, n_pts=n_f)
            frechet = frechet_distance_discrete(P, Q)
        else:
            frechet = 0.0

        rows.append({
            "start_idx": int(start),
            "rmse": float(m_rmse),
            "r2": float(m_r2),
            "area_true": float(area_true),
            "area_pred": float(area_pred),
            "area_err_rel": float(area_err),
            "area_true_simpson": float(area_true_simpson),
            "area_pred_simpson": float(area_pred_simpson),
            "area_err_rel_simpson": float(area_err_simpson),
            "frechet": float(frechet),
        })

        if len(loops_x) < int(args.save_n_loops):
            loops_x.append(np.asarray(x_loop, dtype=np.float32))
            loops_y_true.append(np.asarray(y_true, dtype=np.float32))
            loops_y_pred.append(np.asarray(y_pred_1d, dtype=np.float32))
            loop_starts.append(int(start))

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "start_idx",
                "rmse",
                "r2",
                "area_true",
                "area_pred",
                "area_err_rel",
                "area_true_simpson",
                "area_pred_simpson",
                "area_err_rel_simpson",
                "frechet",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    np.savez(
        out_npz,
        loop_x=np.asarray(loops_x, dtype=object),
        loop_force_true=np.asarray(loops_y_true, dtype=object),
        loop_force_pred=np.asarray(loops_y_pred, dtype=object),
        loop_start_idx=np.asarray(loop_starts, dtype=np.int32),
        fs=np.float32(fs),
        window=np.int32(win),
        stride=np.int32(stride),
        loop_x_mode=str(args.loop_x),
    )

    if rows:
        all_rmse = float(np.mean([r["rmse"] for r in rows]))
        all_r2 = float(np.mean([r["r2"] for r in rows]))
        all_frechet = float(np.mean([r["frechet"] for r in rows]))
        all_area_rel = float(np.mean([r["area_err_rel"] for r in rows]))
        all_area_rel_s = float(np.mean([r["area_err_rel_simpson"] for r in rows]))
        print(
            f"windows={len(rows)}  rmse_mean={all_rmse:.6g}  r2_mean={all_r2:.6g}  "
            f"frechet_mean={all_frechet:.6g}  area_err_rel_mean={all_area_rel:.6g}  area_err_rel_simpson_mean={all_area_rel_s:.6g}"
        )
    print(f"wrote: {out_csv}")
    print(f"wrote: {out_npz}")

    # PDF report
    if (not bool(args.no_pdf)) and (Image is not None):
        pages = []
        res_arr = np.asarray(residual_samples, dtype=np.float32)
        st = residual_stats(res_arr)
        lines = [
            f"raw_csv={raw_csv}",
            f"fs={fs}  window={win}  stride={stride}  shift={shift}",
            f"channel_order={args.channel_order}  loop_x={args.loop_x}  integrator={args.integrator}",
            f"windows={len(rows)}",
        ]
        if rows:
            lines.append(f"rmse_mean={all_rmse:.6g}  r2_mean={all_r2:.6g}  frechet_mean={all_frechet:.6g}")
            lines.append(f"area_err_rel_mean={all_area_rel:.6g}  area_err_rel_simpson_mean={all_area_rel_s:.6g}")
        lines.append(f"residual_mean={st.get('mean', 0.0):.6g}  residual_std={st.get('std', 0.0):.6g}  residual_rmse={st.get('rmse', 0.0):.6g}")
        lines.append(f"residual_skew={st.get('skew', 0.0):.6g}  residual_kurt={st.get('kurt', 0.0):.6g}")
        pages.append(render_text_page("Hailo Validation Report", lines))

        if res_arr.size > 0:
            nb = max(10, int(args.residual_bins))
            counts, bins = np.histogram(res_arr, bins=nb)
            pages.append(render_hist_page("Residual Histogram", bins=bins, counts=counts, stats=st))

        if loops_x:
            n_pages = max(0, int(args.pdf_n_loops))
            n_pages = min(n_pages, len(loops_x))
            if n_pages > 0:
                idxs = np.linspace(0, len(loops_x) - 1, n_pages).round().astype(np.int32)
                for k in idxs.tolist():
                    pages.append(
                        render_loop_page(
                            f"Loop start_idx={loop_starts[k]}",
                            x=loops_x[k],
                            y_true=loops_y_true[k],
                            y_pred=loops_y_pred[k],
                        )
                    )

        ok = write_pdf(pages, out_pdf=out_pdf)
        if ok:
            print(f"wrote: {out_pdf}")
    elif (not bool(args.no_pdf)) and (Image is None):
        print("note: Pillow not available; skipping PDF report")


if __name__ == "__main__":
    main()
