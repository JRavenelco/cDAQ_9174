#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline feature extraction for shaker-based bench characterization.
- Pairs fuerza/vibracion CSVs saved by interfaz_DAQ_V2.py (sesion_YYYYMMDD_HHMMSS_*.csv)
- Computes time-domain and frequency-domain features on acceleration (ai0) and low-frequency features on force (ai0).
- Estimates FRF H1 and coherence where possible (limited by quasi-static force sensor bandwidth).
- Saves summary CSV and plots (PSD, FRF/Coherence) into analisis_resultados/.

Usage examples:
  python offline_feature_extract.py --from 20251006 --to 20251008
  python offline_feature_extract.py --data-dir Pruebas/src/datos_automaticos --save-dir Pruebas/src/analisis_resultados
"""
import os
import re
import csv
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew, spearmanr
import matplotlib.pyplot as plt

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), 'datos_automaticos')
DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'analisis_resultados')

# Analysis band and parameters
BAND_LO = 10.0
BAND_HI = 120.0
COH_THRESH = 0.3
TOPK_FRF = 3

@dataclass
class SessionFiles:
    base: str              # 'sesion_YYYYMMDD_HHMMSS'
    fecha: str             # 'YYYYMMDD'
    hora: str              # 'HHMMSS'
    fuerza_path: str
    vib_path: str


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def parse_base_from_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    # Expect names like: sesion_YYYYMMDD_HHMMSS_fuerza.csv
    m = re.match(r"(sesion_(\d{8})_(\d{6}))_.*\\.csv$", filename)
    if not m:
        m = re.match(r"(sesion_(\d{8})_(\d{6}))_.*\.csv$", filename)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def list_sessions(data_dir: str, date_from: Optional[str], date_to: Optional[str]) -> List[SessionFiles]:
    fuerza_files: Dict[str, str] = {}
    vib_files: Dict[str, str] = {}
    for root, _, files in os.walk(data_dir):
        for f in files:
            if not f.endswith('.csv'): continue
            meta = parse_base_from_filename(f)
            if not meta: continue
            base, fecha, hora = meta
            full = os.path.join(root, f)
            if f.endswith('_fuerza.csv'):
                fuerza_files[base] = full
            elif '_vibracion' in f and f.endswith('.csv'):
                vib_files[base] = full
    bases = sorted(set(fuerza_files.keys()) | set(vib_files.keys()))
    sessions: List[SessionFiles] = []
    for base in bases:
        meta = re.match(r"(sesion_(\d{8})_(\d{6}))$", base)
        if not meta: continue
        fecha, hora = meta.group(2), meta.group(3)
        if date_from and fecha < date_from: continue
        if date_to and fecha > date_to: continue
        if base not in fuerza_files or base not in vib_files:
            continue  # only process paired sessions
        sessions.append(SessionFiles(base=base, fecha=fecha, hora=hora,
                                     fuerza_path=fuerza_files[base], vib_path=vib_files[base]))
    return sorted(sessions, key=lambda s: (s.fecha, s.hora))


def read_csv_with_header(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()
    header = [h.strip() for h in header_line.split(',')]
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return header, data


def pick_col_index(header: List[str], startswith: str) -> int:
    for i, h in enumerate(header):
        if i == 0:  # skip time
            continue
        if h.startswith(startswith):
            return i
    # fallback to first non-time col
    return 1 if len(header) > 1 else 0


def time_features(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x)
    if x.size == 0:
        return {k: np.nan for k in ['mean','std','rms','p2p','max','min','crest','kurt','skew','zcr']}
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    rms = float(np.sqrt(np.mean(x**2)))
    p2p = float(np.ptp(x))
    xmax = float(np.max(x))
    xmin = float(np.min(x))
    crest = float((abs(xmax) if abs(xmax) > abs(xmin) else abs(xmin)) / (rms + 1e-12))
    k = float(kurtosis(x, fisher=True, bias=False)) if x.size > 3 else np.nan
    s = float(skew(x, bias=False)) if x.size > 2 else np.nan
    zc = float(((x[:-1] * x[1:]) < 0).sum() / max(len(x)-1,1))
    return {'mean':mean,'std':std,'rms':rms,'p2p':p2p,'max':xmax,'min':xmin,'crest':crest,'kurt':k,'skew':s,'zcr':zc}


def welch_psd(x: np.ndarray, fs: float, nperseg: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:
    if nperseg is None:
        nperseg = min(len(x), 4096)
        if nperseg < 256:
            nperseg = max(64, len(x)//4)
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg//2, window='hann', detrend='constant')
    return f, Pxx


def bandpower(f: np.ndarray, Pxx: np.ndarray, fmin: float, fmax: float) -> float:
    idx = (f >= fmin) & (f <= fmax)
    if not np.any(idx):
        return 0.0
    # numpy.trapz deprecated; use trapezoid
    return float(np.trapezoid(Pxx[idx], f[idx]))


def top_peaks(f: np.ndarray, Pxx: np.ndarray, k: int = 3) -> List[Tuple[float, float]]:
    peaks, props = signal.find_peaks(Pxx)
    if peaks.size == 0:
        return []
    peak_vals = Pxx[peaks]
    order = np.argsort(peak_vals)[::-1][:k]
    return [(float(f[peaks[i]]), float(Pxx[peaks[i]])) for i in order]


def robust_minmax(x: np.ndarray, q_low: float = 0.01, q_high: float = 0.99) -> np.ndarray:
    """Scale to [-1, 1] using robust percentiles to avoid outliers."""
    x = np.asarray(x)
    if x.size == 0:
        return x
    lo, hi = np.quantile(x, [q_low, q_high])
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    xn = 2.0 * (x - lo) / (hi - lo) - 1.0
    return np.clip(xn, -1.0, 1.0)


def segment_cycles_by_zc(accel_norm: np.ndarray, fs: float) -> List[Tuple[int, int]]:
    """Segment cycles using zero-crossings (positive slope) on normalized acceleration.
    Returns list of (start_idx, end_idx) per cycle. Requires fs>0.
    """
    cycles: List[Tuple[int, int]] = []
    if accel_norm.size < 4 or not np.isfinite(fs) or fs <= 0:
        return cycles
    # positive-slope zero crossings
    x = accel_norm
    zc = np.where((x[:-1] < 0) & (x[1:] >= 0))[0] + 1
    if zc.size < 2:
        return cycles
    # expected samples per cycle for 20-150 Hz
    min_samp = max(8, int(fs / 150.0))
    max_samp = max(min_samp + 1, int(fs / 20.0 * 2.0))  # allow up to ~2 periods of 20 Hz
    prev = zc[0]
    for idx in zc[1:]:
        L = idx - prev
        if L >= min_samp and L <= max_samp:
            cycles.append((prev, idx))
        prev = idx
    return cycles


def lissajous_area(Fn: np.ndarray, An: np.ndarray) -> float:
    """Compute signed loop area ∮ F dA using discrete path integral."""
    if Fn.size < 2 or An.size < 2:
        return np.nan
    return float(np.abs(np.sum(Fn[:-1] * (An[1:] - An[:-1]))))


def poly_fit_metrics(An: np.ndarray, Fn: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Fit linear and cubic models Fn ~ a*An + b and Fn ~ c3*An^3 + c1*An + c0; return coefs and R2."""
    if An.size < 3 or Fn.size < 3:
        return np.array([np.nan, np.nan]), np.nan, np.array([np.nan, np.nan, np.nan, np.nan]), np.nan
    # Linear fit
    lin_coefs = np.polyfit(An, Fn, 1)  # [a1, a0]
    Fn_lin = np.polyval(lin_coefs, An)
    ss_res = float(np.sum((Fn - Fn_lin) ** 2))
    ss_tot = float(np.sum((Fn - np.mean(Fn)) ** 2)) + 1e-18
    lin_r2 = 1.0 - ss_res / ss_tot
    # Cubic fit
    cubic_coefs = np.polyfit(An, Fn, 3)  # [c3, c2, c1, c0]
    Fn_cub = np.polyval(cubic_coefs, An)
    ss_res_c = float(np.sum((Fn - Fn_cub) ** 2))
    cubic_r2 = 1.0 - ss_res_c / ss_tot
    return lin_coefs, lin_r2, cubic_coefs, cubic_r2


def frf_h1(force: np.ndarray, accel: np.ndarray, fs: float, nperseg: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if nperseg is None:
        nperseg = min(len(force), len(accel), 4096)
        if nperseg < 256:
            nperseg = max(64, min(len(force), len(accel))//4)
    f, Pxx = signal.welch(force, fs=fs, nperseg=nperseg, noverlap=nperseg//2, window='hann', detrend='constant')
    f2, Pyy = signal.welch(accel, fs=fs, nperseg=nperseg, noverlap=nperseg//2, window='hann', detrend='constant')
    f3, Pxy = signal.csd(force, accel, fs=fs, nperseg=nperseg, noverlap=nperseg//2, window='hann', detrend='constant')
    # align
    m = min(len(f), len(f2), len(f3))
    f = f[:m]; Pxx = Pxx[:m]; Pyy = Pyy[:m]; Pxy = Pxy[:m]
    H1 = Pxy / (Pxx + 1e-18)
    coh = (np.abs(Pxy)**2) / ((Pxx * Pyy) + 1e-18)
    return f, H1, np.clip(coh.real, 0, 1)


def bandpass_band(x: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase band-pass [BAND_LO, BAND_HI] if feasible, otherwise detrend."""
    if x.size == 0 or not np.isfinite(fs) or fs <= 0:
        return x
    nyq = 0.5 * fs
    lo = float(BAND_LO) / nyq
    hi = float(BAND_HI) / nyq
    if hi >= 1.0 or lo <= 0 or hi <= lo:
        return signal.detrend(x, type='constant')
    sos = signal.butter(4, [lo, hi], btype='band', output='sos')
    try:
        return signal.sosfiltfilt(sos, x)
    except Exception:
        return signal.detrend(x, type='constant')


def process_session(sess: SessionFiles, save_dir: str) -> Dict[str, float]:
    # Load fuerza
    hF, dF = read_csv_with_header(sess.fuerza_path)
    hA, dA = read_csv_with_header(sess.vib_path)
    # Expect time in col 0
    tF = dF[:,0] if dF.shape[1] > 0 else np.array([])
    tA = dA[:,0] if dA.shape[1] > 0 else np.array([])
    fsF = 1.0/np.median(np.diff(tF)) if tF.size > 1 else np.nan
    fsA = 1.0/np.median(np.diff(tA)) if tA.size > 1 else np.nan
    # Pick ai0 columns
    iF = pick_col_index(hF, 'Fuerza_')
    iA = pick_col_index(hA, 'Vibracion_')
    F = dF[:, iF] if dF.shape[1] > iF else np.array([])
    A = dA[:, iA] if dA.shape[1] > iA else np.array([])

    # Basic cleaning
    def clean(x):
        x = np.asarray(x)
        x = x[~np.isnan(x)] if x.size else x
        return x
    F = clean(F); A = clean(A)

    # Detrend (remove DC) but keep 20–150 Hz content for FRF
    if A.size > 0:
        A = signal.detrend(A, type='constant')
    if F.size > 0:
        F = signal.detrend(F, type='constant')

    # If SRs differ, resample A to force SR (or vice versa) for FRF/coherence
    A_rs = A.copy()
    fs = fsA
    if A.size > 0 and F.size > 0 and (not np.isnan(fsA)) and (not np.isnan(fsF)) and abs(fsA - fsF) > 1e-3:
        # resample acceleration to force sampling for coherence below low frequency
        target_N = int(round(len(A) * (fsF/fsA)))
        if target_N > 10:
            A_rs = signal.resample(A, target_N)
            fs = fsF
        else:
            fs = min(fsA, fsF)
    else:
        fs = fsA if not np.isnan(fsA) else fsF

    # Features
    feats_acc_t = time_features(A)
    feats_force_t = time_features(F)

    # PSD of acceleration
    if A.size > 0 and not np.isnan(fsA):
        fA, PxxA = welch_psd(A, fsA)
        peaks = top_peaks(fA, PxxA, k=3)
        # Bandpowers (adaptive): 0-10, 10-50, 50-200, 200-1000, 1k-5k etc up to Nyquist
        bands = [(0,10),(10,50),(50,200),(200,500),(500,1000),(1000,2000),(2000,5000)]
        band_powers = {f"bp_{lo}_{hi}": bandpower(fA, PxxA, lo, min(hi, float(fsA)/2.0)) for lo,hi in bands if lo < float(fsA)/2.0}
    else:
        fA = np.array([]); PxxA = np.array([]); peaks = []; band_powers = {}

    # FRF & coherence in [BAND_LO, BAND_HI] Hz (per user focus)
    frf_freq = np.array([]); H1 = np.array([]); coh = np.array([])
    coh_peak_band = np.nan; frf_mag_at_peak_band = np.nan; phase_deg_at_peak_band = np.nan; lag_samples_band = np.nan
    frf_pk_freqs = [np.nan]*TOPK_FRF
    frf_pk_coh = [np.nan]*TOPK_FRF
    frf_pk_mag_db = [np.nan]*TOPK_FRF
    frf_pk_phase = [np.nan]*TOPK_FRF
    if F.size > 0 and A_rs.size > 0 and not np.isnan(fs):
        # Band-pass both channels for better alignment in band
        F_bp = bandpass_band(F, fs)
        A_bp = bandpass_band(A_rs, fs)
        # Align by maximizing cross-correlation in band
        try:
            c = signal.correlate(A_bp, F_bp, mode='full')
            lags = signal.correlation_lags(len(A_bp), len(F_bp), mode='full')
            lag = int(lags[np.argmax(c)])
            lag_samples_band = float(lag)
            if lag > 0:
                A_bp = A_bp[lag:]
                F_bp = F_bp[:len(A_bp)]
            elif lag < 0:
                F_bp = F_bp[-lag:]
                A_bp = A_bp[:len(F_bp)]
        except Exception:
            pass
        # Compute FRF on aligned band-passed signals
        frf_freq, H1, coh = frf_h1(F_bp, A_bp, fs)
        # Evaluate in [BAND_LO, BAND_HI] Hz region
        idx = (frf_freq >= BAND_LO) & (frf_freq <= BAND_HI)
        if np.any(idx):
            kmax = np.argmax(coh[idx])
            coh_peak_band = float(coh[idx][kmax])
            frf_mag_at_peak_band = float(np.abs(H1[idx][kmax]))
            phase_deg_at_peak_band = float(np.angle(H1[idx][kmax], deg=True))
            # Peaks with coherence filter
            band_f = frf_freq[idx]
            band_mag = np.abs(H1[idx])
            band_phase = np.angle(H1[idx], deg=True)
            band_coh = coh[idx]
            valid = band_coh >= COH_THRESH
            if np.any(valid):
                pk_idx, _ = signal.find_peaks(band_mag[valid])
                # Map peak indices back to band arrays
                valid_idx = np.where(valid)[0]
                peak_global_idx = valid_idx[pk_idx]
                order = np.argsort(band_mag[peak_global_idx])[::-1][:TOPK_FRF]
                for j, oi in enumerate(order):
                    i_global = peak_global_idx[oi]
                    frf_pk_freqs[j] = float(band_f[i_global])
                    frf_pk_coh[j] = float(band_coh[i_global])
                    frf_pk_mag_db[j] = float(20*np.log10(band_mag[i_global]+1e-18))
                    frf_pk_phase[j] = float(band_phase[i_global])

    # Prepare aligned band-limited signals for XY (F vs A)
    # Use the same lag alignment computed above
    A_xy = bandpass_band(A_rs if (A_rs.size > 0 and F.size > 0) else A, fs if not np.isnan(fs) else fsA)
    F_xy = bandpass_band(F, fs if not np.isnan(fs) else fsF)
    if np.isfinite(lag_samples_band):
        lag = int(lag_samples_band)
        if lag > 0 and len(A_xy) > lag:
            A_xy = A_xy[lag:]
            F_xy = F_xy[:len(A_xy)]
        elif lag < 0 and len(F_xy) > -lag:
            F_xy = F_xy[-lag:]
            A_xy = A_xy[:len(F_xy)]
    Nxy = min(len(A_xy), len(F_xy))
    if Nxy > 0:
        A_xy = A_xy[:Nxy]
        F_xy = F_xy[:Nxy]
    # Choose sampling rate for cycle segmentation
    fs_xy = fs if (not np.isnan(fs)) else (fsA if not np.isnan(fsA) else fsF)

    # Normalization (robust [-1,1]) and nonlinear relations F vs A
    F_n = robust_minmax(F_xy)
    A_n = robust_minmax(A_xy)
    # Linear/cubic fits and correlations
    if F_n.size > 0 and A_n.size > 0:
        lin_coefs, lin_r2, cubic_coefs, cubic_r2 = poly_fit_metrics(A_n, F_n)
        pear = float(np.corrcoef(A_n, F_n)[0,1]) if A_n.size > 1 else np.nan
        spear = float(spearmanr(A_n, F_n).correlation) if A_n.size > 1 else np.nan
    else:
        lin_coefs = np.array([np.nan, np.nan]); lin_r2 = np.nan
        cubic_coefs = np.array([np.nan, np.nan, np.nan, np.nan]); cubic_r2 = np.nan
        pear = np.nan; spear = np.nan

    # Cycle-based loop areas using acceleration zero-crossings
    Eloop_vals: List[float] = []
    n_cycles = 0
    if A_n.size > 0 and not np.isnan(fs_xy):
        cycles = segment_cycles_by_zc(A_n, fs_xy)
        for (i0, i1) in cycles[:50]:
            el = lissajous_area(F_n[i0:i1], A_n[i0:i1])
            if np.isfinite(el):
                Eloop_vals.append(el)
        n_cycles = len(cycles)
    Eloop_mean = float(np.mean(Eloop_vals)) if Eloop_vals else np.nan
    Eloop_std = float(np.std(Eloop_vals, ddof=1)) if len(Eloop_vals) > 1 else np.nan

    # Save plots
    ensure_dir(save_dir)
    base_out = os.path.join(save_dir, sess.base)
    # PSD plot
    if fA.size > 0:
        plt.figure(figsize=(8,4))
        plt.semilogy(fA, PxxA + 1e-18)
        if peaks:
            for fp, ap in peaks:
                plt.semilogy([fp], [ap+1e-18], 'ro')
        # Highlight analysis band
        plt.axvspan(BAND_LO, BAND_HI, color='orange', alpha=0.1, linewidth=0)
        plt.grid(True, which='both')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('PSD (unit^2/Hz)')
        plt.title(f'PSD Aceleración (ai0) - {sess.fecha} {sess.hora}')
        plt.tight_layout()
        plt.savefig(base_out + '_PSD_acc.png', dpi=150)
        plt.close()
    # FRF/Coherence plot
    if frf_freq.size > 0:
        fig, ax = plt.subplots(2,1, figsize=(8,6), sharex=True)
        ax[0].plot(frf_freq, 20*np.log10(np.abs(H1)+1e-18))
        ax[0].set_ylabel('|H1| (dB)')
        ax[0].grid(True)
        ax[1].plot(frf_freq, coh)
        ax[1].set_xlabel('Frecuencia (Hz)')
        ax[1].set_ylabel('Coherencia')
        ax[1].grid(True)
        ax[1].set_ylim(0,1)
        # Highlight band
        for a in ax:
            a.axvspan(BAND_LO, BAND_HI, color='orange', alpha=0.1, linewidth=0)
        ax[1].set_xlim(0, 300)
        lag_ms = (lag_samples_band/fs*1000.0) if (np.isfinite(lag_samples_band) and fs and fs>0) else np.nan
        ax[0].set_title(f'FRF/Coherencia (F->A) - {sess.fecha} {sess.hora}  [Banda {BAND_LO:.0f}-{BAND_HI:.0f} Hz, lag={lag_ms:.2f} ms]')
        plt.tight_layout()
        plt.savefig(base_out + '_FRF_coh.png', dpi=150)
        plt.close()

    # STFT (spectrogram) of acceleration up to 200 Hz
    if A.size > 0 and not np.isnan(fsA):
        nperseg = min(len(A), 4096)
        nperseg = max(256, nperseg)
        fS, tS, Sxx = signal.spectrogram(A, fs=fsA, nperseg=nperseg, noverlap=nperseg//2, window='hann', scaling='density', detrend='constant')
        f_mask = fS <= 200.0
        plt.figure(figsize=(8,4))
        plt.pcolormesh(tS, fS[f_mask], 10*np.log10(Sxx[f_mask]+1e-18), shading='gouraud', cmap='viridis')
        plt.colorbar(label='PSD (dB/Hz)')
        plt.axhspan(BAND_LO, BAND_HI, color='orange', alpha=0.15)
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Frecuencia (Hz)')
        plt.title(f'STFT Aceleración (ai0) - {sess.fecha} {sess.hora} [Banda {BAND_LO:.0f}-{BAND_HI:.0f} Hz]')
        plt.tight_layout()
        plt.savefig(base_out + '_STFT_acc.png', dpi=150)
        plt.close()

    # Lissajous (normalized) scatter and fits
    if F_n.size > 0 and A_n.size > 0:
        # Scatter
        plt.figure(figsize=(6,6))
        ds = max(1, int(len(A_n)/20000))  # downsample for large arrays
        plt.scatter(A_n[::ds], F_n[::ds], s=3, alpha=0.2, color='tab:blue', label='Datos')
        # Fits
        xg = np.linspace(-1, 1, 300)
        if np.all(np.isfinite(lin_coefs)):
            plt.plot(xg, np.polyval(lin_coefs, xg), 'r--', lw=1.5, label=f'Lin R2={lin_r2:.2f}')
        if np.all(np.isfinite(cubic_coefs)):
            plt.plot(xg, np.polyval(cubic_coefs, xg), 'g-', lw=1.0, label=f'Cúbica R2={cubic_r2:.2f}')
        plt.xlim(-1.1, 1.1); plt.ylim(-1.1, 1.1)
        plt.xlabel('a_norm'); plt.ylabel('F_norm')
        plt.title(f'Lissajous F_norm vs a_norm - {sess.fecha} {sess.hora} [Banda {BAND_LO:.0f}-{BAND_HI:.0f} Hz, lag={lag_ms:.2f} ms]')
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(base_out + '_Lissajous_norm.png', dpi=150)
        plt.close()

        # Overlay of first cycles
        if A_n.size > 0 and not np.isnan(fs_xy):
            cycles = segment_cycles_by_zc(A_n, fs_xy)
            if cycles:
                plt.figure(figsize=(6,6))
                for k, (i0, i1) in enumerate(cycles[:12]):
                    plt.plot(A_n[i0:i1], F_n[i0:i1], lw=1.0, alpha=0.8, label=f'ciclo {k+1}' if k < 3 else None)
                plt.xlim(-1.1, 1.1); plt.ylim(-1.1, 1.1)
                plt.xlabel('a_norm'); plt.ylabel('F_norm')
                plt.title(f'Lazos por ciclo (norm) - {sess.fecha} {sess.hora} [Banda {BAND_LO:.0f}-{BAND_HI:.0f} Hz, lag={lag_ms:.2f} ms]\nEloop_mean={Eloop_mean:.3f} ± {Eloop_std if np.isfinite(Eloop_std) else np.nan:.3f} (n={n_cycles})')
                plt.grid(True)
                if len(cycles) >= 1:
                    plt.legend(loc='lower right')
                plt.tight_layout()
                plt.savefig(base_out + '_Loops_norm.png', dpi=150)
                plt.close()

    # Aggregate summary
    summary = {
        'session': sess.base,
        'fecha': sess.fecha,
        'hora': sess.hora,
        'fs_force': float(fsF) if not np.isnan(fsF) else np.nan,
        'fs_accel': float(fsA) if not np.isnan(fsA) else np.nan,
        # time acc
        **{f'acc_{k}': v for k,v in feats_acc_t.items()},
        # time force
        **{f'force_{k}': v for k,v in feats_force_t.items()},
        # psd peaks
        **{f'peak{i+1}_Hz': (peaks[i][0] if i < len(peaks) else np.nan) for i in range(3)},
        **{f'peak{i+1}_PSD': (peaks[i][1] if i < len(peaks) else np.nan) for i in range(3)},
        # bandpowers
        **band_powers,
        # frf/coherence in 20-150 Hz
        'coh_peak_band': coh_peak_band,
        'frf_mag_at_peak_band': frf_mag_at_peak_band,
        'phase_deg_at_peak_band': phase_deg_at_peak_band,
        'lag_samples_band': lag_samples_band,
        'lag_ms_band': (lag_samples_band/fs*1000.0) if (np.isfinite(lag_samples_band) and fs and fs>0) else np.nan,
        # frf peaks with coherence filter
        **{f'frfpk{i+1}_Hz': frf_pk_freqs[i] for i in range(TOPK_FRF)},
        **{f'frfpk{i+1}_coh': frf_pk_coh[i] for i in range(TOPK_FRF)},
        **{f'frfpk{i+1}_mag_dB': frf_pk_mag_db[i] for i in range(TOPK_FRF)},
        **{f'frfpk{i+1}_phase_deg': frf_pk_phase[i] for i in range(TOPK_FRF)},
        # nonlinear relations (normalized)
        'lin_slope': float(lin_coefs[0]) if np.all(np.isfinite(lin_coefs)) else np.nan,
        'lin_intercept': float(lin_coefs[1]) if np.all(np.isfinite(lin_coefs)) else np.nan,
        'lin_R2': float(lin_r2) if np.isfinite(lin_r2) else np.nan,
        'cubic_c3': float(cubic_coefs[0]) if np.all(np.isfinite(cubic_coefs)) else np.nan,
        'cubic_c2': float(cubic_coefs[1]) if np.all(np.isfinite(cubic_coefs)) else np.nan,
        'cubic_c1': float(cubic_coefs[2]) if np.all(np.isfinite(cubic_coefs)) else np.nan,
        'cubic_c0': float(cubic_coefs[3]) if np.all(np.isfinite(cubic_coefs)) else np.nan,
        'cubic_R2': float(cubic_r2) if np.isfinite(cubic_r2) else np.nan,
        'corr_pearson': pear,
        'corr_spearman': spear,
        'Eloop_norm_mean': Eloop_mean,
        'Eloop_norm_std': Eloop_std,
        'n_cycles': int(n_cycles),
    }
    return summary


def write_summary(save_dir: str, summaries: List[Dict[str, float]]):
    if not summaries: return
    ensure_dir(save_dir)
    # Collect all keys for current run
    keys = sorted(set().union(*[s.keys() for s in summaries]))
    out_csv = os.path.join(save_dir, 'features_resumen.csv')
    # Check existing header
    need_new_file = False
    if os.path.exists(out_csv):
        try:
            with open(out_csv, 'r', encoding='utf-8') as f:
                header_line = f.readline().strip()
            old_keys = [h.strip() for h in header_line.split(',')]
            if set(old_keys) != set(keys):
                need_new_file = True
        except Exception:
            need_new_file = True
    if need_new_file:
        out_csv = os.path.join(save_dir, 'features_resumen_v2.csv')
        print(f"Esquema cambió: escribiendo salida en {out_csv}")
    write_header = not os.path.exists(out_csv)
    with open(out_csv, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            w.writeheader()
        for s in summaries:
            w.writerow(s)
    print(f"Resumen de características guardado en: {out_csv}")


def main():
    ap = argparse.ArgumentParser(description='Offline feature extraction for shaker bench data')
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR, help='Directorio con CSVs (default: datos_automaticos)')
    ap.add_argument('--save-dir', default=DEFAULT_SAVE_DIR, help='Directorio de salida para plots y resumen')
    ap.add_argument('--from', dest='date_from', default=None, help='Fecha inicial YYYYMMDD (opcional)')
    ap.add_argument('--to', dest='date_to', default=None, help='Fecha final YYYYMMDD (opcional)')
    args = ap.parse_args()

    sessions = list_sessions(args.data_dir, args.date_from, args.date_to)
    if not sessions:
        print(f"No se encontraron sesiones emparejadas fuerza/vibración en {args.data_dir} con el filtro de fechas.")
        return

    print(f"Procesando {len(sessions)} sesiones...")
    summaries: List[Dict[str, float]] = []
    for sess in sessions:
        try:
            print(f"- {sess.base} ...")
            s = process_session(sess, args.save_dir)
            summaries.append(s)
        except Exception as e:
            print(f"Error procesando {sess.base}: {e}")
    write_summary(args.save_dir, summaries)


if __name__ == '__main__':
    main()
