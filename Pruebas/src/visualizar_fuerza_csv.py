#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import re
import csv
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Opcional
try:
    import chardet  # type: ignore
except Exception:
    chardet = None

# Opcional
try:
    from scipy.signal import welch
except Exception:
    welch = None

def _nanmean(x):
    x = np.asarray(x, dtype=float)
    if np.isfinite(x).any():
        return np.nanmean(x)
    return np.nan

def estimate_sr_from_time(t):
    """Estima SR a partir del vector de tiempo (s) usando la mediana del paso."""
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    sr = 1.0 / np.median(dt)
    return float(sr)


def detect_encoding(path, sample_bytes=262144):
    with open(path, 'rb') as f:
        raw = f.read(sample_bytes)
    # Intentos rápidos
    for enc in ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1'):
        try:
            raw.decode(enc)
            return enc
        except Exception:
            pass
    # chardet si está disponible
    if chardet is not None:
        res = chardet.detect(raw)
        enc = res.get('encoding')
        if enc:
            return enc
    # Último recurso
    return 'latin-1'


def detect_delimiter(text_sample):
    try:
        dialect = csv.Sniffer().sniff(text_sample, delimiters=[',', ';', '\t', '|'])
        return dialect.delimiter
    except Exception:
        # Heurística simple
        counts = {d: text_sample.count(d) for d in [',', ';', '\t', '|']}
        return max(counts, key=counts.get)


def extract_sr_from_comments(lines):
    # Busca "Sample Rate: XXX Hz" o "SR (Hz),..." en comentarios
    sr = None
    for ln in lines:
        if not ln.lstrip().startswith('#'):
            continue
        m = re.search(r"Sample\s*Rate\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*Hz", ln, re.IGNORECASE)
        if m:
            try:
                sr = float(m.group(1))
                break
            except Exception:
                pass
        m2 = re.search(r"SR\s*\(Hz\)\s*[:,]\s*([0-9]+(?:\.[0-9]+)?)", ln, re.IGNORECASE)
        if m2:
            try:
                sr = float(m2.group(1))
                break
            except Exception:
                pass
    return sr


def load_csv_force(path, force_col=None, time_col=None, sr_cli=None, limit=None):
    enc = detect_encoding(path)
    # Leer un bloque pequeño para detectar delimitador y comentarios
    with open(path, 'r', encoding=enc, errors='ignore', newline='') as f:
        head_text = f.read(131072)
    lines = head_text.splitlines()
    sr_in_comments = extract_sr_from_comments(lines)

    # Determinar primera línea no-comentario (header)
    header_line = None
    first_data_offset = 0
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith('#') or len(ln.strip()) == 0:
            continue
        header_line = ln
        first_data_offset = i
        break
    if header_line is None:
        raise RuntimeError('No se encontró encabezado en el archivo (primeras líneas son comentarios o están vacías).')

    delimiter = detect_delimiter('\n'.join(lines[first_data_offset:first_data_offset+10]))

    # Leer completo usando csv.DictReader, saltando comentarios
    headers = []
    t_vals = []
    f_vals = []
    raw_cols = {}
    with open(path, 'r', encoding=enc, errors='ignore', newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        # Saltar comentarios hasta header
        for row in reader:
            if not row:
                continue
            if row[0].lstrip().startswith('#'):
                continue
            headers = [h.strip() for h in row]
            break
        if not headers:
            raise RuntimeError('No se pudo leer encabezado de columnas.')
        # Leer datos
        data_rows = []
        for row in reader:
            if not row:
                continue
            if row[0].lstrip().startswith('#'):
                continue
            data_rows.append(row)
            if limit and len(data_rows) >= limit:
                break
    # Construir columnas como dict
    cols = {h: [] for h in headers}
    for r in data_rows:
        for i, h in enumerate(headers):
            if i < len(r):
                cols[h].append(r[i])
            else:
                cols[h].append('')

    # Elegir columnas
    def find_col(candidates):
        for patt in candidates:
            for h in headers:
                if re.search(patt, h, re.IGNORECASE):
                    return h
        return None

    if force_col is None:
        force_col = find_col([r"fuerza", r"force", r"load", r"N\)?$", r"F\s*\(?N\)?"]) or headers[1 if len(headers) > 1 else 0]
    if time_col is None:
        time_col = find_col([r"tiempo", r"time", r"timestamp", r"seg", r"ms\b"])  # puede ser None

    # Convertir a float
    def to_float(arr):
        out = []
        for x in arr:
            try:
                out.append(float(str(x).replace(',', '.')))
            except Exception:
                out.append(np.nan)
        return np.array(out, dtype=float)

    # Convertir todas las columnas a float cuando sea posible
    numeric_cols = {}
    for h in headers:
        arr = to_float(cols[h])
        finite_ratio = np.isfinite(arr).mean() if arr.size else 0
        if finite_ratio > 0.5:  # considerar numérica si al menos 50% es convertible
            numeric_cols[h] = arr

    f_arr = numeric_cols.get(force_col) if force_col in numeric_cols else None
    if time_col and time_col in cols:
        t_arr = to_float(cols[time_col])
        # Si está en ms, normalizar a s
        if re.search(r"ms\b", time_col, re.IGNORECASE):
            t_arr = t_arr / 1000.0
    else:
        t_arr = None

    # Si no hay tiempo, generar a partir de SR si se conoce
    sr = sr_cli or sr_in_comments
    if t_arr is None:
        if sr and sr > 0:
            t_arr = np.arange(len(f_arr)) / float(sr)
        else:
            t_arr = np.arange(len(f_arr))
    else:
        # Si tenemos tiempo pero no SR, estimar
        if not sr:
            sr = estimate_sr_from_time(t_arr)

    # Preparar diccionario de señales (todas las columnas numéricas excepto tiempo)
    signals = {}
    for h, arr in numeric_cols.items():
        if time_col and h == time_col:
            continue
        signals[h] = arr

    return {
        'encoding': enc,
        'delimiter': delimiter,
        'headers': headers,
        'force_col': force_col,
        'time_col': time_col,
        'sr': sr,
        't': t_arr,
        'f': f_arr,
        'signals': signals,
    }


def compute_psd(x, sr=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None, None
    if welch and sr:
        f, Pxx = welch(x, fs=sr, nperseg=min(4096, max(256, (x.size // 8) // 2 * 2)))
        return f, Pxx
    # Fallback simple FFT (sin ventanas/overlap)
    N = x.size
    X = np.fft.rfft(x - np.nanmean(x))
    Pxx = (np.abs(X) ** 2) / (N * (sr if sr else 1.0))
    f = np.fft.rfftfreq(N, d=(1.0 / sr) if sr else 1.0)
    return f, Pxx


def _skew_kurtosis(x):
    """Calcula skewness y kurtosis (exceso) sin depender de scipy."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    m = np.mean(x)
    xc = x - m
    s2 = np.mean(xc**2)
    if s2 <= 0:
        return np.nan, np.nan
    s = np.sqrt(s2)
    skew = np.mean(xc**3) / (s**3)
    kurt = np.mean(xc**4) / (s**4) - 3.0
    return float(skew), float(kurt)


def compute_time_features(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            'mean': np.nan, 'std': np.nan, 'rms': np.nan, 'p2p': np.nan,
            'min': np.nan, 'max': np.nan, 'median': np.nan,
            'skew': np.nan, 'kurt': np.nan, 'crest_factor': np.nan,
            'zcr': np.nan, 'abs_mean': np.nan
        }
    mean = float(np.mean(x))
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x**2)))
    p2p = float(np.ptp(x))
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    med = float(np.median(x))
    skew, kurt = _skew_kurtosis(x)
    crest = float(np.max(np.abs(x))/rms) if np.isfinite(rms) and rms > 0 else np.nan
    # Zero-crossing rate
    zc = 0.0
    if x.size > 1:
        zc = float(np.mean((x[:-1] * x[1:]) < 0))
    abs_mean = float(np.mean(np.abs(x)))
    return {
        'mean': mean, 'std': std, 'rms': rms, 'p2p': p2p,
        'min': x_min, 'max': x_max, 'median': med,
        'skew': skew, 'kurt': kurt, 'crest_factor': crest,
        'zcr': zc, 'abs_mean': abs_mean
    }


def _bandpower(f, Pxx, fmin, fmax):
    if f is None or Pxx is None:
        return np.nan
    idx = (f >= fmin) & (f <= fmax)
    if not np.any(idx):
        return 0.0
    return float(np.trapz(Pxx[idx], f[idx]))


def compute_freq_features(x, sr, bands=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or not sr or sr <= 0:
        return {'dom_freq': np.nan, 'dom_amp': np.nan, 'spec_centroid': np.nan}

    # Periodograma rápido
    X = np.fft.rfft(x - np.mean(x))
    f = np.fft.rfftfreq(x.size, d=1.0/sr)
    mag = np.abs(X)
    # Evitar DC
    if mag.size > 1:
        i_max = int(np.argmax(mag[1:]) + 1)
    else:
        i_max = 0
    dom_freq = float(f[i_max]) if i_max < f.size else np.nan
    dom_amp = float(mag[i_max]) if i_max < mag.size else np.nan
    # Centroide espectral (potencia)
    P = (mag**2)
    denom = float(np.sum(P)) if np.sum(P) > 0 else np.nan
    spec_centroid = float(np.sum(f * P) / denom) if np.isfinite(denom) and denom > 0 else np.nan

    feats = {
        'dom_freq': dom_freq,
        'dom_amp': dom_amp,
        'spec_centroid': spec_centroid,
        'total_power': float(np.trapz(P, f)) if f.size and np.isfinite(P).any() else np.nan,
    }
    if bands:
        for (lo, hi) in bands:
            feats[f'bandpower_{lo}_{hi}'] = _bandpower(f, P, lo, hi)
    return feats


def extract_features_all_signals(data, win_samples, step_samples, bands=None, tone_freqs=None):
    """Extrae características por ventanas para todas las señales numéricas.
    Devuelve la lista de filas (dict) y el nombre de la métrica de tono principal si aplica.
    """
    t = np.asarray(data['t'], dtype=float)
    sr = data['sr'] if data['sr'] else estimate_sr_from_time(t)
    signals = data.get('signals', {})
    idxs = segment_indices(len(t), win_samples, step_samples)
    results = []
    for col, x in signals.items():
        x = np.asarray(x, dtype=float)
        for i_win, (i0, i1) in enumerate(idxs):
            seg = x[i0:i1]
            tf = compute_time_features(seg)
            ff = compute_freq_features(seg, sr, bands=bands)
            tones = compute_tone_features(seg, sr, tone_freqs) if tone_freqs else {}
            t0 = t[i0] if i0 < len(t) else np.nan
            t1 = t[i1-1] if (i1-1) < len(t) else np.nan
            row = {
                'signal': col,
                'i_win': i_win + 1,
                'i0': i0, 'i1': i1,
                't0': float(t0) if np.isfinite(t0) else np.nan,
                't1': float(t1) if np.isfinite(t1) else np.nan,
            }
            row.update(tf)
            row.update(ff)
            row.update(tones)
            results.append(row)
    main_tone = None
    if tone_freqs and len(tone_freqs) > 0:
        main_tone = f"tone_{int(round(tone_freqs[0]))}Hz"
    return results, main_tone


def parse_bands(bands_str):
    """Convierte '0-10,10-30,30-60' -> [(0,10),(10,30),(30,60)]"""
    if not bands_str:
        return None
    out = []
    for part in bands_str.split(','):
        part = part.strip()
        m = re.match(r"\s*([0-9]*\.?[0-9]+)\s*[-:]\s*([0-9]*\.?[0-9]+)\s*", part)
        if m:
            out.append((float(m.group(1)), float(m.group(2))))
    return out if out else None


def extract_features_over_windows(t, x, sr, win_samples, step_samples, bands=None, t0_offset=True):
    idxs = segment_indices(len(x), win_samples, step_samples)
    results = []
    for (i0, i1) in idxs:
        seg = x[i0:i1]
        tf = compute_time_features(seg)
        ff = compute_freq_features(seg, sr, bands=bands)
        t0 = t[i0] if i0 < len(t) else np.nan
        t1 = t[i1-1] if (i1-1) < len(t) else np.nan
        row = {
            'i0': i0, 'i1': i1, 't0': float(t0) if np.isfinite(t0) else np.nan, 't1': float(t1) if np.isfinite(t1) else np.nan,
        }
        row.update(tf)
        row.update(ff)
        results.append(row)
    return results


def parse_freqs(freqs_str):
    if not freqs_str:
        return None
    out = []
    for part in freqs_str.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except Exception:
            pass
    return out if out else None


def goertzel_mag(x, sr, freq):
    """Amplitud a una frecuencia específica (Goertzel)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    N = x.size
    if N == 0 or not sr or sr <= 0:
        return np.nan
    k = int(0.5 + (N * freq) / sr)
    if k <= 0:
        k = 1
    if k >= N:
        k = N - 1
    omega = (2.0 * np.pi * k) / N
    coeff = 2.0 * np.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for sample in x:
        s = sample + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    real = s_prev - s_prev2 * np.cos(omega)
    imag = s_prev2 * np.sin(omega)
    mag = np.sqrt(real * real + imag * imag) / N
    return float(mag)


def compute_tone_features(x, sr, freqs):
    feats = {}
    if not freqs:
        return feats
    for f0 in freqs:
        feats[f'tone_{int(round(f0))}Hz'] = goertzel_mag(x, sr, f0)
    return feats


def plot_force(data, title=None, out_png=None):
    t = data['t']
    f = data['f']
    sr = data['sr']
    force_col = data['force_col']

    fig, axs = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)

    # Serie de tiempo
    axs[0].plot(t, f, lw=0.9)
    axs[0].set_title(f"Fuerza vs tiempo ({force_col})")
    axs[0].set_xlabel('Tiempo [s]' if (data['time_col'] or sr) else 'Muestra')
    axs[0].set_ylabel('F [N] (asumido)')

    # Estadísticos
    finite = f[np.isfinite(f)]
    if finite.size:
        mean = np.mean(finite)
        rms = np.sqrt(np.mean((finite - np.mean(finite)) ** 2))
        p2p = np.ptp(finite)
    else:
        mean = rms = p2p = np.nan
    axs[1].hist(finite, bins=60, color='tab:gray', alpha=0.8)
    axs[1].set_title(f"Histograma  |  mean={mean:.3g}, rms={rms:.3g}, p2p={p2p:.3g}")
    axs[1].set_xlabel('F [N]')

    # PSD
    ff, Pxx = compute_psd(f, sr)
    if ff is not None:
        axs[2].semilogx(ff[1:], 10*np.log10(Pxx[1:]+1e-20), lw=0.9)
        axs[2].set_title('PSD (dB)')
        axs[2].set_xlabel('Frecuencia [Hz]' if sr else 'Bin (sin SR)')
        axs[2].set_ylabel('dB')

    if title:
        fig.suptitle(title)

    if out_png:
        fig.savefig(out_png, dpi=120)
        print(f"✅ Figura guardada: {out_png}")

    plt.show()


def plot_psd_windows_all_signals(data, win_sec=1.0, step_sec=None, max_windows=12, out_dir=None, base_title="", fmax=None):
    """Grafica espectros (PSD simple) por ventana para todas las señales numéricas."""
    t = np.asarray(data['t'])
    sr = data['sr'] if data['sr'] else estimate_sr_from_time(t)
    if not sr:
        print("⚠️ No se puede graficar PSD por ventanas sin SR estimada.")
        return

    win_samples = int(round(win_sec * sr)) if win_sec else max(1, int(len(t) // 10))
    if win_samples <= 0:
        win_samples = max(1, int(len(t) // 10))
    if step_sec is None:
        step_samples = win_samples
    else:
        step_samples = max(1, int(round(step_sec * sr)))

    idxs = segment_indices(len(t), win_samples, step_samples)
    if len(idxs) > max_windows:
        idxs = idxs[:max_windows]

    signals = data.get('signals', {})
    if not signals:
        signals = {data['force_col'] or 'signal': np.asarray(data['f'])}

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for col, x in signals.items():
        x = np.asarray(x, dtype=float)
        n = len(idxs)
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4*ncols, 2.6*nrows), squeeze=False, constrained_layout=True)
        for k, (i0, i1) in enumerate(idxs):
            r = k // ncols
            c = k % ncols
            ax = axes[r][c]
            yy = x[i0:i1]
            f, Pxx = compute_psd(yy, sr)
            if f is None:
                ax.text(0.5, 0.5, 'sin datos', ha='center', va='center')
                ax.axis('off')
                continue
            # dB
            ydb = 10*np.log10(Pxx + 1e-20)
            ax.semilogx(f[1:], ydb[1:], lw=0.9)
            if fmax:
                ax.set_xlim(0.5, fmax)
            ax.set_xlabel('Hz')
            ax.set_ylabel('dB')
            # Marcar frecuencia dominante
            if yy.size > 1:
                feats = compute_freq_features(yy, sr)
                fd = feats.get('dom_freq', np.nan)
                if np.isfinite(fd):
                    ax.axvline(fd, color='r', ls='--', lw=0.9)
                    ax.set_title(f"win {k+1} | fdom={fd:.2f} Hz")
                else:
                    ax.set_title(f"win {k+1}")
            else:
                ax.set_title(f"win {k+1}")

        for k in range(n, nrows*ncols):
            r = k // ncols
            c = k % ncols
            axes[r][c].axis('off')

        fig.suptitle(f"PSD por ventanas {col}  |  win={win_sec}s step={step_sec if step_sec is not None else win_sec}s  (SR={sr:.3g} Hz)")
        if out_dir:
            base = os.path.splitext(os.path.basename(base_title))[0] if base_title else 'signal'
            out_png = os.path.join(out_dir, f"{base}_psdwindows_{re.sub(r'[^A-Za-z0-9_]+','_',col)}.png")
            fig.savefig(out_png, dpi=130)
            print(f"✅ Figura PSD-ventanas guardada: {out_png}")
    plt.show()


def segment_indices(n, win_samples, step_samples):
    """Genera índices (i0, i1) para ventanas deslizantes."""
    idx = []
    i = 0
    while i + win_samples <= n:
        idx.append((i, i + win_samples))
        i += step_samples
    if not idx and n > 0:  # si la señal es más corta que la ventana, devolver una única ventana
        idx.append((0, n))
    return idx


def plot_windows_all_signals(data, win_sec=1.0, step_sec=None, max_windows=12, out_dir=None, base_title=""):
    """Grafica ventanas (series de tiempo) para todas las señales numéricas.
    - Crea una figura por señal con una cuadrícula de subplots (cada subplot = una ventana).
    - Limita a max_windows ventanas para no saturar.
    """
    t = np.asarray(data['t'])
    sr = data['sr'] if data['sr'] else estimate_sr_from_time(t)
    if not sr:
        # Si no hay SR y el tiempo no es uniforme, usamos número de muestras por ventana aproximado
        if t.size > 1:
            sr = estimate_sr_from_time(t)
        else:
            sr = 1.0

    win_samples = int(round(win_sec * sr)) if win_sec else max(1, int(len(t) // 10))
    if win_samples <= 0:
        win_samples = max(1, int(len(t) // 10))
    if step_sec is None:
        step_samples = win_samples  # sin solapamiento
    else:
        step_samples = max(1, int(round(step_sec * sr)))

    idxs = segment_indices(len(t), win_samples, step_samples)
    if len(idxs) > max_windows:
        idxs = idxs[:max_windows]

    signals = data.get('signals', {})
    if not signals:
        # fallback: usar columna 'f'
        signals = {data['force_col'] or 'signal': np.asarray(data['f'])}

    # Crear carpeta de salida si se solicitó
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for col, x in signals.items():
        x = np.asarray(x, dtype=float)
        # Configurar cuadrícula
        n = len(idxs)
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4*ncols, 2.6*nrows), squeeze=False, constrained_layout=True)
        for k, (i0, i1) in enumerate(idxs):
            r = k // ncols
            c = k % ncols
            ax = axes[r][c]
            tt = t[i0:i1]
            yy = x[i0:i1]
            # Re-indexar tiempo relativo a la ventana
            if tt.size:
                tt_rel = tt - tt[0]
            else:
                tt_rel = tt
            ax.plot(tt_rel, yy, lw=0.9)
            # Frecuencia dominante (si hay SR)
            dom_info = ""
            if sr and yy.size > 1:
                ff = compute_freq_features(yy, sr, bands=None)
                if np.isfinite(ff.get('dom_freq', np.nan)):
                    dom_info = f" | fdom={ff['dom_freq']:.2f} Hz"
            ax.set_title(f"win {k+1}: t=[{i0/sr:.2f},{i1/sr:.2f}]s{dom_info}")
            ax.set_xlabel('t [s]')
            ax.set_ylabel(col)

        # Desactivar subplots vacíos
        for k in range(n, nrows*ncols):
            r = k // ncols
            c = k % ncols
            axes[r][c].axis('off')

        title = base_title or os.path.basename(col)
        fig.suptitle(f"Ventanas {col}  |  win={win_sec}s step={step_sec if step_sec is not None else win_sec}s  (SR≈{sr:.3g} Hz)")

        if out_dir:
            base = os.path.splitext(os.path.basename(base_title))[0] if base_title else 'signal'
            out_png = os.path.join(out_dir, f"{base}_windows_{re.sub(r'[^A-Za-z0-9_]+','_',col)}.png")
            fig.savefig(out_png, dpi=130)
            print(f"✅ Figura de ventanas guardada: {out_png}")

    plt.show()


def main():
    ap = argparse.ArgumentParser(description='Visualizador rápido de CSV de fuerza')
    ap.add_argument('csv_path', help='Ruta al CSV')
    ap.add_argument('--force-col', help='Nombre de columna de fuerza (si no se detecta)')
    ap.add_argument('--time-col', help='Nombre de columna de tiempo (si existe)')
    ap.add_argument('--sr', type=float, help='Sample rate Hz (si no viene en archivo)')
    ap.add_argument('--limit', type=int, help='Leer solo N filas (debug)')
    ap.add_argument('--windows', action='store_true', help='Graficar ventanas para todas las señales numéricas')
    ap.add_argument('--win-sec', type=float, default=1.0, help='Tamaño de ventana en segundos')
    ap.add_argument('--step-sec', type=float, help='Paso entre ventanas en segundos (por defecto = win-sec, sin solape)')
    ap.add_argument('--max-windows', type=int, default=12, help='Máximo de ventanas a graficar')
    ap.add_argument('--columns', help='Lista separada por coma de columnas a incluir (opcional)')
    ap.add_argument('--out-dir', help='Carpeta para guardar figuras de ventanas')
    ap.add_argument('--freq-windows', action='store_true', help='Graficar PSD por ventanas de todas las señales')
    ap.add_argument('--fmax', type=float, help='Frecuencia máxima para el eje X en PSD por ventanas')
    # Features por ventanas (p/ canal objetivo)
    ap.add_argument('--features', action='store_true', help='Calcular y guardar características por ventanas del canal objetivo')
    ap.add_argument('--target-col', help='Columna objetivo (ej. "Channel 2")')
    ap.add_argument('--win-samples', type=int, help='Tamaño de ventana en muestras (por defecto 200 si no hay win-sec)')
    ap.add_argument('--step-samples', type=int, help='Paso entre ventanas en muestras (por defecto = win-samples)')
    ap.add_argument('--bands', help='Bandas de frecuencia, ej. "0-10,10-30,30-60,60-120"')
    ap.add_argument('--out-csv', help='Ruta CSV para guardar las características')
    ap.add_argument('--top', type=int, default=10, help='Mostrar TOP-N ventanas por métrica de ranking (solo --features)')
    ap.add_argument('--rank-metric', default='auto', help="Métrica de ranking: 'auto', 'dom_amp', 'total_power', 'bandpower_45_55', etc.")
    ap.add_argument('--summary-plot', action='store_true', help='Guardar plot t0 vs dom_freq con TOP-N marcado (solo --features)')
    # All-channel features
    ap.add_argument('--features-all', action='store_true', help='Extraer características por ventanas para todas las señales')
    ap.add_argument('--freqs', default='50', help='Frecuencias de tono para Goertzel, ej. "50,100,150" (por defecto 50 Hz)')
    args = ap.parse_args()

    path = os.path.abspath(args.csv_path)
    if not os.path.isfile(path):
        print(f"❌ Archivo no existe: {path}")
        sys.exit(1)

    data = load_csv_force(path, force_col=args.force_col, time_col=args.time_col, sr_cli=args.sr, limit=args.limit)
    base_title = os.path.basename(path)

    # Filtrar columnas si se especificó --columns
    if args.columns:
        cols_req = [c.strip() for c in args.columns.split(',') if c.strip()]
        signals = {k: v for k, v in data.get('signals', {}).items() if k in cols_req}
        if signals:
            data['signals'] = signals

    if args.windows:
        out_dir = args.out_dir or os.path.join(os.path.dirname(path), 'fig_windows')
        plot_windows_all_signals(
            data,
            win_sec=args.win_sec,
            step_sec=args.step_sec,
            max_windows=args.max_windows,
            out_dir=out_dir,
            base_title=base_title,
        )
    elif args.freq_windows:
        out_dir = args.out_dir or os.path.join(os.path.dirname(path), 'fig_windows')
        plot_psd_windows_all_signals(
            data,
            win_sec=args.win_sec,
            step_sec=args.step_sec,
            max_windows=args.max_windows,
            out_dir=out_dir,
            base_title=base_title,
            fmax=args.fmax,
        )
    elif args.features:
        # Determinar columna objetivo
        target = args.target_col
        if not target:
            # intentar "Channel 2"
            for h in data['headers']:
                if re.search(r'channel\s*2\b', h, re.IGNORECASE):
                    target = h
                    break
        if not target:
            target = data['force_col']  # fallback

        signals = data.get('signals', {})
        if target not in signals:
            print(f"❌ Columna objetivo no encontrada: {target}")
            print(f"   Columnas disponibles: {list(signals.keys())[:10]} ...")
            sys.exit(1)

        x = np.asarray(signals[target], dtype=float)
        t = np.asarray(data['t'], dtype=float)
        sr = data['sr'] if data['sr'] else estimate_sr_from_time(t)

        # Ventaneo
        if args.win_samples:
            win_samples = max(1, int(args.win_samples))
        elif args.win_sec and sr:
            win_samples = max(1, int(round(args.win_sec * sr)))
        else:
            win_samples = 200  # por solicitud
        if args.step_samples:
            step_samples = max(1, int(args.step_samples))
        elif args.step_sec and sr:
            step_samples = max(1, int(round(args.step_sec * sr)))
        else:
            step_samples = win_samples

        # Bandas de frecuencia
        bands = parse_bands(args.bands) if args.bands else [(0, 10), (10, 30), (30, 60), (60, 120)] if sr else None

        feats = extract_features_over_windows(t, x, sr, win_samples, step_samples, bands=bands)

        # Guardar CSV
        out_csv = args.out_csv
        if not out_csv:
            safe_col = re.sub(r'[^A-Za-z0-9_]+', '_', target)
            out_csv = os.path.splitext(path)[0] + f"_features_{safe_col}_win{win_samples}.csv"
        fieldnames = list(feats[0].keys()) if feats else []
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in feats:
                writer.writerow(row)
        print(f"✅ Features por ventanas guardadas: {out_csv}")
        print(f"   Columna: {target} | ventanas: {len(feats)} | win={win_samples} step={step_samples} (sr≈{sr:.3g} Hz)")

        # Resumen rápido TOP-N
        if feats:
            # Elegir métrica de ranking
            cand_metric = None
            if args.rank_metric != 'auto':
                cand_metric = args.rank_metric
            else:
                # preferir banda 45-55 si existe
                for k in feats[0].keys():
                    if re.match(r'bandpower_45_55$', k):
                        cand_metric = k
                        break
                if not cand_metric:
                    # buscar cualquier bandpower que contenga 50 Hz (ej 40-60, 48-52, etc.)
                    for k in feats[0].keys():
                        if k.startswith('bandpower_') and ('_50' in k or '45_55' in k or '40_60' in k):
                            cand_metric = k
                            break
                if not cand_metric:
                    cand_metric = 'dom_amp'

            # Ordenar y mostrar
            feats_sorted = sorted(feats, key=lambda r: (r.get(cand_metric) if r.get(cand_metric) is not None else -np.inf), reverse=True)
            topN = min(args.top, len(feats_sorted))
            print(f"\n📈 TOP {topN} por '{cand_metric}':")
            print("i  t0[s]   t1[s]   dom_freq[Hz]  dom_amp   " + cand_metric)
            for i, row in enumerate(feats_sorted[:topN], 1):
                print(f"{i:>2} {row['t0']:.3f} {row['t1']:.3f} {row.get('dom_freq', np.nan):>10.3f} {row.get('dom_amp', np.nan):>8.3g} {row.get(cand_metric, np.nan):>10.3g}")

            # Plot resumen dom_freq vs t0
            if args.summary_plot:
                t0s = np.array([r['t0'] for r in feats], dtype=float)
                fdom = np.array([r.get('dom_freq', np.nan) for r in feats], dtype=float)
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.plot(t0s, fdom, '.', ms=3, alpha=0.8)
                ax.set_xlabel('t0 [s]')
                ax.set_ylabel('dom_freq [Hz]')
                ax.grid(True, ls='--', alpha=0.3)
                # marcar TOP-N
                for row in feats_sorted[:topN]:
                    ax.axvspan(row['t0'], row['t1'], color='tab:orange', alpha=0.15)
                out_png_summary = os.path.splitext(out_csv)[0] + '_summary.png'
                fig.savefig(out_png_summary, dpi=130)
                print(f"✅ Resumen guardado: {out_png_summary}")

        # Mostrar una figura de ventanas de esta columna
        plot_windows_all_signals({**data, 'signals': {target: x}}, win_sec=win_samples/(sr if sr else 1.0), step_sec=step_samples/(sr if sr else 1.0), max_windows=min(12, len(feats)), out_dir=args.out_dir or os.path.join(os.path.dirname(path), 'fig_windows'), base_title=base_title)
    elif args.features_all:
        # Ventanas en muestras
        t = np.asarray(data['t'], dtype=float)
        sr = data['sr'] if data['sr'] else estimate_sr_from_time(t)
        if args.win_samples:
            win_samples = max(1, int(args.win_samples))
        elif args.win_sec and sr:
            win_samples = max(1, int(round(args.win_sec * sr)))
        else:
            win_samples = 200
        if args.step_samples:
            step_samples = max(1, int(args.step_samples))
        elif args.step_sec and sr:
            step_samples = max(1, int(round(args.step_sec * sr)))
        else:
            step_samples = win_samples

        bands = parse_bands(args.bands) if args.bands else [(45,55),(40,60),(0,10),(10,30),(30,60),(60,120)] if sr else None
        tone_freqs = parse_freqs(args.freqs) if args.freqs else [50.0]

        feats_all, main_tone = extract_features_all_signals(data, win_samples, step_samples, bands=bands, tone_freqs=tone_freqs)

        out_csv = args.out_csv or (os.path.splitext(path)[0] + f"_features_all_win{win_samples}.csv")
        fieldnames = list(feats_all[0].keys()) if feats_all else []
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in feats_all:
                writer.writerow(row)
        print(f"✅ Features por ventanas (todas las señales) guardadas: {out_csv}")
        print(f"   Señales: {len(data.get('signals', {}))} | ventanas por señal: {sum(1 for r in feats_all if r['signal']==list(data['signals'].keys())[0])}")

        # Resumen por señal: media de la magnitud a 50 Hz (o primera en --freqs)
        if feats_all and main_tone:
            import collections
            agg = collections.defaultdict(list)
            for r in feats_all:
                if main_tone in r:
                    agg[r['signal']].append(r[main_tone])
            print("\n📊 Resumen por señal (media de", main_tone, "):")
            for sig, vals in agg.items():
                vals = np.array(vals, dtype=float)
                mean_val = float(np.nanmean(vals)) if vals.size else np.nan
                print(f" - {sig}: {mean_val:.4g}")
    else:
        title = f"{os.path.basename(path)}  |  enc={data['encoding']}  sep='"+data['delimiter']+f"'  SR={data['sr']}"
        out_png = os.path.splitext(path)[0] + '_preview.png'
        plot_force(data, title=title, out_png=out_png)


if __name__ == '__main__':
    main()
