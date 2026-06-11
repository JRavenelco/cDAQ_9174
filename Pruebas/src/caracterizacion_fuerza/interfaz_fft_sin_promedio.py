#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interfaz simple para cargar CSV y obtener FFT directa sin promediar.

La FFT se calcula sobre un solo tramo de la senal seleccionada. No usa Welch,
no divide en ventanas y no promedia espectros.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

import numpy as np

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

try:
    from scipy import signal as scipy_signal
except Exception:  # pragma: no cover - fallback para equipos sin scipy
    scipy_signal = None


WORKDIR = Path(__file__).resolve().parent
RED_FREQS_HZ = [60.0, 120.0, 180.0, 240.0]


@dataclass
class LoadedData:
    path: Path
    columns: dict[str, np.ndarray]
    meta: dict | None = None


def sniff_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        return dialect.delimiter
    except Exception:
        first = sample.splitlines()[0] if sample else ""
        if "\t" in first:
            return "\t"
        if ";" in first:
            return ";"
        return ","


def load_numeric_csv(path: Path) -> LoadedData:
    delimiter = sniff_delimiter(path)
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("El CSV no tiene encabezados.")
        raw = {name.strip(): [] for name in reader.fieldnames if name}
        for row in reader:
            for name in raw:
                text = (row.get(name) or "").strip().replace(",", ".")
                try:
                    raw[name].append(float(text))
                except ValueError:
                    raw[name].append(np.nan)

    columns: dict[str, np.ndarray] = {}
    for name, values in raw.items():
        if name.strip().lower() in {"aceleracion_bancada_g", "accel_aceleracion_bancada_g"}:
            continue
        arr = np.asarray(values, dtype=float)
        if np.count_nonzero(np.isfinite(arr)) > max(10, 0.2 * len(arr)):
            columns[name] = arr
    if not columns:
        raise ValueError("No encontre columnas numericas.")
    return LoadedData(path=path, columns=columns)


def load_ads1219_meta(path: Path) -> LoadedData:
    """Carga una prueba desde su *_meta.json (los archivos están en la misma carpeta).

    Expone:
      - tiempo_comun_s, fuerza_V, aceleracion_g  -> archivo ALINEADO (doc): misma base
        de tiempo = reloj medido del Pico (~990 Hz). Para fuerza-vs-aceleración y
        frecuencias bajas (hasta el Nyquist del doc ~495 Hz).
      - tiempo_accel_s, aceleracion_full_g       -> aceleración NATIVA del 9234 (tasa
        real, p.ej. 2048 Hz). Para análisis espectral de frecuencias altas.
    """
    meta = json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
    base = path.parent
    columns: dict[str, np.ndarray] = {}

    def add_csv(fname: str, time_key: str, sig_map: dict[str, str]) -> None:
        f = base / fname
        if not f.exists():
            return
        d = load_numeric_csv(f)
        for name, arr in d.columns.items():
            low = name.lower()
            if low in {"tiempo_s", "time", "t"}:
                columns.setdefault(time_key, arr)
            else:
                key = sig_map.get(low, name)
                if key:
                    columns.setdefault(key, arr)

    # 1) Archivo ALINEADO (doc) = reloj del Pico
    if meta.get("file_doc_txt"):
        add_csv(meta["file_doc_txt"], "tiempo_comun_s",
                {"fuerza_v": "fuerza_V", "aceleracion_g": "aceleracion_g"})
    # 2) Fuerza nativa (solo si no hubo doc)
    if "fuerza_V" not in columns and meta.get("file_force"):
        add_csv(meta["file_force"], "tiempo_fuerza_s", {"fuerza_v": "fuerza_V"})
    # 3) Aceleración NATIVA (tasa real del 9234) para frecuencias altas
    if meta.get("file_accel"):
        add_csv(meta["file_accel"], "tiempo_accel_s",
                {"aceleracion_sensor_g": "aceleracion_full_g",
                 "aceleracion_g": "aceleracion_full_g"})

    if not columns:
        raise ValueError("El meta JSON no apunta a archivos de datos válidos.")
    return LoadedData(path=path, columns=columns, meta=meta)


def choose_default_column(names: list[str], needles: list[str]) -> str:
    lowered = [(name, name.lower()) for name in names]
    for needle in needles:
        for name, low in lowered:
            if needle in low:
                return name
    return names[0] if names else ""


def estimate_fs(time_s: np.ndarray) -> float:
    finite = np.isfinite(time_s)
    t = time_s[finite]
    if len(t) < 3:
        raise ValueError("No hay suficientes puntos de tiempo para estimar fs.")
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("La columna de tiempo no es monotona.")
    return float(1.0 / np.median(dt))


def direct_fft(
    time_s: np.ndarray,
    x: np.ndarray,
    fs: float,
    window_name: str,
    detrend: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(time_s) & np.isfinite(x)
    time_s = time_s[mask]
    x = x[mask]
    if len(x) < 8:
        raise ValueError("Necesito al menos 8 muestras para calcular FFT.")

    x = np.asarray(x, dtype=float)
    if detrend:
        if scipy_signal is not None:
            x_proc = scipy_signal.detrend(x, type="constant")
        else:
            x_proc = x - np.mean(x)
    else:
        x_proc = x.copy()

    if window_name.lower().startswith("hann"):
        window = np.hanning(len(x_proc))
    elif window_name.lower().startswith("hamming"):
        window = np.hamming(len(x_proc))
    else:
        window = np.ones(len(x_proc))

    coherent_gain = np.sum(window) / len(window)
    if coherent_gain <= 0:
        coherent_gain = 1.0

    spectrum = np.fft.rfft(x_proc * window)
    freq = np.fft.rfftfreq(len(x_proc), d=1.0 / fs)
    amp = np.abs(spectrum) / (len(x_proc) * coherent_gain)
    if len(amp) > 2:
        amp[1:-1] *= 2.0
    amp_db = 20.0 * np.log10(np.maximum(amp, 1e-20))
    return freq, amp, amp_db


def simple_spectrogram(
    x: np.ndarray,
    fs: float,
    nperseg: int,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scipy_signal is not None:
        noverlap = int(nperseg * overlap)
        f, t, sxx = scipy_signal.spectrogram(
            x,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="spectrum",
            mode="magnitude",
        )
        return f, t, sxx

    step = max(1, int(nperseg * (1.0 - overlap)))
    chunks = []
    times = []
    for start in range(0, max(1, len(x) - nperseg + 1), step):
        chunk = x[start : start + nperseg]
        if len(chunk) < nperseg:
            break
        _, amp, _ = direct_fft(np.arange(nperseg) / fs, chunk, fs, "Hann", True)
        chunks.append(amp)
        times.append((start + nperseg / 2) / fs)
    if not chunks:
        raise ValueError("No hay suficientes muestras para espectrograma.")
    return np.fft.rfftfreq(nperseg, d=1.0 / fs), np.asarray(times), np.asarray(chunks).T


def clean_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "senal"


class FFTApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FFT directa sin promedio")
        self.geometry("1220x780")
        self.minsize(980, 640)

        self.data: LoadedData | None = None
        self.last_result: dict[str, object] | None = None

        self.file_var = tk.StringVar(value="Sin archivo")
        self.time_var = tk.StringVar()
        self.signal_var = tk.StringVar()
        self.fs_var = tk.StringVar(value="")
        self.rpm_var = tk.StringVar(value="0")
        self.t0_var = tk.StringVar(value="")
        self.t1_var = tk.StringVar(value="")
        self.fmax_var = tk.StringVar(value="500")
        self.window_var = tk.StringVar(value="Hann")
        self.detrend_var = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(main)
        controls.pack(fill=tk.X)

        ttk.Button(controls, text="Cargar CSV", command=self.load_file).grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(controls, text="Cargar carpeta", command=self.load_folder).grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(controls, textvariable=self.file_var).grid(row=0, column=2, columnspan=7, sticky="w", padx=4)

        ttk.Label(controls, text="Tiempo").grid(row=1, column=0, sticky="e", padx=4)
        self.time_combo = ttk.Combobox(controls, textvariable=self.time_var, width=22, state="readonly")
        self.time_combo.grid(row=1, column=1, sticky="w", padx=4)

        ttk.Label(controls, text="Senal").grid(row=1, column=2, sticky="e", padx=4)
        self.signal_combo = ttk.Combobox(controls, textvariable=self.signal_var, width=28, state="readonly")
        self.signal_combo.grid(row=1, column=3, sticky="w", padx=4)
        self.signal_combo.bind("<<ComboboxSelected>>", self._auto_time_for_signal)

        ttk.Label(controls, text="fs si no hay tiempo (Hz)").grid(row=1, column=4, sticky="e", padx=4)
        ttk.Entry(controls, textvariable=self.fs_var, width=10).grid(row=1, column=5, sticky="w", padx=4)

        ttk.Label(controls, text="RPM").grid(row=1, column=6, sticky="e", padx=4)
        ttk.Entry(controls, textvariable=self.rpm_var, width=8).grid(row=1, column=7, sticky="w", padx=4)

        ttk.Label(controls, text="t0").grid(row=2, column=0, sticky="e", padx=4)
        ttk.Entry(controls, textvariable=self.t0_var, width=10).grid(row=2, column=1, sticky="w", padx=4)
        ttk.Label(controls, text="t1").grid(row=2, column=2, sticky="e", padx=4)
        ttk.Entry(controls, textvariable=self.t1_var, width=10).grid(row=2, column=3, sticky="w", padx=4)
        ttk.Label(controls, text="fmax (Hz)").grid(row=2, column=4, sticky="e", padx=4)
        ttk.Entry(controls, textvariable=self.fmax_var, width=10).grid(row=2, column=5, sticky="w", padx=4)

        ttk.Label(controls, text="Ventana").grid(row=2, column=6, sticky="e", padx=4)
        ttk.Combobox(
            controls,
            textvariable=self.window_var,
            width=12,
            state="readonly",
            values=["Hann", "Hamming", "Rectangular"],
        ).grid(row=2, column=7, sticky="w", padx=4)

        ttk.Checkbutton(controls, text="Quitar DC", variable=self.detrend_var).grid(row=2, column=8, padx=4)
        ttk.Button(controls, text="Graficar FFT", command=self.plot).grid(row=1, column=8, padx=4, pady=4)
        ttk.Button(controls, text="Guardar TXT/PNG", command=self.save_results).grid(row=1, column=9, padx=4, pady=4)

        for i in range(10):
            controls.columnconfigure(i, weight=0)
        controls.columnconfigure(1, weight=1)

        self.summary = tk.Text(main, height=5, wrap="word")
        self.summary.pack(fill=tk.X, pady=(6, 8))
        self.summary.insert(
            "1.0",
            "Carga un CSV. La FFT es directa: una sola ventana, sin Welch y sin promedio.\n",
        )

        self.tabs = ttk.Notebook(main)
        self.tabs.pack(fill=tk.BOTH, expand=True)

        self.fig_fft = Figure(figsize=(10, 6), dpi=100)
        self.ax_time = self.fig_fft.add_subplot(211)
        self.ax_fft = self.fig_fft.add_subplot(212)
        self.canvas_fft = self._add_figure_tab("Tiempo y FFT", self.fig_fft)

        self.fig_spec = Figure(figsize=(10, 6), dpi=100)
        self.ax_spec = self.fig_spec.add_subplot(111)
        self.canvas_spec = self._add_figure_tab("Espectrograma", self.fig_spec)

    def _add_figure_tab(self, title: str, fig: Figure) -> FigureCanvasTkAgg:
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text=title)
        canvas = FigureCanvasTkAgg(fig, master=tab)
        toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        return canvas

    def load_file(self) -> None:
        path_str = filedialog.askopenfilename(
            title="Seleccionar CSV / TXT / meta JSON",
            initialdir=str(WORKDIR),
            filetypes=[("Datos", "*.csv *.txt *.json"), ("CSV/TXT", "*.csv *.txt"), ("Meta JSON", "*.json"), ("Todos", "*.*")],
        )
        if path_str:
            self._load_path(Path(path_str))

    def load_folder(self) -> None:
        dir_str = filedialog.askdirectory(title="Selecciona la carpeta de la prueba", initialdir=str(WORKDIR))
        if not dir_str:
            return
        metas = sorted(Path(dir_str).glob("*_meta.json"))
        if not metas:
            messagebox.showerror("Sin meta", f"No encontré *_meta.json en:\n{dir_str}")
            return
        self._load_path(metas[0])

    def _load_path(self, path: Path) -> None:
        try:
            if path.suffix.lower() == ".json":
                self.data = load_ads1219_meta(path)
            else:
                self.data = load_numeric_csv(path)
        except Exception as exc:
            messagebox.showerror("No se pudo cargar", str(exc))
            return

        names = list(self.data.columns)
        self.time_combo["values"] = ["Indice/muestras"] + names
        self.signal_combo["values"] = names
        self.time_var.set(choose_default_column(["Indice/muestras"] + names, ["tiempo_comun", "tiempo", "time", "t_s"]))
        self.signal_var.set(
            choose_default_column(
                names,
                ["fuerza_v", "fuerza", "force", "voltaje", "aceleracion_g", "aceleracion", "accel", "ai"],
            )
        )
        self._auto_time_for_signal()
        self.file_var.set(str(self.data.path))
        self.summary.delete("1.0", tk.END)
        msg = f"Archivo cargado: {self.data.path.name}\nColumnas: {', '.join(names)}\n"
        meta = getattr(self.data, "meta", None)
        if meta:
            fsf = meta.get("fs_hz_force")
            fsa = meta.get("fs_hz_accel")
            msg += (f"fs fuerza/doc (reloj Pico) = {fsf} Hz  |  fs accel nativa (9234) = {fsa} Hz\n"
                    f"Alineado (~{fsf} Hz): fuerza_V / aceleracion_g + tiempo_comun_s.  "
                    f"Frec. altas: aceleracion_full_g + tiempo_accel_s.\n")
        self.summary.insert("1.0", msg)

    def _auto_time_for_signal(self, _event=None) -> None:
        signal_name = self.signal_var.get().lower()
        values = list(self.time_combo["values"])

        def setif(*candidates: str) -> None:
            for c in candidates:
                if c in values:
                    self.time_var.set(c)
                    return

        if "full" in signal_name and "tiempo_accel_s" in values:
            setif("tiempo_accel_s")                       # aceleración nativa (frec. altas)
        elif "fuerza" in signal_name or "force" in signal_name or "voltaje" in signal_name:
            setif("tiempo_comun_s", "tiempo_fuerza_s")    # fuerza (rejilla del Pico)
        elif "acel" in signal_name or "accel" in signal_name:
            setif("tiempo_comun_s", "tiempo_accel_s")     # aceleración alineada (doc)

    def _get_selection(self) -> tuple[np.ndarray, np.ndarray, float, str]:
        if self.data is None:
            raise ValueError("Primero carga un CSV.")
        signal_name = self.signal_var.get()
        if signal_name not in self.data.columns:
            raise ValueError("Selecciona una columna de senal valida.")

        x = self.data.columns[signal_name]
        time_name = self.time_var.get()
        if time_name and time_name != "Indice/muestras" and time_name in self.data.columns:
            time_s = self.data.columns[time_name]
            fs = estimate_fs(time_s)
        else:
            fs = float(self.fs_var.get())
            if fs <= 0:
                raise ValueError("Ingresa fs en Hz cuando no usas columna de tiempo.")
            time_s = np.arange(len(x), dtype=float) / fs

        mask = np.isfinite(time_s) & np.isfinite(x)
        time_s = time_s[mask]
        x = x[mask]

        t0_text = self.t0_var.get().strip()
        t1_text = self.t1_var.get().strip()
        if t0_text:
            time0 = float(t0_text)
            keep = time_s >= time0
            time_s, x = time_s[keep], x[keep]
        if t1_text:
            time1 = float(t1_text)
            keep = time_s <= time1
            time_s, x = time_s[keep], x[keep]

        if len(x) < 8:
            raise ValueError("El tramo seleccionado tiene muy pocas muestras.")
        return time_s, x, fs, signal_name

    def plot(self) -> None:
        try:
            time_s, x, fs, signal_name = self._get_selection()
            freq, amp, amp_db = direct_fft(
                time_s=time_s,
                x=x,
                fs=fs,
                window_name=self.window_var.get(),
                detrend=self.detrend_var.get(),
            )
            fmax = float(self.fmax_var.get() or np.nanmax(freq))
            rpm = float(self.rpm_var.get() or 0.0)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        nyquist = fs / 2.0
        fmax = min(fmax, nyquist)
        in_band = freq <= fmax
        if not np.any(in_band):
            in_band = np.ones_like(freq, dtype=bool)

        self.ax_time.clear()
        self.ax_fft.clear()
        self.ax_time.plot(time_s, x, color="black", linewidth=0.8)
        self.ax_time.set_title(f"{signal_name} en tiempo", fontfamily="serif", fontstyle="italic")
        self.ax_time.set_xlabel("Tiempo (s)", fontfamily="serif", fontstyle="italic")
        self.ax_time.set_ylabel(signal_name, fontfamily="serif", fontstyle="italic")
        self.ax_time.grid(True, alpha=0.35)

        self.ax_fft.plot(freq[in_band], amp[in_band], color="black", linewidth=0.8)
        self._mark_reference_lines(self.ax_fft, fmax, rpm)
        self.ax_fft.set_title("FFT directa sin promedio", fontfamily="serif", fontstyle="italic")
        self.ax_fft.set_xlabel("Frecuencia (Hz)", fontfamily="serif", fontstyle="italic")
        self.ax_fft.set_ylabel("Amplitud", fontfamily="serif", fontstyle="italic")
        self.ax_fft.grid(True, alpha=0.35)
        self.ax_fft.set_xlim(0, fmax)
        self.ax_fft.legend(loc="upper right", fontsize=8)
        self.fig_fft.tight_layout()
        self.canvas_fft.draw()

        self._plot_spectrogram(time_s, x, fs, fmax, rpm, signal_name)

        peaks = self._find_top_peaks(freq[in_band], amp[in_band])
        rms_ac = float(np.std(x - np.mean(x)))
        red_lines = self._line_values(freq, amp, RED_FREQS_HZ)
        self.last_result = {
            "time_s": time_s,
            "x": x,
            "fs": fs,
            "freq": freq,
            "amp": amp,
            "amp_db": amp_db,
            "signal_name": signal_name,
            "fmax": fmax,
            "rpm": rpm,
            "rms_ac": rms_ac,
            "peaks": peaks,
            "red_lines": red_lines,
        }

        lines = [
            f"Archivo: {self.data.path.name if self.data else ''}",
            f"Senal: {signal_name} | muestras: {len(x)} | fs: {fs:.6g} Hz | Nyquist: {nyquist:.6g} Hz",
            f"Tramo: {time_s[0]:.6g} a {time_s[-1]:.6g} s | RMS AC: {rms_ac:.6g}",
            "Picos principales FFT directa: "
            + ", ".join([f"{f:.3g} Hz ({a:.3g})" for f, a in peaks[:8]]),
            "Lineas red: " + ", ".join([f"{f:.0f} Hz={a:.3g}" for f, a in red_lines.items()]),
        ]
        self.summary.delete("1.0", tk.END)
        self.summary.insert("1.0", "\n".join(lines))

    def _plot_spectrogram(self, time_s: np.ndarray, x: np.ndarray, fs: float, fmax: float, rpm: float, name: str) -> None:
        self.ax_spec.clear()
        nperseg = int(min(max(64, 2 ** math.floor(math.log2(max(8, len(x) // 16)))), 4096, len(x)))
        try:
            f, t, sxx = simple_spectrogram(x - np.mean(x), fs=fs, nperseg=nperseg, overlap=0.75)
            keep = f <= fmax
            extent_t = time_s[0] + t
            z = 20.0 * np.log10(np.maximum(sxx[keep], 1e-20))
            self.ax_spec.contourf(extent_t, f[keep], z, levels=40, cmap="jet")
            self._mark_reference_lines(self.ax_spec, fmax, rpm)
            self.ax_spec.set_title(f"Espectrograma {name}", fontfamily="serif", fontstyle="italic")
            self.ax_spec.set_xlabel("Tiempo (s)", fontfamily="serif", fontstyle="italic")
            self.ax_spec.set_ylabel("Frecuencia (Hz)", fontfamily="serif", fontstyle="italic")
            self.ax_spec.set_ylim(0, fmax)
            self.ax_spec.grid(True, alpha=0.35)
            self.ax_spec.legend(loc="upper right", fontsize=8)
        except Exception as exc:
            self.ax_spec.text(0.05, 0.5, f"No se pudo calcular espectrograma: {exc}", transform=self.ax_spec.transAxes)
        self.fig_spec.tight_layout()
        self.canvas_spec.draw()

    def _mark_reference_lines(self, ax, fmax: float, rpm: float) -> None:
        for f in RED_FREQS_HZ:
            if 0 < f <= fmax:
                if ax is self.ax_fft:
                    ax.axvline(f, color="red", linestyle="--", linewidth=0.8, alpha=0.65, label="red 60 Hz" if f == 60 else None)
                else:
                    ax.axhline(f, color="red", linestyle="--", linewidth=0.8, alpha=0.65, label="red 60 Hz" if f == 60 else None)
        if rpm > 0:
            base = rpm / 60.0
            k = 1
            while k * base <= fmax:
                freq = k * base
                if ax is self.ax_fft:
                    ax.axvline(freq, color="royalblue", linestyle=":", linewidth=0.8, alpha=0.75, label="armonicos rpm" if k == 1 else None)
                else:
                    ax.axhline(freq, color="royalblue", linestyle=":", linewidth=0.8, alpha=0.75, label="armonicos rpm" if k == 1 else None)
                k += 1

    def _find_top_peaks(self, freq: np.ndarray, amp: np.ndarray) -> list[tuple[float, float]]:
        if len(freq) < 3:
            return []
        valid = freq > 0
        freq = freq[valid]
        amp = amp[valid]
        if len(freq) == 0:
            return []
        if scipy_signal is not None and len(freq) > 5:
            idx, _ = scipy_signal.find_peaks(amp)
            if len(idx) == 0:
                idx = np.arange(len(amp))
        else:
            idx = np.arange(len(amp))
        order = idx[np.argsort(amp[idx])[::-1]]
        return [(float(freq[i]), float(amp[i])) for i in order[:20]]

    def _line_values(self, freq: np.ndarray, amp: np.ndarray, lines_hz: list[float]) -> dict[float, float]:
        out: dict[float, float] = {}
        for target in lines_hz:
            if target <= freq[-1]:
                idx = int(np.argmin(np.abs(freq - target)))
                out[target] = float(amp[idx])
        return out

    def save_results(self) -> None:
        if self.last_result is None:
            self.plot()
            if self.last_result is None:
                return
        if self.data is None:
            return

        result = self.last_result
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        signal_name = clean_name(str(result["signal_name"]))
        outdir = WORKDIR / f"fft_manual_{self.data.path.stem}_{signal_name}_{stamp}"
        outdir.mkdir(exist_ok=True)

        freq = result["freq"]
        amp = result["amp"]
        amp_db = result["amp_db"]
        fft_txt = outdir / f"datos_fft_{signal_name}.txt"
        with fft_txt.open("w", encoding="utf-8") as f:
            f.write("# FFT directa sin promedio\n")
            f.write(f"# archivo_origen: {self.data.path}\n")
            f.write(f"# senal: {result['signal_name']}\n")
            f.write(f"# fs_Hz: {result['fs']}\n")
            f.write(f"# rpm: {result['rpm']}\n")
            f.write("# columnas: frecuencia_Hz\tamplitud\tamplitud_dBV\n")
            for fi, ai, di in zip(freq, amp, amp_db):
                f.write(f"{fi:.12g}\t{ai:.12g}\t{di:.12g}\n")

        resumen_txt = outdir / "resumen_fft.txt"
        with resumen_txt.open("w", encoding="utf-8") as f:
            f.write(self.summary.get("1.0", tk.END).strip() + "\n")
            f.write("\nNota: FFT calculada en una sola ventana, sin Welch y sin promedio.\n")

        png_fft = outdir / f"fft_directa_{signal_name}.png"
        png_spec = outdir / f"espectrograma_{signal_name}.png"
        self.fig_fft.savefig(png_fft, dpi=220, bbox_inches="tight")
        self.fig_spec.savefig(png_spec, dpi=220, bbox_inches="tight")

        messagebox.showinfo(
            "Guardado",
            f"Archivos creados:\n{fft_txt}\n{resumen_txt}\n{png_fft}\n{png_spec}",
        )


def main() -> None:
    app = FFTApp()
    app.mainloop()


if __name__ == "__main__":
    main()
