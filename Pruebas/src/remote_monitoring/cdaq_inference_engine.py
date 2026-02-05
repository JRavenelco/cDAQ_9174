#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motores de inferencia para cDAQ UDP Receiver (Jetson).

Nivel 1 – EnvelopeInferenceEngine:
  • Envolvente de Hilbert de la aceleración
  • FFT → frecuencia dominante + THD
  • Estimación lineal de fuerza: F_est = k·E + b
  • Detección de corte activo (umbral de energía)

Uso:
    from cdaq_inference_engine import EnvelopeInferenceEngine
    engine = EnvelopeInferenceEngine(fs=2500)
"""

import numpy as np
from scipy.signal import hilbert, butter, sosfilt

# Importar clase base del receiver
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdaq_udp_receiver import InferenceEngine


# ═══════════════════════════════════════════════════════════════
# Nivel 1: Envolvente + FFT + Estimación lineal de fuerza
# ═══════════════════════════════════════════════════════════════

class EnvelopeInferenceEngine(InferenceEngine):
    """
    Procesamiento de señal en tiempo real sobre cada bloque UDP.

    Resultados devueltos (12 floats):
      [0] F_est_mean    – Fuerza estimada media (V) del bloque
      [1] F_est_peak    – Fuerza estimada pico (V) del bloque
      [2] envelope_rms  – RMS de la envolvente de aceleración (g)
      [3] freq_dom      – Frecuencia dominante de aceleración (Hz)
      [4] freq_dom_mag  – Magnitud de la freq dominante
      [5] thd_accel     – THD de aceleración (%)
      [6] thd_force     – THD de fuerza (%)
      [7] cutting_flag  – 1.0 si hay corte activo, 0.0 si no
      [8] force_rms     – RMS de fuerza cruda (V)
      [9] accel_rms     – RMS de aceleración cruda (g)
     [10] force_pp      – Pico-a-pico de fuerza (V)
     [11] accel_pp      – Pico-a-pico de aceleración (g)
    """

    # Parámetros del modelo lineal F = k_env * E + b_env
    # Identificados offline en caracterizacion_fuerza/
    K_ENV = 0.0669      # V/g  (pendiente envolvente→fuerza)
    B_ENV = 0.8837      # V    (offset)

    # Umbral de energía para detección de corte activo
    CUTTING_THRESHOLD_G = 0.05   # RMS de envolvente > este valor → corte

    # Número de armónicos para THD
    N_HARMONICS = 5

    def __init__(self, fs: float = 2500.0, envelope_fc: float = 15.0):
        super().__init__()
        self.fs = fs
        self.envelope_fc = envelope_fc

        # Buffer circular para acumular muestras entre bloques
        # (la envolvente de Hilbert necesita contexto)
        self._buf_len = int(fs * 0.5)   # 0.5 s de contexto
        self._accel_buf = np.zeros(0, dtype=np.float32)
        self._force_buf = np.zeros(0, dtype=np.float32)

        # Filtro pasa-bajos para envolvente (SOS para estabilidad)
        self._env_sos = butter(4, envelope_fc / (fs / 2),
                               btype='low', output='sos')

    def predict(self, force: np.ndarray, accel: np.ndarray,
                fs: float, seq: int):
        # Actualizar fs si cambió
        if abs(fs - self.fs) > 1.0:
            self.fs = fs
            self._env_sos = butter(4, self.envelope_fc / (fs / 2),
                                   btype='low', output='sos')
            self._buf_len = int(fs * 0.5)

        # ── Acumular en buffer circular ──
        self._accel_buf = np.concatenate(
            [self._accel_buf, accel])[-self._buf_len:]
        self._force_buf = np.concatenate(
            [self._force_buf, force])[-self._buf_len:]

        n = len(self._accel_buf)
        if n < 64:
            return None   # no hay suficientes datos aún

        a_buf = self._accel_buf
        f_buf = self._force_buf

        # ── 1. Envolvente de Hilbert ──
        analytic = hilbert(a_buf)
        env_raw = np.abs(analytic).astype(np.float32)
        # Filtrar solo si hay suficientes muestras para el filtro
        if n > 30:
            env = sosfilt(self._env_sos, env_raw).astype(np.float32)
        else:
            env = env_raw
        envelope_rms = float(np.sqrt(np.mean(env ** 2)))

        # ── 2. Estimación lineal de fuerza ──
        F_est = self.K_ENV * env + self.B_ENV
        F_est_mean = float(np.mean(F_est[-len(accel):]))
        F_est_peak = float(np.max(F_est[-len(accel):]))

        # ── 3. FFT de aceleración → freq dominante + THD ──
        n_fft = min(n, 2048)
        window = np.hanning(n_fft)
        a_win = a_buf[-n_fft:] * window
        A_fft = np.abs(np.fft.rfft(a_win)) * 2.0 / n_fft
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.fs)

        # Saltar DC (índice 0)
        A_fft_nodc = A_fft[1:]
        freqs_nodc = freqs[1:]

        if len(A_fft_nodc) > 0:
            pk_idx = int(np.argmax(A_fft_nodc))
            freq_dom = float(freqs_nodc[pk_idx])
            freq_dom_mag = float(A_fft_nodc[pk_idx])
            thd_accel = self._calc_thd(A_fft_nodc, pk_idx)
        else:
            freq_dom = 0.0
            freq_dom_mag = 0.0
            thd_accel = 0.0

        # FFT de fuerza → THD
        f_win = f_buf[-n_fft:] * window
        F_fft = np.abs(np.fft.rfft(f_win)) * 2.0 / n_fft
        F_fft_nodc = F_fft[1:]
        if len(F_fft_nodc) > 0:
            pk_f = int(np.argmax(F_fft_nodc))
            thd_force = self._calc_thd(F_fft_nodc, pk_f)
        else:
            thd_force = 0.0

        # ── 4. Detección de corte activo ──
        cutting_flag = 1.0 if envelope_rms > self.CUTTING_THRESHOLD_G else 0.0

        # ── 5. Estadísticas crudas (sobre el bloque actual) ──
        force_rms = float(np.sqrt(np.mean(force ** 2)))
        accel_rms = float(np.sqrt(np.mean(accel ** 2)))
        force_pp = float(np.ptp(force))
        accel_pp = float(np.ptp(accel))

        return np.array([
            F_est_mean, F_est_peak, envelope_rms,
            freq_dom, freq_dom_mag, thd_accel, thd_force,
            cutting_flag,
            force_rms, accel_rms, force_pp, accel_pp,
        ], dtype=np.float32)

    def _calc_thd(self, spectrum: np.ndarray, fundamental_idx: int) -> float:
        """Calcula THD (%) a partir del espectro y el índice del fundamental."""
        if fundamental_idx < 1 or fundamental_idx >= len(spectrum):
            return 0.0
        fund_mag = spectrum[fundamental_idx]
        if fund_mag < 1e-10:
            return 0.0

        harm_sum_sq = 0.0
        for h in range(2, self.N_HARMONICS + 2):
            idx = fundamental_idx * h
            if idx >= len(spectrum):
                break
            # Buscar pico en ventana ±2 bins
            lo = max(0, idx - 2)
            hi = min(len(spectrum), idx + 3)
            harm_sum_sq += float(np.max(spectrum[lo:hi])) ** 2

        thd = np.sqrt(harm_sum_sq) / fund_mag * 100.0
        return float(min(thd, 999.0))   # cap para evitar outliers
