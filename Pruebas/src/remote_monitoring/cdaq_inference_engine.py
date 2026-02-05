#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motores de inferencia para cDAQ UDP Receiver (Jetson).

Nivel 1 – EnvelopeInferenceEngine:
  • Envolvente de Hilbert de la aceleración
  • FFT → frecuencia dominante + THD
  • Estimación lineal de fuerza: F_est = k·E + b
  • Detección de corte activo (umbral de energía)

Nivel 2 – BoucWenInferenceEngine (hereda de Nivel 1):
  • Modelo KAN-PINN: F = m·a + k·x + c·v + α·k·E + (1-α)·k·z
  • Variable de histéresis z persistente entre bloques
  • Área del lazo de histéresis (indicador de desgaste)
  • Integración numérica accel → vel → disp con filtro anti-drift

Uso:
    from cdaq_inference_engine import EnvelopeInferenceEngine, BoucWenInferenceEngine
    engine = BoucWenInferenceEngine(fs=2500)

Parámetros identificados offline en:
    Pruebas/src/caracterizacion_fuerza/test_modelos_envolvente.py
    Pruebas/src/caracterizacion_fuerza/README.md
"""

import numpy as np
from scipy.signal import hilbert, butter, sosfilt
from scipy.integrate import cumulative_trapezoid

# Importar clase base del receiver
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdaq_udp_receiver import InferenceEngine


# ═══════════════════════════════════════════════════════════════
# Nombres de columnas para CSV (usados por el receiver)
# ═══════════════════════════════════════════════════════════════

ENVELOPE_RESULT_NAMES = [
    "F_est_mean_V", "F_est_peak_V", "envelope_rms_g",
    "freq_dom_Hz", "freq_dom_mag", "THD_accel_pct", "THD_force_pct",
    "cutting_flag",
    "force_rms_V", "accel_rms_g", "force_pp_V", "accel_pp_g",
]

BOUCWEN_RESULT_NAMES = ENVELOPE_RESULT_NAMES + [
    "F_bw_mean_V", "F_bw_peak_V",
    "z_mean", "z_last", "z_abs_max",
    "alpha", "R2_block",
    "hyst_area", "hyst_energy_pct",
    "vel_rms", "disp_rms",
]


# ═══════════════════════════════════════════════════════════════
# Utilidades compartidas
# ═══════════════════════════════════════════════════════════════

def _calc_thd(spectrum, fundamental_idx, n_harmonics=5):
    """Calcula THD (%) a partir del espectro y el índice del fundamental."""
    if fundamental_idx < 1 or fundamental_idx >= len(spectrum):
        return 0.0
    fund_mag = spectrum[fundamental_idx]
    if fund_mag < 1e-10:
        return 0.0
    harm_sum_sq = 0.0
    for h in range(2, n_harmonics + 2):
        idx = fundamental_idx * h
        if idx >= len(spectrum):
            break
        lo = max(0, idx - 2)
        hi = min(len(spectrum), idx + 3)
        harm_sum_sq += float(np.max(spectrum[lo:hi])) ** 2
    thd = np.sqrt(harm_sum_sq) / fund_mag * 100.0
    return float(min(thd, 999.0))


# ═══════════════════════════════════════════════════════════════
# Nivel 1: Envolvente + FFT + Estimación lineal de fuerza
# ═══════════════════════════════════════════════════════════════

class EnvelopeInferenceEngine(InferenceEngine):
    """
    Procesamiento de señal en tiempo real sobre cada bloque UDP.
    Devuelve 12 floats (ver ENVELOPE_RESULT_NAMES).
    """

    RESULT_NAMES = ENVELOPE_RESULT_NAMES

    # Parámetros del modelo lineal F = k_env * E + b_env
    # Identificados offline en caracterizacion_fuerza/
    K_ENV = 0.0669      # V/g  (pendiente envolvente→fuerza)
    B_ENV = 0.8837      # V    (offset)

    # Umbral de energía para detección de corte activo
    CUTTING_THRESHOLD_G = 0.05   # RMS de envolvente > este valor → corte

    N_HARMONICS = 5

    def __init__(self, fs: float = 2500.0, envelope_fc: float = 15.0):
        super().__init__()
        self.fs = fs
        self.envelope_fc = envelope_fc
        self._buf_len = int(fs * 0.5)   # 0.5 s de contexto
        self._accel_buf = np.zeros(0, dtype=np.float32)
        self._force_buf = np.zeros(0, dtype=np.float32)
        self._env_sos = butter(4, envelope_fc / (fs / 2),
                               btype='low', output='sos')

    def _update_fs(self, fs):
        if abs(fs - self.fs) > 1.0:
            self.fs = fs
            self._env_sos = butter(4, self.envelope_fc / (fs / 2),
                                   btype='low', output='sos')
            self._buf_len = int(fs * 0.5)

    def _accumulate(self, force, accel):
        self._accel_buf = np.concatenate(
            [self._accel_buf, accel])[-self._buf_len:]
        self._force_buf = np.concatenate(
            [self._force_buf, force])[-self._buf_len:]

    def _compute_envelope(self):
        """Retorna (env_filtered, envelope_rms) sobre el buffer completo."""
        a_buf = self._accel_buf
        n = len(a_buf)
        analytic = hilbert(a_buf)
        env_raw = np.abs(analytic).astype(np.float32)
        if n > 30:
            env = sosfilt(self._env_sos, env_raw).astype(np.float32)
        else:
            env = env_raw
        envelope_rms = float(np.sqrt(np.mean(env ** 2)))
        return env, envelope_rms

    def _compute_fft(self):
        """Retorna (freq_dom, freq_dom_mag, thd_accel, thd_force)."""
        n = len(self._accel_buf)
        n_fft = min(n, 2048)
        window = np.hanning(n_fft)

        a_win = self._accel_buf[-n_fft:] * window
        A_fft = np.abs(np.fft.rfft(a_win)) * 2.0 / n_fft
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.fs)
        A_nodc = A_fft[1:]
        f_nodc = freqs[1:]

        if len(A_nodc) > 0:
            pk = int(np.argmax(A_nodc))
            freq_dom = float(f_nodc[pk])
            freq_dom_mag = float(A_nodc[pk])
            thd_accel = _calc_thd(A_nodc, pk, self.N_HARMONICS)
        else:
            freq_dom = freq_dom_mag = thd_accel = 0.0

        f_win = self._force_buf[-n_fft:] * window
        F_fft = np.abs(np.fft.rfft(f_win)) * 2.0 / n_fft
        F_nodc = F_fft[1:]
        if len(F_nodc) > 0:
            thd_force = _calc_thd(F_nodc, int(np.argmax(F_nodc)), self.N_HARMONICS)
        else:
            thd_force = 0.0

        return freq_dom, freq_dom_mag, thd_accel, thd_force

    def predict(self, force: np.ndarray, accel: np.ndarray,
                fs: float, seq: int):
        self._update_fs(fs)
        self._accumulate(force, accel)

        n = len(self._accel_buf)
        if n < 64:
            return None

        env, envelope_rms = self._compute_envelope()
        F_est = self.K_ENV * env + self.B_ENV
        blk = len(accel)
        F_est_mean = float(np.mean(F_est[-blk:]))
        F_est_peak = float(np.max(F_est[-blk:]))

        freq_dom, freq_dom_mag, thd_accel, thd_force = self._compute_fft()
        cutting_flag = 1.0 if envelope_rms > self.CUTTING_THRESHOLD_G else 0.0

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


# ═══════════════════════════════════════════════════════════════
# Nivel 2: Bouc-Wen / KAN-PINN en tiempo real
# ═══════════════════════════════════════════════════════════════

class BoucWenInferenceEngine(EnvelopeInferenceEngine):
    """
    Modelo KAN-PINN completo ejecutado en tiempo real:
        F = m·a + k·x + c·v + α·k·E + (1-α)·k·z

    Ecuación de histéresis Bouc-Wen:
        dz/dt = v·(A_bw - |z|^n · (B_bw·sign(v·z) + C_bw))

    Parámetros FRF identificados (sesión 135910):
        m = 0.107 kg, k = 1424 N/m, c = 3.73 Ns/m

    Devuelve 23 floats (ver BOUCWEN_RESULT_NAMES):
        [0..11]  = mismos que EnvelopeInferenceEngine
        [12..22] = Bouc-Wen específicos
    """

    RESULT_NAMES = BOUCWEN_RESULT_NAMES

    # ── Parámetros físicos identificados offline ──
    # FRF (sesión 135910)
    MASS = 0.107        # kg
    STIFFNESS = 1424.0  # N/m
    DAMPING = 3.73      # Ns/m

    # Bouc-Wen (KAN-PINN, tabla comparativa README.md)
    # α ≈ 1.0 para sistema lineal; usar 0.81 (BW Simple corte) como default
    ALPHA = 0.811
    A_BW = 1.0
    B_BW = 0.5
    C_BW = 0.5
    N_BW = 1.0

    # Filtro anti-drift para integración accel→vel→disp
    DRIFT_FC = 5.0      # Hz, pasa-altos

    def __init__(self, fs: float = 2500.0, envelope_fc: float = 15.0,
                 alpha: float = None, mass: float = None,
                 stiffness: float = None, damping: float = None):
        super().__init__(fs=fs, envelope_fc=envelope_fc)

        # Permitir override de parámetros
        if alpha is not None:
            self.ALPHA = alpha
        if mass is not None:
            self.MASS = mass
        if stiffness is not None:
            self.STIFFNESS = stiffness
        if damping is not None:
            self.DAMPING = damping

        # Estado persistente de histéresis entre bloques
        self._z = 0.0

        # Filtro pasa-altos para integración (SOS)
        self._drift_sos = butter(2, self.DRIFT_FC / (fs / 2),
                                 btype='high', output='sos')

        # Buffer de envolvente anterior (para área del lazo)
        self._prev_env = None
        self._prev_force = None

    def _update_fs(self, fs):
        super()._update_fs(fs)
        if abs(fs - self.fs) > 1.0:
            self._drift_sos = butter(2, self.DRIFT_FC / (fs / 2),
                                     btype='high', output='sos')

    def _integrate_accel(self, accel_buf):
        """Integra aceleración → velocidad → desplazamiento con filtro anti-drift."""
        dt = 1.0 / self.fs
        n = len(accel_buf)
        if n < 10:
            return np.zeros(n), np.zeros(n)

        # Integrar con cumtrapz + filtro pasa-altos
        vel_raw = cumulative_trapezoid(accel_buf, dx=dt, initial=0)
        vel = sosfilt(self._drift_sos, vel_raw).astype(np.float32)
        disp_raw = cumulative_trapezoid(vel, dx=dt, initial=0)
        disp = sosfilt(self._drift_sos, disp_raw).astype(np.float32)
        return vel, disp

    def _run_bouc_wen(self, vel, env, dt):
        """Ejecuta Bouc-Wen sobre el bloque, mantiene estado z entre llamadas."""
        n = len(vel)
        z_arr = np.zeros(n, dtype=np.float64)
        z = self._z

        A = self.A_BW
        B = self.B_BW
        C = self.C_BW
        n_bw = self.N_BW

        for i in range(n):
            v = vel[i]
            abs_z = abs(z) + 1e-8
            sign_vz = 1.0 if v * z >= 0 else -1.0
            dz = v * (A - (abs_z ** n_bw) * (B * sign_vz + C))
            z = z + dz * dt
            z = max(-1.0, min(1.0, z))   # clip
            z_arr[i] = z

        self._z = z
        return z_arr

    def _calc_hysteresis_area(self, env, force):
        """Calcula el área del lazo F vs E usando la fórmula del shoelace."""
        n = len(env)
        if n < 10:
            return 0.0
        # Shoelace simplificado
        area = 0.0
        for i in range(n - 1):
            area += env[i] * force[i + 1] - env[i + 1] * force[i]
        return abs(area) * 0.5

    def predict(self, force: np.ndarray, accel: np.ndarray,
                fs: float, seq: int):
        self._update_fs(fs)
        self._accumulate(force, accel)

        n = len(self._accel_buf)
        if n < 64:
            return None

        # ── Nivel 1: envolvente + FFT + stats ──
        env, envelope_rms = self._compute_envelope()
        F_est_lin = self.K_ENV * env + self.B_ENV
        blk = len(accel)
        F_est_mean = float(np.mean(F_est_lin[-blk:]))
        F_est_peak = float(np.max(F_est_lin[-blk:]))

        freq_dom, freq_dom_mag, thd_accel, thd_force = self._compute_fft()
        cutting_flag = 1.0 if envelope_rms > self.CUTTING_THRESHOLD_G else 0.0

        force_rms = float(np.sqrt(np.mean(force ** 2)))
        accel_rms = float(np.sqrt(np.mean(accel ** 2)))
        force_pp = float(np.ptp(force))
        accel_pp = float(np.ptp(accel))

        # ── Nivel 2: Bouc-Wen / KAN-PINN ──
        dt = 1.0 / self.fs
        a_buf = self._accel_buf

        # Integrar aceleración → velocidad, desplazamiento
        vel, disp = self._integrate_accel(a_buf)

        # Ejecutar Bouc-Wen sobre el bloque actual (últimas blk muestras)
        z_arr = self._run_bouc_wen(vel[-blk:], env[-blk:], dt)

        # Modelo KAN-PINN: F = m·a + k·x + c·v + α·k·E + (1-α)·k·z
        m = self.MASS
        k = self.STIFFNESS
        c = self.DAMPING
        alpha = self.ALPHA

        a_blk = a_buf[-blk:]
        v_blk = vel[-blk:]
        x_blk = disp[-blk:]
        E_blk = env[-blk:]

        F_bw = (m * a_blk + k * x_blk + c * v_blk
                + alpha * k * E_blk + (1 - alpha) * k * z_arr)

        # Normalizar F_bw al rango de fuerza medida para comparación
        f_blk = force
        f_mean = np.mean(f_blk)
        f_std = np.std(f_blk) + 1e-10
        bw_mean_raw = np.mean(F_bw)
        bw_std_raw = np.std(F_bw) + 1e-10
        F_bw_scaled = (F_bw - bw_mean_raw) / bw_std_raw * f_std + f_mean

        F_bw_mean = float(np.mean(F_bw_scaled))
        F_bw_peak = float(np.max(F_bw_scaled))

        # R² del bloque
        ss_res = np.sum((f_blk - F_bw_scaled) ** 2)
        ss_tot = np.sum((f_blk - f_mean) ** 2) + 1e-10
        R2_block = float(1.0 - ss_res / ss_tot)
        R2_block = max(-1.0, min(1.0, R2_block))

        # Estadísticas de z
        z_mean = float(np.mean(z_arr))
        z_last = float(z_arr[-1])
        z_abs_max = float(np.max(np.abs(z_arr)))

        # Área del lazo de histéresis (F vs E)
        hyst_area = self._calc_hysteresis_area(E_blk, F_bw_scaled)

        # Energía de histéresis como % de energía total
        total_energy = float(np.sum(np.abs(f_blk))) + 1e-10
        hyst_energy_pct = float(hyst_area / total_energy * 100.0)
        hyst_energy_pct = min(hyst_energy_pct, 100.0)

        vel_rms = float(np.sqrt(np.mean(v_blk ** 2)))
        disp_rms = float(np.sqrt(np.mean(x_blk ** 2)))

        return np.array([
            # [0..11] Nivel 1
            F_est_mean, F_est_peak, envelope_rms,
            freq_dom, freq_dom_mag, thd_accel, thd_force,
            cutting_flag,
            force_rms, accel_rms, force_pp, accel_pp,
            # [12..22] Nivel 2 Bouc-Wen
            F_bw_mean, F_bw_peak,
            z_mean, z_last, z_abs_max,
            alpha, R2_block,
            hyst_area, hyst_energy_pct,
            vel_rms, disp_rms,
        ], dtype=np.float32)
