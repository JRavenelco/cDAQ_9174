#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cDAQ UDP Receiver  –  Daemon ligero para JETSON.

Recibe datos de fuerza (NI 9205) y aceleración (NI 9234) vía UDP desde
el sender en Windows.  Guarda a CSV, corre inferencia en GPU
(JAX / PINN / KAN / Hailo – pluggable) y devuelve los resultados al
sender por INFER_PORT para que Windows los muestre en su GUI.

Sin GUI – el monitoreo visual se hace desde Windows.

Uso:
    python cdaq_udp_receiver.py
    python cdaq_udp_receiver.py --sender-ip 192.168.137.1
    python cdaq_udp_receiver.py --no-csv --no-infer   # solo recibir
"""

import sys
import os
import time
import socket
import threading
import argparse
import csv
import signal
import logging
from collections import deque
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass, field, asdict
import math

import numpy as np

# Protocolo compartido
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdaq_udp_protocol import (
    DATA_PORT, CTRL_PORT, INFER_PORT, MAGIC,
    DEFAULT_FS, DEFAULT_SPR, DEFAULT_NCH,
    MSG_SYNC_REQ, MSG_SYNC_RSP, MSG_START, MSG_STOP,
    unpack_data, pack_start, pack_stop, pack_sync_req,
    unpack_sync_rsp, pack_cfg, pack_infer,
)

# ─── Logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rx")


# ═══════════════════════════════════════════════════════════════
# Time Sync (NTP-like)
# ═══════════════════════════════════════════════════════════════
class TimeSynchronizer:
    """Estima offset = t_sender - t_local usando intercambios NTP-like."""

    def __init__(self):
        self.offset = 0.0
        self.rtt = float("inf")
        self.n_samples = 0

    def do_sync(self, sender_ip: str, n_rounds: int = 8, timeout: float = 0.5):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        offsets, rtts = [], []
        for _ in range(n_rounds):
            t1 = time.perf_counter()
            sock.sendto(pack_sync_req(t1), (sender_ip, CTRL_PORT))
            try:
                data, _ = sock.recvfrom(256)
                t4 = time.perf_counter()
                result = unpack_sync_rsp(data)
                if result is None:
                    continue
                t1_echo, t2, t3 = result
                rtt = (t4 - t1) - (t3 - t2)
                offset = ((t2 - t1) + (t3 - t4)) / 2.0
                offsets.append(offset)
                rtts.append(rtt)
            except socket.timeout:
                continue
        sock.close()
        if offsets:
            self.offset = float(np.median(offsets))
            self.rtt = float(np.median(rtts))
            self.n_samples = len(offsets)
        return self.n_samples > 0

    def to_local(self, t_sender: float) -> float:
        return t_sender - self.offset


# ═══════════════════════════════════════════════════════════════
# Condiciones de corte  (modelo mecanicista Altintas & Budak)
# ═══════════════════════════════════════════════════════════════

@dataclass
class CuttingConditions:
    """Parámetros de corte para el modelo mecanicista de fuerzas.

    Modelo:  F_t = K_tc · a_p · f_z · sin(φ) + K_te · a_p
    Media (slotting):  F̄_t = N_f · K_tc · a_p · f_z / π

    Herramienta default: Korloy AMSA3100HS (fresa escuadrar)
      D=25.4mm (1"), Z=2 insertos, κ=90° → sin(κ)=1
    Material default: Al 6061-T6  (Ktc≈800 N/mm²)
    """
    cut_id: int = 0                # Identificador de condición (se incrementa)
    label: str = "idle"            # Etiqueta libre ("corte1", "prof_0.5mm", etc.)
    ap_mm: float = 0.0            # Profundidad axial de corte (mm)
    rpm_spindle: float = 0.0      # RPM del husillo
    rpm_feed: float = 0.0         # RPM del eje de avance X
    feed_mmrev: float = 0.0       # Avance por revolución (mm/rev) – si se conoce
    tool_diam_mm: float = 25.4    # Korloy AMSA3100HS: 1" = 25.4 mm
    n_flutes: int = 2             # 2 insertos
    Ktc: float = 800.0            # Al 6061-T6 (N/mm²)
    Kte: float = 10.0             # Coef. de filo (N/mm)

    @property
    def fz_mm(self) -> float:
        """Avance por diente (mm/diente)."""
        if self.feed_mmrev > 0:
            return self.feed_mmrev / self.n_flutes
        if self.rpm_spindle > 0 and self.rpm_feed > 0:
            # Estimar: avance lineal ≈ rpm_feed * paso_tornillo
            # Asumimos tornillo de paso 2mm (típico mesa CNC hobby)
            feed_mm_min = self.rpm_feed * 2.0
            return feed_mm_min / (self.n_flutes * self.rpm_spindle)
        return 0.0

    @property
    def Ft_mean_N(self) -> float:
        """Fuerza tangencial media teórica (N) para slotting completo."""
        fz = self.fz_mm
        if fz <= 0 or self.ap_mm <= 0:
            return 0.0
        return (self.n_flutes * self.Ktc * self.ap_mm * fz) / math.pi

    @property
    def Ft_peak_N(self) -> float:
        """Fuerza tangencial pico teórica (N) – sin(φ)=1."""
        fz = self.fz_mm
        if fz <= 0 or self.ap_mm <= 0:
            return 0.0
        return self.Ktc * self.ap_mm * fz + self.Kte * self.ap_mm

    def header_lines(self) -> list:
        """Líneas de metadatos para escribir al inicio del CSV."""
        fz = self.fz_mm
        return [
            f"# cut_id={self.cut_id}  label={self.label}",
            f"# ap_mm={self.ap_mm}  rpm_spindle={self.rpm_spindle}"
            f"  rpm_feed={self.rpm_feed}  feed_mmrev={self.feed_mmrev}",
            f"# tool_diam_mm={self.tool_diam_mm}  n_flutes={self.n_flutes}",
            f"# Ktc={self.Ktc} N/mm²  Kte={self.Kte} N/mm",
            f"# fz={fz:.4f} mm/diente  Ft_mean={self.Ft_mean_N:.2f} N"
            f"  Ft_peak={self.Ft_peak_N:.2f} N",
        ]

    def summary(self) -> str:
        fz = self.fz_mm
        return (f"[CUT {self.cut_id}] {self.label}: "
                f"ap={self.ap_mm}mm  RPM={self.rpm_spindle}  "
                f"feed_RPM={self.rpm_feed}  fz={fz:.4f}mm  "
                f"Ft_mean={self.Ft_mean_N:.1f}N  Ft_peak={self.Ft_peak_N:.1f}N")


# ═══════════════════════════════════════════════════════════════
# CSV Logger  (datos crudos muestra-a-muestra)
# ═══════════════════════════════════════════════════════════════
class CSVLogger:
    """Escribe datos recibidos a CSV de forma eficiente (flush periódico)."""

    def __init__(self, path: str, cutting: CuttingConditions = None):
        self.path = path
        self._file = None
        self._writer = None
        self.rows = 0
        self.cutting = cutting or CuttingConditions()

    def open(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8",
                          buffering=1)   # line-buffered
        # Metadatos de condiciones de corte
        for line in self.cutting.header_lines():
            self._file.write(line + "\n")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "cut_id", "seq", "t_sender_s", "t_local_s", "sample_idx",
            "force_v", "accel_g",
        ])
        self.rows = 0
        log.info("CSV abierto: %s", self.path)

    def write_block(self, seq, t_sender, t_local, force, accel, cut_id=0):
        if self._writer is None:
            return
        for i in range(len(force)):
            self._writer.writerow([
                cut_id,
                seq, f"{t_sender:.9f}", f"{t_local:.9f}", i,
                f"{force[i]:.7f}", f"{accel[i]:.7f}",
            ])
        self.rows += len(force)

    def write_cut_change(self, cutting: CuttingConditions):
        """Escribe un marcador de cambio de condición en el CSV."""
        self.cutting = cutting
        if self._file:
            for line in cutting.header_lines():
                self._file.write(line + "\n")

    def flush(self):
        if self._file:
            self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            log.info("CSV cerrado: %d filas → %s", self.rows, self.path)
            self._file = None
            self._writer = None

    @property
    def is_open(self):
        return self._file is not None


# ═══════════════════════════════════════════════════════════════
# Inference CSV Logger  (un renglón por bloque con resultados)
# ═══════════════════════════════════════════════════════════════
class InferenceCSVLogger:
    """Escribe un CSV con una fila por bloque UDP: metadatos + inferencia.
    Permite que cualquier persona reproduzca y valide el experimento."""

    def __init__(self, path: str, result_names: list = None,
                 cutting: CuttingConditions = None):
        self.path = path
        self._file = None
        self._writer = None
        self.rows = 0
        self._result_names = result_names or []
        self.cutting = cutting or CuttingConditions()

    def open(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8",
                          buffering=1)
        # Metadatos de condiciones de corte
        for line in self.cutting.header_lines():
            self._file.write(line + "\n")
        self._writer = csv.writer(self._file)
        header = [
            "cut_id", "seq", "t_sender_s", "t_local_s", "fs_hz", "n_samples",
            "infer_us", "Ft_mean_N", "Ft_peak_N",
        ] + list(self._result_names)
        self._writer.writerow(header)
        self.rows = 0
        log.info("Inference CSV abierto: %s  (%d columnas de inferencia)",
                 self.path, len(self._result_names))

    def write_row(self, seq, t_sender, t_local, fs_hz, n_samples,
                  infer_us, result, cut_id=0, Ft_mean=0.0, Ft_peak=0.0):
        if self._writer is None:
            return
        row = [
            cut_id,
            seq, f"{t_sender:.9f}", f"{t_local:.9f}",
            f"{fs_hz:.1f}", n_samples, f"{infer_us:.1f}",
            f"{Ft_mean:.4f}", f"{Ft_peak:.4f}",
        ]
        if result is not None:
            row.extend(f"{v:.7f}" for v in result)
        self._writer.writerow(row)
        self.rows += 1

    def write_cut_change(self, cutting: CuttingConditions):
        """Escribe un marcador de cambio de condición en el CSV."""
        self.cutting = cutting
        if self._file:
            for line in cutting.header_lines():
                self._file.write(line + "\n")

    def flush(self):
        if self._file:
            self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            log.info("Inference CSV cerrado: %d filas → %s",
                     self.rows, self.path)
            self._file = None
            self._writer = None

    @property
    def is_open(self):
        return self._file is not None


# ═══════════════════════════════════════════════════════════════
# Inference Engine  (pluggable – JAX / PINN / KAN / Hailo)
# ═══════════════════════════════════════════════════════════════
class InferenceEngine:
    """
    Clase base / placeholder para inferencia en GPU.

    Para usar tu modelo real, hereda esta clase y sobreescribe `predict`.
    El método recibe force(N,) y accel(N,) y devuelve un array float32
    con los resultados que se enviarán a Windows.

    Ejemplo de subclase:
        class PINNInference(InferenceEngine):
            def __init__(self):
                super().__init__()
                import jax
                self.model = ...   # cargar modelo JAX/Flax
            def predict(self, force, accel, fs, seq):
                x = jax.numpy.stack([force, accel])
                return self.model(x)   # devuelve array 1-D
    """

    def __init__(self):
        self.calls = 0
        self.total_time = 0.0

    def predict(self, force: np.ndarray, accel: np.ndarray,
                fs: float, seq: int) -> Optional[np.ndarray]:
        """Devuelve resultados de inferencia o None para no enviar nada.
        Override en subclases con tu modelo real.
        Por defecto: echo de estadísticas básicas (mean, std, rms por canal).
        """
        f_rms = np.sqrt(np.mean(force ** 2))
        a_rms = np.sqrt(np.mean(accel ** 2))
        return np.array([
            np.mean(force), np.std(force), f_rms,
            np.mean(accel), np.std(accel), a_rms,
        ], dtype=np.float32)

    def run(self, force, accel, fs, seq):
        """Wrapper con timing."""
        t0 = time.perf_counter()
        result = self.predict(force, accel, fs, seq)
        dt = time.perf_counter() - t0
        self.calls += 1
        self.total_time += dt
        return result, dt


# ═══════════════════════════════════════════════════════════════
# Receiver Daemon
# ═══════════════════════════════════════════════════════════════
class ReceiverDaemon:
    """Daemon principal: recibe datos UDP, loggea CSV, corre inferencia,
    y devuelve resultados a Windows."""

    def __init__(self, sender_ip: str, csv_path: str,
                 enable_csv: bool = True,
                 enable_infer: bool = True,
                 engine: Optional[InferenceEngine] = None,
                 infer_csv_path: str = None,
                 cutting: CuttingConditions = None):
        self.sender_ip = sender_ip
        self.enable_csv = enable_csv
        self.enable_infer = enable_infer
        self.engine = engine or InferenceEngine()
        self.cutting = cutting or CuttingConditions()

        # Sockets
        self.data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                                      4 * 1024 * 1024)
        except Exception:
            pass
        self.data_sock.bind(("0.0.0.0", DATA_PORT))
        self.data_sock.settimeout(0.5)

        self.infer_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # CSV
        self.csv_logger = (
            CSVLogger(csv_path, self.cutting) if enable_csv else None
        )

        # Inference CSV (resultados por bloque)
        result_names = getattr(self.engine, 'RESULT_NAMES', [])
        self.infer_csv = (
            InferenceCSVLogger(infer_csv_path, result_names, self.cutting)
            if infer_csv_path and enable_infer else None
        )

        # Sync
        self.sync = TimeSynchronizer()

        # State
        self.running = False
        self.stats = {
            "packets": 0, "errors": 0, "last_seq": -1,
            "infer_calls": 0, "infer_us_avg": 0.0,
        }

    # ── Main loop ─────────────────────────────────────────────
    def run_forever(self):
        self.running = True
        if self.csv_logger:
            self.csv_logger.open()
        if self.infer_csv:
            self.infer_csv.open()

        log.info("Escuchando datos en :%d  |  Inferencia → %s:%d",
                 DATA_PORT, self.sender_ip, INFER_PORT)
        log.info("CSV=%s  INFER=%s", self.enable_csv, self.enable_infer)

        flush_interval = 5.0
        last_flush = time.time()
        last_print = time.time()

        while self.running:
            # Recibir paquete
            try:
                buf, addr = self.data_sock.recvfrom(65535)
            except socket.timeout:
                self._periodic(last_flush, last_print)
                continue
            except OSError:
                break

            parsed = unpack_data(buf)
            if parsed is None:
                self.stats["errors"] += 1
                continue

            seq, t_s, fs_hz, n_samp, n_ch, force, accel = parsed
            t_local = time.perf_counter()
            self.stats["packets"] += 1
            self.stats["last_seq"] = seq

            # CSV logging
            if self.csv_logger:
                self.csv_logger.write_block(
                    seq, t_s, t_local, force, accel,
                    cut_id=self.cutting.cut_id)

            # Inferencia → devolver resultado a Windows
            if self.enable_infer:
                result, dt_infer = self.engine.run(force, accel, fs_hz, seq)
                if result is not None:
                    pkt = pack_infer(seq, t_local, result)
                    try:
                        self.infer_sock.sendto(pkt, (addr[0], INFER_PORT))
                    except Exception as e:
                        log.warning("Error enviando inferencia: %s", e)
                # Guardar en inference CSV
                if self.infer_csv:
                    self.infer_csv.write_row(
                        seq, t_s, t_local, fs_hz, n_samp,
                        dt_infer * 1e6, result,
                        cut_id=self.cutting.cut_id,
                        Ft_mean=self.cutting.Ft_mean_N,
                        Ft_peak=self.cutting.Ft_peak_N)
                self.stats["infer_calls"] += 1
                # Media móvil exponencial del tiempo de inferencia
                alpha = 0.05
                prev = self.stats["infer_us_avg"]
                self.stats["infer_us_avg"] = (
                    prev * (1 - alpha) + dt_infer * 1e6 * alpha)

            # Flush / print periódico
            now = time.time()
            if now - last_flush > flush_interval:
                if self.csv_logger:
                    self.csv_logger.flush()
                if self.infer_csv:
                    self.infer_csv.flush()
                last_flush = now
            if now - last_print > 2.0:
                self._print_stats()
                last_print = now

        # Cleanup
        if self.csv_logger:
            self.csv_logger.close()
        if self.infer_csv:
            self.infer_csv.close()
        self.data_sock.close()
        self.infer_sock.close()
        log.info("Daemon detenido.")

    def _periodic(self, last_flush, last_print):
        now = time.time()
        if now - last_print > 5.0:
            if self.stats["packets"] > 0:
                self._print_stats()

    def _print_stats(self):
        s = self.stats
        parts = [
            f"pkts={s['packets']}",
            f"seq={s['last_seq']}",
        ]
        if s["errors"]:
            parts.append(f"err={s['errors']}")
        if self.csv_logger and self.csv_logger.is_open:
            parts.append(f"csv={self.csv_logger.rows}")
        if self.infer_csv and self.infer_csv.is_open:
            parts.append(f"icsv={self.infer_csv.rows}")
        if self.enable_infer and s["infer_calls"] > 0:
            parts.append(f"infer={s['infer_calls']}"
                         f"({s['infer_us_avg']:.0f}us)")
        if self.sync.n_samples > 0:
            parts.append(f"offset={self.sync.offset*1000:.2f}ms")
        if self.cutting.cut_id > 0:
            parts.append(f"cut={self.cutting.cut_id}:{self.cutting.label}")
        log.info("  ".join(parts))

    def stop(self):
        self.running = False

    def set_cutting(self, **kwargs):
        """Cambia condiciones de corte en vivo. Incrementa cut_id automáticamente."""
        self.cutting.cut_id += 1
        for k, v in kwargs.items():
            if hasattr(self.cutting, k):
                setattr(self.cutting, k, v)
        # Notificar a los CSV loggers
        if self.csv_logger:
            self.csv_logger.write_cut_change(self.cutting)
        if self.infer_csv:
            self.infer_csv.write_cut_change(self.cutting)
        log.info(self.cutting.summary())

    # ── Comandos de control ───────────────────────────────────
    def do_sync(self):
        ok = self.sync.do_sync(self.sender_ip)
        if ok:
            log.info("SYNC OK: offset=%.3f ms  RTT=%.3f ms  (%d muestras)",
                     self.sync.offset * 1000, self.sync.rtt * 1000,
                     self.sync.n_samples)
        else:
            log.warning("SYNC fallido – sin respuesta de %s", self.sender_ip)
        return ok

    def send_start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(pack_start(), (self.sender_ip, CTRL_PORT))
        sock.close()
        log.info("START enviado a %s", self.sender_ip)

    def send_stop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(pack_stop(), (self.sender_ip, CTRL_PORT))
        sock.close()
        log.info("STOP enviado a %s", self.sender_ip)


# ═══════════════════════════════════════════════════════════════
# CLI Control Thread  (comandos interactivos por stdin)
# ═══════════════════════════════════════════════════════════════
class CLIThread(threading.Thread):
    """Lee comandos desde stdin para controlar el daemon."""

    def __init__(self, daemon: ReceiverDaemon):
        super().__init__(daemon=True)
        self.d = daemon

    def _parse_cut_cmd(self, args_str: str):
        """Parsea 'cut label ap rpm feed' o 'cut key=val key=val ...'."""
        parts = args_str.split()
        kwargs = {}
        positional = []
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                try:
                    v = float(v)
                except ValueError:
                    pass
                kwargs[k] = v
            else:
                positional.append(p)
        # Positional: label ap_mm rpm_spindle rpm_feed
        if len(positional) >= 1:
            kwargs.setdefault("label", positional[0])
        if len(positional) >= 2:
            kwargs.setdefault("ap_mm", float(positional[1]))
        if len(positional) >= 3:
            kwargs.setdefault("rpm_spindle", float(positional[2]))
        if len(positional) >= 4:
            kwargs.setdefault("rpm_feed", float(positional[3]))
        return kwargs

    def run(self):
        print("\n─── Comandos: start | stop | sync | stats | cut | quit ───\n")
        while self.d.running:
            try:
                cmd = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.d.stop()
                break
            cmd_lower = cmd.lower()
            if cmd_lower == "start":
                self.d.send_start()
            elif cmd_lower == "stop":
                self.d.send_stop()
            elif cmd_lower == "sync":
                self.d.do_sync()
            elif cmd_lower == "stats":
                self.d._print_stats()
            elif cmd_lower.startswith("cut"):
                rest = cmd[3:].strip()
                if not rest:
                    print(self.d.cutting.summary())
                    print("  Uso: cut <label> <ap_mm> <rpm_spindle> <rpm_feed>")
                    print("    o: cut label=corte1 ap_mm=0.5 rpm_spindle=300 rpm_feed=20")
                    print("  Ejemplo: cut prof_0.5mm 0.5 300 20")
                else:
                    kwargs = self._parse_cut_cmd(rest)
                    self.d.set_cutting(**kwargs)
            elif cmd_lower in ("quit", "exit", "q"):
                self.d.send_stop()
                time.sleep(0.3)
                self.d.stop()
                break
            elif cmd_lower == "help":
                print("  start  – enviar START al sender Windows")
                print("  stop   – enviar STOP al sender Windows")
                print("  sync   – sincronizar relojes NTP-like")
                print("  stats  – mostrar estadísticas")
                print("  cut    – ver/cambiar condiciones de corte")
                print("           cut <label> <ap_mm> <rpm> <feed_rpm>")
                print("           cut ap_mm=1.0 rpm_spindle=300")
                print("  quit   – salir")
            elif cmd:
                print(f"  Comando desconocido: '{cmd}'  (escribe 'help')")



# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def _default_csv_path():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"cdaq_remote_{ts}.csv")


def main():
    ap = argparse.ArgumentParser(
        description="cDAQ UDP Receiver – Daemon ligero para Jetson")
    ap.add_argument("--sender-ip", default="192.168.137.1",
                    help="IP del sender Windows")
    ap.add_argument("--csv", default="",
                    help="Ruta CSV (auto-generada si vacía)")
    ap.add_argument("--no-csv", action="store_true",
                    help="Desactivar logging CSV")
    ap.add_argument("--no-infer", action="store_true",
                    help="Desactivar inferencia (solo recibir y guardar)")
    ap.add_argument("--engine", default="basic",
                    choices=["basic", "envelope", "bouc-wen"],
                    help="Motor de inferencia: basic | envelope | bouc-wen")
    ap.add_argument("--infer-csv", default="auto",
                    help="Ruta CSV de inferencia ('auto'=junto al raw, 'none'=desactivar)")
    ap.add_argument("--auto-start", action="store_true",
                    help="Enviar START al sender al iniciar")
    ap.add_argument("--sync", action="store_true",
                    help="Hacer sync de reloj al iniciar")
    # ── Condiciones de corte (modelo mecanicista) ──
    cut = ap.add_argument_group("Condiciones de corte")
    cut.add_argument("--ap", type=float, default=0.0,
                     help="Profundidad axial de corte (mm)")
    cut.add_argument("--rpm", type=float, default=0.0,
                     help="RPM del husillo")
    cut.add_argument("--feed-rpm", type=float, default=0.0,
                     help="RPM del eje de avance X")
    cut.add_argument("--tool-diam", type=float, default=25.4,
                     help="Diámetro del cortador (mm) [Korloy AMSA3100HS=25.4]")
    cut.add_argument("--n-flutes", type=int, default=2,
                     help="Número de filos del cortador")
    cut.add_argument("--Ktc", type=float, default=800.0,
                     help="Coef. tangencial específico (N/mm²) [Al6061≈800, Acero≈2000]")
    cut.add_argument("--Kte", type=float, default=10.0,
                     help="Coef. de filo (N/mm)")
    cut.add_argument("--cut-label", default="idle",
                     help="Etiqueta para la condición de corte inicial")
    args = ap.parse_args()

    csv_path = args.csv if args.csv else _default_csv_path()

    # Seleccionar motor de inferencia
    engine = None
    if not args.no_infer:
        if args.engine == "bouc-wen":
            try:
                from cdaq_inference_engine import BoucWenInferenceEngine
                engine = BoucWenInferenceEngine(fs=2500.0)
                log.info("Motor: BoucWenInferenceEngine (KAN-PINN + Hilbert + FFT)")
            except ImportError as e:
                log.warning("No se pudo cargar BoucWenInferenceEngine: %s", e)
                log.warning("Usando motor básico (stats)")
        elif args.engine == "envelope":
            try:
                from cdaq_inference_engine import EnvelopeInferenceEngine
                engine = EnvelopeInferenceEngine(fs=2500.0)
                log.info("Motor: EnvelopeInferenceEngine (Hilbert+FFT+THD)")
            except ImportError as e:
                log.warning("No se pudo cargar EnvelopeInferenceEngine: %s", e)
                log.warning("Usando motor básico (stats)")
        else:
            log.info("Motor de inferencia: básico (mean/std/rms)")

    # Ruta del CSV de inferencia
    infer_csv_path = None
    if not args.no_infer and args.infer_csv != "none":
        if args.infer_csv == "auto":
            infer_csv_path = csv_path.replace(".csv", "_inference.csv")
        else:
            infer_csv_path = args.infer_csv

    # Condiciones de corte iniciales
    cutting = CuttingConditions(
        cut_id=0,
        label=args.cut_label,
        ap_mm=args.ap,
        rpm_spindle=args.rpm,
        rpm_feed=args.feed_rpm,
        tool_diam_mm=args.tool_diam,
        n_flutes=args.n_flutes,
        Ktc=args.Ktc,
        Kte=args.Kte,
    )
    if cutting.ap_mm > 0:
        log.info(cutting.summary())

    daemon = ReceiverDaemon(
        sender_ip=args.sender_ip,
        csv_path=csv_path,
        enable_csv=not args.no_csv,
        enable_infer=not args.no_infer,
        engine=engine,
        infer_csv_path=infer_csv_path,
        cutting=cutting,
    )

    # Ctrl+C graceful
    def _sigint(sig, frame):
        log.info("Ctrl+C – cerrando…")
        daemon.stop()
    signal.signal(signal.SIGINT, _sigint)

    # Sync y auto-start opcionales
    if args.sync:
        daemon.do_sync()
    if args.auto_start:
        daemon.send_start()

    # CLI interactivo en hilo separado
    cli = CLIThread(daemon)
    cli.start()

    # Loop principal
    daemon.run_forever()


if __name__ == "__main__":
    main()
