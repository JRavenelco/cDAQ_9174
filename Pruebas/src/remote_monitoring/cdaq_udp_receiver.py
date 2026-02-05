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
# CSV Logger
# ═══════════════════════════════════════════════════════════════
class CSVLogger:
    """Escribe datos recibidos a CSV de forma eficiente (flush periódico)."""

    def __init__(self, path: str):
        self.path = path
        self._file = None
        self._writer = None
        self.rows = 0

    def open(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8",
                          buffering=1)   # line-buffered
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "seq", "t_sender_s", "t_local_s", "sample_idx",
            "force_v", "accel_g",
        ])
        self.rows = 0
        log.info("CSV abierto: %s", self.path)

    def write_block(self, seq, t_sender, t_local, force, accel):
        if self._writer is None:
            return
        for i in range(len(force)):
            self._writer.writerow([
                seq, f"{t_sender:.9f}", f"{t_local:.9f}", i,
                f"{force[i]:.7f}", f"{accel[i]:.7f}",
            ])
        self.rows += len(force)

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
                 engine: Optional[InferenceEngine] = None):
        self.sender_ip = sender_ip
        self.enable_csv = enable_csv
        self.enable_infer = enable_infer
        self.engine = engine or InferenceEngine()

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
        self.csv_logger = CSVLogger(csv_path) if enable_csv else None

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
                self.csv_logger.write_block(seq, t_s, t_local, force, accel)

            # Inferencia → devolver resultado a Windows
            if self.enable_infer:
                result, dt_infer = self.engine.run(force, accel, fs_hz, seq)
                if result is not None:
                    pkt = pack_infer(seq, t_local, result)
                    try:
                        self.infer_sock.sendto(pkt, (addr[0], INFER_PORT))
                    except Exception as e:
                        log.warning("Error enviando inferencia: %s", e)
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
                last_flush = now
            if now - last_print > 2.0:
                self._print_stats()
                last_print = now

        # Cleanup
        if self.csv_logger:
            self.csv_logger.close()
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
        if self.enable_infer and s["infer_calls"] > 0:
            parts.append(f"infer={s['infer_calls']}"
                         f"({s['infer_us_avg']:.0f}us)")
        if self.sync.n_samples > 0:
            parts.append(f"offset={self.sync.offset*1000:.2f}ms")
        log.info("  ".join(parts))

    def stop(self):
        self.running = False

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

    def run(self):
        print("\n─── Comandos: start | stop | sync | stats | quit ───\n")
        while self.d.running:
            try:
                cmd = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.d.stop()
                break
            if cmd == "start":
                self.d.send_start()
            elif cmd == "stop":
                self.d.send_stop()
            elif cmd == "sync":
                self.d.do_sync()
            elif cmd == "stats":
                self.d._print_stats()
            elif cmd in ("quit", "exit", "q"):
                self.d.send_stop()
                time.sleep(0.3)
                self.d.stop()
                break
            elif cmd == "help":
                print("  start  – enviar START al sender Windows")
                print("  stop   – enviar STOP al sender Windows")
                print("  sync   – sincronizar relojes NTP-like")
                print("  stats  – mostrar estadísticas")
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
                    choices=["basic", "envelope"],
                    help="Motor de inferencia: basic (stats) o envelope (Hilbert+FFT+THD)")
    ap.add_argument("--auto-start", action="store_true",
                    help="Enviar START al sender al iniciar")
    ap.add_argument("--sync", action="store_true",
                    help="Hacer sync de reloj al iniciar")
    args = ap.parse_args()

    csv_path = args.csv if args.csv else _default_csv_path()

    # Seleccionar motor de inferencia
    engine = None
    if not args.no_infer:
        if args.engine == "envelope":
            try:
                from cdaq_inference_engine import EnvelopeInferenceEngine
                engine = EnvelopeInferenceEngine(fs=2500.0)
                log.info("Motor de inferencia: EnvelopeInferenceEngine (Hilbert+FFT+THD)")
            except ImportError as e:
                log.warning("No se pudo cargar EnvelopeInferenceEngine: %s", e)
                log.warning("Usando motor básico (stats)")
        else:
            log.info("Motor de inferencia: básico (mean/std/rms)")

    daemon = ReceiverDaemon(
        sender_ip=args.sender_ip,
        csv_path=csv_path,
        enable_csv=not args.no_csv,
        enable_infer=not args.no_infer,
        engine=engine,
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
