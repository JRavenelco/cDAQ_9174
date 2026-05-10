#!/usr/bin/env python3
"""
Sender sintetico CNC — corre en Windows (o Jetson para pruebas).

Genera señales de fuerza + aceleración que imitan fresado CNC y las envía
al receiver Jetson usando el protocolo cdaq_udp_protocol.pack_data().

Modos disponibles (--mode):
  nuevo      : cortador nuevo, lazo de histeresis limpio y cerrado
  medio_uso  : algo de desgaste, lazo más ancho, componentes de alta freq
  desgastado : desgaste severo, chatter, lazo irregular
  idle       : máquina encendida sin corte (referencia)
  ciclo      : rota automaticamente nuevo→medio_uso→desgastado→idle cada N seg

Uso:
  python synthetic_udp_sender.py --jetson-ip 192.168.137.2
  python synthetic_udp_sender.py --jetson-ip 192.168.137.2 --mode ciclo --rpm 600
  python synthetic_udp_sender.py --jetson-ip 192.168.137.2 --mode desgastado --rpm 400 --duration 60
"""
from __future__ import annotations

import sys
import argparse
import math
import random
import socket
import numpy as np

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import time

try:
    from cdaq_udp_protocol import pack_data, DATA_PORT
except ImportError:
    import struct
    DATA_PORT = 12001
    MAGIC = 0xCDAC0001
    HDR_FMT = "<I d d f H H"

    def pack_data(seq: int, t_s: float, fs_hz: float,
                  force: list, accel: list) -> bytes:
        n = len(force)
        hdr = struct.pack(HDR_FMT, MAGIC, seq, t_s, fs_hz, n, 2)
        samples = b"".join(struct.pack("<ff", f, a) for f, a in zip(force, accel))
        return hdr + samples


FS_HZ    = 2500
SPR      = 100          # samples per block (same as real cDAQ)
BLOCK_DT = SPR / FS_HZ  # ~0.04 s

CYCLE_DURATIONS = {     # segundos por modo en ciclo automático
    "nuevo":     20,
    "medio_uso": 20,
    "desgastado":20,
    "idle":      10,
}
CYCLE_ORDER = ["nuevo", "medio_uso", "desgastado", "idle"]


def _spindle_hz(rpm: float) -> float:
    return rpm / 60.0


def generate_block(
    t0: float,
    mode: str,
    rpm: float,
    n_flutes: int = 2,
) -> tuple[list[float], list[float]]:
    """Genera SPR muestras de (fuerza_V, accel_g) para el modo indicado."""
    sh = _spindle_hz(rpm)
    tooth_hz = sh * n_flutes

    force_block = []
    accel_block = []

    for i in range(SPR):
        t = t0 + i / FS_HZ

        if mode == "idle":
            f = 0.01 * math.sin(2 * math.pi * 60 * t) + 0.005 * random.gauss(0, 1)
            a = 0.01 * math.sin(2 * math.pi * 120 * t) + 0.003 * random.gauss(0, 1)

        elif mode == "nuevo":
            # Lazo limpio: fuerza sigue a accel con desfase (histeresis pura)
            f = (0.45 * math.sin(2 * math.pi * sh * t)
                 + 0.20 * math.sin(2 * math.pi * tooth_hz * t)
                 + 0.02 * random.gauss(0, 1))
            a = (0.40 * math.sin(2 * math.pi * sh * t + 0.8)   # desfase ~45°
                 + 0.15 * math.sin(2 * math.pi * tooth_hz * t + 0.5)
                 + 0.01 * random.gauss(0, 1))

        elif mode == "medio_uso":
            # Desgaste parcial: más ruido, componentes extra, lazo más ancho
            f = (0.50 * math.sin(2 * math.pi * sh * t)
                 + 0.25 * math.sin(2 * math.pi * tooth_hz * t)
                 + 0.08 * math.sin(2 * math.pi * tooth_hz * 2 * t)
                 + 0.04 * random.gauss(0, 1))
            a = (0.42 * math.sin(2 * math.pi * sh * t + 1.1)
                 + 0.18 * math.sin(2 * math.pi * tooth_hz * t + 0.7)
                 + 0.06 * math.sin(2 * math.pi * 3 * sh * t)
                 + 0.02 * random.gauss(0, 1))

        elif mode == "desgastado":
            # Desgaste severo: chatter, alta vibración, lazo muy irregular
            chatter_hz = tooth_hz * 3.5  # frecuencia de chatter
            f = (0.60 * math.sin(2 * math.pi * sh * t)
                 + 0.30 * math.sin(2 * math.pi * tooth_hz * t)
                 + 0.25 * math.sin(2 * math.pi * chatter_hz * t)
                 + 0.06 * random.gauss(0, 1))
            a = (0.55 * math.sin(2 * math.pi * sh * t + 1.4)
                 + 0.28 * math.sin(2 * math.pi * tooth_hz * t + 0.9)
                 + 0.30 * math.sin(2 * math.pi * chatter_hz * t + 0.3)
                 + 0.04 * random.gauss(0, 1))
        else:
            f, a = 0.0, 0.0

        force_block.append(f)
        accel_block.append(a)

    return force_block, accel_block


def main() -> int:
    ap = argparse.ArgumentParser(description="Sender sintetico CNC para probar el receiver Jetson")
    ap.add_argument("--jetson-ip", default="192.168.137.2",
                    help="IP del Jetson (receiver)")
    ap.add_argument("--port", type=int, default=DATA_PORT,
                    help=f"Puerto UDP del receiver (default: {DATA_PORT})")
    ap.add_argument("--mode", default="ciclo",
                    choices=["nuevo", "medio_uso", "desgastado", "idle", "ciclo"],
                    help="Modo de señal a generar")
    ap.add_argument("--rpm", type=float, default=600.0,
                    help="RPM del husillo (default: 600)")
    ap.add_argument("--n-flutes", type=int, default=2,
                    help="Número de filos del cortador (default: 2)")
    ap.add_argument("--duration", type=float, default=0,
                    help="Duración en segundos (0 = infinito)")
    ap.add_argument("--realtime", action="store_true", default=True,
                    help="Enviar a velocidad real (default: sí)")
    ap.add_argument("--no-realtime", dest="realtime", action="store_false",
                    help="Enviar tan rápido como sea posible")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.jetson_ip, args.port)

    print(f"Sender sintetico CNC")
    print(f"  Destino : {args.jetson_ip}:{args.port}")
    print(f"  Modo    : {args.mode}")
    print(f"  RPM     : {args.rpm}")
    print(f"  Filos   : {args.n_flutes}")
    print(f"  Dur.    : {'inf' if args.duration == 0 else f'{args.duration}s'}")
    print()

    seq = 0
    t_sim = 0.0
    t_wall_start = time.perf_counter()

    # Ciclo automático
    cycle_idx = 0
    cycle_start = time.perf_counter()
    current_mode = CYCLE_ORDER[0] if args.mode == "ciclo" else args.mode

    try:
        while True:
            now = time.perf_counter()

            # Cambiar modo en ciclo automático
            if args.mode == "ciclo":
                elapsed_in_mode = now - cycle_start
                mode_dur = CYCLE_DURATIONS.get(current_mode, 20)
                if elapsed_in_mode >= mode_dur:
                    cycle_idx = (cycle_idx + 1) % len(CYCLE_ORDER)
                    current_mode = CYCLE_ORDER[cycle_idx]
                    cycle_start = now
                    print(f"  → Modo: {current_mode}")

            # Generar bloque
            force, accel = generate_block(t_sim, current_mode, args.rpm, args.n_flutes)
            pkt = pack_data(seq, t_sim, float(FS_HZ),
                           np.asarray(force, dtype=np.float32),
                           np.asarray(accel, dtype=np.float32))
            sock.sendto(pkt, dest)

            seq += 1
            t_sim += BLOCK_DT

            if args.duration > 0 and (time.perf_counter() - t_wall_start) >= args.duration:
                print("Duración alcanzada. Deteniendo.")
                break

            if args.realtime:
                t_next = t_wall_start + t_sim
                wait = t_next - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)

            if seq % 250 == 0:  # cada ~10s
                elapsed = time.perf_counter() - t_wall_start
                print(f"  seq={seq}  t_sim={t_sim:.1f}s  elapsed={elapsed:.1f}s  modo={current_mode}")

    except KeyboardInterrupt:
        print("\nDetenido por usuario.")

    sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
