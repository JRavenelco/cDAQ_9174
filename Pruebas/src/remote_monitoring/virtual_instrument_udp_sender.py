#!/usr/bin/env python3
"""Virtual force/acceleration instrument that streams UDP packets to the Jetson.

Runs on Windows. It emulates a DAQ-like device without NI hardware and sends
small binary UDP frames that the Jetson receiver can log and analyze.
"""

from __future__ import annotations

import argparse
import math
import random
import socket
import struct
import time
from dataclasses import dataclass

MAGIC = b"VDAQ1"
HEADER = struct.Struct("<5sIdfHBx")
# magic, seq uint32, unix_time double, fs float32, n uint16, mode uint8, pad
SAMPLE = struct.Struct("<ff")
MODE_IDS = {"normal": 1, "chatter": 2, "wear": 3, "impact": 4}
MODE_NAMES = {v: k for k, v in MODE_IDS.items()}


@dataclass
class SimState:
    fs: float
    spindle_hz: float
    tooth_hz: float
    z: float = 0.0
    last_u: float = 0.0

    def sample(self, t: float, mode: str) -> tuple[float, float]:
        base = math.sin(2.0 * math.pi * self.spindle_hz * t)
        tooth = math.sin(2.0 * math.pi * self.tooth_hz * t)
        accel = 0.35 * base + 0.18 * tooth
        accel += 0.04 * random.gauss(0.0, 1.0)

        if mode == "chatter":
            accel += 0.45 * math.sin(2.0 * math.pi * 310.0 * t)
        elif mode == "wear":
            accel += 0.10 * math.sin(2.0 * math.pi * 85.0 * t)
        elif mode == "impact" and int(t * 3) % 5 == 0:
            phase = (t * 3.0) % 1.0
            if phase < 0.035:
                accel += 1.2 * math.exp(-phase * 95.0)

        # Small Bouc-Wen-like memory term so force does not follow acceleration
        # instantaneously. This gives the receiver useful hysteretic loops.
        dt = 1.0 / self.fs
        u = accel
        du = (u - self.last_u) / dt
        dz = 0.85 * du - 0.55 * abs(du) * self.z - 0.25 * du * abs(self.z)
        self.z = max(-2.5, min(2.5, self.z + dz * dt))
        self.last_u = u

        alpha = 0.82 if mode != "wear" else 0.65
        force = 0.78 + alpha * 0.38 * u + (1.0 - alpha) * 0.38 * self.z
        force += 0.02 * random.gauss(0.0, 1.0)
        if mode == "wear":
            force += 0.10 * min(1.0, t / 20.0)
        return force, accel


def pack_frame(seq: int, fs: float, mode: str, samples: list[tuple[float, float]]) -> bytes:
    payload = bytearray()
    payload += HEADER.pack(MAGIC, seq, time.time(), float(fs), len(samples), MODE_IDS[mode])
    for force_v, accel_g in samples:
        payload += SAMPLE.pack(float(force_v), float(accel_g))
    return bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream simulated force/acceleration over UDP")
    parser.add_argument("--jetson-ip", default="192.168.55.1", help="Jetson IP address")
    parser.add_argument("--port", type=int, default=12011, help="Jetson UDP port")
    parser.add_argument("--fs", type=float, default=2500.0, help="Sample rate in Hz")
    parser.add_argument("--block-size", type=int, default=100, help="Samples per UDP packet")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to stream; 0 means forever")
    parser.add_argument("--mode", choices=sorted(MODE_IDS), default="normal")
    parser.add_argument("--spindle-hz", type=float, default=18.31)
    parser.add_argument("--tooth-hz", type=float, default=36.62)
    parser.add_argument("--no-realtime", action="store_true", help="Send as fast as possible")
    args = parser.parse_args()

    if args.block_size < 1 or args.block_size > 160:
        raise SystemExit("--block-size must be between 1 and 160 to avoid UDP fragmentation")

    target = (args.jetson_ip, args.port)
    state = SimState(fs=args.fs, spindle_hz=args.spindle_hz, tooth_hz=args.tooth_hz)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"Virtual instrument -> UDP {target[0]}:{target[1]}")
    print(f"mode={args.mode}, fs={args.fs:g} Hz, block={args.block_size}, duration={args.duration:g}s")
    print("Ctrl+C to stop")

    seq = 0
    sample_index = 0
    t_start = time.perf_counter()
    next_send = t_start
    try:
        while True:
            elapsed = time.perf_counter() - t_start
            if args.duration > 0 and elapsed >= args.duration:
                break
            samples = []
            for _ in range(args.block_size):
                t = sample_index / args.fs
                samples.append(state.sample(t, args.mode))
                sample_index += 1
            sock.sendto(pack_frame(seq, args.fs, args.mode, samples), target)
            seq += 1
            if seq % 25 == 0:
                print(f"sent seq={seq} samples={sample_index} t={sample_index / args.fs:.2f}s")
            if not args.no_realtime:
                next_send += args.block_size / args.fs
                delay = next_send - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        sock.close()
    print(f"Done. sent frames={seq}, samples={sample_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
