#!/usr/bin/env python3
"""Jetson UDP receiver for the virtual force/acceleration instrument."""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import struct
import time
from collections import deque
from pathlib import Path

MAGIC = b"VDAQ1"
HEADER = struct.Struct("<5sIdfHBx")
SAMPLE = struct.Struct("<ff")
MODE_NAMES = {1: "normal", 2: "chatter", 3: "wear", 4: "impact"}


def parse_frame(data: bytes):
    if len(data) < HEADER.size:
        raise ValueError("packet too small")
    magic, seq, sent_unix, fs, n, mode_id = HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError("bad magic")
    expected = HEADER.size + n * SAMPLE.size
    if len(data) != expected:
        raise ValueError(f"bad packet size {len(data)} != {expected}")
    samples = []
    offset = HEADER.size
    for _ in range(n):
        samples.append(SAMPLE.unpack_from(data, offset))
        offset += SAMPLE.size
    return seq, sent_unix, float(fs), MODE_NAMES.get(mode_id, f"mode_{mode_id}"), samples


def rms(values):
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def corr(xs, ys):
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    xs = list(xs)[-n:]
    ys = list(ys)[-n:]
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def main() -> int:
    parser = argparse.ArgumentParser(description="Receive virtual DAQ UDP frames and log them")
    parser.add_argument("--bind-ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12011)
    parser.add_argument("--out-dir", type=Path, default=Path("logs"))
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--max-packets", type=int, default=0, help="0 means forever")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = args.out_dir / f"virtual_instrument_{stamp}.csv"
    cases_path = args.out_dir / f"virtual_instrument_{stamp}_cases.jsonl"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_ip, args.port))
    sock.settimeout(1.0)

    print(f"Listening UDP {args.bind_ip}:{args.port}")
    print(f"CSV: {csv_path}")
    print(f"Cases: {cases_path}")
    print("Ctrl+C to stop")

    force_window = deque()
    accel_window = deque()
    sample_clock = 0.0
    packet_count = 0
    sample_count = 0
    last_seq = None
    dropped = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_fh, cases_path.open("w", encoding="utf-8") as cases_fh:
        writer = csv.writer(csv_fh)
        writer.writerow(["received_unix", "sent_unix", "seq", "sample_index", "t_sample_s", "fs_hz", "mode", "force_v", "accel_g"])

        try:
            while True:
                if args.max_packets and packet_count >= args.max_packets:
                    break
                try:
                    data, addr = sock.recvfrom(2048)
                except socket.timeout:
                    continue
                received = time.time()
                seq, sent_unix, fs, mode, samples = parse_frame(data)
                if last_seq is not None and seq != last_seq + 1:
                    dropped += max(0, seq - last_seq - 1)
                last_seq = seq
                packet_count += 1

                for force_v, accel_g in samples:
                    writer.writerow([received, sent_unix, seq, sample_count, sample_clock, fs, mode, force_v, accel_g])
                    force_window.append(float(force_v))
                    accel_window.append(float(accel_g))
                    sample_count += 1
                    sample_clock += 1.0 / fs

                max_window = max(8, int(round(args.window_seconds * fs)))
                while len(force_window) > max_window:
                    force_window.popleft()
                    accel_window.popleft()

                if packet_count % 10 == 0:
                    feature = {
                        "case_id": f"udp_{stamp}_{packet_count:06d}",
                        "source": "virtual_udp_instrument",
                        "sender": addr[0],
                        "seq": seq,
                        "mode": mode,
                        "fs_hz": fs,
                        "samples_total": sample_count,
                        "dropped_packets_est": dropped,
                        "window_seconds": args.window_seconds,
                        "force_mean": sum(force_window) / len(force_window),
                        "force_rms": rms(force_window),
                        "force_min": min(force_window),
                        "force_max": max(force_window),
                        "accel_mean": sum(accel_window) / len(accel_window),
                        "accel_rms": rms(accel_window),
                        "accel_min": min(accel_window),
                        "accel_max": max(accel_window),
                        "corr_force_accel": corr(force_window, accel_window),
                    }
                    cases_fh.write(json.dumps(feature, ensure_ascii=False) + "\n")
                    cases_fh.flush()
                    print(
                        f"seq={seq} mode={mode} samples={sample_count} "
                        f"force_rms={feature['force_rms']:.4f} accel_rms={feature['accel_rms']:.4f} dropped={dropped}"
                    )
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            sock.close()
    print(f"Done. packets={packet_count}, samples={sample_count}, dropped_est={dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
