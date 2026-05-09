#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import socket
import time
from typing import Iterator, Optional, Tuple

import numpy as np

from cdaq_udp_protocol import DATA_PORT, pack_data


def _iter_blocks_from_raw_csv(path: str) -> Iterator[Tuple[int, float, np.ndarray, np.ndarray]]:
    """Yield (seq, t_sender_s, force, accel) blocks from a raw cdaq_remote_*.csv.

    Raw CSV is sample-per-row with columns:
      cut_id,seq,t_sender_s,t_local_s,sample_idx,force_v,accel_g

    Lines starting with '#' are metadata and are skipped.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        # Skip metadata/comment lines but keep the header
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                return
            if not line.startswith("#"):
                f.seek(pos)
                break

        reader = csv.DictReader(f)
        cur_seq: Optional[int] = None
        cur_t_s: Optional[float] = None
        force_list = []
        accel_list = []

        def flush():
            nonlocal cur_seq, cur_t_s, force_list, accel_list
            if cur_seq is None:
                return None
            n = max(len(force_list), len(accel_list))
            if n <= 0:
                return None
            f_arr = np.asarray(force_list + [np.nan] * (n - len(force_list)), dtype=np.float32)
            a_arr = np.asarray(accel_list + [np.nan] * (n - len(accel_list)), dtype=np.float32)
            # Replace NaNs if any gaps
            if not np.isfinite(f_arr).all():
                bad = ~np.isfinite(f_arr)
                f_arr[bad] = 0.0
            if not np.isfinite(a_arr).all():
                bad = ~np.isfinite(a_arr)
                a_arr[bad] = 0.0
            out = (int(cur_seq), float(cur_t_s or 0.0), f_arr, a_arr)
            cur_seq = None
            cur_t_s = None
            force_list = []
            accel_list = []
            return out

        for row in reader:
            try:
                seq = int(row.get("seq", "0"))
            except Exception:
                continue

            if cur_seq is None:
                cur_seq = seq

            if seq != cur_seq:
                blk = flush()
                if blk is not None:
                    yield blk
                cur_seq = seq

            try:
                if cur_t_s is None:
                    cur_t_s = float(row.get("t_sender_s", "0"))
            except Exception:
                if cur_t_s is None:
                    cur_t_s = 0.0

            try:
                idx = int(row.get("sample_idx", "0"))
            except Exception:
                idx = len(force_list)

            if idx < 0:
                idx = 0

            if idx >= len(force_list):
                force_list.extend([np.nan] * (idx + 1 - len(force_list)))
                accel_list.extend([np.nan] * (idx + 1 - len(accel_list)))

            try:
                force_list[idx] = float(row.get("force_v", "0"))
            except Exception:
                force_list[idx] = 0.0
            try:
                accel_list[idx] = float(row.get("accel_g", "0"))
            except Exception:
                accel_list[idx] = 0.0

        blk = flush()
        if blk is not None:
            yield blk


def main():
    ap = argparse.ArgumentParser(description="UDP sender emulator: replay raw CSV blocks to a receiver")
    ap.add_argument("--csv", required=True, help="Path to raw cdaq_remote_*.csv")
    ap.add_argument("--dst-ip", default="127.0.0.1", help="Receiver IP (default localhost)")
    ap.add_argument("--dst-port", type=int, default=DATA_PORT, help="Receiver DATA_PORT")
    ap.add_argument("--fs", type=float, default=2500.0, help="Sample rate to encode in UDP packets")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Replay speed factor. 1.0=real time based on t_sender_s. 0=as fast as possible")
    ap.add_argument("--loop", action="store_true", help="Loop replay forever")
    ap.add_argument("--max-packets", type=int, default=0, help="Stop after N packets (0=no limit)")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")

    fs = float(args.fs)
    speed = float(args.speed)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (args.dst_ip, int(args.dst_port))

    sent = 0
    while True:
        prev_t_s: Optional[float] = None
        t0_wall = time.perf_counter()

        for seq, t_s, force, accel in _iter_blocks_from_raw_csv(args.csv):
            if args.max_packets and sent >= int(args.max_packets):
                sock.close()
                return

            if speed > 0.0 and prev_t_s is not None:
                dt_sender = float(t_s) - float(prev_t_s)
                if np.isfinite(dt_sender) and dt_sender > 0:
                    time.sleep(max(0.0, dt_sender / speed))
            prev_t_s = float(t_s)

            pkt = pack_data(int(seq), float(t_s), float(fs), force, accel)
            sock.sendto(pkt, dst)
            sent += 1

            if sent % 50 == 0:
                dt = time.perf_counter() - t0_wall
                rate = sent / max(1e-6, dt)
                print(f"sent={sent}  last_seq={seq}  rate={rate:.1f} pkt/s", flush=True)

        if not args.loop:
            break

    sock.close()


if __name__ == "__main__":
    main()
