#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protocolo UDP compartido entre Windows (sender) y Jetson (receiver).

Paquetes de DATOS  (Windows → Jetson)  puerto DATA_PORT
Paquetes de CONTROL (bidireccional)     puerto CTRL_PORT

Formato de datos:
  Header  (24 bytes):
    magic    uint32   0xCDAC0001
    seq      uint32   número de paquete
    t_s      float64  perf_counter del sender
    fs_hz    float32  sample rate
    n_samp   uint16   muestras por canal
    n_ch     uint16   canales (2: force_v + accel_g)
  Payload:
    float32[n_ch * n_samp]  intercalado [ch0_s0, ch1_s0, ch0_s1, ch1_s1, …]

Mensajes de control:
  SYNC_REQ   (Jetson → Windows): magic(4) + 'S'(1) + t1(8)           = 13 bytes
  SYNC_RSP   (Windows → Jetson): magic(4) + 'R'(1) + t1(8) + t2(8) + t3(8) = 29 bytes
  START      (Jetson → Windows): magic(4) + 'G'(1)                    = 5 bytes
  STOP       (Jetson → Windows): magic(4) + 'X'(1)                    = 5 bytes
  CFG        (Jetson → Windows): magic(4) + 'C'(1) + fs(4) + spr(4)  = 13 bytes

Inferencia (Jetson → Windows)  puerto INFER_PORT:
  Header  (17 bytes):
    magic    uint32   0xCDAC0001
    type     uint8    'I' (73)
    seq      uint32   seq del bloque de datos original
    t_s      float64  timestamp local de la inferencia
  Payload:
    float32[N]  resultados de inferencia (flexible)
"""

import struct
import numpy as np

# ─── Puertos ────────────────────────────────────────────────────
DATA_PORT  = 12001
CTRL_PORT  = 12000
INFER_PORT = 12002

# ─── Constantes ─────────────────────────────────────────────────
MAGIC = 0xCDAC0001
HDR_FMT = "<IIdfHH"        # magic, seq, t_s, fs_hz, n_samp, n_ch
HDR_SIZE = struct.calcsize(HDR_FMT)   # 24 bytes

# Control message types
MSG_SYNC_REQ = ord("S")
MSG_SYNC_RSP = ord("R")
MSG_START    = ord("G")
MSG_STOP     = ord("X")
MSG_CFG      = ord("C")
MSG_INFER    = ord("I")

# ─── Defaults ───────────────────────────────────────────────────
DEFAULT_FS   = 2500       # Hz
DEFAULT_SPR  = 100        # samples per read (per channel)
DEFAULT_NCH  = 2          # force_v, accel_g

MAX_UDP_PAYLOAD = 1400    # bytes, stay below typical MTU


# ═══════════════════════════════════════════════════════════════
# DATA packets
# ═══════════════════════════════════════════════════════════════

def pack_data(seq: int, t_s: float, fs_hz: float,
              force: np.ndarray, accel: np.ndarray) -> bytes:
    """Empaqueta un bloque de datos force(N,) + accel(N,) en un datagrama UDP."""
    n_samp = len(force)
    n_ch = 2
    hdr = struct.pack(HDR_FMT, MAGIC, seq, t_s, fs_hz, n_samp, n_ch)
    # Intercalar: [f0, a0, f1, a1, …]
    interleaved = np.empty(n_ch * n_samp, dtype=np.float32)
    interleaved[0::2] = force.astype(np.float32)
    interleaved[1::2] = accel.astype(np.float32)
    return hdr + interleaved.tobytes()


def unpack_data(buf: bytes):
    """Desempaqueta un datagrama de datos.
    Returns: (seq, t_s, fs_hz, n_samp, n_ch, force, accel) or None on error.
    """
    if len(buf) < HDR_SIZE:
        return None
    magic, seq, t_s, fs_hz, n_samp, n_ch = struct.unpack(HDR_FMT, buf[:HDR_SIZE])
    if magic != MAGIC:
        return None
    expected = HDR_SIZE + n_ch * n_samp * 4
    if len(buf) < expected:
        return None
    payload = np.frombuffer(buf[HDR_SIZE:HDR_SIZE + n_ch * n_samp * 4],
                            dtype=np.float32)
    force = payload[0::2].copy()
    accel = payload[1::2].copy()
    return seq, t_s, fs_hz, n_samp, n_ch, force, accel


# ═══════════════════════════════════════════════════════════════
# CONTROL packets
# ═══════════════════════════════════════════════════════════════

def pack_sync_req(t1: float) -> bytes:
    return struct.pack("<IB d", MAGIC, MSG_SYNC_REQ, t1)

def unpack_sync_req(buf: bytes):
    if len(buf) < 13 or buf[4] != MSG_SYNC_REQ:
        return None
    _, _, t1 = struct.unpack("<IB d", buf[:13])
    return t1

def pack_sync_rsp(t1: float, t2: float, t3: float) -> bytes:
    return struct.pack("<IB ddd", MAGIC, MSG_SYNC_RSP, t1, t2, t3)

def unpack_sync_rsp(buf: bytes):
    if len(buf) < 29 or buf[4] != MSG_SYNC_RSP:
        return None
    _, _, t1, t2, t3 = struct.unpack("<IB ddd", buf[:29])
    return t1, t2, t3

def pack_start() -> bytes:
    return struct.pack("<IB", MAGIC, MSG_START)

def pack_stop() -> bytes:
    return struct.pack("<IB", MAGIC, MSG_STOP)

def pack_cfg(fs_hz: int, samples_per_read: int) -> bytes:
    return struct.pack("<IB II", MAGIC, MSG_CFG, fs_hz, samples_per_read)

def unpack_cfg(buf: bytes):
    if len(buf) < 13 or buf[4] != MSG_CFG:
        return None
    _, _, fs, spr = struct.unpack("<IB II", buf[:13])
    return fs, spr

def ctrl_msg_type(buf: bytes) -> int:
    """Devuelve el tipo de mensaje de control, o -1 si es inválido."""
    if len(buf) < 5:
        return -1
    magic = struct.unpack("<I", buf[:4])[0]
    if magic != MAGIC:
        return -1
    return buf[4]


# ═════════════════════════════════════════════════════════════
# INFERENCE packets  (Jetson → Windows)
# ═════════════════════════════════════════════════════════════

INFER_HDR_FMT  = "<IB I d H"   # magic, type, seq, t_s, n_values
INFER_HDR_SIZE = struct.calcsize(INFER_HDR_FMT)  # 19 bytes

def pack_infer(seq: int, t_s: float, values: np.ndarray) -> bytes:
    """Empaqueta resultados de inferencia para enviar a Windows.
    values: array 1-D float32 con los resultados (e.g. fuerza predicha,
    estado estimado, features, etc.).
    """
    vals = np.asarray(values, dtype=np.float32).ravel()
    hdr = struct.pack(INFER_HDR_FMT, MAGIC, MSG_INFER, seq, t_s, len(vals))
    return hdr + vals.tobytes()


def unpack_infer(buf: bytes):
    """Desempaqueta un paquete de inferencia.
    Returns: (seq, t_s, values_array) or None.
    """
    if len(buf) < INFER_HDR_SIZE:
        return None
    magic, mtype, seq, t_s, n_vals = struct.unpack(
        INFER_HDR_FMT, buf[:INFER_HDR_SIZE])
    if magic != MAGIC or mtype != MSG_INFER:
        return None
    expected = INFER_HDR_SIZE + n_vals * 4
    if len(buf) < expected:
        return None
    vals = np.frombuffer(
        buf[INFER_HDR_SIZE:INFER_HDR_SIZE + n_vals * 4],
        dtype=np.float32).copy()
    return seq, t_s, vals
