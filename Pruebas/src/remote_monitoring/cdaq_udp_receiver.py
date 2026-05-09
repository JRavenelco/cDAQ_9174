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
import json
import queue
import urllib.request
import urllib.error
import base64
import io
from collections import deque
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass, field, asdict
import math
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw
    _HAVE_PIL = True
except Exception:
    Image = None
    ImageDraw = None
    _HAVE_PIL = False

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


class JSONLCaseStore:
    def __init__(self, path: str):
        self.path = path
        self._file = None
        self._lock = threading.Lock()

    def open(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")

    def append(self, record: dict):
        if not self._file:
            return
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self):
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


class OllamaReasoner:
    def __init__(self,
                 url: str = "http://127.0.0.1:11434/api/generate",
                 model: str = "llama3.2",
                 timeout_s: float = 30.0):
        self.url = url
        self.model = model
        self.timeout_s = float(timeout_s)

    def reason(self, meta: dict, features: dict) -> dict:
        payload = {
            "model": self.model,
            "stream": False,
            "prompt": (
                "Eres un sistema de razonamiento para monitoreo de mecanizado.\n"
                "Devuelve SOLO un JSON válido (sin markdown) con llaves: "
                "cutting_state, should_store_case, store_segment, anomaly, label, notes.\n"
                "meta=" + json.dumps(meta, ensure_ascii=False) + "\n"
                "features=" + json.dumps(features, ensure_ascii=False) + "\n"
            ),
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        outer = json.loads(raw)
        text = (outer.get("response") or "").strip()
        if not text:
            return {"raw": raw}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"response": text}


class OllamaGenerator:
    def __init__(self,
                 url: str = "http://127.0.0.1:11434/api/generate",
                 model: str = "llama3.2",
                 timeout_s: float = 30.0):
        self.url = url
        self.model = model
        self.timeout_s = float(timeout_s)

    def generate(self, prompt: str, images_b64: Optional[list] = None) -> dict:
        if images_b64:
            # Modelos de vision modernos (moondream, gemma4, llava) requieren /api/chat
            chat_url = self.url.replace("/api/generate", "/api/chat")
            payload = {
                "model": self.model,
                "stream": False,
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": images_b64,
                }],
            }
            req = urllib.request.Request(
                chat_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            outer = json.loads(raw)
            text = (outer.get("message", {}).get("content") or "").strip()
        else:
            payload = {"model": self.model, "stream": False, "prompt": prompt}
            req = urllib.request.Request(
                self.url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            outer = json.loads(raw)
            text = (outer.get("response") or "").strip()
        return {"raw": raw, "response": text}

    @staticmethod
    def parse_json_from_response_text(text: str) -> Optional[dict]:
        t = (text or "").strip()
        if not t:
            return None
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            return None


class VisionPanelRenderer:
    def __init__(self,
                 width: int = 768,
                 height: int = 512,
                 spec_nfft: int = 256,
                 spec_hop: int = 64,
                 spec_fmax_hz: float = 1200.0):
        self.width = int(width)
        self.height = int(height)
        self.spec_nfft = int(spec_nfft)
        self.spec_hop = int(spec_hop)
        self.spec_fmax_hz = float(spec_fmax_hz)

    def _robust_scale(self, x: np.ndarray) -> tuple:
        if x.size == 0:
            return -1.0, 1.0
        a = np.asarray(x, dtype=np.float32)
        a = a[np.isfinite(a)]
        if a.size < 16:
            m = float(np.nanmean(a)) if a.size else 0.0
            s = float(np.nanstd(a)) if a.size else 1.0
            s = max(s, 1e-6)
            return m - 3.0 * s, m + 3.0 * s
        lo = float(np.percentile(a, 2.0))
        hi = float(np.percentile(a, 98.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-6:
            m = float(np.nanmean(a))
            s = float(np.nanstd(a))
            s = max(s, 1e-6)
            return m - 3.0 * s, m + 3.0 * s
        pad = 0.05 * (hi - lo)
        return lo - pad, hi + pad

    def _draw_waveform(self, draw: "ImageDraw.ImageDraw", x: np.ndarray,
                       rect: tuple, color=(0, 200, 255), label: str = ""):
        x0, y0, x1, y1 = rect
        w = max(1, int(x1 - x0))
        h = max(1, int(y1 - y0))
        if x.size < 2:
            return
        lo, hi = self._robust_scale(x)
        if abs(hi - lo) < 1e-9:
            hi = lo + 1.0

        n = x.size
        for px in range(w):
            i0 = int(px * n / w)
            i1 = int((px + 1) * n / w)
            if i1 <= i0:
                i1 = i0 + 1
            seg = x[i0:i1]
            seg = seg[np.isfinite(seg)]
            if seg.size == 0:
                continue
            vmin = float(np.min(seg))
            vmax = float(np.max(seg))
            yy0 = y1 - int((vmin - lo) / (hi - lo) * h)
            yy1 = y1 - int((vmax - lo) / (hi - lo) * h)
            yy0 = max(y0, min(y1, yy0))
            yy1 = max(y0, min(y1, yy1))
            draw.line([(x0 + px, yy0), (x0 + px, yy1)], fill=color)

        if label:
            draw.text((x0 + 6, y0 + 4), label, fill=(230, 230, 230))

    def _spectrogram(self, x: np.ndarray, fs: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x = x[np.isfinite(x)]
        if x.size < self.spec_nfft:
            return np.zeros((64, 64), dtype=np.uint8)

        nfft = self.spec_nfft
        hop = max(1, self.spec_hop)
        win = np.hanning(nfft).astype(np.float32)

        n_frames = 1 + (x.size - nfft) // hop
        if n_frames < 1:
            n_frames = 1

        freqs = np.fft.rfftfreq(nfft, d=1.0 / float(fs))
        fmax = float(self.spec_fmax_hz)
        fmask = freqs <= fmax
        n_f = int(np.sum(fmask))
        if n_f < 8:
            n_f = min(len(freqs), 64)
            fmask = np.arange(len(freqs)) < n_f

        spec = np.zeros((n_f, n_frames), dtype=np.float32)
        for i in range(n_frames):
            s = i * hop
            frame = x[s:s + nfft]
            if frame.size < nfft:
                pad = np.zeros((nfft,), dtype=np.float32)
                pad[:frame.size] = frame
                frame = pad
            frame = (frame - float(np.mean(frame))) * win
            mag = np.abs(np.fft.rfft(frame))
            mag = mag[fmask]
            spec[:, i] = mag

        spec_db = 20.0 * np.log10(spec + 1e-6)
        lo = float(np.percentile(spec_db, 10.0))
        hi = float(np.percentile(spec_db, 99.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-6:
            lo, hi = float(np.min(spec_db)), float(np.max(spec_db) + 1e-6)
        spec_n = (spec_db - lo) / (hi - lo)
        spec_n = np.clip(spec_n, 0.0, 1.0)
        return (spec_n * 255.0).astype(np.uint8)

    def render_png_bytes(self,
                         force: np.ndarray,
                         accel: np.ndarray,
                         fs: float,
                         meta: Optional[dict] = None) -> Optional[bytes]:
        if not _HAVE_PIL:
            return None

        meta = meta or {}
        W, H = self.width, self.height
        img = Image.new("RGB", (W, H), (20, 20, 24))
        draw = ImageDraw.Draw(img)

        pad = 10
        title_h = 36
        top = pad + title_h
        inner_h = H - top - pad
        row_h = inner_h // 3

        r_force = (pad, top, W - pad, top + row_h - 4)
        r_accel = (pad, top + row_h, W - pad, top + 2 * row_h - 4)
        r_spec = (pad, top + 2 * row_h, W - pad, H - pad)

        cut_id = meta.get("cut_id", "")
        cut_label = meta.get("cut_label", "")
        title = f"cut={cut_id} {cut_label}  fs={fs:.0f}Hz  N={len(accel)}"
        draw.text((pad, pad + 6), title, fill=(240, 240, 240))

        self._draw_waveform(
            draw,
            np.asarray(force).ravel(),
            r_force,
            color=(0, 200, 255),
            label=f"force_v (std={float(np.std(force)):.3g})",
        )
        self._draw_waveform(
            draw,
            np.asarray(accel).ravel(),
            r_accel,
            color=(255, 170, 0),
            label=f"accel_g (std={float(np.std(accel)):.3g})",
        )

        spec = self._spectrogram(np.asarray(accel).ravel(), fs=float(fs))
        spec_img = Image.fromarray(spec[::-1, :], mode="L")
        spec_w = max(1, int(r_spec[2] - r_spec[0]))
        spec_h = max(1, int(r_spec[3] - r_spec[1]))
        spec_img = spec_img.resize((spec_w, spec_h), resample=Image.BILINEAR)
        spec_rgb = Image.merge("RGB", (spec_img, spec_img, spec_img))
        img.paste(spec_rgb, (int(r_spec[0]), int(r_spec[1])))
        draw.text((r_spec[0] + 6, r_spec[1] + 4), "Spectrogram accel_g", fill=(230, 230, 230))

        for r in (r_force, r_accel, r_spec):
            draw.rectangle(r, outline=(60, 60, 70), width=1)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


class MultiModalCaseWorker(threading.Thread):
    def __init__(self,
                 vlm: OllamaGenerator,
                 llm: OllamaReasoner,
                 store: JSONLCaseStore,
                 stop_flag: threading.Event,
                 renderer: VisionPanelRenderer):
        super().__init__(daemon=True)
        self.vlm = vlm
        self.llm = llm
        self.store = store
        self.stop_flag = stop_flag
        self.renderer = renderer
        self.q: "queue.Queue[dict]" = queue.Queue(maxsize=16)

    def submit(self, task: dict):
        try:
            self.q.put_nowait(task)
        except queue.Full:
            pass

    def _vlm_prompt(self, meta: dict, features: dict) -> str:
        return (
            "Eres un inspector visual de señales de mecanizado.\n"
            "Ves un panel con: fuerza, aceleración y espectrograma (aceleración).\n"
            "Detecta chatter/vibración anómala, golpes, inestabilidad.\n"
            "Devuelve SOLO un JSON válido (sin markdown) con llaves: "
            "stable, chatter, severity_0_3, dominant_band_hz, notes.\n"
            "meta=" + json.dumps(meta, ensure_ascii=False) + "\n"
            "features=" + json.dumps(features, ensure_ascii=False) + "\n"
        )

    def run(self):
        while not self.stop_flag.is_set():
            try:
                task = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                meta = task.get("meta", {}) or {}
                features = task.get("features", {}) or {}
                force = task.get("force")
                accel = task.get("accel")
                fs_hz = float(task.get("fs_hz") or 0.0)

                vlm_obj = None
                vlm_text = ""
                if force is not None and accel is not None and fs_hz > 1.0:
                    png = self.renderer.render_png_bytes(force=force, accel=accel, fs=fs_hz, meta=meta)
                    if png is not None:
                        img_b64 = base64.b64encode(png).decode("ascii")
                        vr = self.vlm.generate(prompt=self._vlm_prompt(meta, features), images_b64=[img_b64])
                        vlm_text = (vr.get("response") or "").strip()
                        vlm_obj = OllamaGenerator.parse_json_from_response_text(vlm_text)

                merged_features = dict(features)
                if vlm_obj is not None:
                    merged_features["vlm"] = vlm_obj
                elif vlm_text:
                    merged_features["vlm_text"] = vlm_text

                llm = self.llm.reason(meta=meta, features=merged_features)
                case = {
                    "t_local_s": task.get("t_local_s"),
                    "seq": task.get("seq"),
                    "meta": meta,
                    "features": features,
                    "vlm": (vlm_obj if vlm_obj is not None else {"response": vlm_text} if vlm_text else None),
                    "llm": llm,
                }
                self.store.append(case)
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as e:
                log.warning("Multimodal worker error: %s", e)
            except Exception as e:
                log.warning("Multimodal worker unknown error: %s", e)


# ═══════════════════════════════════════════════════════════════
# Hysteresis Renderer  (PIL, sin matplotlib, para daemon)
# ═══════════════════════════════════════════════════════════════

class HisteresisRenderer:
    """
    Renderiza el lazo de histeresis F vs a usando PIL.
    Usa gradiente de color (azul→rojo) para mostrar la direccion del lazo.
    Compatible con VLMs de vision: imagen clara sobre fondo oscuro.
    """

    def __init__(self, width: int = 512, height: int = 512):
        self.width = int(width)
        self.height = int(height)

    @staticmethod
    def _robust_range(x: np.ndarray) -> tuple[float, float]:
        x = x[np.isfinite(x)]
        if x.size < 4:
            return -1.0, 1.0
        lo = float(np.percentile(x, 2.0))
        hi = float(np.percentile(x, 98.0))
        span = hi - lo
        if span < 1e-9:
            m = (lo + hi) / 2
            return m - 1.0, m + 1.0
        pad = 0.08 * span
        return lo - pad, hi + pad

    def render_png_bytes(
        self,
        force: np.ndarray,
        accel: np.ndarray,
        meta: Optional[dict] = None,
    ) -> Optional[bytes]:
        if not _HAVE_PIL:
            return None

        f = np.asarray(force, dtype=np.float32).ravel()
        a = np.asarray(accel, dtype=np.float32).ravel()
        n = min(len(f), len(a))
        if n < 4:
            return None
        f, a = f[:n], a[:n]

        meta = meta or {}
        W, H = self.width, self.height
        PAD = 40

        img = Image.new("RGB", (W, H), (18, 18, 22))
        draw = ImageDraw.Draw(img)

        # Rangos con margen
        a_lo, a_hi = self._robust_range(a)
        f_lo, f_hi = self._robust_range(f)
        a_span = a_hi - a_lo
        f_span = f_hi - f_lo

        inner_w = W - 2 * PAD
        inner_h = H - 2 * PAD - 28  # 28 para titulo

        def to_px(ai: float, fi: float) -> tuple[int, int]:
            px = int(PAD + (ai - a_lo) / a_span * inner_w)
            py = int(H - PAD - (fi - f_lo) / f_span * inner_h)
            return (
                max(PAD, min(W - PAD, px)),
                max(28 + PAD, min(H - PAD, py)),
            )

        # Ejes
        zero_x = to_px(0.0, f_lo)[0]
        zero_y = to_px(a_lo, 0.0)[1]
        draw.line([(PAD, zero_y), (W - PAD, zero_y)], fill=(55, 55, 65), width=1)
        draw.line([(zero_x, 28 + PAD), (zero_x, H - PAD)], fill=(55, 55, 65), width=1)

        # Lazo con gradiente azul→rojo (inicio→fin)
        pts = [to_px(float(a[i]), float(f[i])) for i in range(n)]
        for i in range(n - 1):
            t = i / max(n - 2, 1)
            r = int(40 + 215 * t)
            g = int(120 * (1 - abs(t - 0.5) * 2))
            b = int(255 * (1 - t) + 40 * t)
            draw.line([pts[i], pts[i + 1]], fill=(r, g, b), width=2)

        # Punto de inicio (verde) y fin (blanco)
        sx, sy = pts[0]
        draw.ellipse([(sx - 4, sy - 4), (sx + 4, sy + 4)], fill=(0, 230, 100))
        ex, ey = pts[-1]
        draw.ellipse([(ex - 3, ey - 3), (ex + 3, ey + 3)], fill=(255, 255, 255))

        # Etiquetas de ejes
        draw.text((W // 2 - 20, H - 14), "accel_g", fill=(180, 180, 180))
        draw.text((4, H // 2 - 8), "F", fill=(180, 180, 180))

        # Titulo
        rpm = meta.get("rpm_spindle", meta.get("rpm_husillo", 0))
        cutter = meta.get("cutter_condition", meta.get("cut_label", ""))
        title = f"Histeresis F vs a  RPM={rpm}  {cutter}"
        draw.text((PAD, 6), title[:60], fill=(220, 220, 220))

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


# ═══════════════════════════════════════════════════════════════
# Experto CNC — guarda casos individuales en JSON
# ═══════════════════════════════════════════════════════════════

class ExpertoCasoStore:
    """
    Guarda cada segmento aprobado por el VLM como un JSON individual en
    casos_experto/{cutter}_{rpm}_{timestamp}.json
    con: datos (t,F,a), metadata, decision VLM.
    """

    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.saved = 0

    def save(
        self,
        t: np.ndarray,
        force: np.ndarray,
        accel: np.ndarray,
        fs_hz: float,
        meta: dict,
        vlm_decision: dict,
        loop_area: float,
    ) -> str:
        cutter = str(meta.get("cutter_condition", meta.get("cut_label", "unknown"))).strip()
        rpm = int(meta.get("rpm_spindle", meta.get("rpm_husillo", 0)))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]
        safe_cutter = cutter.replace(" ", "_").replace("/", "_")
        fname = f"{safe_cutter}_{rpm}rpm_{ts}.json"

        record = {
            "case_id": fname.replace(".json", ""),
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "cutter_condition": cutter,
                "rpm_husillo": int(meta.get("rpm_spindle", meta.get("rpm_husillo", 0))),
                "rpm_avance": int(meta.get("rpm_feed", meta.get("rpm_avance", 0))),
                "ap_mm": float(meta.get("ap_mm", 0.0)),
                "fs_hz": float(fs_hz),
                "cut_label": str(meta.get("cut_label", "")),
            },
            "datos": {
                "t": [float(v) for v in t[:1250]],
                "fuerza_V": [float(v) for v in force[:1250]],
                "accel_g": [float(v) for v in accel[:1250]],
            },
            "vlm": vlm_decision,
            "loop_area_norm": float(loop_area),
        }

        out_path = self.out_dir / fname
        with self._lock:
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False, indent=2)
            self.saved += 1
        log.info("EXPERTO guardado: %s  (total=%d)", fname, self.saved)
        return str(out_path)


class ExpertoCNCWorker(threading.Thread):
    """
    Worker daemon que:
    1. Recibe tareas con signal snapshot (force, accel, fs, meta).
    2. Renderiza lazo de histeresis F vs a con PIL.
    3. Llama al VLM llama3.2-vision con prompt de experto CNC.
    4. Si should_store=true → guarda JSON via ExpertoCasoStore.
    """

    _VLM_PROMPT_TMPL = (
        "Eres un experto en mecanizado CNC. Observas el lazo de histéresis "
        "Fuerza vs Aceleración de una operacion de fresado.\n"
        "RPM={rpm}  Condicion cortador={cutter}  Area lazo={area:.4f}\n"
        "El gradiente azul->rojo indica la direccion temporal del lazo.\n"
        "Devuelve SOLO un JSON valido (sin markdown) con estas llaves:\n"
        "  cutting_state  : true/false  (hay corte activo?)\n"
        "  quality        : \"good\"|\"marginal\"|\"bad\"\n"
        "  should_store   : true/false  (vale la pena guardar este segmento?)\n"
        "  anomaly        : null o string describiendo la anomalia\n"
        "  notes          : string breve con observaciones\n"
    )

    def __init__(
        self,
        vlm: OllamaGenerator,
        store: "ExpertoCasoStore",
        stop_flag: threading.Event,
        renderer: "HisteresisRenderer",
    ):
        super().__init__(daemon=True)
        self.vlm = vlm
        self.store = store
        self.stop_flag = stop_flag
        self.renderer = renderer
        self.q: "queue.Queue[dict]" = queue.Queue(maxsize=8)
        self.approved = 0
        self.rejected = 0

    def submit(self, task: dict) -> None:
        try:
            self.q.put_nowait(task)
        except queue.Full:
            pass

    @staticmethod
    def _loop_area(force: np.ndarray, accel: np.ndarray) -> float:
        if len(force) < 4:
            return 0.0
        f = force.astype(np.float64)
        a = accel.astype(np.float64)
        f -= f.mean(); a -= a.mean()
        _trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        area = float(abs(_trap(f, a)))
        bbox = (f.max() - f.min()) * (a.max() - a.min())
        return float(np.clip(area / bbox, 0.0, 1.0)) if bbox > 1e-15 else 0.0

    def run(self) -> None:
        while not self.stop_flag.is_set():
            try:
                task = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                self._process(task)
            except (urllib.error.URLError, TimeoutError) as exc:
                log.warning("ExpertoCNCWorker VLM error: %s", exc)
            except Exception as exc:
                log.warning("ExpertoCNCWorker error: %s", exc)

    def _process(self, task: dict) -> None:
        force = np.asarray(task.get("force", []), dtype=np.float32)
        accel = np.asarray(task.get("accel", []), dtype=np.float32)
        fs_hz = float(task.get("fs_hz") or 2500.0)
        meta = task.get("meta", {}) or {}

        if force.size < 16 or accel.size < 16:
            return

        la = self._loop_area(force, accel)
        rpm = int(meta.get("rpm_spindle", meta.get("rpm_husillo", 0)))
        cutter = str(meta.get("cutter_condition", meta.get("cut_label", "")))

        # Renderizar lazo de histeresis
        png = self.renderer.render_png_bytes(force=force, accel=accel, meta=meta)
        if png is None:
            log.warning("ExpertoCNCWorker: render fallo (PIL no disponible?)")
            return

        img_b64 = base64.b64encode(png).decode("ascii")
        prompt = self._VLM_PROMPT_TMPL.format(rpm=rpm, cutter=cutter or "?", area=la)

        resp = self.vlm.generate(prompt=prompt, images_b64=[img_b64])
        text = (resp.get("response") or "").strip()
        vlm_obj = OllamaGenerator.parse_json_from_response_text(text)

        if vlm_obj is None:
            vlm_obj = {"response": text, "should_store": False}

        should_store = bool(vlm_obj.get("should_store", False))
        quality = str(vlm_obj.get("quality", ""))
        cutting = bool(vlm_obj.get("cutting_state", False))

        log.info(
            "EXPERTO: cutting=%s quality=%s should_store=%s area=%.4f",
            cutting, quality, should_store, la
        )

        if should_store:
            t = np.arange(min(len(force), len(accel)), dtype=np.float32) / max(fs_hz, 1.0)
            self.store.save(
                t=t, force=force, accel=accel, fs_hz=fs_hz,
                meta=meta, vlm_decision=vlm_obj, loop_area=la,
            )
            self.approved += 1
        else:
            self.rejected += 1


class SentinelTrigger:
    def __init__(self,
                 threshold: float = 3.0,
                 persist_n: int = 3,
                 cooldown_s: float = 8.0,
                 alpha: float = 0.02,
                 band_low_hz: float = 120.0,
                 band_high_hz: float = 1200.0,
                 require_cutting_flag: bool = False):
        self.threshold = float(threshold)
        self.persist_n = int(persist_n)
        self.cooldown_s = float(cooldown_s)
        self.alpha = float(alpha)
        self.band_low_hz = float(band_low_hz)
        self.band_high_hz = float(band_high_hz)
        self.require_cutting_flag = bool(require_cutting_flag)

        self._rms_ema = None
        self._rms_dev = 0.0
        self._thd_ema = None
        self._thd_dev = 0.0
        self._counter = 0
        self._last_trigger_t = 0.0

    def _ema_update(self, ema: Optional[float], x: float) -> float:
        if ema is None or not np.isfinite(ema):
            return float(x)
        return float((1.0 - self.alpha) * float(ema) + self.alpha * float(x))

    def update(self, feats: dict, now_s: float) -> bool:
        cutting_flag = feats.get("cutting_flag", None)
        if self.require_cutting_flag and cutting_flag is not None:
            try:
                if float(cutting_flag) < 0.5:
                    self._counter = max(0, self._counter - 1)
                    return False
            except Exception:
                pass

        accel_rms = feats.get("accel_rms_g", feats.get("accel_rms", None))
        thd = feats.get("THD_accel_pct", feats.get("thd_accel_pct", None))
        freq_dom = feats.get("freq_dom_Hz", feats.get("freq_dom", None))

        score_parts = []

        if accel_rms is not None:
            try:
                x = float(accel_rms)
                if np.isfinite(x):
                    self._rms_ema = self._ema_update(self._rms_ema, x)
                    dev = abs(x - float(self._rms_ema))
                    self._rms_dev = self._ema_update(self._rms_dev, dev)
                    denom = max(1e-6, float(self._rms_dev))
                    z = (x - float(self._rms_ema)) / denom
                    score_parts.append(max(0.0, float(z)))
            except Exception:
                pass

        if thd is not None:
            try:
                x = float(thd)
                if np.isfinite(x):
                    self._thd_ema = self._ema_update(self._thd_ema, x)
                    dev = abs(x - float(self._thd_ema))
                    self._thd_dev = self._ema_update(self._thd_dev, dev)
                    denom = max(1e-6, float(self._thd_dev))
                    z = (x - float(self._thd_ema)) / denom
                    score_parts.append(0.5 * max(0.0, float(z)))
            except Exception:
                pass

        score = float(max(score_parts) if score_parts else 0.0)

        if freq_dom is not None:
            try:
                f = float(freq_dom)
                if np.isfinite(f) and (self.band_low_hz <= f <= self.band_high_hz):
                    score += 0.5
            except Exception:
                pass

        if score >= self.threshold:
            self._counter += 1
        else:
            self._counter = max(0, self._counter - 1)

        if self._counter < self.persist_n:
            return False

        if (now_s - self._last_trigger_t) < max(0.0, self.cooldown_s):
            return False

        self._last_trigger_t = float(now_s)
        self._counter = 0
        return True


class LLMCaseWorker(threading.Thread):
    def __init__(self,
                 reasoner: OllamaReasoner,
                 store: JSONLCaseStore,
                 stop_flag: threading.Event):
        super().__init__(daemon=True)
        self.reasoner = reasoner
        self.store = store
        self.stop_flag = stop_flag
        self.q: "queue.Queue[dict]" = queue.Queue(maxsize=64)

    def submit(self, task: dict):
        try:
            self.q.put_nowait(task)
        except queue.Full:
            pass

    def run(self):
        while not self.stop_flag.is_set():
            try:
                task = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                meta = task.get("meta", {})
                features = task.get("features", {})
                llm = self.reasoner.reason(meta=meta, features=features)
                case = {
                    "t_local_s": task.get("t_local_s"),
                    "seq": task.get("seq"),
                    "meta": meta,
                    "features": features,
                    "llm": llm,
                }
                self.store.append(case)
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as e:
                log.warning("LLM worker error: %s", e)
            except Exception as e:
                log.warning("LLM worker unknown error: %s", e)


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

    RESULT_NAMES = [
        "force_mean",
        "force_std",
        "force_rms",
        "accel_mean",
        "accel_std",
        "accel_rms",
    ]
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
                 cutting: CuttingConditions = None,
                 llm_enable: bool = False,
                 llm_every_s: float = 1.0,
                 llm_timeout_s: float = 30.0,
                 llm_model: str = "llama3.2",
                 llm_url: str = "http://127.0.0.1:11434/api/generate",
                 case_store_path: str = None,
                 vlm_enable: bool = False,
                 vlm_model: str = "moondream",
                 vlm_every_s: float = 2.0,
                 vlm_timeout_s: float = 30.0,
                 vlm_window_s: float = 2.0,
                 sentinel_threshold: float = 3.0,
                 sentinel_persist_n: int = 3,
                 sentinel_cooldown_s: float = 8.0,
                 sentinel_band_low_hz: float = 120.0,
                 sentinel_band_high_hz: float = 1200.0,
                 sentinel_require_cutting_flag: bool = False,
                 experto_enable: bool = False,
                 experto_dir: str = "./casos_experto",
                 experto_cutter_condition: str = "",
                 experto_every_s: float = 2.0):
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

        self._llm_enable = bool(llm_enable)
        self._llm_every_s = float(llm_every_s) if llm_every_s is not None else 1.0
        self._llm_last_t = 0.0
        self._llm_stop = threading.Event()
        self._llm_worker: Optional[LLMCaseWorker] = None
        self._mm_worker: Optional[MultiModalCaseWorker] = None
        self._case_store: Optional[JSONLCaseStore] = None

        self._vlm_enable = bool(vlm_enable)
        self._vlm_model = str(vlm_model or "moondream")
        self._vlm_every_s = float(vlm_every_s) if vlm_every_s is not None else 2.0
        self._vlm_timeout_s = float(vlm_timeout_s) if vlm_timeout_s is not None else 30.0
        self._vlm_window_s = float(vlm_window_s) if vlm_window_s is not None else 2.0
        self._vlm_last_t = 0.0

        self._sentinel: Optional[SentinelTrigger] = None
        if self._vlm_enable:
            self._sentinel = SentinelTrigger(
                threshold=sentinel_threshold,
                persist_n=sentinel_persist_n,
                cooldown_s=sentinel_cooldown_s,
                band_low_hz=sentinel_band_low_hz,
                band_high_hz=sentinel_band_high_hz,
                require_cutting_flag=sentinel_require_cutting_flag,
            )

        self._sigbuf_force: deque = deque()
        self._sigbuf_accel: deque = deque()
        self._sigbuf_n = 0
        self._sigbuf_fs = None
        if self._llm_enable and case_store_path:
            try:
                self._case_store = JSONLCaseStore(case_store_path)
                self._case_store.open()
                if self._vlm_enable:
                    if not _HAVE_PIL:
                        raise RuntimeError("Pillow no disponible: no se puede habilitar VLM")
                    vlm = OllamaGenerator(url=llm_url, model=self._vlm_model, timeout_s=self._vlm_timeout_s)
                    llm_reasoner = OllamaReasoner(url=llm_url, model=llm_model, timeout_s=llm_timeout_s)
                    renderer = VisionPanelRenderer()
                    self._mm_worker = MultiModalCaseWorker(vlm, llm_reasoner, self._case_store, self._llm_stop, renderer)
                    self._mm_worker.start()
                    log.info("Multimodal habilitado: vlm=%s llm=%s cases=%s", self._vlm_model, llm_model, case_store_path)
                else:
                    reasoner = OllamaReasoner(url=llm_url, model=llm_model, timeout_s=llm_timeout_s)
                    self._llm_worker = LLMCaseWorker(reasoner, self._case_store, self._llm_stop)
                    self._llm_worker.start()
                    log.info("LLM habilitado: model=%s  cases=%s", llm_model, case_store_path)
            except Exception as e:
                log.warning("No se pudo iniciar LLM worker: %s", e)
                self._llm_enable = False
                self._llm_worker = None
                self._mm_worker = None
                if self._case_store:
                    try:
                        self._case_store.close()
                    except Exception:
                        pass
                    self._case_store = None

        self._experto_enable = bool(experto_enable)
        self._experto_cutter_condition = str(experto_cutter_condition)
        self._experto_every_s = float(experto_every_s)
        self._experto_last_t = 0.0
        self._experto_worker: Optional[ExpertoCNCWorker] = None
        if self._experto_enable:
            if not _HAVE_PIL:
                log.warning("--experto requiere Pillow; desactivado")
            else:
                try:
                    _exp_store = ExpertoCasoStore(experto_dir)
                    _exp_renderer = HisteresisRenderer()
                    _exp_vlm = OllamaGenerator(
                        url=llm_url, model=vlm_model, timeout_s=vlm_timeout_s
                    )
                    self._experto_worker = ExpertoCNCWorker(
                        _exp_vlm, _exp_store, self._llm_stop, _exp_renderer
                    )
                    self._experto_worker.start()
                    log.info(
                        "Experto CNC habilitado: dir=%s  cutter=%s  cada=%.1fs",
                        experto_dir, experto_cutter_condition or "?", experto_every_s,
                    )
                except Exception as e:
                    log.warning("No se pudo iniciar ExpertoCNCWorker: %s", e)
                    self._experto_worker = None

        # State
        self.running = False
        self.stats = {
            "packets": 0, "errors": 0, "last_seq": -1,
            "infer_calls": 0, "infer_us_avg": 0.0,
        }

    def _sigbuf_push(self, force: np.ndarray, accel: np.ndarray, fs_hz: float):
        if force is None or accel is None:
            return
        if fs_hz <= 1.0:
            return
        if self._sigbuf_fs is None:
            self._sigbuf_fs = float(fs_hz)
        if abs(float(fs_hz) - float(self._sigbuf_fs)) > 1e-3:
            self._sigbuf_force.clear()
            self._sigbuf_accel.clear()
            self._sigbuf_n = 0
            self._sigbuf_fs = float(fs_hz)

        f = np.asarray(force, dtype=np.float32).ravel()
        a = np.asarray(accel, dtype=np.float32).ravel()
        self._sigbuf_force.append(f)
        self._sigbuf_accel.append(a)
        self._sigbuf_n += int(min(f.size, a.size))

        max_n = int(max(1.0, self._vlm_window_s) * float(self._sigbuf_fs))
        while self._sigbuf_n > max_n and self._sigbuf_force:
            old_f = self._sigbuf_force.popleft()
            old_a = self._sigbuf_accel.popleft()
            self._sigbuf_n -= int(min(old_f.size, old_a.size))

    def _sigbuf_snapshot(self) -> tuple:
        if not self._sigbuf_force or not self._sigbuf_accel or not self._sigbuf_fs:
            return None, None, None
        try:
            f = np.concatenate(list(self._sigbuf_force), axis=0)
            a = np.concatenate(list(self._sigbuf_accel), axis=0)
            return f, a, float(self._sigbuf_fs)
        except Exception:
            return None, None, None

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

            if (self._vlm_enable and self._mm_worker) or self._experto_worker:
                self._sigbuf_push(force, accel, fs_hz)

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

                if self._llm_worker and result is not None:
                    now_s = time.time()
                    if (now_s - self._llm_last_t) >= max(0.1, self._llm_every_s):
                        names = getattr(self.engine, "RESULT_NAMES", []) or []
                        feats = {}
                        arr = np.asarray(result).ravel()
                        if names and len(names) == len(arr):
                            for k, v in zip(names, arr):
                                feats[str(k)] = float(v)
                        else:
                            feats["result"] = [float(v) for v in arr]

                        meta_llm = {
                            "cut_id": int(self.cutting.cut_id),
                            "cut_label": str(self.cutting.label),
                            "ap_mm": float(self.cutting.ap_mm),
                            "rpm_spindle": float(self.cutting.rpm_spindle),
                            "rpm_feed": float(self.cutting.rpm_feed),
                            "tool_diam_mm": float(self.cutting.tool_diam_mm),
                            "n_flutes": int(self.cutting.n_flutes),
                            "Ktc": float(self.cutting.Ktc),
                            "Kte": float(self.cutting.Kte),
                            "fz_mm": float(self.cutting.fz_mm),
                            "Ft_mean_N": float(self.cutting.Ft_mean_N),
                            "Ft_peak_N": float(self.cutting.Ft_peak_N),
                        }

                        self._llm_worker.submit({
                            "t_local_s": float(t_local),
                            "seq": int(seq),
                            "meta": meta_llm,
                            "features": feats,
                        })
                        self._llm_last_t = now_s

                if self._mm_worker and result is not None:
                    now_s = time.time()
                    if (now_s - self._vlm_last_t) >= max(0.2, self._vlm_every_s):
                        names = getattr(self.engine, "RESULT_NAMES", []) or []
                        feats = {}
                        arr = np.asarray(result).ravel()
                        if names and len(names) == len(arr):
                            for k, v in zip(names, arr):
                                feats[str(k)] = float(v)
                        else:
                            feats["result"] = [float(v) for v in arr]

                        should_trigger = True
                        if self._sentinel is not None:
                            try:
                                should_trigger = bool(self._sentinel.update(feats, now_s))
                            except Exception:
                                should_trigger = True

                        if should_trigger:
                            f_snap, a_snap, fs_snap = self._sigbuf_snapshot()
                            if f_snap is not None and a_snap is not None and fs_snap is not None:
                                meta_llm = {
                                    "cut_id": int(self.cutting.cut_id),
                                    "cut_label": str(self.cutting.label),
                                    "ap_mm": float(self.cutting.ap_mm),
                                    "rpm_spindle": float(self.cutting.rpm_spindle),
                                    "rpm_feed": float(self.cutting.rpm_feed),
                                    "tool_diam_mm": float(self.cutting.tool_diam_mm),
                                    "n_flutes": int(self.cutting.n_flutes),
                                    "Ktc": float(self.cutting.Ktc),
                                    "Kte": float(self.cutting.Kte),
                                    "fz_mm": float(self.cutting.fz_mm),
                                    "Ft_mean_N": float(self.cutting.Ft_mean_N),
                                    "Ft_peak_N": float(self.cutting.Ft_peak_N),
                                }

                                self._mm_worker.submit({
                                    "t_local_s": float(t_local),
                                    "seq": int(seq),
                                    "meta": meta_llm,
                                    "features": feats,
                                    "force": f_snap,
                                    "accel": a_snap,
                                    "fs_hz": float(fs_snap),
                                })
                                self._vlm_last_t = now_s

            if self._experto_worker:
                now_s = time.time()
                if (now_s - self._experto_last_t) >= max(0.5, self._experto_every_s):
                    f_snap, a_snap, fs_snap = self._sigbuf_snapshot()
                    if f_snap is not None and a_snap is not None and fs_snap is not None:
                        meta_exp = {
                            "cutter_condition": self._experto_cutter_condition,
                            "cut_label": str(self.cutting.label),
                            "ap_mm": float(self.cutting.ap_mm),
                            "rpm_spindle": float(self.cutting.rpm_spindle),
                            "rpm_husillo": float(self.cutting.rpm_spindle),
                            "rpm_avance": float(self.cutting.rpm_feed),
                            "rpm_feed": float(self.cutting.rpm_feed),
                            "fs_hz": float(fs_snap),
                        }
                        self._experto_worker.submit({
                            "force": f_snap,
                            "accel": a_snap,
                            "fs_hz": float(fs_snap),
                            "meta": meta_exp,
                        })
                        self._experto_last_t = now_s

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
        if self._llm_worker:
            self._llm_stop.set()
            try:
                self._llm_worker.join(timeout=1.0)
            except Exception:
                pass
        if self._mm_worker:
            self._llm_stop.set()
            try:
                self._mm_worker.join(timeout=1.0)
            except Exception:
                pass
        if self._case_store:
            try:
                self._case_store.close()
            except Exception:
                pass
        if self._experto_worker:
            self._llm_stop.set()
            try:
                self._experto_worker.join(timeout=2.0)
            except Exception:
                pass
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
        self._llm_stop.set()

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
    ap.add_argument("--llm", action="store_true",
                    help="Habilitar razonador LLM (Ollama) para generar casos JSONL")
    ap.add_argument("--vlm", action="store_true",
                    help="Habilitar pipeline multimodal (VLM+LLM) via Ollama con panel PNG")
    ap.add_argument("--llm-model", default="llama3.2",
                    help="Modelo de Ollama (ej: llama3.2)")
    ap.add_argument("--vlm-model", default="moondream",
                    help="Modelo VLM en Ollama (ej: moondream)")
    ap.add_argument("--llm-url", default="http://127.0.0.1:11434/api/generate",
                    help="Endpoint Ollama /api/generate")
    ap.add_argument("--llm-every-s", type=float, default=1.0,
                    help="Periodo en segundos entre consultas al LLM")
    ap.add_argument("--vlm-every-s", type=float, default=2.0,
                    help="Periodo en segundos entre consultas al VLM")
    ap.add_argument("--llm-timeout-s", type=float, default=30.0,
                    help="Timeout en segundos para la consulta a Ollama")
    ap.add_argument("--vlm-timeout-s", type=float, default=30.0,
                    help="Timeout en segundos para la consulta al VLM")
    ap.add_argument("--vlm-window-s", type=float, default=2.0,
                    help="Ventana en segundos de señales para render del panel")
    ap.add_argument("--sentinel-threshold", type=float, default=3.0,
                    help="Umbral del centinela (score z) para disparar VLM/LLM")
    ap.add_argument("--sentinel-persist-n", type=int, default=3,
                    help="Número de bloques consecutivos para confirmar trigger")
    ap.add_argument("--sentinel-cooldown-s", type=float, default=8.0,
                    help="Cooldown en segundos entre triggers")
    ap.add_argument("--sentinel-band-low-hz", type=float, default=120.0,
                    help="Banda mínima (Hz) para dar peso extra al trigger")
    ap.add_argument("--sentinel-band-high-hz", type=float, default=1200.0,
                    help="Banda máxima (Hz) para dar peso extra al trigger")
    ap.add_argument("--sentinel-require-cutting-flag", action="store_true",
                    help="Requerir cutting_flag>=0.5 (si el motor lo provee) para disparar")
    # ── Modo experto (VLM evalúa histeresis y acumula dataset) ──
    ap.add_argument("--experto", action="store_true",
                    help="Habilitar modo experto: VLM evalúa lazo de histéresis y guarda casos aprobados")
    ap.add_argument("--casos-experto-dir", default="./casos_experto",
                    help="Directorio donde guardar casos aprobados por el experto VLM")
    ap.add_argument("--cutter-condition", default="",
                    help="Condicion del cortador (nuevo|medio_uso|desgastado) para contexto del VLM")
    ap.add_argument("--experto-every-s", type=float, default=2.0,
                    help="Segundos entre evaluaciones del experto VLM")
    ap.add_argument("--cases", default="auto",
                    help="Ruta para guardar casos JSONL ('auto'=junto al raw, 'none'=desactivar)")
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
    ap.add_argument("--cut-label", default="idle",
                     help="Etiqueta para la condición de corte inicial")
    args = ap.parse_args()

    if args.vlm and not args.llm:
        args.llm = True

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

    case_store_path = None
    if (args.llm or args.vlm) and args.cases != "none":
        if args.cases == "auto":
            case_store_path = csv_path.replace(".csv", "_cases.jsonl")
        else:
            case_store_path = args.cases

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
        llm_enable=bool(args.llm or args.vlm),
        llm_every_s=args.llm_every_s,
        llm_timeout_s=args.llm_timeout_s,
        llm_model=args.llm_model,
        llm_url=args.llm_url,
        case_store_path=case_store_path,
        vlm_enable=bool(args.vlm),
        vlm_model=args.vlm_model,
        vlm_every_s=args.vlm_every_s,
        vlm_timeout_s=args.vlm_timeout_s,
        vlm_window_s=args.vlm_window_s,
        sentinel_threshold=args.sentinel_threshold,
        sentinel_persist_n=args.sentinel_persist_n,
        sentinel_cooldown_s=args.sentinel_cooldown_s,
        sentinel_band_low_hz=args.sentinel_band_low_hz,
        sentinel_band_high_hz=args.sentinel_band_high_hz,
        sentinel_require_cutting_flag=bool(args.sentinel_require_cutting_flag),
        experto_enable=bool(args.experto),
        experto_dir=args.casos_experto_dir,
        experto_cutter_condition=args.cutter_condition,
        experto_every_s=args.experto_every_s,
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
