#!/usr/bin/env python3
"""
serve_cerebro.py — Servidor HTTP ligero que expone los casos del VLM experto.

Corre en la Jetson. La app 3D en Windows se conecta a este endpoint para ver
las "neuronas de memoria" en tiempo real.

Endpoints:
  GET /api/casos           → lista de todos los casos guardados
  GET /api/casos?since=TS  → solo casos nuevos desde timestamp ISO
  GET /api/status          → stats: total casos, última actualización

Uso:
  python3 serve_cerebro.py --port 8765 --dir ./casos_experto
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs


class CerebroHandler(BaseHTTPRequestHandler):
    casos_dir: Path = Path("casos_experto")

    def log_message(self, fmt, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}")

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self._send_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _load_cases(self, since: str | None = None) -> list[dict]:
        casos = []
        if not self.casos_dir.exists():
            return casos

        since_dt = None
        if since:
            try:
                since_dt = datetime.fromisoformat(since)
            except ValueError:
                pass

        for fpath in sorted(self.casos_dir.glob("*.json")):
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime, tz=timezone.utc)
            if since_dt and mtime <= since_dt:
                continue
            try:
                caso = json.loads(fpath.read_text(encoding="utf-8"))
                caso["_file"] = fpath.name
                caso["_mtime"] = mtime.isoformat()
                casos.append(caso)
            except Exception as e:
                print(f"  skip {fpath.name}: {e}")

        return casos

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/api/casos":
            since = params.get("since", [None])[0]
            casos = self._load_cases(since=since)
            self._json_response({"casos": casos, "total": len(casos)})

        elif parsed.path == "/api/status":
            casos = self._load_cases()
            last_ts = max(
                (c.get("_mtime", "") for c in casos), default=None
            )
            self._json_response({
                "total_casos": len(casos),
                "ultimo_caso": last_ts,
                "directorio": str(self.casos_dir.resolve()),
            })

        elif parsed.path == "/health":
            self._json_response({"ok": True, "ts": datetime.now(timezone.utc).isoformat()})

        else:
            self._json_response({"error": "not found"}, status=404)


def main():
    ap = argparse.ArgumentParser(description="Servidor cerebro — expone casos VLM")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dir", type=Path, default=Path("casos_experto"))
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()

    args.dir = args.dir.resolve()
    args.dir.mkdir(parents=True, exist_ok=True)
    CerebroHandler.casos_dir = args.dir

    server = HTTPServer((args.bind, args.port), CerebroHandler)
    print(f"🧠 Cerebro VLM sirviendo en http://{args.bind}:{args.port}")
    print(f"   Directorio: {args.dir}")
    print(f"   Endpoints:")
    print(f"     GET /api/casos")
    print(f"     GET /api/casos?since=2026-05-09T19:50:00")
    print(f"     GET /api/status")
    print(f"     GET /health")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCerebro detenido.")
        server.shutdown()


if __name__ == "__main__":
    main()
