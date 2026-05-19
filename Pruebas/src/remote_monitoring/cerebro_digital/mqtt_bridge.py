#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mqtt_bridge.py — Puente entre serve_cerebro.py y MQTT para n8n.

Corre en la Jetson JUNTO con el stack existente (cdaq_udp_receiver + serve_cerebro).
Cada N segundos consulta serve_cerebro.py por casos nuevos y los publica a MQTT
para que n8n los consuma via MQTT Trigger.

También publica features en tiempo real si se conecta al UDP receiver directamente.

Uso:
    python3 mqtt_bridge.py
    python3 mqtt_bridge.py --cerebro-url http://localhost:8765 --broker localhost
    python3 mqtt_bridge.py --poll-interval 5
"""

import argparse
import json
import time
import threading
import logging
from datetime import datetime, timezone

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("ERROR: pip install paho-mqtt")
    raise SystemExit(1)

try:
    import urllib.request
    import urllib.error
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mqtt_bridge")

# ── Topics MQTT ─────────────────────────────────────────────────
TOPIC_CASES      = "cidesi/cnc01/cases"        # casos VLM del cerebro
TOPIC_STATUS     = "cidesi/cnc01/status"        # status del cerebro
TOPIC_FEATURES   = "cidesi/cnc01/features"      # features en tiempo real
TOPIC_HEARTBEAT  = "cidesi/cnc01/heartbeat"     # latido del bridge


class CerebroBridge:
    """Puente serve_cerebro → MQTT."""

    def __init__(self, cerebro_url: str, broker: str, port: int,
                 poll_interval: float):
        self.cerebro_url = cerebro_url.rstrip("/")
        self.poll_interval = poll_interval
        self.last_since = None
        self._running = False

        # MQTT client
        self.client = mqtt.Client(
            client_id="cerebro-bridge",
            protocol=mqtt.MQTTv311,
        )
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.broker = broker
        self.port = port

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            log.info(f"Conectado a MQTT broker {self.broker}:{self.port}")
        else:
            log.error(f"Error MQTT rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        log.warning(f"Desconectado de MQTT (rc={rc}), reconectando...")

    def _http_get(self, path: str) -> dict | None:
        """GET a serve_cerebro.py y devuelve JSON."""
        url = f"{self.cerebro_url}{path}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            log.debug(f"HTTP GET {url} falló: {e}")
            return None

    def _poll_cerebro(self):
        """Consulta casos nuevos y los publica a MQTT."""
        path = "/api/casos"
        if self.last_since:
            path += f"?since={self.last_since}"

        data = self._http_get(path)
        if not data:
            return

        casos = data.get("casos", [])
        if not casos:
            return

        log.info(f"Nuevos casos del cerebro: {len(casos)}")

        for caso in casos:
            payload = json.dumps(caso, ensure_ascii=False)
            self.client.publish(TOPIC_CASES, payload, qos=1)
            # Actualizar since al más reciente
            mtime = caso.get("_mtime")
            if mtime:
                self.last_since = mtime

    def _poll_status(self):
        """Publica status del cerebro a MQTT."""
        data = self._http_get("/api/status")
        if data:
            self.client.publish(TOPIC_STATUS, json.dumps(data), qos=0)

    def _heartbeat(self):
        """Publica heartbeat periódico."""
        payload = json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "bridge": "cerebro-mqtt",
            "cerebro_url": self.cerebro_url,
        })
        self.client.publish(TOPIC_HEARTBEAT, payload, qos=0)

    def run(self):
        """Loop principal del bridge."""
        log.info(f"Iniciando bridge: {self.cerebro_url} → mqtt://{self.broker}:{self.port}")
        log.info(f"  Poll interval: {self.poll_interval}s")
        log.info(f"  Topics: {TOPIC_CASES}, {TOPIC_STATUS}, {TOPIC_HEARTBEAT}")

        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_start()
        self._running = True

        cycle = 0
        try:
            while self._running:
                self._heartbeat()
                self._poll_cerebro()

                # Status cada 5 ciclos
                if cycle % 5 == 0:
                    self._poll_status()

                cycle += 1
                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            log.info("Bridge detenido por usuario.")
        finally:
            self._running = False
            self.client.loop_stop()
            self.client.disconnect()


def main():
    ap = argparse.ArgumentParser(description="Puente serve_cerebro → MQTT")
    ap.add_argument("--cerebro-url", default="http://localhost:8765",
                    help="URL de serve_cerebro.py")
    ap.add_argument("--broker", default="localhost",
                    help="MQTT broker host")
    ap.add_argument("--mqtt-port", type=int, default=1883,
                    help="MQTT broker port")
    ap.add_argument("--poll-interval", type=float, default=3.0,
                    help="Segundos entre polls al cerebro")
    args = ap.parse_args()

    bridge = CerebroBridge(
        cerebro_url=args.cerebro_url,
        broker=args.broker,
        port=args.mqtt_port,
        poll_interval=args.poll_interval,
    )
    bridge.run()


if __name__ == "__main__":
    main()
