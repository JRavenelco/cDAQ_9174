#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mqtt_test_windows.py — Script de prueba para Windows.

Publica datos simulados de sensores a MQTT para verificar que el
stack del cerebro digital (Mosquitto → n8n → InfluxDB) funciona.

También puede suscribirse para ver los mensajes que llegan.

Uso:
    pip install paho-mqtt numpy

    # Publicar datos de prueba:
    python mqtt_test_windows.py --publish --broker 192.168.137.164

    # Suscribirse y ver mensajes:
    python mqtt_test_windows.py --subscribe --broker 192.168.137.164

    # Ambos (publicar y ver):
    python mqtt_test_windows.py --publish --subscribe --broker 192.168.137.164
"""

import argparse
import json
import time
import threading
import math
from datetime import datetime, timezone

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("ERROR: pip install paho-mqtt")
    raise SystemExit(1)

try:
    import numpy as np
    _HAVE_NP = True
except ImportError:
    np = None
    _HAVE_NP = False

# Topics
TOPIC_PREFIX = "cidesi/cnc01"


def generate_fake_features(t: float) -> dict:
    """Genera features simulados de un corte de fresado."""
    rpm = 800
    tooth_freq = rpm * 2 / 60  # 2 dientes, Hz
    # Señal de fuerza simulada con envolvente
    cutting = 1.0 if (t % 10.0) < 7.0 else 0.0  # 7s corte, 3s aire
    force_rms = (2.5 + 0.3 * math.sin(2 * math.pi * 0.05 * t)) * cutting
    accel_rms = (1.2 + 0.15 * math.sin(2 * math.pi * 0.08 * t)) * cutting
    envelope_rms = force_rms * 0.4
    freq_dom = tooth_freq if cutting > 0.5 else 0.0
    thd = 12.5 + 3.0 * math.sin(2 * math.pi * 0.02 * t) if cutting > 0.5 else 0.0

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "spindle_id": "CNC-01",
        "rpm": rpm,
        "cutting_flag": int(cutting),
        "force_rms_V": round(force_rms, 4),
        "accel_rms_g": round(accel_rms, 4),
        "envelope_rms_g": round(envelope_rms, 4),
        "freq_dom_Hz": round(freq_dom, 2),
        "THD_pct": round(thd, 2),
        "F_est_mean_V": round(force_rms * 0.85, 4),
        "t_elapsed_s": round(t, 2),
    }


def publisher_loop(client, interval: float):
    """Publica features simulados periódicamente."""
    t0 = time.time()
    seq = 0
    print(f"\n📡 Publicando features cada {interval}s a {TOPIC_PREFIX}/features ...")
    print("   (Ctrl+C para detener)\n")

    while True:
        t = time.time() - t0
        features = generate_fake_features(t)
        features["seq"] = seq

        payload = json.dumps(features, ensure_ascii=False)
        client.publish(f"{TOPIC_PREFIX}/features", payload, qos=1)

        if seq % 10 == 0:
            status = {
                "ts": features["ts"],
                "total_blocks": seq,
                "uptime_s": round(t, 1),
            }
            client.publish(f"{TOPIC_PREFIX}/status", json.dumps(status), qos=0)

        cutting = "CORTE" if features["cutting_flag"] else "AIRE"
        print(f"  [{seq:04d}] {cutting}  F={features['force_rms_V']:.3f}V  "
              f"A={features['accel_rms_g']:.3f}g  THD={features['THD_pct']:.1f}%")

        seq += 1
        time.sleep(interval)


def on_message(client, userdata, msg):
    """Callback para mensajes recibidos."""
    try:
        data = json.loads(msg.payload.decode("utf-8"))
        ts = data.get("ts", "?")
        print(f"  📨 [{msg.topic}] ts={ts}  keys={list(data.keys())}")
    except Exception:
        print(f"  📨 [{msg.topic}] {msg.payload[:200]}")


def main():
    ap = argparse.ArgumentParser(description="Test MQTT desde Windows")
    ap.add_argument("--broker", default="192.168.137.164",
                    help="IP del broker MQTT (Jetson)")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--publish", action="store_true",
                    help="Publicar datos simulados")
    ap.add_argument("--subscribe", action="store_true",
                    help="Suscribirse a todos los topics")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="Intervalo de publicación (s)")
    args = ap.parse_args()

    if not args.publish and not args.subscribe:
        args.publish = True
        args.subscribe = True

    client = mqtt.Client(client_id="windows-test", protocol=mqtt.MQTTv311)

    def on_connect(c, ud, flags, rc):
        if rc == 0:
            print(f"✅ Conectado a MQTT broker {args.broker}:{args.port}")
            if args.subscribe:
                c.subscribe(f"{TOPIC_PREFIX}/#")
                print(f"👂 Suscrito a {TOPIC_PREFIX}/#")
        else:
            print(f"❌ Error de conexión MQTT rc={rc}")

    client.on_connect = on_connect
    if args.subscribe:
        client.on_message = on_message

    print(f"Conectando a mqtt://{args.broker}:{args.port} ...")
    try:
        client.connect(args.broker, args.port, keepalive=60)
    except Exception as e:
        print(f"❌ No se pudo conectar: {e}")
        print(f"   Verifica que Mosquitto esté corriendo en {args.broker}")
        return

    client.loop_start()

    try:
        if args.publish:
            publisher_loop(client, args.interval)
        else:
            print("\n👂 Escuchando mensajes... (Ctrl+C para salir)\n")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nDetenido.")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
