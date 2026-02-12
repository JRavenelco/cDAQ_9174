# Remote Monitoring: cDAQ ↔ Jetson + RPi Hailo (UDP)

Sistema de monitoreo remoto con dos flujos de datos:

1. **Datos de sensores**: NI cDAQ-9174 (Windows) → Jetson (inferencia GPU) → Windows (visualización)
2. **Video con detección**: RPi + Hailo (YOLOv8n) → Windows (ffplay, baja latencia)

## Arquitectura

```
┌──────────────────────────┐        UDP         ┌──────────────────────────┐
│  WINDOWS  (Sender + GUI) │ ── DATA:12001 ──→  │  JETSON  (Daemon ligero) │
│                          │                     │                          │
│  NI 9205 (ai0) → fuerza │ ←─ INFER:12002 ──  │  Recibe datos            │
│  NI 9234 (ai0) → accel  │                     │  Guarda CSV              │
│                          │ ←─ CTRL:12000 ──→  │  Corre inferencia GPU    │
│  Monitoreo + gráficas    │                     │  Devuelve resultados     │
│  Muestra inferencia      │                     │                          │
│  cdaq_udp_sender.py      │                     │  cdaq_udp_receiver.py    │
└──────────────────────────┘                     └──────────────────────────┘
        ▲
        │  UDP :5000 (H.264)
        │
┌──────────────────────────┐
│  RPi + Hailo 8L          │
│                          │
│  Cámara → YOLOv8n (HEF)  │
│  Detección de objetos    │
│  H.264 encode → UDP      │
│                          │
│  object_detection (C++)  │
└──────────────────────────┘
```

**Windows** maneja toda la visualización y monitoreo (sensores + video).
**Jetson** es un daemon sin GUI: recibe, guarda, infiere y responde.
**RPi** corre detección de objetos con Hailo y transmite video anotado por UDP.

## Archivos

| Archivo | Dónde corre | Descripción |
|---------|------------|-------------|
| `cdaq_udp_protocol.py` | Ambos | Protocolo compartido (pack/unpack datos, control, inferencia) |
| `cdaq_udp_sender.py` | Windows | Adquisición NI + GUI + envío UDP + recibe inferencia |
| `cdaq_udp_receiver.py` | Jetson | Daemon: recibe UDP, CSV, inferencia GPU, devuelve resultados |
| `cdaq_inference_engine.py` | Jetson | Motores de inferencia: Envolvente, Bouc-Wen/KAN-PINN |
| `test_loopback.py` | Cualquiera | Tests de validación del protocolo |
| `play_rpi_stream.bat` | Windows | Lanza ffplay para recibir video del RPi (baja latencia) |
| `stream_rpi.sdp` | Windows | Archivo SDP para modo RTP (opcional) |

## Instalación

### Windows (Sender + GUI)
```bash
pip install numpy PyQt5 nidaqmx
```
Copiar `cdaq_udp_protocol.py` y `cdaq_udp_sender.py` al PC con Windows.

### Jetson (Receiver daemon)
```bash
pip install numpy
# Para inferencia (según modelo):
# pip install jax jaxlib  # o hailo-rt, etc.
```

## Uso

### 1. Iniciar Receiver en Jetson
```bash
# Daemon básico (recibe, guarda CSV, corre inferencia placeholder):
python3 cdaq_udp_receiver.py --sender-ip 192.168.137.1

# Con sync automático y auto-start:
python3 cdaq_udp_receiver.py --sender-ip 192.168.137.1 --sync --auto-start

# Solo recibir y guardar (sin inferencia):
python3 cdaq_udp_receiver.py --sender-ip 192.168.137.1 --no-infer

# Sin CSV (solo inferencia):
python3 cdaq_udp_receiver.py --sender-ip 192.168.137.1 --no-csv
```

Comandos interactivos en la terminal del Jetson:
- `start` – enviar START al sender Windows
- `stop` – enviar STOP al sender
- `sync` – sincronizar relojes
- `stats` – mostrar estadísticas
- `quit` – salir

### 2. Iniciar Sender en Windows
```bash
# Con GUI (muestra datos + resultados de inferencia del Jetson):
python cdaq_udp_sender.py --jetson-ip 192.168.137.164

# Simulación (sin hardware NI):
python cdaq_udp_sender.py --jetson-ip 192.168.137.164 --simulate

# Headless:
python cdaq_udp_sender.py --jetson-ip 192.168.137.164 --headless --auto-start
```

## Protocolo UDP

| Puerto | Dirección | Contenido |
|--------|-----------|-----------|
| 12001 (DATA) | Windows → Jetson | Bloques de fuerza + aceleración (float32 intercalado) |
| 12000 (CTRL) | Bidireccional | SYNC_REQ/RSP, START, STOP, CFG |
| 12002 (INFER) | Jetson → Windows | Resultados de inferencia GPU |

### Inferencia pluggable
La clase `InferenceEngine` en el receiver es un placeholder. Para usar tu
modelo real, hereda y sobreescribe `predict()`:

```python
class PINNInference(InferenceEngine):
    def __init__(self):
        super().__init__()
        import jax
        self.model = ...  # cargar modelo

    def predict(self, force, accel, fs, seq):
        x = jax.numpy.stack([force, accel])
        return self.model(x)  # array 1-D float32
```

## Parámetros de hardware

| Módulo | Canal | Descripción | Sample Rate |
|--------|-------|-------------|-------------|
| NI 9205 (cDAQ1Mod1) | ai0 | Fuerza / Voltaje analógico | 2500 Hz |
| NI 9234 (cDAQ1Mod2) | ai0 | Acelerómetro IEPE (PCB 352C33) | 2500 Hz |

---

## RPi + Hailo: Video con Detección de Objetos (UDP)

Stream de video anotado con YOLOv8n corriendo en el acelerador Hailo 8L del
Raspberry Pi. Se envía H.264 por UDP al puerto 5000 de Windows.

### Arquitectura del stream de video

```
RPi (192.168.137.xxx)                    Windows (192.168.137.1)
┌─────────────────────┐                  ┌─────────────────────┐
│  Cámara RPi         │                  │  ffplay             │
│       ↓             │   UDP :5000      │  (baja latencia)    │
│  Hailo 8L (YOLOv8n) │ ──── H.264 ───→ │  ~50-100ms latencia │
│       ↓             │                  │                     │
│  H.264 encode       │                  │  play_rpi_stream.bat│
│  object_detection   │                  └─────────────────────┘
└─────────────────────┘
```

### RPi: Compilar y ejecutar

```bash
# Compilar la app de detección
cd ~/hailo-apps/hailo_apps/cpp/object_detection
./build.sh

# Ejecutar con stream UDP hacia Windows
./build/x86_64/object_detection \
  --net /usr/local/hailo/resources/hef_files_costum/yolov8n.hef \
  --input rpi \
  --labels-json /usr/local/hailo/resources/hef_files_costum/detect.json \
  --udp --udp-host 192.168.137.1 --udp-port 5000 --udp-bitrate 4000 \
  --headless
```

### Windows: Recibir video con ffplay

#### Opción 1: Script automático (recomendado)
```cmd
play_rpi_stream.bat
```
El script busca `ffplay.exe` automáticamente en la subcarpeta `ffmpeg-*/bin/`.

#### Opción 2: ffplay manual (UDP directo)
```cmd
ffplay -fflags nobuffer -flags low_delay -framedrop -probesize 32 -analyzeduration 0 udp://@:5000
```

#### Opción 3: ffplay con SDP (modo RTP)
```cmd
ffplay -fflags nobuffer -flags low_delay -framedrop -probesize 32 -analyzeduration 0 -protocol_whitelist file,rtp,udp -i stream_rpi.sdp
```

### Instalación de ffmpeg en Windows

1. Descargar ffmpeg essentials de https://www.gyan.dev/ffmpeg/builds/
2. Extraer el `.7z` en esta carpeta (`remote_monitoring/`)
3. Debe quedar: `ffmpeg-XXXX/bin/ffplay.exe`
4. El script `play_rpi_stream.bat` lo detecta automáticamente

### Reducir latencia en VLC (alternativa)

Si prefieres VLC en vez de ffplay:
1. Herramientas → Preferencias → **Todo** (abajo izquierda)
2. Entrada/Codecs → **Caché de red** → cambiar de 1000 ms a **100 ms**

> **Nota**: ffplay tiene buffers mínimos comparado con VLC, la latencia será
> ~50-100ms vs los segundos que se ven con MJPEG/TCP o VLC con cache default.
