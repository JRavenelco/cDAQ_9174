# Remote Monitoring: cDAQ ↔ Jetson (UDP)

Sistema de monitoreo remoto para adquirir datos del NI cDAQ-9174 en Windows,
transmitirlos al Jetson para inferencia en GPU (JAX/PINN/KAN/Hailo), y
devolver los resultados a Windows para visualización.

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
```

**Windows** maneja toda la visualización y monitoreo.
**Jetson** es un daemon sin GUI: recibe, guarda, infiere y responde.

## Archivos

| Archivo | Dónde corre | Descripción |
|---------|------------|-------------|
| `cdaq_udp_protocol.py` | Ambos | Protocolo compartido (pack/unpack datos, control, inferencia) |
| `cdaq_udp_sender.py` | Windows | Adquisición NI + GUI + envío UDP + recibe inferencia |
| `cdaq_udp_receiver.py` | Jetson | Daemon: recibe UDP, CSV, inferencia GPU, devuelve resultados |
| `test_loopback.py` | Cualquiera | Tests de validación del protocolo |

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
