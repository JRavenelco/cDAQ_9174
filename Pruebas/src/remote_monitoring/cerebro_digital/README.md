# Cerebro Digital — CIDESI TCM

Stack de orquestación para Tool Condition Monitoring en la Jetson Orin NX.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│  JETSON ORIN NX  (192.168.137.164)                          │
│                                                             │
│  ┌─────────────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │ cdaq_udp_receiver│──▶│serve_cerebro │──▶│ mqtt_bridge  │  │
│  │ (existente)      │   │ :8765        │   │ (Python)     │  │
│  │ UDP :12001       │   └──────────────┘   └──────┬───────┘  │
│  └─────────────────┘                              │ MQTT     │
│                                                   ▼          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Docker Compose                                         │ │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐│ │
│  │  │Mosquitto │◀─│  n8n     │─▶│  InfluxDB 2            ││ │
│  │  │ :1883    │  │  :5678   │  │  :8086                 ││ │
│  │  └──────────┘  │ AI Agent │  │  bucket: tcm           ││ │
│  │                │ LangChain│  └────────────────────────┘│ │
│  │                └────┬─────┘                             │ │
│  │                     │ HTTP                              │ │
│  │                     ▼                                   │ │
│  │              Ollama :11434                               │ │
│  │              (llama3.2 / qwen2.5)                        │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
         ▲ UDP :12001                    MQTT :1883 ▲
         │                                          │
┌────────┴──────────────────────────────────────────┴─────────┐
│  WINDOWS  (192.168.137.1)                                    │
│                                                              │
│  cdaq_udp_sender.py  (GUI + NI cDAQ-9174)                   │
│  mqtt_test_windows.py (verificación MQTT)                    │
└──────────────────────────────────────────────────────────────┘
```

## Requisitos en la Jetson

- Docker y Docker Compose instalados
- Ollama corriendo (`ollama serve`)
- Stack existente corriendo: `cdaq_udp_receiver.py` + `serve_cerebro.py`
- Python 3.10+ con `paho-mqtt` (`pip3 install paho-mqtt`)

## Setup rápido

```bash
# 1. Ir al repo y hacer pull
cd ~/cDAQ_9174
git pull

# 2. Levantar los contenedores
cd Pruebas/src/remote_monitoring/cerebro_digital
docker compose up -d

# 3. Verificar que están corriendo
docker compose ps

# 4. Iniciar el bridge (en otra terminal)
#    Requiere que serve_cerebro.py ya esté corriendo en :8765
python3 mqtt_bridge.py --cerebro-url http://localhost:8765 --broker localhost

# 5. Abrir n8n en el browser
#    http://192.168.137.164:5678
#    Usuario: cidesi  /  Password: cerebro2026
```

## Credenciales por defecto

| Servicio   | URL                              | Usuario  | Password       |
|------------|----------------------------------|----------|----------------|
| n8n        | http://jetson-ip:5678            | cidesi   | cerebro2026    |
| InfluxDB   | http://jetson-ip:8086            | cidesi   | cidesi2026     |
| Mosquitto  | mqtt://jetson-ip:1883            | (anónimo)| —              |

**InfluxDB API Token:** `cidesi-cerebro-token-2026`
**InfluxDB Org:** `cidesi`  |  **Bucket:** `tcm`

## Topics MQTT

| Topic                        | Origen       | Contenido                    |
|------------------------------|--------------|------------------------------|
| `cidesi/cnc01/cases`         | mqtt_bridge  | Casos VLM de serve_cerebro   |
| `cidesi/cnc01/status`        | mqtt_bridge  | Status del cerebro           |
| `cidesi/cnc01/features`      | Windows/test | Features en tiempo real      |
| `cidesi/cnc01/heartbeat`     | mqtt_bridge  | Latido del bridge            |

## Configurar n8n (primer uso)

1. Abrir http://jetson-ip:5678 e iniciar sesión
2. Crear un nuevo workflow
3. Agregar nodo **MQTT Trigger**:
   - Broker: `mosquitto` (nombre del contenedor)
   - Port: 1883
   - Topic: `cidesi/cnc01/#`
4. Conectar a un nodo **Code** para parsear el JSON
5. Conectar a un nodo **HTTP Request** para escribir a InfluxDB:
   - URL: `http://influxdb:8086/api/v2/write?org=cidesi&bucket=tcm`
   - Header: `Authorization: Token cidesi-cerebro-token-2026`

## Verificar desde Windows

```powershell
pip install paho-mqtt
cd Pruebas\src\remote_monitoring\cerebro_digital
python mqtt_test_windows.py --broker 192.168.137.164
```

## Detener

```bash
# Detener bridge
Ctrl+C

# Detener contenedores
docker compose down

# Detener y borrar datos
docker compose down -v
```
