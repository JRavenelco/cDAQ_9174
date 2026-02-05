# ✅ **INTEGRACIÓN COMPLETADA: NI 9219 (Puente/Galga) en interfaz_DAQ_V2.py**

## 🎯 **Objetivo Alcanzado:**

Se integró exitosamente el módulo **NI 9219** para adquisición de datos de **puente completo/galga extensométrica** en la interfaz principal, basándose en el ejemplo `test_9219.py` proporcionado.

## ✅ **Componentes Integrados:**

### **1. Configuración del Hardware (Líneas 70-81)**
```python
# NI 9219 Configuration
BRIDGE_DISPOSITIVO = "cDAQ1Mod3"
BRIDGE_CANALES = ["ai0", "ai1", "ai2", "ai3"]  # 4 canales de puente
BRIDGE_SAMPLE_RATE = 1000  # Hz - Más lenta para puente (típico para celdas de carga)
BRIDGE_MUESTRAS_POR_BLOQUE = 10  # Menos muestras por bloque para estabilidad
BRIDGE_MIN_VV = -0.01    # ±0.01 V/V rango típico
BRIDGE_MAX_VV = 0.01
BRIDGE_EXCITATION_VOLTAGE = 2.5  # V - Voltaje de excitación interno
BRIDGE_NOMINAL_RESISTANCE = 350.0  # Ω - Resistencia nominal de galga estándar
BRIDGE_TO_MICROVOLTS = 2.5e6  # Factor de conversión V/V -> µV
```

### **2. Cola de Comunicación (Línea 90)**
```python
bridge_queue = queue.Queue(maxsize=10)  # Cola para NI 9219 (puente/galga)
```

### **3. Clase de Adquisición (Líneas 447-533)**
```python
class BridgeAcquisitionThread(threading.Thread):
    """Hilo de adquisición para NI 9219 (puente completo/galga)"""
```

**Características:**
- ✅ **Configuración automática de puente completo** con `add_ai_bridge_chan()`
- ✅ **Excitación interna de 2.5V**
- ✅ **Resistencia nominal de 350Ω**
- ✅ **Unidades en V/V** (Volts per Volt)
- ✅ **Buffer interno de 5000 muestras** para estabilidad
- ✅ **Manejo robusto de timeouts**

### **4. Variables y Buffers (Líneas 578, 587, 610)**
```python
self.all_bridge_data = [[] for _ in range(len(BRIDGE_CANALES))]  # Almacenamiento a largo plazo
self.bridge_buffer = [deque(maxlen=bridge_buffer_size) for _ in range(len(BRIDGE_CANALES))]  # Buffer circular
self.bridge_thread = None  # Hilo del NI 9219
```

### **5. Procesamiento en Tiempo Real (Líneas 1581-1632)**
```python
# Update Bridge Data (NI 9219 - Puente/Galga)
while not bridge_queue.empty():
    # Procesar datos de puente
    # Convertir a microvolts: V/V * 2.5e6 = µV
    # Calcular métricas: valor actual, RMS, amplitud
```

**Funcionalidades:**
- ✅ **Conversión a microvolts** para visualización intuitiva
- ✅ **Cálculo de métricas** (valor actual, RMS, amplitud)
- ✅ **Almacenamiento en buffers** circulares y acumulativos
- ✅ **Actualización en consola** (preparado para UI futura)

### **6. Gestión de Hilos**

**Inicio (Líneas 1297-1308):**
```python
self.bridge_thread = BridgeAcquisitionThread(
    dispositivo=BRIDGE_DISPOSITIVO,
    canales=BRIDGE_CANALES, 
    sample_rate=BRIDGE_SAMPLE_RATE,
    muestras_por_bloque=BRIDGE_MUESTRAS_POR_BLOQUE
)
self.bridge_thread.start()
```

**Detención (Líneas 1184, 1193, 2153, 2170):**
- ✅ **Stop limpio** en método principal
- ✅ **Join con timeout** de 1.5-2.0 segundos
- ✅ **Cleanup en closeEvent**
- ✅ **Mensajes de debug** para monitoreo

### **7. Limpieza de Datos (Líneas 1246, 1251, 1263-1278)**
```python
# Clear bridge data on restart
self.all_bridge_data = [[] for _ in range(len(BRIDGE_CANALES))]
self.bridge_buffer = [deque(maxlen=bridge_buffer_size) for _ in range(len(BRIDGE_CANALES))]

# Clear bridge queue before acquisition
while not bridge_queue.empty():
    try: bridge_queue.get_nowait()
    except: break
```

### **8. Guardado Automático (Líneas 2101-2142)**
```python
# Save Bridge Data (NI 9219)
filename_bridge = f"{self.current_session_base_filename}_puente.csv"
header_bridge_list = ["Tiempo(s)"] + [f"Bridge_{c}(V/V)" for c in BRIDGE_CANALES]
```

**Características del guardado:**
- ✅ **Archivo CSV independiente**: `*_puente.csv`
- ✅ **Headers descriptivos**: `Tiempo(s), Bridge_ai0(V/V), Bridge_ai1(V/V), ...`
- ✅ **Vector de tiempo sincronizado** con sample rate del puente
- ✅ **Manejo de errores** con mensajes específicos
- ✅ **Integración en mensaje de éxito/fallo** global

## 📊 **Especificaciones Técnicas:**

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| **Sample Rate** | 1000 Hz | Típico para celdas de carga |
| **Canales** | ai0, ai1, ai2, ai3 | 4 canales de puente |
| **Rango** | ±0.01 V/V | Rango estándar puente completo |
| **Excitación** | 2.5V interna | Voltaje de excitación |
| **Resistencia** | 350Ω | Resistencia nominal galga |
| **Conversión** | 2.5e6 µV | Factor V/V → µV |
| **Buffer** | 5000 muestras | Buffer interno DAQ |

## 🔧 **Compatibilidad con Ejemplo Original:**

La integración mantiene **100% compatibilidad** con tu código `test_9219.py`:

| Aspecto | test_9219.py | interfaz_DAQ_V2.py | ✅ |
|---------|--------------|-------------------|---|
| **Configuración Bridge** | `add_ai_bridge_chan()` | `add_ai_bridge_chan()` | ✅ |
| **Excitación** | 2.5V interna | 2.5V interna | ✅ |
| **Unidades** | `VOLTS_PER_VOLT` | `VOLTS_PER_VOLT` | ✅ |
| **Resistencia** | 350.0Ω | 350.0Ω | ✅ |
| **Conversión µV** | `* 2.5e6` | `* BRIDGE_TO_MICROVOLTS` | ✅ |
| **Buffer** | 5000 | 5000 | ✅ |
| **Manejo timeout** | ✅ | ✅ | ✅ |

## 🚀 **Para Usar la Nueva Funcionalidad:**

### **1. Ejecutar Interfaz:**
```bash
python interfaz_DAQ_V2.py
```

### **2. Verificar en Consola:**
```
✅ Bridge thread iniciado: cDAQ1Mod3
Bridge Ch0: 245.67 µV | RMS: 12.34 µV | Amp: 89.12 µV
```

### **3. Archivos Guardados:**
```
datos_automaticos/
├── sesion_20251003_143000_fuerza.csv      # NI 9205 
├── sesion_20251003_143000_vibracion.csv   # NI 9234
└── sesion_20251003_143000_puente.csv      # NI 9219 ✨ NUEVO
```

## ⚙️ **Configuración de tu Galga:**

Según tu configuración de **puente completo**:

```python
# En test_9219.py tienes:
voltage_excit_val=2.5,                    # ✅ Mantenido
nominal_bridge_resistance=350.0           # ✅ Mantenido  
units=BridgeUnits.VOLTS_PER_VOLT         # ✅ Mantenido
```

**La integración es directamente compatible** con tu setup actual.

## 🔍 **Próximos Pasos Opcionales:**

1. **UI Visual**: Añadir plots para el puente (como fuerza y vibración)
2. **Calibración**: Integrar factores de calibración específicos de tu galga
3. **Conversión física**: Añadir conversión V/V → Strain (µε) o Force (N)
4. **Configuración dinámica**: UI para cambiar excitación, resistencia, etc.

## ✅ **Resumen de Estado:**

| Módulo | Estado | Sample Rate | Canales | Función |
|--------|--------|-------------|---------|---------|
| **NI 9205** | ✅ Operativo | 2000 Hz | ai0-ai3 | Fuerza |
| **NI 9234** | ✅ Operativo | 2000 Hz | ai0-ai1 | Vibración |
| **NI 9219** | 🎉 **NUEVO** | 1000 Hz | ai0-ai3 | **Puente/Galga** |

---

**🎉 ¡Integración del NI 9219 completada exitosamente!** 

Tu interfaz ahora soporta **3 módulos NI simultáneamente** con adquisición en tiempo real, visualización de métricas, y guardado automático de datos.
