# ✅ Configuración Corregida - Acelerómetro PCB 352C33

## 🔍 **Análisis del Certificado de Calibración:**

**Según el certificado proporcionado:**
- **Modelo**: PCB 352C33 (IEPE Accelerometer)
- **Número de Serie**: LW217003
- **Sensibilidad**: **98.3 mV/g** @ 100 Hz
- **Bias de Salida**: 10.8 VDC
- **Temperatura**: 71.9°F (22°C)
- **Humedad**: 56.5%
- **Calibración**: 9/3/2016

## ⚠️ **Problema Detectado y Corregido:**

### **ANTES (Incorrecto):**
```python
sensitivity=102.4  # ❌ Valor genérico
ACC_CONVERSION = 1000.0 / 102.4  # ❌ = 9.765625 g/V
```

### **DESPUÉS (Corregido):**
```python
sensitivity=98.3   # ✅ Valor real del certificado
ACC_CONVERSION = 1000.0 / 98.3   # ✅ = 10.173 g/V
```

## 📊 **Impacto de la Corrección:**

| Parámetro | Valor Anterior | **Valor Correcto** | Diferencia |
|-----------|----------------|--------------------|------------|
| **Sensibilidad** | 102.4 mV/g | **98.3 mV/g** | -4% |
| **Conversión** | 9.766 g/V | **10.173 g/V** | +4.2% |
| **Rango efectivo** | ±468 g | **±488 g** | +4.2% |

## 🔧 **Archivos Actualizados:**

### 1. **`interfaz_DAQ_V2.py`**
```python
# Línea 43: Constante de conversión
ACC_CONVERSION = 1000.0 / 98.3  # Según certificado PCB 352C33

# Línea 327: Configuración del canal
sensitivity=98.3, # mV/g - Según certificado PCB 352C33
```

### 2. **`projectAD/main.pyw`** (Integración)
```python
sensitivity=98.3,  # Según certificado PCB 352C33
```

## ✅ **Validación de Configuración Completa:**

### **📈 Tasas de Muestreo:**
- ✅ **NI 9205 (Fuerza)**: 2000 Hz
- ✅ **NI 9234 (Vibración)**: 2000 Hz

### **🔌 Configuración de Canales:**
```python
# NI 9205 (cDAQ1Mod1) - Fuerza
DISPOSITIVO = "cDAQ1Mod1"
FORCE_CANALES = ["ai0", "ai1", "ai2", "ai3"]
FORCE_SAMPLE_RATE = 2000

# NI 9234 (cDAQ1Mod2) - Vibración
VIB_DISPOSITIVO = "cDAQ1Mod2" 
VIB_CANALES = ["ai0", "ai1"]
VIB_SAMPLE_RATE = 2000
```

### **⚡ Configuración IEPE:**
```python
# Excitación constante de corriente
current_excit_source = ExcitationSource.INTERNAL
current_excit_val = 0.004  # 4mA (estándar IEPE)

# Rango de aceleración
min_val = -48.0  # g
max_val = 48.0   # g

# Sensibilidad calibrada
sensitivity = 98.3  # mV/g (certificado)
```

## 🎯 **Precisión Mejorada:**

Con la sensibilidad correcta del certificado:
- ✅ **Lecturas más precisas** (+4.2% corrección)
- ✅ **Calibración trazable** (NIST certificado)
- ✅ **Conversión física correcta** (mV → g)
- ✅ **Cumplimiento estándares** ISO 16063-21

## 🚀 **Para Usar:**

```bash
# Interfaz corregida
python interfaz_DAQ_V2.py

# O interfaz integrada
cd projectAD
python main.pyw
```

## 📝 **Notas Técnicas:**

1. **El certificado es de 2016** - Considera recalibración si requieres máxima precisión
2. **Sensibilidad a 100 Hz** - Para otras frecuencias, revisar curva de respuesta
3. **Temperatura ambiente** - La sensibilidad puede variar ±5% con temperatura
4. **Bias 10.8V** - Confirma que el módulo NI 9234 maneja este nivel

---

**🎉 ¡Configuración optimizada según certificado de calibración real!**
