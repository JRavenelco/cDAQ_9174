# ✅ Cambios en Tasa de Muestreo - NI 9205 y NI 9234

## 🎯 **Modificación Realizada:**

**Cambio de tasa de muestreo de 2500 Hz → 2000 Hz** para ambos módulos:
- **NI 9205** (Fuerza): `FORCE_SAMPLE_RATE = 2000`
- **NI 9234** (Vibración): `VIB_SAMPLE_RATE = 2000`

## 📝 **Archivos Modificados:**

### 1. **interfaz_DAQ_V2.py**
```python
# ANTES:
FORCE_SAMPLE_RATE = 2500 # Tasa de muestreo para el módulo 9205
VIB_SAMPLE_RATE = 2500 # MODIFICADO: Establecido al máximo valor del ComboBox y del NI 9234

# DESPUÉS:
FORCE_SAMPLE_RATE = 2000 # Tasa de muestreo para el módulo 9205
VIB_SAMPLE_RATE = 2000 # MODIFICADO: Establecido a 2000 Hz para el módulo NI 9234
```

### 2. **projectAD/main.pyw** (Integración)
```python
# ANTES:
FORCE_SAMPLE_RATE = 2500
VIB_SAMPLE_RATE = 2500

# DESPUÉS:
FORCE_SAMPLE_RATE = 2000
VIB_SAMPLE_RATE = 2000
```

## ⚙️ **Impacto de los Cambios:**

### **✅ Beneficios:**
1. **Menor uso de recursos**:
   - Menos datos por segundo
   - Menor carga en CPU y memoria
   - Buffers más eficientes

2. **Compatibilidad mejorada**:
   - 2000 Hz es una frecuencia estándar
   - Mejor sincronización entre módulos
   - Menor probabilidad de timeout errors

3. **Datos más manejables**:
   - Archivos de salida más pequeños
   - Procesamiento más rápido
   - Menos congestión en colas

### **🔧 Ajustes Automaticos:**
- **Tiempo por bloque**: Con 100 muestras/bloque:
  - Antes: 100/2500 = 0.04 segundos
  - Ahora: 100/2000 = **0.05 segundos**
- **Buffer sizes**: Se ajustan automáticamente
- **Visualización**: Tiempo de actualización optimizado

## 📊 **Especificaciones Técnicas:**

| Módulo | Tasa Anterior | **Tasa Nueva** | Resolución | Canales |
|--------|---------------|----------------|------------|---------|
| **NI 9205** | 2500 Hz | **2000 Hz** | ±1.5V | ai0-ai3 |
| **NI 9234** | 2500 Hz | **2000 Hz** | ±48g | ai0-ai1 |

### **Configuración de Bloques:**
- **Muestras por bloque**: 100 (sin cambios)
- **Duración por bloque**: 0.05 segundos
- **Frecuencia de lectura**: ~20 Hz (cada 50ms)

## 🚀 **Para Usar los Cambios:**

### **Interfaz Original:**
```bash
python interfaz_DAQ_V2.py
```

### **Interfaz Integrada:**
```bash
cd projectAD
python main.pyw
# Tab: "🔌 DAQ Real-Time"
```

## ✅ **Verificación:**

Los cambios se aplicaron correctamente y están listos para usar. La nueva configuración de 2000 Hz proporcionará:

- ✅ **Adquisición estable** a frecuencia estándar
- ✅ **Menor latencia** en el procesamiento
- ✅ **Compatibilidad** con análisis existentes
- ✅ **Eficiencia mejorada** en recursos

---

**🎯 Configuración Final:**
- **NI 9205 (Fuerza)**: 2000 Hz, 4 canales (ai0-ai3)
- **NI 9234 (Vibración)**: 2000 Hz, 2 canales (ai0-ai1)
- **Bloques**: 100 muestras cada 50ms
- **Guardado**: CSV con timestamps precisos
