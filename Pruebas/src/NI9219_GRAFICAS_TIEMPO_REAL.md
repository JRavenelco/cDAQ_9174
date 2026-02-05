# 📊 **GRÁFICAS EN TIEMPO REAL - NI 9219 AÑADIDAS**

## 🎉 **Nuevas Funcionalidades:**

### **📈 Visualización Gráfica Completa del Puente**

Se han añadido **4 gráficas individuales** para cada canal del puente NI 9219 con **actualización en tiempo real**:

```python
# 4 Plot Widgets independientes
self.bridge_plot_widgets = []  # Contenedores de gráficas
self.bridge_plot_curves = []   # Curvas de datos

# Configuración visual
bridge_colors = ['#FF6B35', '#F7931E', '#FFD23F', '#06FFA5']  # Naranjas/Verdes
```

## 🎨 **Especificaciones Visuales:**

### **📊 Layout de Gráficas:**
```
┌─────────────────┬─────────────────┐
│ Bridge ai0 (µV) │ Bridge ai1 (µV) │
│     📈          │     📈          │
└─────────────────┼─────────────────┤
│ Bridge ai2 (µV) │ Bridge ai3 (µV) │
│     📈          │     📈          │
└─────────────────┴─────────────────┘
```

### **⚙️ Configuración Técnica:**
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| **Número de Plots** | 4 | Uno por cada canal del puente |
| **Layout** | 2×2 Grid | 2 columnas, 2 filas |
| **Unidades Y** | µV (microvolts) | Conversión automática V/V → µV |
| **Eje X** | Tiempo (s) | Ventana deslizante configurable |
| **Colores** | Naranjas/Verdes | Diferenciado de fuerza/vibración |
| **Actualización** | ~50ms | Tiempo real sincronizado |

### **🎯 Características Avanzadas:**

#### **1. 📏 Auto-escalado Inteligente:**
```python
# Ajuste automático del rango Y basado en datos reales
if len(bridge_data_microvolts) > 10:
    data_min = np.min(bridge_data_microvolts)
    data_max = np.max(bridge_data_microvolts)
    margin = max(abs(data_max - data_min) * 0.1, 5.0)  # Mínimo 5µV
    self.bridge_plot_widgets[i].setYRange(data_min - margin, data_max + margin)
```

#### **2. 🕐 Eje Temporal Sincronizado:**
```python
current_bridge_time_axis = np.linspace(-self.time_window, 0, len(bridge_data_microvolts))
self.bridge_plot_curves[i].setData(current_bridge_time_axis, bridge_data_microvolts)
```

#### **3. 🎨 Identificación Visual:**
```python
# Colores únicos para cada canal
'#FF6B35' -> Canal ai0 (Naranja vibrante)
'#F7931E' -> Canal ai1 (Naranja clásico)  
'#FFD23F' -> Canal ai2 (Amarillo dorado)
'#06FFA5' -> Canal ai3 (Verde mint)
```

## 📊 **Vista de la Interfaz Actualizada:**

### **Tab "Módulo 9234 (Vibración)" - Sección NI 9219:**
```
┌─────────────────────────────────────────────────────────────┐
│ NI 9219 - Puente/Galga                                     │
│ ┌─────────┬─────────┬─────────┬─────────┐                  │
│ │Bridge(ai0)│Bridge(ai1)│Bridge(ai2)│Bridge(ai3)│        │
│ │Valor:2.66µV│Valor:2.58µV│Valor:2.71µV│Valor:2.62µV│    │
│ │Amp: 0.04µV │Amp: 0.03µV │Amp: 0.02µV │Amp: 0.05µV │    │
│ │RMS: 2.67µV │RMS: 2.59µV │RMS: 2.72µV │RMS: 2.63µV │    │
│ └─────────┴─────────┴─────────┴─────────┘                  │
│ Estado: ✅ Datos activos                                    │
│                                                             │
│ ┌──────────────────┬──────────────────┐ ← NUEVAS GRÁFICAS │
│ │ Bridge ai0 (µV)  │ Bridge ai1 (µV)  │                   │
│ │ ╭─────────────╮   │ ╭─────────────╮   │                   │
│ │ │ ~~~~~~~~~~~ │   │ │ ~~~~~~~~~~~ │   │  📈 Tiempo Real │
│ │ │ ──────────~ │   │ │ ─~~~~~~~~~~ │   │                   │
│ │ ╰─────────────╯   │ ╰─────────────╯   │                   │
│ ├──────────────────┼──────────────────┤                   │
│ │ Bridge ai2 (µV)  │ Bridge ai3 (µV)  │                   │
│ │ ╭─────────────╮   │ ╭─────────────╮   │                   │
│ │ │ ~~~~~~~~~~~ │   │ │ ~~~~~~~~~~~ │   │  📊 Auto-scale  │
│ │ │ ──~──────~~ │   │ │ ~────────── │   │                   │
│ │ ╰─────────────╯   │ ╰─────────────╯   │                   │
│ └──────────────────┴──────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 **Para Ver las Nuevas Gráficas:**

### **1. Ejecutar Interfaz:**
```bash
python interfaz_DAQ_V2.py
```

### **2. Configurar NI 9219:**
- Tab **"Configuración"** → ✅ "Habilitar NI 9219"
- Presionar **"Iniciar"**

### **3. Ver Gráficas en Tiempo Real:**
- Tab **"Módulo 9234 (Vibración)"**
- Scroll hacia abajo para ver sección **"NI 9219 - Puente/Galga"**
- **4 gráficas individuales** actualizándose en tiempo real

### **4. Interacción con Gráficas:**
- ✅ **Zoom**: Rueda del mouse sobre las gráficas
- ✅ **Pan**: Click y arrastra para mover vista
- ✅ **Auto-range**: Doble click para resetear zoom
- ✅ **Grid**: Cuadrícula visible para mejor lectura

## 📈 **Comportamiento de las Gráficas:**

### **🔄 Actualización Continua:**
```
Frecuencia: ~50ms (20 Hz)
Datos: Rolling window de 500ms (configurable)
Buffer: Deque circular auto-gestionado
Sincronía: Perfecto con métricas numéricas
```

### **📏 Rangos Adaptativos:**
```
Inicial: ±50µV (rango conservador)
Auto-ajuste: Basado en datos reales + 10% margen
Mínimo margen: 5µV (para señales muy pequeñas)
Actualización: Cada vez que hay datos nuevos
```

### **🎨 Estilo Visual:**
```
Líneas: 2px de grosor para claridad
Colores: Únicos por canal, alta visibilidad
Grid: Activado por defecto
Labels: "Puente µV" y "Tiempo s"
Títulos: "Bridge ai0", "Bridge ai1", etc.
```

## ✅ **Estado Final Completo:**

| Módulo | Métricas | Gráficas | Configuración | Guardado |
|--------|----------|----------|---------------|----------|
| **NI 9205** | ✅ | ✅ | ✅ | ✅ |
| **NI 9234** | ✅ | ✅ | ✅ | ✅ |
| **NI 9219** | ✅ | 🎉 **SÍ** | ✅ | ✅ |

---

**🎉 ¡NI 9219 AHORA TIENE VISUALIZACIÓN COMPLETA!**

Tu interfaz tiene **métricas numéricas** + **gráficas en tiempo real** + **configuración visual** + **guardado automático** para todos los 3 módulos NI simultáneamente.
