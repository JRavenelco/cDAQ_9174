# 🔄 **REORGANIZACIÓN DE TABS + AJUSTES DE RANGO**

## ✅ **CAMBIOS IMPLEMENTADOS:**

### **📊 1. Rangos Ajustados Según Imágenes:**

#### **🏃 Acelerómetro (NI 9234):**
```python
# ANTES: ±48g (insuficiente)
ACCEL_MIN_G = -48.0
ACCEL_MAX_G = 48.0

# DESPUÉS: ±200g (según imagen mostrada)
ACCEL_MIN_G = -200.0  ✅
ACCEL_MAX_G = 200.0   ✅
```

#### **⚡ Módulo 9205 (Fuerza):**
```python
# Ya configurado correctamente: ±1.5V
FORCE_MIN_VOLTAGE = -1.5  ✅ (coincide con imagen)
FORCE_MAX_VOLTAGE = 1.5   ✅ (coincide con imagen)
```

### **🗂️ 2. Reorganización Completa de Tabs:**

#### **ANTES (3 tabs):**
```
📂 Tab 1: "Módulo 9205 (Voltaje)" → 4 gráficas fuerza
📂 Tab 2: "Módulo 9234 (Vibración)" → 2 vibración + 4 puente
📂 Tab 3: "Configuración"
📂 Tab 4: "Modelado Bouc-Wen"
```

#### **DESPUÉS (5 tabs reorganizados):**
```
📂 Tab 1: "Módulo 9205 (Voltaje)" → 4 gráficas fuerza (sin cambios)
📂 Tab 2: "Vibración + Fuerza (9234 + 9205)" → ✅ 6 gráficas integradas
📂 Tab 3: "Módulo 9219 (Puente/Galga)" → ✅ 4 gráficas puente separadas  
📂 Tab 4: "Configuración"
📂 Tab 5: "Modelado Bouc-Wen"
```

### **🎯 3. Layout 3×2 Optimizado (Tab 2):**

```
Tab "Vibración + Fuerza (9234 + 9205)":
┌─────────────┬─────────────┬─────────────┐
│ Vibración   │ Vibración   │ Fuerza      │
│    ai0      │    ai1      │   ai0       │
│   (±200g)   │   (±200g)   │  (±1.5V)    │
├─────────────┼─────────────┼─────────────┤
│ Fuerza      │ Fuerza      │ Fuerza      │
│   ai1       │   ai2       │   ai3       │
│  (±1.5V)    │  (±1.5V)    │  (±1.5V)    │
└─────────────┴─────────────┴─────────────┘
```

### **🔧 4. Layout 2×2 para Puente (Tab 3):**

```
Tab "Módulo 9219 (Puente/Galga)":
┌─────────────┬─────────────┐
│ Bridge ai0  │ Bridge ai1  │
│   (µV)      │   (µV)      │
├─────────────┼─────────────┤
│ Bridge ai2  │ Bridge ai3  │
│   (µV)      │   (µV)      │
└─────────────┴─────────────┘
```

## 🎨 **Colores Diferenciados:**

### **🌈 Por Módulo:**
```python
# Vibración (Dorado/Cian)
vib_colors = ['#FFD700', '#00FFFF']

# Fuerza (Rojos/Morados) - NUEVOS
force_colors = ['#FF5722', '#E91E63', '#9C27B0', '#673AB7']

# Puente (Naranjas/Verdes)
bridge_colors = ['#FF6B35', '#F7931E', '#FFD23F', '#06FFA5']
```

## 📊 **Especificaciones Técnicas:**

### **📐 Configuración de Gráficas:**

| Módulo | Canales | Rango Y | Ubicación | Layout |
|--------|---------|---------|-----------|--------|
| **NI 9234** | ai0, ai1 | ±200g | Tab 2, Fila 0 (0-1) | 3×2 |
| **NI 9205** | ai0-ai3 | ±1.5V | Tab 2, Fila 0(2) + Fila 1(0-2) | 3×2 |
| **NI 9219** | ai0-ai3 | ±50µV | Tab 3, Grid completo | 2×2 |

### **🎯 Ventajas de la Reorganización:**

✅ **Separación lógica**: Cada módulo tiene su espacio dedicado  
✅ **Visualización optimizada**: 6 ventanas iguales para comparación directa  
✅ **Rangos correctos**: Basados en datos reales mostrados  
✅ **Colores diferenciados**: Fácil identificación visual  
✅ **Escalabilidad**: Fácil añadir más módulos  

## 🚀 **Para Probar los Cambios:**

1. **Ejecutar**: `python interfaz_DAQ_V2.py`
2. **Verificar rangos**:
   - Acelerómetro: ±200g (amplitud apropiada)
   - Fuerza: ±1.5V (según imagen mostrada)
3. **Navegar tabs**:
   - **Tab 2**: Ver 6 gráficas integradas (2 vibración + 4 fuerza)
   - **Tab 3**: Ver 4 gráficas del puente en layout 2×2

## 📈 **Configuración Aplicada:**

### **Acelerómetro Ampliado:**
```python
# Captura completa del rango dinámico
w.setYRange(-200.0, 200.0)  # ±200g según imagen
```

### **Integración Visual:**
```python
# Layout 3×2 para visualización comparativa
all_plots_layout = QGridLayout()
# Fila 0: [Vib ai0] [Vib ai1] [Fuerza ai0]  
# Fila 1: [Fuerza ai1] [Fuerza ai2] [Fuerza ai3]
```

---

**🎉 ¡REORGANIZACIÓN COMPLETA!**

Ahora tienes:
- ✅ **Rangos correctos** según las imágenes mostradas
- ✅ **Tabs organizados lógicamente** por funcionalidad  
- ✅ **Visualización optimizada** para comparación directa
- ✅ **Colores diferenciados** para fácil identificación
