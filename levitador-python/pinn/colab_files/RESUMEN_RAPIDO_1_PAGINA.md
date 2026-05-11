# 🧲 KAN-PINN para Levitador Magnético - RESUMEN 1 PÁGINA

## 🎯 QUÉ ES
Red neuronal **KAN** (activaciones B-spline) + **PINN** (física informada) para modelar levitador magnético.

---

## 📐 ECUACIÓN FÍSICA
```
m·ÿ = k0·i² / (k·y)^a - m·g

Objetivos:
1. Predecir (y, i) dado tiempo t
2. Satisfacer ecuación física
3. Identificar parámetros (k0, k, a)
```

---

## 🧠 ARQUITECTURA
```
KANLevitator:
  Input: t (tiempo)
  Layers: 128 → 128 → 128 (B-splines)
  Output: (y, i)
  Params: 364,675
  
Parámetros físicos:
  k0, k, a (3 valores a optimizar)
```

---

## 🔬 METODOLOGÍA PINN
```
Loss = λ₁·Loss_Data + λ₂·Loss_Physics

Loss_Data = ||predicciones - mediciones||²
Loss_Physics = ||m·ÿ - F_magnética + m·g||²
```

**Ventaja**: Aprende de datos + física (mejor generalización, menos datos requeridos)

---

## 🚀 ENTRENAMIENTO (2 fases)

### Phase 1: Differential Evolution (~90 min)
- Optimiza solo (k0, k, a)
- 120 evaluaciones
- Explora espacio global

### Phase 2: Adam (~75 min)
- Optimiza TODO (red + parámetros)
- 500 epochs
- Refina solución local

**Total**: ~2.5 horas

---

## 📊 DATOS
- **7673 puntos** experimentales
- Columnas: tiempo, posición, corriente, voltaje
- Normalización: Z-score

---

## 💻 HARDWARE
- Google Colab A100 (80 GB VRAM)
- Uso actual: 11.7 GB (15%)
- CPU: Bajo

---

## 🎨 VISUALIZACIONES

### Código Manim (animaciones):
```bash
manim -pql visualizacion_manim.py KANPINNVisualization
```

**Escenas**:
1. Sistema físico del levitador
2. Arquitectura KAN vs MLP
3. Concepto PINN
4. Proceso entrenamiento (DE + Adam)
5. Comparativa configuraciones

---

## 📈 CONFIGURACIONES PROBADAS

| Script | Params | GPU | Tiempo | Estado |
|--------|--------|-----|---------|---------|
| train_visual.py | 75K | 2 GB | 40 min | ✅ Funcional |
| train_A100_fast.py | 365K | 12 GB | 2h | ⚡ Actual |
| train_A100_safe.py | 1.2M | 55 GB | 20h | ❌ Muy lento |

---

## 🎓 CONCEPTOS CLAVE

**KAN**: Red con activaciones B-spline aprendibles (vs ReLU fijas)  
**PINN**: Restricción física en loss function (datos + ecuaciones)  
**DE**: Metaheurístico para exploración global  
**Adam**: Optimización local con gradientes  
**Autograd**: Cálculo automático de derivadas (ÿ)  

---

## 📁 ARCHIVOS ESENCIALES

```
kan_model.py              # Arquitectura KAN
losses.py                 # PINN loss functions
train_A100_fast.py        # Script actual
visualizacion_manim.py    # Animaciones
CONTEXTO_COMPLETO.md      # Documentación completa
PROMPT_PARA_ASISTENTE.md  # Para enseñanza
```

---

## 🔄 FLUJO
```
Datos → Normalizar → KAN → Loss(Data+Physics) → Optimizer(DE/Adam) → Modelo
```

---

## 💡 LECCIONES

1. **Balance modelo**: Ni muy grande (OOM) ni muy pequeño (infrautiliza GPU)
2. **Costo por eval**: ~0.8 min independiente del metaheurístico
3. **Híbrido DE+Adam**: Exploración global + refinamiento local
4. **PINN ventaja**: Física = regularización + mejor extrapolación

---

## 🎯 RESULTADOS ESPERADOS

```python
Parámetros físicos identificados:
  k0 ≈ 0.01-0.1
  k  ≈ 0.001-0.01
  a  ≈ 0.5-2.0

Loss final: ~1e-04
Predicciones: Error <5% vs datos reales
```

---

## 📚 PARA PROFUNDIZAR

**Usa este prompt con tu asistente IA**:
```
Lee PROMPT_PARA_ASISTENTE_IA.md y explícame:
1. ¿Por qué es difícil este problema?
2. ¿Qué aporta PINN sobre redes normales?
3. ¿Por qué híbrido DE + Adam?
```

**Visualiza con Manim**:
```bash
manim -pql visualizacion_manim.py KANPINNVisualization
```

---

**Estado actual**: Entrenamiento al 85% Phase 1 (1h 11min / ~2h total)  
**GPU**: 11.7 GB estable  
**Finalización estimada**: 11:30 PM - 12:00 AM
