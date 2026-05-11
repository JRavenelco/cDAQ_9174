# 🧲 PROYECTO: KAN-PINN para Levitador Magnético

## 📋 RESUMEN EJECUTIVO

### Objetivo
Entrenar una red neuronal Physics-Informed (PINN) con arquitectura KAN (Kolmogorov-Arnold Network) para modelar la dinámica de un levitador magnético usando datos experimentales y restricciones físicas.

---

## 🎯 PROBLEMA A RESOLVER

### Sistema Físico: Levitador Magnético
```
Componentes:
├─ Bola metálica levitada (masa m = 0.018 kg)
├─ Electroimán (genera campo magnético)
├─ Sensor de posición (mide altura y)
├─ Controlador (aplica corriente i)
└─ Voltaje de control u

Ecuación física (ODE no-lineal):
  m·ÿ = k0·i² / (k·y)^a - m·g
  
Donde:
  - k0: constante magnética base
  - k:  factor de escala posición
  - a:  exponente no-lineal
  - g:  gravedad = 9.81 m/s²
  - R:  resistencia = 2.72 Ω
```

### Datos Disponibles
```
Archivo: datos_levitador_20251024_162706.txt
Puntos: 7673 mediciones experimentales
Columnas:
  - t: tiempo [s]
  - y: posición [m]
  - i: corriente [A]
  - u: voltaje [V]
```

---

## 🧠 ARQUITECTURA: KAN (Kolmogorov-Arnold Network)

### ¿Qué es KAN?
```
KAN = Red neuronal con activaciones B-spline aprendibles

Red Clásica (MLP):
  y = σ(W₂·σ(W₁·x + b₁) + b₂)
  Activación fija: σ(x) = ReLU/tanh/sigmoid

KAN:
  y = Φ₂(Φ₁(x))
  Activación aprendible: Φ(x) = B-spline con knots ajustables
  
Ventaja:
  ✓ Más expresiva para funciones no-lineales
  ✓ Interpola mejor entre datos
  ✓ Menos parámetros para misma capacidad
```

### Configuración del Modelo
```python
class KANLevitator:
    input_dim = 1        # tiempo t
    hidden = 128         # neuronas por capa
    depth = 3            # capas ocultas
    num_knots = 10       # puntos de control B-spline
    output_dim = 2       # posición y, corriente i
    
    total_params = 364,675 parámetros
    
    # Parámetros físicos entrenables:
    log_k0: tensor([...])  # k0 en log-space
    log_k:  tensor([...])  # k en log-space
    log_a:  tensor([...])  # a en log-space
```

---

## 🔬 METODOLOGÍA: PINN (Physics-Informed Neural Network)

### Concepto PINN
```
PINN = Red que satisface DATOS + FÍSICA simultáneamente

Loss Total = λ₁·Loss_Data + λ₂·Loss_Physics

Loss_Data:
  - MSE entre predicciones y mediciones reales
  - L_data = ||y_pred - y_real||² + ||i_pred - i_real||²

Loss_Physics:
  - Residuo de la ecuación diferencial
  - Usa autograd para calcular ÿ (segunda derivada)
  - L_phys = ||m·ÿ - (k0·i²/(k·y)^a - m·g)||²
  
El modelo debe:
  ✓ Predecir bien los datos observados
  ✓ Satisfacer la ecuación física
```

### Pesos de Loss
```python
LAMBDA_DATA = 1.0      # Peso datos experimentales
LAMBDA_PHYSICS = 1e-3  # Peso restricción física

Loss_Total = 1.0 × Loss_Data + 0.001 × Loss_Physics
```

---

## 🚀 ENTRENAMIENTO: Híbrido DE + Adam

### Phase 1: Differential Evolution (DE)
```
Objetivo: Optimizar parámetros físicos (k0, k, a)

Algoritmo DE:
  1. Población inicial: 12 individuos
  2. Cada individuo = (k0, k, a) en log-space
  3. Evoluciona por 10 generaciones
  4. Operadores: mutación + crossover
  5. Selección: sobrevive el mejor
  
Total evaluaciones: 10 × 12 = 120 evals
Tiempo estimado: 120 × 0.8 min = 96 min

Por qué DE:
  ✓ Explora espacio global (evita mínimos locales)
  ✓ No requiere gradientes
  ✓ Robusto a inicialización
```

### Phase 2: Adam (Gradient Descent)
```
Objetivo: Refinar TODOS los parámetros (red + físicos)

Configuración:
  - Optimizer: Adam
  - Learning rate: 5e-5
  - Epochs: 500
  - Batch size: 1000 puntos
  
Tiempo estimado: 500 × 0.15 min = 75 min

Por qué Adam:
  ✓ Refina solución local
  ✓ Optimiza red neuronal + params físicos
  ✓ Converge más rápido que DE
```

---

## 📊 CONFIGURACIONES PROBADAS

### train_visual.py (Gemini - Conservador)
```python
Modelo:
  - hidden: 64
  - depth: 3
  - num_knots: 8
  - params: 75,459

Entrenamiento:
  - DE: 15 iters × 10 pop = 150 evals
  - Adam: 500 epochs
  
GPU: 2-5 GB (3% de A100)
Tiempo: ~40 min ✅
Loss: ~1.8e-04
```

### train_visual_A100.py (Agresivo - FALLÓ)
```python
Modelo:
  - hidden: 256
  - depth: 5
  - num_knots: 12
  - params: 3,417,859

Entrenamiento:
  - DE: 20 iters × 30 pop = 600 evals
  - Adam: 1000 epochs
  
GPU: 78 GB (98% de A100) 💀
Resultado: OOM (Out of Memory) crash
```

### train_A100_safe.py (Balanceado - MUY LENTO)
```python
Modelo:
  - hidden: 192
  - depth: 4
  - num_knots: 10
  - params: 1,222,851

Entrenamiento:
  - DE: 20 iters × 20 pop = 400 evals
  - Adam: 800 epochs
  
GPU: 50-60 GB (70% de A100)
Tiempo: ~20 HORAS 😱
Problema: Demasiadas evaluaciones DE
```

### train_A100_fast.py (Actual - EN EJECUCIÓN)
```python
Modelo:
  - hidden: 128
  - depth: 3
  - num_knots: 10
  - params: 364,675

Entrenamiento:
  - DE: 10 iters × 12 pop = 120 evals
  - Adam: 500 epochs
  
GPU: 11.7 GB (15% de A100)
Tiempo: ~1.5-2 horas (real)
Estimado inicial: 15-20 min (falló)
```

---

## 🔄 FLUJO DE EJECUCIÓN

```
1. SETUP (2 min)
   ├─ Cargar datos (7673 puntos)
   ├─ Normalizar (Z-score)
   ├─ Crear modelo KAN (364K params)
   └─ Transferir a GPU

2. PHASE 1: DIFFERENTIAL EVOLUTION (90-100 min)
   ├─ Inicializar población (12 individuos)
   ├─ Loop 120 evaluaciones:
   │   ├─ Asignar (k0, k, a) al modelo
   │   ├─ Calcular Loss_Data
   │   ├─ Calcular Loss_Physics (batch 1000)
   │   ├─ Evaluar fitness
   │   └─ Evolucionar población
   └─ Retornar mejores (k0, k, a)

3. PHASE 2: ADAM (75 min)
   ├─ Inicializar con mejores params de DE
   ├─ Loop 500 epochs:
   │   ├─ Forward pass
   │   ├─ Calcular Loss_Total
   │   ├─ Backpropagation
   │   └─ Update weights
   └─ Modelo final optimizado

4. GUARDADO (1 min)
   ├─ Checkpoint: best_model_colab.pt
   ├─ Historial: adam_loss_history.npy
   └─ Gráficas: progress_de.png

TIEMPO TOTAL: ~2.5 horas
```

---

## 📈 RESULTADOS ESPERADOS

### Outputs del Modelo
```python
checkpoint = {
    'model_state_dict': dict,      # Pesos de la red
    'physical_params': {
        'k0': 0.0123,              # Constante magnética
        'k': 0.00456,              # Factor escala
        'a': 0.789                 # Exponente no-lineal
    },
    'loss': 1.23e-04,              # Loss final
    'stats': {                     # Para desnormalizar
        'y_mu': ..., 'y_std': ...,
        'i_mu': ..., 'i_std': ...
    },
    'architecture': {...},         # Config del modelo
    'de_history': {...},           # Historial Phase 1
    'adam_history': [...]          # Historial Phase 2
}
```

### Visualizaciones
```
progress_de.png:
  ├─ Gráfica convergencia DE
  └─ Evolución parámetros (k0, k, a)

training_results_complete.png:
  ├─ Predicciones vs datos reales (y, i)
  ├─ Historial loss (DE + Adam)
  ├─ Evolución parámetros físicos
  └─ Residuos físicos
```

---

## ⚙️ HARDWARE: Google Colab A100

```
Especificaciones:
├─ GPU: NVIDIA A100-SXM4-80GB
├─ VRAM: 80 GB
├─ RAM: 167 GB
├─ Compute: 19.5 TFLOPS (FP32)
└─ Arquitectura: Ampere

Uso actual:
├─ GPU: 11.7 GB (15%)
├─ CPU: Bajo
└─ Tiempo sesión: ~2 horas
```

---

## 🎓 LECCIONES APRENDIDAS

### 1. Balance GPU vs Tiempo
```
Más parámetros ≠ Mejor
Más GPU ≠ Más rápido

Óptimo: Modelo mediano que converge rápido
train_visual.py (75K): Muy pequeño, infrautiliza GPU
train_A100_fast (365K): Balance razonable
train_A100_safe (1.2M): Muy lento por evaluaciones
```

### 2. Costo por Evaluación
```
Cuello de botella: Cálculo de Loss_Physics
  - Autograd para derivadas
  - Batching de 7673 puntos
  - Forward + backward pass

Tiempo/eval ≈ 0.8-1.0 min (constante)
Total tiempo ∝ Número de evaluaciones
```

### 3. Metaheurísticos
```
DE (actual): 120 evals → ~96 min
CMA-ES (propuesto): 30-50 evals → ~30-40 min
Bayesian Opt: 20-30 evals → ~20-25 min

Trade-off: Menos evals = Riesgo convergencia
```

---

## 📚 ARCHIVOS DEL PROYECTO

### Código Principal
```
kan_model.py           # Arquitectura KAN
losses.py              # PINN loss functions
dataset.py             # Carga y normalización datos
train_visual.py        # Script conservador (40 min)
train_A100_fast.py     # Script actual (2h)
train_A100_safe.py     # Script lento (20h)
train_A100_cmaes.py    # Alternativa CMA-ES (30 min)
visualize_results.py   # Generación gráficas
```

### Datos
```
datos_levitador_20251024_162706.txt  # Datos experimentales
best_model_colab.pt                  # Checkpoint modelo
adam_loss_history.npy                # Historial Adam
progress_de.png                      # Gráfica DE
```

### Documentación
```
README_COLAB_COMPLETO.md         # Guía completa
COMPARATIVA_VELOCIDAD.md         # Comparación scripts
GEMINI_VS_NOSOTROS.md           # Análisis alternativas
CONTEXTO_COMPLETO_PROYECTO.md   # Este documento
```

---

## 🎯 ESTADO ACTUAL (23 Nov 2025, 9:41 PM)

```
Script en ejecución: train_A100_fast.py
Tiempo transcurrido: 1h 11min
Estado: Phase 1 - Differential Evolution
Progreso estimado: 85-90% Phase 1
GPU: 11.7 GB estable
Tiempo restante: ~1.5 horas

Expectativa:
  ├─ Phase 1 termina: ~10:00 PM (20 min)
  ├─ Phase 2 completa: ~11:30 PM (90 min)
  └─ Total finaliza: 11:30 PM - 12:00 AM
```

---

## 🚀 PRÓXIMOS PASOS

### Al Terminar Entrenamiento
1. Ejecutar `visualize_results.py`
2. Descargar `best_model_colab.pt`
3. Analizar parámetros físicos (k0, k, a)
4. Comparar predicciones vs datos reales
5. Evaluar residuos físicos

### Mejoras Futuras
1. Probar CMA-ES para reducir tiempo
2. Implementar early stopping
3. Validación cruzada
4. Análisis de incertidumbre
5. Deploy como controlador en tiempo real

---

## 📖 REFERENCIAS

```
KAN Networks:
  - Liu et al. (2024) "KAN: Kolmogorov-Arnold Networks"
  
PINN:
  - Raissi et al. (2019) "Physics-informed neural networks"
  
Differential Evolution:
  - Storn & Price (1997) "Differential Evolution algorithm"
```

---

**Última actualización**: 23 Nov 2025, 9:41 PM  
**Estado**: Entrenamiento en progreso (85%)  
**Tiempo estimado finalización**: 11:30 PM - 12:00 AM
