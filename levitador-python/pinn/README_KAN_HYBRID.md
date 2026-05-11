# KAN-PINN con Optimización Híbrida

Sistema completo de entrenamiento para el levitador magnético usando Kolmogorov-Arnold Networks (KAN) y optimización híbrida.

## 🏗️ Arquitectura

### Modelo: KAN (Kolmogorov-Arnold Network)
- **Archivo**: `kan_model.py`
- **Clase principal**: `KANLevitator`
- **Características**:
  - Funciones de activación B-spline **aprendibles** en los bordes
  - Reemplazo de activaciones fijas (Tanh/ReLU) por funciones adaptativas
  - Compatible con GPU mediante `nn.Parameter` para todos los coeficientes
  - Misma interfaz que `TimeMLP` para compatibilidad

### Optimización Híbrida
- **Archivo**: `run_global_search.py`
- **Estrategia**: Global → Local (dos fases)

#### Fase 1: Exploración Global (Differential Evolution)
- Algoritmo: `scipy.optimize.differential_evolution`
- Configuración robusta:
  - `strategy='best1bin'`
  - `mutation=(0.5, 1.0)` - alta mutación para evitar estancamiento
  - `recombination=0.7`
  - Límites logarítmicos estrictos para parámetros físicos (±2.0)
- **Objetivo**: Encontrar región prometedora del espacio de parámetros

#### Fase 2: Refinamiento Local (Adam)
- Toma el mejor `θ` de Fase 1
- Ejecuta descenso de gradiente (500 épocas, lr=1e-4)
- **Objetivo**: Afinar solución al mínimo exacto del valle

## 📁 Archivos Clave

```
levitador-python/pinn/
├── kan_model.py              # Implementación KAN
├── train_meta.py             # Interfaz fitness optimizada GPU
├── run_global_search.py      # Orquestador híbrido DE→Adam
├── validate_kan_hybrid.py    # Evaluación y visualización
└── README_KAN_HYBRID.md      # Este archivo
```

## 🚀 Uso

### 1. Ejecutar Optimización Híbrida

```bash
cd levitador-python/pinn
python run_global_search.py
```

**Salidas** (en `runs_hybrid/`):
- `phase1_best_theta.npy` - Mejor solución de DE
- `phase2_loss_history.npy` - Historial de pérdida de Adam
- `hybrid_best_theta.npy` - Vector de parámetros final
- `best_model_hybrid.pt` - Checkpoint completo del modelo

### 2. Validar Modelo

```bash
python validate_kan_hybrid.py --ckpt runs_hybrid/best_model_hybrid.pt
```

**Salidas**:
- Métricas RMSE y R² en consola
- `hybrid_validation.png` - Predicciones y residuales
- `phase2_loss_curve.png` - Curva de convergencia

## ⚡ Optimizaciones GPU

El sistema está diseñado para maximizar el uso de GPU:

1. **Precarga de datos**: Todos los tensores se cargan en GPU una sola vez
2. **Fitness sin transferencias**: `fitness_function()` opera completamente en GPU
3. **Una sola transferencia**: Solo el escalar final se mueve a CPU
4. **Todos los parámetros en GPU**: `nn.Parameter` en B-splines garantiza movilidad automática

## 🔬 Sistema Físico

Ecuaciones del levitador magnético:

```
L(y) = k₀ + k/(1 + y/a)
Fₘ = 0.5 · (∂L/∂y) · i²
R_elect = u - R·i - Φ̇
R_mech = m·ÿ - Fₘ + mg
```

- Diferenciación automática para `ẏ, ÿ, i̇`
- Parámetros físicos entrenables: `k₀, k, a` (en log-space)

## 📊 Ventajas de KAN sobre MLP

1. **Mayor expresividad**: Aprende las activaciones óptimas
2. **Menor número de parámetros**: Más eficiente que MLPs profundos
3. **Mejor para física**: Puede representar relaciones no lineales complejas
4. **Interpretabilidad**: Splines visualizables revelan comportamiento aprendido

## 🔧 Configuración Actual

```python
# KAN Architecture
hidden = 64
depth = 3
num_knots = 8

# Hybrid Optimization
DE_maxiter = 50
DE_popsize = 15
Adam_epochs = 500
Adam_lr = 1e-4

# Loss weights
LAMBDA_DATA = 1.0
LAMBDA_PHYSICS = 1e-3
```

## 🎯 Próximos Pasos (Fase 2 del Proyecto)

Una vez obtenido el mejor modelo PINN (gemelo digital):

1. Usar el modelo como "planta virtual"
2. Diseñar controlador FTZNN (convergencia finita)
3. Implementar `u_control = FTZNN(y_target, y_current, model)`

---

**Nota**: Este sistema complementa (no reemplaza) a `train_pinn.py`. Úsalo cuando busques la mejor solución global posible.
