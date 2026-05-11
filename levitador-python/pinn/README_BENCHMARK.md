# 🏆 Benchmark de Metaheurísticas Bio-Inspiradas

Comparación rigurosa de algoritmos de optimización para identificación de parámetros físicos del levitador magnético.

## 🦊 Algoritmos Comparados

### 1. **Differential Evolution** (Evo. Diferencial) ⭐
- **Inspiración**: Evolución biológica y selección natural
- **Características**: Robusto, buen balance exploración/explotación
- **Uso típico**: Problemas multimodales complejos

### 2. **Simulated Annealing** (Recocido Simulado)
- **Inspiración**: Proceso de recocido metalúrgico
- **Características**: Acepta soluciones peores con probabilidad decreciente
- **Uso típico**: Escapar de mínimos locales

### 3. **Artificial Bee Colony (ABC)** 🐝
- **Inspiración**: Comportamiento de abejas buscando néctar
- **Características**: 3 tipos de abejas (employed, onlooker, scout)
- **Uso típico**: Optimización continua

### 4. **Wolf Search Optimization (WSO)** 🐺
- **Inspiración**: Jerarquía social y caza en manadas de lobos
- **Características**: Líder alfa guía la búsqueda
- **Uso típico**: Problemas de alta dimensión

### 5. **Honey Badger Algorithm (HBA)** 🦡
- **Inspiración**: Comportamiento de búsqueda del tejón de miel
- **Características**: Agresivo en explotación, persistente
- **Uso típico**: Optimización global

### 6. **Krill Herd Algorithm** 🦐
- **Inspiración**: Movimiento de cardúmenes de krill
- **Características**: Influencia de vecinos + búsqueda de comida
- **Uso típico**: Optimización continua

### 7. **Random Search** (Baseline)
- **Baseline**: Para comparación de performance mínimo aceptable

---

## 🚀 Uso

### Requisitos
```bash
pip install numpy scipy matplotlib
```

### Ejecutar Benchmark Completo
```bash
C:\Python312\python.exe benchmark_mho.py
```

**Duración**: ~30-45 minutos (7 algoritmos × 5 runs × 1500 evaluaciones)

---

## 📊 Métricas Evaluadas

| Métrica | Descripción | Objetivo |
|---------|-------------|----------|
| **Best Fitness** | Mejor pérdida encontrada | Minimizar |
| **Std(Fitness)** | Desv. estándar entre runs | Minimizar (↓ = más robusto) |
| **Tiempo [s]** | Duración de ejecución | Minimizar |
| **N° Evaluaciones** | Llamadas a función objetivo | Reportar |
| **Tasa de Éxito** | % runs convergentes | 100% |

---

## 📈 Interpretación de Resultados

### Gráficas Generadas (`benchmark_mho/comparison.png`):

#### 1. **Calidad de Solución**
- **Eje Y**: Best Fitness (menor = mejor)
- **Barras de error**: Desviación estándar
- **Interpretación**: El algoritmo con menor barra es el mejor

#### 2. **Eficiencia Computacional**
- **Eje Y**: Tiempo promedio [s]
- **Interpretación**: Más bajo = más rápido

#### 3. **Robustez**
- **Eje Y**: Std(Fitness) entre runs
- **Interpretación**: Menor std = más consistente (no depende de suerte)

#### 4. **Ranking General**
- **Score combinado**: 50% calidad + 30% robustez + 20% eficiencia
- **Interpretación**: Mayor score = mejor algoritmo overall

---

## 📄 Datos Guardados

### `benchmark_results.json`
```json
{
  "Differential Evolution": [
    {
      "best_fitness": 0.001234,
      "best_params": [-3.21, -5.45, -5.12],
      "n_evaluations": 1500,
      "time": 18.3,
      "success": true
    },
    ...  // 5 runs totales
  ],
  ...  // Otros algoritmos
}
```

---

## 🎓 Para tu Artículo/Tesis

### Tabla de Resultados

| Algoritmo | Best Fitness (mean ± std) | Tiempo [s] | Robustez (CV) | Rank |
|-----------|---------------------------|------------|---------------|------|
| DE        | 1.234e-03 ± 2.1e-05      | 18.3 ± 1.2 | 1.7%          | 1    |
| ABC       | 1.456e-03 ± 5.7e-05      | 22.1 ± 2.3 | 3.9%          | 2    |
| ...       | ...                      | ...        | ...           | ...  |

### Justificación del Método Elegido

Si usas **Differential Evolution** (ejemplo):

> *"Se compararon 7 algoritmos metaheurísticos bajo condiciones equivalentes 
> (1500 evaluaciones, 5 runs independientes). Differential Evolution demostró 
> superioridad en:*
> - *Calidad de solución: 15% mejor fitness que el segundo mejor*
> - *Robustez: CV = 1.7% (más consistente)*
> - *Tasa de éxito: 100% de convergencia*
>
> *Esto valida que los parámetros encontrados corresponden a un óptimo global."*

---

## 🔬 Validación Científica

### Test de Hipótesis

**H₀**: Todos los algoritmos encuentran soluciones equivalentes  
**H₁**: Existen diferencias significativas

**Prueba**: ANOVA + post-hoc Tukey (si distribución normal) o Kruskal-Wallis (no paramétrico)

### Ejemplo de Código (post-procesamiento)

```python
import scipy.stats as stats

# Leer resultados
with open('benchmark_mho/benchmark_results.json') as f:
    results = json.load(f)

# Extraer fitness de cada algoritmo
fitness_de = [r['best_fitness'] for r in results['Differential Evolution']]
fitness_abc = [r['best_fitness'] for r in results['ABC (Abejas)']]
# ... etc

# ANOVA
f_stat, p_value = stats.f_oneway(fitness_de, fitness_abc, ...)

if p_value < 0.05:
    print("Diferencias significativas entre algoritmos (p < 0.05)")
```

---

## ⚙️ Configuración Avanzada

### Ajustar Budget por Algoritmo

Edita `benchmark_mho.py`:

```python
benchmark = MHOBenchmark(
    budget=3000,  # Más evaluaciones = mejor pero más lento
    n_runs=10,    # Más runs = mejor robustez estadística
)
```

### Agregar Tu Propio Algoritmo

```python
def run_mi_algoritmo(self, seed: int) -> dict:
    """
    Mi Algoritmo Custom
    """
    np.random.seed(seed)
    start_time = time.time()
    
    # Tu implementación aquí
    best_params = ...
    best_fitness = ...
    n_evals = ...
    
    elapsed = time.time() - start_time
    
    return {
        'best_fitness': best_fitness,
        'best_params': best_params.tolist(),
        'n_evaluations': n_evals,
        'time': elapsed,
        'success': True
    }

# Agregar a la lista en run_benchmark():
algorithms['Mi Algoritmo'] = self.run_mi_algoritmo
```

---

## 📚 Referencias

- **Differential Evolution**: Storn & Price (1997)
- **Simulated Annealing**: Kirkpatrick et al. (1983)
- **Artificial Bee Colony**: Karaboga (2005)
- **Wolf Search**: Tang et al. (2012)
- **Honey Badger**: Hashim et al. (2022)
- **Krill Herd**: Gandomi & Alavi (2012)

---

## 🤝 Soporte

Si algún algoritmo falla:
1. Verifica que `train_meta.py` esté funcionando
2. Reduce `budget` y `n_runs` para prueba rápida
3. Revisa logs de error específicos

---

**Última actualización**: 2025-11-23  
**Versión**: 1.0  
**Autor**: Sistema KAN-PINN para Levitador Magnético
