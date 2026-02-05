"""
Benchmark de Metaheurísticas de Optimización (MHO) para identificación de parámetros físicos.

Algoritmos Bio-inspirados Comparados:
- Differential Evolution (Evo. Diferencial)
- Simulated Annealing (Recocido Simulado)
- Artificial Bee Colony (ABC - Abejas)
- Wolf Search Optimization (WSO - Lobos)
- Honey Badger Algorithm (HBA - Tejón de Miel)
- Krill Herd Algorithm (Camarón)
- Random Search (baseline)

Métricas: Best fitness, N° evaluaciones, tiempo, robustez, convergencia
"""
from __future__ import annotations

import numpy as np
import time
import json
from pathlib import Path
from typing import Callable
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, minimize, dual_annealing

# Importar fitness function de train_meta
import sys
sys.path.insert(0, '.')
from train_meta import fitness_function, model, device, get_initial_theta


class MHOBenchmark:
    """
    Benchmark para comparar metaheurísticas en identificación de parámetros.
    """
    
    def __init__(
        self,
        fitness_func: Callable,
        bounds: list[tuple[float, float]],
        budget: int = 3000,  # Max evaluaciones
        n_runs: int = 10,
        save_dir: str = 'benchmark_mho'
    ):
        self.fitness_func = fitness_func
        self.bounds = bounds
        self.budget = budget
        self.n_runs = n_runs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.results = {}
    
    def run_differential_evolution(self, seed: int) -> dict:
        """
        Differential Evolution (scipy)
        """
        start_time = time.time()
        
        # Calcular popsize e iteraciones para cumplir budget
        popsize = 15
        maxiter = self.budget // (popsize * len(self.bounds)) - 1
        
        result = differential_evolution(
            self.fitness_func,
            self.bounds,
            strategy='best1bin',
            maxiter=maxiter,
            popsize=popsize,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=seed,
            disp=False,
            workers=1,
            polish=False
        )
        
        elapsed = time.time() - start_time
        
        return {
            'best_fitness': result.fun,
            'best_params': result.x.tolist(),
            'n_evaluations': result.nfev,
            'time': elapsed,
            'success': result.success,
            'message': result.message
        }
    
    def run_simulated_annealing(self, seed: int) -> dict:
        """
        Simulated Annealing (Recocido Simulado)
        """
        np.random.seed(seed)
        start_time = time.time()
        
        # Usar dual_annealing de scipy (versión moderna de SA)
        result = dual_annealing(
            self.fitness_func,
            self.bounds,
            maxiter=self.budget // 10,  # Cada iteración hace ~10 evaluaciones
            seed=seed,
            no_local_search=False
        )
        
        elapsed = time.time() - start_time
        
        return {
            'best_fitness': result.fun,
            'best_params': result.x.tolist(),
            'n_evaluations': result.nfev,
            'time': elapsed,
            'success': result.success
        }
    
    def run_abc_algorithm(self, seed: int) -> dict:
        """
        Artificial Bee Colony (ABC) - Algoritmo de Colonia de Abejas
        """
        np.random.seed(seed)
        start_time = time.time()
        
        n_bees = 20
        n_iter = self.budget // (n_bees * 2)  # Employed + onlooker bees
        dim = len(self.bounds)
        lb = np.array([b[0] for b in self.bounds])
        ub = np.array([b[1] for b in self.bounds])
        
        # Inicializar colmena
        food_sources = np.random.uniform(lb, ub, (n_bees, dim))
        fitness = np.array([self.fitness_func(x) for x in food_sources])
        trials = np.zeros(n_bees)
        
        best_idx = np.argmin(fitness)
        best_solution = food_sources[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        limit = 20  # Abandonment limit
        n_evals = n_bees
        
        for _ in range(n_iter):
            # Employed bees phase
            for i in range(n_bees):
                # Seleccionar vecino aleatorio
                k = np.random.choice([j for j in range(n_bees) if j != i])
                phi = np.random.uniform(-1, 1, dim)
                
                # Generar nueva solución
                new_solution = food_sources[i] + phi * (food_sources[i] - food_sources[k])
                new_solution = np.clip(new_solution, lb, ub)
                
                new_fitness = self.fitness_func(new_solution)
                n_evals += 1
                
                # Greedy selection
                if new_fitness < fitness[i]:
                    food_sources[i] = new_solution
                    fitness[i] = new_fitness
                    trials[i] = 0
                    
                    if new_fitness < best_fitness:
                        best_solution = new_solution.copy()
                        best_fitness = new_fitness
                else:
                    trials[i] += 1
            
            # Onlooker bees phase
            probabilities = 1 / (1 + fitness)
            probabilities /= probabilities.sum()
            
            for _ in range(n_bees):
                i = np.random.choice(n_bees, p=probabilities)
                k = np.random.choice([j for j in range(n_bees) if j != i])
                phi = np.random.uniform(-1, 1, dim)
                
                new_solution = food_sources[i] + phi * (food_sources[i] - food_sources[k])
                new_solution = np.clip(new_solution, lb, ub)
                
                new_fitness = self.fitness_func(new_solution)
                n_evals += 1
                
                if new_fitness < fitness[i]:
                    food_sources[i] = new_solution
                    fitness[i] = new_fitness
                    trials[i] = 0
                    
                    if new_fitness < best_fitness:
                        best_solution = new_solution.copy()
                        best_fitness = new_fitness
                else:
                    trials[i] += 1
            
            # Scout bees phase (abandon exhausted sources)
            for i in range(n_bees):
                if trials[i] > limit:
                    food_sources[i] = np.random.uniform(lb, ub, dim)
                    fitness[i] = self.fitness_func(food_sources[i])
                    trials[i] = 0
                    n_evals += 1
        
        elapsed = time.time() - start_time
        
        return {
            'best_fitness': best_fitness,
            'best_params': best_solution.tolist(),
            'n_evaluations': n_evals,
            'time': elapsed,
            'success': True
        }
    
    def run_wolf_search(self, seed: int) -> dict:
        """
        Wolf Search Optimization (WSO) - Algoritmo de Búsqueda de Lobos
        """
        np.random.seed(seed)
        start_time = time.time()
        
        n_wolves = 15
        n_iter = self.budget // n_wolves
        dim = len(self.bounds)
        lb = np.array([b[0] for b in self.bounds])
        ub = np.array([b[1] for b in self.bounds])
        
        # Inicializar manada
        wolves = np.random.uniform(lb, ub, (n_wolves, dim))
        fitness = np.array([self.fitness_func(x) for x in wolves])
        
        # Alpha, Beta, Delta wolves (mejores 3)
        sorted_idx = np.argsort(fitness)
        alpha_pos = wolves[sorted_idx[0]].copy()
        alpha_fitness = fitness[sorted_idx[0]]
        
        n_evals = n_wolves
        
        for it in range(n_iter):
            a = 2 - it * 2 / n_iter  # Linearly decrease from 2 to 0
            
            for i in range(n_wolves):
                # Update position based on alpha, beta, delta
                sorted_idx = np.argsort(fitness)
                alpha_pos = wolves[sorted_idx[0]]
                beta_pos = wolves[sorted_idx[1]] if n_wolves > 1 else alpha_pos
                delta_pos = wolves[sorted_idx[2]] if n_wolves > 2 else alpha_pos
                
                r1, r2 = np.random.rand(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * alpha_pos - wolves[i])
                X1 = alpha_pos - A1 * D_alpha
                
                r1, r2 = np.random.rand(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * beta_pos - wolves[i])
                X2 = beta_pos - A2 * D_beta
                
                r1, r2 = np.random.rand(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * delta_pos - wolves[i])
                X3 = delta_pos - A3 * D_delta
                
                # Average position
                wolves[i] = (X1 + X2 + X3) / 3
                wolves[i] = np.clip(wolves[i], lb, ub)
                
                fitness[i] = self.fitness_func(wolves[i])
                n_evals += 1
                
                if fitness[i] < alpha_fitness:
                    alpha_pos = wolves[i].copy()
                    alpha_fitness = fitness[i]
        
        elapsed = time.time() - start_time
        
        return {
            'best_fitness': alpha_fitness,
            'best_params': alpha_pos.tolist(),
            'n_evaluations': n_evals,
            'time': elapsed,
            'success': True
        }
    
    def run_honey_badger(self, seed: int) -> dict:
        """
        Honey Badger Algorithm (HBA) - Algoritmo del Tejón de Miel
        """
        np.random.seed(seed)
        start_time = time.time()
        
        n_badgers = 15
        n_iter = self.budget // n_badgers
        dim = len(self.bounds)
        lb = np.array([b[0] for b in self.bounds])
        ub = np.array([b[1] for b in self.bounds])
        
        # Inicializar población
        badgers = np.random.uniform(lb, ub, (n_badgers, dim))
        fitness = np.array([self.fitness_func(x) for x in badgers])
        
        best_idx = np.argmin(fitness)
        best_pos = badgers[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        n_evals = n_badgers
        
        for it in range(n_iter):
            alpha = 2 * np.exp(-it / n_iter)  # Decreasing parameter
            
            for i in range(n_badgers):
                r = np.random.rand()
                
                if r < 0.5:
                    # Digging phase (exploitation)
                    r2 = np.random.rand()
                    if r2 < 0.5:
                        new_pos = best_pos + alpha * np.random.randn(dim)
                    else:
                        j = np.random.randint(n_badgers)
                        new_pos = badgers[j] + alpha * np.random.randn(dim)
                else:
                    # Honey finding phase (exploration)
                    r3 = np.random.rand()
                    if r3 < 0.5:
                        new_pos = best_pos + np.random.randn(dim) * (ub - lb) / 10
                    else:
                        new_pos = np.random.uniform(lb, ub, dim)
                
                new_pos = np.clip(new_pos, lb, ub)
                new_fitness = self.fitness_func(new_pos)
                n_evals += 1
                
                if new_fitness < fitness[i]:
                    badgers[i] = new_pos
                    fitness[i] = new_fitness
                    
                    if new_fitness < best_fitness:
                        best_pos = new_pos.copy()
                        best_fitness = new_fitness
        
        elapsed = time.time() - start_time
        
        return {
            'best_fitness': best_fitness,
            'best_params': best_pos.tolist(),
            'n_evaluations': n_evals,
            'time': elapsed,
            'success': True
        }
    
    def run_krill_herd(self, seed: int) -> dict:
        """
        Krill Herd Algorithm (Algoritmo de Camarón/Krill)
        """
        np.random.seed(seed)
        start_time = time.time()
        
        n_krill = 15
        n_iter = self.budget // n_krill
        dim = len(self.bounds)
        lb = np.array([b[0] for b in self.bounds])
        ub = np.array([b[1] for b in self.bounds])
        
        # Inicializar población de krill
        krill = np.random.uniform(lb, ub, (n_krill, dim))
        fitness = np.array([self.fitness_func(x) for x in krill])
        
        best_idx = np.argmin(fitness)
        best_krill = krill[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        n_evals = n_krill
        dt = 0.5  # Time step
        
        for it in range(n_iter):
            for i in range(n_krill):
                # Motion induced by other krill
                nn_effect = np.zeros(dim)
                for j in range(n_krill):
                    if i != j:
                        dist = np.linalg.norm(krill[i] - krill[j])
                        if dist > 0:
                            nn_effect += (krill[j] - krill[i]) / dist * (fitness[i] - fitness[j])
                
                # Motion induced by food location
                food_effect = (best_krill - krill[i])
                
                # Random diffusion
                diffusion = np.random.randn(dim) * 0.01
                
                # Update position
                new_krill = krill[i] + dt * (nn_effect + food_effect + diffusion)
                new_krill = np.clip(new_krill, lb, ub)
                
                new_fitness = self.fitness_func(new_krill)
                n_evals += 1
                
                if new_fitness < fitness[i]:
                    krill[i] = new_krill
                    fitness[i] = new_fitness
                    
                    if new_fitness < best_fitness:
                        best_krill = new_krill.copy()
                        best_fitness = new_fitness
        
        elapsed = time.time() - start_time
        
        return {
            'best_fitness': best_fitness,
            'best_params': best_krill.tolist(),
            'n_evaluations': n_evals,
            'time': elapsed,
            'success': True
        }
    
    def run_random_search(self, seed: int) -> dict:
        """
        Random Search (baseline simple)
        """
        np.random.seed(seed)
        start_time = time.time()
        
        best_fitness = np.inf
        best_params = None
        
        for i in range(self.budget):
            # Samplear uniformemente en bounds
            params = [np.random.uniform(b[0], b[1]) for b in self.bounds]
            fitness = self.fitness_func(params)
            
            if fitness < best_fitness:
                best_fitness = fitness
                best_params = params
        
        elapsed = time.time() - start_time
        
        return {
            'best_fitness': best_fitness,
            'best_params': best_params,
            'n_evaluations': self.budget,
            'time': elapsed,
            'success': True
        }
    
    def run_adam_random_init(self, seed: int, n_steps: int = 100) -> dict:
        """
        Adam optimizer desde inicialización aleatoria (baseline local)
        """
        np.random.seed(seed)
        start_time = time.time()
        
        # Inicialización aleatoria en bounds
        x0 = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
        
        # Minimización con L-BFGS-B (similar a Adam pero sin PyTorch)
        result = minimize(
            self.fitness_func,
            x0,
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'maxiter': n_steps, 'disp': False}
        )
        
        elapsed = time.time() - start_time
        
        return {
            'best_fitness': result.fun,
            'best_params': result.x.tolist(),
            'n_evaluations': result.nfev,
            'time': elapsed,
            'success': result.success
        }
    
    def run_benchmark(self):
        """
        Ejecuta benchmark completo: múltiples algoritmos, múltiples runs.
        """
        print(f"\n{'='*80}")
        print("BENCHMARK DE METAHEURÍSTICAS")
        print(f"{'='*80}\n")
        print(f"Configuración:")
        print(f"  Dimensiones: {len(self.bounds)}")
        print(f"  Budget: {self.budget} evaluaciones")
        print(f"  Runs por algoritmo: {self.n_runs}")
        print(f"  Bounds: {self.bounds}")
        print()
        
        algorithms = {
            'Differential Evolution': self.run_differential_evolution,
            'Simulated Annealing': self.run_simulated_annealing,
            'ABC (Abejas)': self.run_abc_algorithm,
            'WSO (Lobos)': self.run_wolf_search,
            'HBA (Tejón de Miel)': self.run_honey_badger,
            'Krill Herd (Camarón)': self.run_krill_herd,
            'Random Search': self.run_random_search,
        }
        
        for algo_name, algo_func in algorithms.items():
            print(f"\n{'─'*80}")
            print(f"Running: {algo_name}")
            print(f"{'─'*80}")
            
            runs_results = []
            
            for run in range(self.n_runs):
                print(f"  Run {run+1}/{self.n_runs}...", end=' ')
                
                try:
                    result = algo_func(seed=42 + run)
                    runs_results.append(result)
                    print(f"✓ Loss: {result['best_fitness']:.6e}")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    runs_results.append({'error': str(e)})
            
            self.results[algo_name] = runs_results
        
        print(f"\n{'='*80}")
        print("BENCHMARK COMPLETADO")
        print(f"{'='*80}\n")
    
    def analyze_results(self) -> dict:
        """
        Analiza resultados y genera estadísticas comparativas.
        """
        print(f"\n{'='*80}")
        print("ANÁLISIS DE RESULTADOS")
        print(f"{'='*80}\n")
        
        summary = {}
        
        for algo_name, runs in self.results.items():
            valid_runs = [r for r in runs if 'error' not in r]
            
            if not valid_runs:
                print(f"{algo_name}: Sin runs válidos")
                continue
            
            fitnesses = [r['best_fitness'] for r in valid_runs]
            times = [r['time'] for r in valid_runs]
            n_evals = [r['n_evaluations'] for r in valid_runs]
            
            summary[algo_name] = {
                'best_fitness_mean': np.mean(fitnesses),
                'best_fitness_std': np.std(fitnesses),
                'best_fitness_min': np.min(fitnesses),
                'best_fitness_max': np.max(fitnesses),
                'time_mean': np.mean(times),
                'time_std': np.std(times),
                'n_evals_mean': np.mean(n_evals),
                'success_rate': len(valid_runs) / len(runs) * 100,
                'n_valid_runs': len(valid_runs)
            }
            
            print(f"{algo_name}:")
            print(f"  Best Fitness: {summary[algo_name]['best_fitness_mean']:.6e} "
                  f"± {summary[algo_name]['best_fitness_std']:.6e}")
            print(f"    Min: {summary[algo_name]['best_fitness_min']:.6e}")
            print(f"    Max: {summary[algo_name]['best_fitness_max']:.6e}")
            print(f"  Time: {summary[algo_name]['time_mean']:.2f}s "
                  f"± {summary[algo_name]['time_std']:.2f}s")
            print(f"  Evaluations: {summary[algo_name]['n_evals_mean']:.0f}")
            print(f"  Success Rate: {summary[algo_name]['success_rate']:.1f}%")
            print()
        
        return summary
    
    def plot_comparison(self, summary: dict, save_path: str = None):
        """
        Genera gráficas comparativas de los algoritmos.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        algo_names = list(summary.keys())
        
        # 1. Best Fitness (con barras de error)
        ax = axes[0, 0]
        means = [summary[a]['best_fitness_mean'] for a in algo_names]
        stds = [summary[a]['best_fitness_std'] for a in algo_names]
        
        x = np.arange(len(algo_names))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(algo_names, rotation=45, ha='right')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Calidad de Solución (menor es mejor)')
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Tiempo de ejecución
        ax = axes[0, 1]
        times = [summary[a]['time_mean'] for a in algo_names]
        time_stds = [summary[a]['time_std'] for a in algo_names]
        
        ax.bar(x, times, yerr=time_stds, capsize=5, alpha=0.7, color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(algo_names, rotation=45, ha='right')
        ax.set_ylabel('Tiempo [s]')
        ax.set_title('Eficiencia Computacional')
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Robustez (std de fitness)
        ax = axes[1, 0]
        robustness = [summary[a]['best_fitness_std'] for a in algo_names]
        
        ax.bar(x, robustness, alpha=0.7, color='lightgreen')
        ax.set_xticks(x)
        ax.set_xticklabels(algo_names, rotation=45, ha='right')
        ax.set_ylabel('Std(Fitness)')
        ax.set_title('Robustez (menor std = más consistente)')
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Ranking general
        ax = axes[1, 1]
        
        # Normalizar métricas y calcular score
        all_means = np.array([summary[a]['best_fitness_mean'] for a in algo_names])
        all_stds = np.array([summary[a]['best_fitness_std'] for a in algo_names])
        all_times = np.array([summary[a]['time_mean'] for a in algo_names])
        
        # Score: penalizar fitness alto, std alto, tiempo alto
        scores = (1 - (all_means - all_means.min()) / (all_means.max() - all_means.min() + 1e-10)) * 0.5 + \
                 (1 - (all_stds / (all_stds.max() + 1e-10))) * 0.3 + \
                 (1 - (all_times / (all_times.max() + 1e-10))) * 0.2
        
        sorted_idx = np.argsort(scores)[::-1]
        sorted_names = [algo_names[i] for i in sorted_idx]
        sorted_scores = scores[sorted_idx]
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_names)))
        ax.barh(sorted_names, sorted_scores, color=colors)
        ax.set_xlabel('Score Global')
        ax.set_title('Ranking General (mayor es mejor)')
        ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Benchmark de Metaheurísticas: Identificación de Parámetros Físicos',
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n📊 Gráficas guardadas: {save_path}")
        
        plt.show()
    
    def save_results(self):
        """
        Guarda resultados completos en JSON.
        """
        save_path = self.save_dir / 'benchmark_results.json'
        
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Resultados guardados: {save_path}")


def main():
    print("\n" + "="*80)
    print("BENCHMARK DE METAHEURÍSTICAS BIO-INSPIRADAS")
    print("Identificación de Parámetros Físicos (k0, k, a) del Levitador Magnético")
    print("="*80 + "\n")
    
    print("Algoritmos a comparar:")
    print("  1. Differential Evolution (Evo. Diferencial) ⭐")
    print("  2. Simulated Annealing (Recocido Simulado)")
    print("  3. Artificial Bee Colony (ABC - Abejas) 🐝")
    print("  4. Wolf Search Optimization (WSO - Lobos) 🐺")
    print("  5. Honey Badger Algorithm (HBA - Tejón de Miel) 🦡")
    print("  6. Krill Herd Algorithm (Camarón) 🦐")
    print("  7. Random Search (baseline)")
    print()
    
    # Definir bounds para k0, k, a (en log space)
    log_k0_init = np.log(36.3e-3)
    log_k_init = np.log(3.5e-3)
    log_a_init = np.log(5.2e-3)
    
    bounds = [
        (log_k0_init - 3.0, log_k0_init + 3.0),  # k0
        (log_k_init - 3.0, log_k_init + 3.0),     # k
        (log_a_init - 3.0, log_a_init + 3.0)      # a
    ]
    
    print("Configuración del Benchmark:")
    print(f"  Dimensiones: 3 parámetros (k0, k, a)")
    print(f"  Budget: 1500 evaluaciones por algoritmo")
    print(f"  Runs independientes: 5 (para robustez estadística)")
    print(f"  Total de evaluaciones: ~10,500")
    print()
    
    input("Presiona ENTER para comenzar el benchmark (durará ~30-45 minutos)...")
    
    # Crear benchmark
    benchmark = MHOBenchmark(
        fitness_func=fitness_function,
        bounds=bounds,
        budget=1500,
        n_runs=5,
        save_dir='benchmark_mho'
    )
    
    # Ejecutar
    benchmark.run_benchmark()
    
    # Analizar
    summary = benchmark.analyze_results()
    
    # Visualizar
    benchmark.plot_comparison(summary, save_path='benchmark_mho/comparison.png')
    
    # Guardar
    benchmark.save_results()
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETADO EXITOSAMENTE")
    print("="*80)
    print("\nResultados guardados en:")
    print("  - benchmark_mho/comparison.png (gráficas comparativas)")
    print("  - benchmark_mho/benchmark_results.json (datos completos)")
    print("\nConsulta el análisis arriba para ver qué algoritmo fue mejor.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
