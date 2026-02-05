from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

# Importar la interfaz de aptitud y los componentes necesarios
from train_meta import fitness_function, model, get_initial_theta

def main():
    # 1. Definir Límites (Bounds) para cada parámetro en theta
    print("Defining parameter bounds for Differential Evolution...")
    initial_theta = get_initial_theta(model)
    n_params = len(initial_theta)

    # Límites genéricos para los pesos y sesgos de la red neuronal
    # Usamos un rango razonable; DE es bueno para explorar.
    bounds = [(-2.0, 2.0)] * (n_params - 3)

    # Límites específicos para los parámetros físicos (en su forma logarítmica)
    # Los centramos en sus valores iniciales con un amplio margen para la búsqueda.
    log_k0_init = np.log(36.3e-3)
    log_k_init = np.log(3.5e-3)
    log_a_init = np.log(5.2e-3)

    bounds.append((log_k0_init - 5.0, log_k0_init + 5.0))  # Límite para log_k0
    bounds.append((log_k_init - 5.0, log_k_init + 5.0))    # Límite para log_k
    bounds.append((log_a_init - 5.0, log_a_init + 5.0))    # Límite para log_a

    print(f"Total parameters to optimize: {n_params}")

    # Crear directorio para guardar resultados si no existe
    save_dir = Path("runs_meta")
    save_dir.mkdir(exist_ok=True)

    # 2. Ejecutar Optimizador
    print("\nStarting Differential Evolution...")
    print("This may take a significant amount of time depending on maxiter and popsize.")

    # Smoke test corto (ajusta luego para una corrida larga)
    result = differential_evolution(
        fitness_function,
        bounds,
        strategy='best1bin',
        maxiter=5,   # prueba rápida
        popsize=6,   # prueba rápida
        disp=True,   # Muestra el progreso en cada iteración
        workers=1    # evitar multiprocessing en Windows en primera instancia
    )

    # 3. Guardar Resultados
    print("\nOptimization Finished.")
    print(f"Best loss (fitness) found: {result.fun:.6e}")

    # Guardar el mejor vector theta encontrado
    best_theta = result.x
    save_path = save_dir / "de_model.npy"
    np.save(save_path, best_theta)

    print(f"\nBest parameter vector saved to: {save_path}")
    print("Next step: Run a validation script to load this vector and evaluate the model.")


if __name__ == "__main__":
    main()
