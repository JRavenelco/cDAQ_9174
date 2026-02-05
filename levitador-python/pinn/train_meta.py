from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from dataset import MonitDataset, concat_datasets
from kan_model import KANLevitator  # Using KAN instead of TimeMLP
from losses import loss_data_from_model, loss_phys_from_model

# --- PASO 1: Inicializar todo (se hace una vez) ---
print("Initializing fitness function interface with KAN...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Cargar datos
# ruta de datos de entrenamiento 
data_path = "../../levitador valentin/experimentos/datos_levitador_20251024_162706.txt"
ds = MonitDataset(data_path)

# DataLoader para la pérdida de datos (un solo lote con todo el dataset)
data_loader = DataLoader(ds, batch_size=len(ds))

# Puntos de colocación para la pérdida de física (reusamos los puntos de datos)
# PRECARGAR EN GPU para evitar transferencias repetidas
# IMPORTANTE: phys_t necesita requires_grad=True para calcular derivadas
phys_t = torch.tensor(ds.t_norm, dtype=torch.float32, requires_grad=True).reshape(-1, 1).to(device)
phys_u = torch.tensor(ds.u, dtype=torch.float32).reshape(-1, 1).to(device)

# Precargar TODOS los datos de entrenamiento en GPU una vez
print("Preloading training data to GPU...")
t_data_gpu, y_real_gpu, i_real_gpu, u_data_gpu = next(iter(data_loader))
t_data_gpu = t_data_gpu.to(device)
y_real_gpu = y_real_gpu.to(device)
i_real_gpu = i_real_gpu.to(device)
u_data_gpu = u_data_gpu.to(device)

# Instanciar el modelo KAN
model = KANLevitator(
    y_mu=ds.stats.y_mu, y_std=ds.stats.y_std,
    i_mu=ds.stats.i_mu, i_std=ds.stats.i_std,
    hidden=64, depth=3, num_knots=8
).to(device)

print(f"KAN model created with {sum(p.numel() for p in model.parameters())} parameters")

# Parámetros fijos y pesos de las pérdidas
m, g, R = 0.018, 9.81, 2.72
LAMBDA_DATA = 1.0
LAMBDA_PHYSICS = 1e-3  # Usamos el mejor lambda que encontramos


# --- PASO 2: Funciones auxiliares para manejar 'theta' ---

def set_model_weights_from_theta(model: KANLevitator, theta_vector: list[float]):
    """Carga un vector 1D de parámetros en el modelo siguiendo el mismo orden de model.parameters()."""
    # Convertir lista a tensor con dtype y device del modelo
    first_param = next(model.parameters())
    theta_tensor = torch.tensor(theta_vector, dtype=first_param.dtype, device=first_param.device)
    # Mapear vector a parámetros del modelo de forma segura
    vector_to_parameters(theta_tensor, model.parameters())

def get_initial_theta(model: KANLevitator) -> list[float]:
    """Extrae los parámetros actuales del modelo como un vector 1D en el mismo orden de model.parameters()."""
    return parameters_to_vector(model.parameters()).detach().cpu().numpy().tolist()


# --- PASO 3: La Función de Aptitud Principal ---

def fitness_function(theta: list[float]) -> float:
    """
    Función de Aptitud optimizada para GPU.
    Recibe un vector 1D de parámetros (theta) y devuelve un solo float (la pérdida).
    
    Optimizaciones:
    - Usa datos precargados en GPU (t_data_gpu, y_real_gpu, etc.)
    - No hace transferencias CPU<->GPU innecesarias
    - Solo devuelve el escalar final a CPU
    
    IMPORTANTE: No usamos @torch.no_grad() porque la pérdida de física
    necesita calcular derivadas automáticas (dy/dt, d²y/dt²).
    """
    # 1. Cargar los nuevos pesos en el modelo (ya está en GPU)
    with torch.no_grad():
        set_model_weights_from_theta(model, theta)
    model.eval()

    # 2. Calcular Loss_data (usa datos ya en GPU)
    loss_data = loss_data_from_model(model, t_data_gpu, y_real_gpu, i_real_gpu)

    # 3. Calcular Loss_physics (usa datos ya en GPU, necesita autograd!)
    loss_physics = loss_phys_from_model(
        model, phys_t, phys_u,
        m=m, g=g, R=R
    )

    # 4. Calcular Pérdida Total
    loss_total = (LAMBDA_DATA * loss_data) + (LAMBDA_PHYSICS * loss_physics)

    # 5. Devolver solo el escalar (única transferencia GPU->CPU)
    return loss_total.item()


# --- PASO 4: Ejemplo de cómo lo usaría el optimizador ---
if __name__ == "__main__":
    print("\nFitness function interface is ready.")

    # Obtener el vector de pesos iniciales
    initial_solution = get_initial_theta(model)
    print(f"Dimension of the parameter vector (theta): {len(initial_solution)}")

    # Evaluar la aptitud de la solución inicial
    print("\nEvaluating fitness of the initial random weights...")
    initial_loss = fitness_function(initial_solution)
    print(f"Initial loss: {initial_loss:.6e}")

    print("\nThis script is now ready to be imported by a metaheuristic optimizer.")
    print("Example: from train_meta import fitness_function, get_initial_theta")

    # --- AQUÍ ES DONDE LLAMARÍAS A TUS ALGORITMOS ---
    # print("\nInitiating metaheuristic optimizer (e.g., Grey Wolf)...")
    #
    # best_theta_found, best_loss_found = GreyWolfOptimizer(
    #     fitness_function=fitness_function,
    #     dim=len(initial_solution),
    #     pop_size=50,
    #     max_iter=100
    # )
    #
    # print(f"Optimization finished. Best loss: {best_loss_found:.6e}")
    #
    # # Guardar la mejor solución encontrada
    # import numpy as np
    # np.save("best_theta.npy", best_theta_found)
    # ----------------------------------------------------
