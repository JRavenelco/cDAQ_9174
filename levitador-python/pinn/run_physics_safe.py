"""
GPU-SAFE Physics-Only Optimization for KAN-PINN.
Optimized for GTX 1060 (3GB VRAM) - reduced model, smaller batches, temperature safe.
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import differential_evolution

from dataset import MonitDataset
from kan_model import KANLevitator
from losses import loss_data_from_model, loss_phys_from_model
from torch.utils.data import DataLoader

# GPU Safety: Clear cache before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Initializing GPU-SAFE fitness function...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load data
data_path = "../../levitador valentin/experimentos/datos_levitador_20251024_162706.txt"
ds = MonitDataset(data_path)
data_loader = DataLoader(ds, batch_size=len(ds))

# Physics collocation points with requires_grad
phys_t = torch.tensor(ds.t_norm, dtype=torch.float32, requires_grad=True).reshape(-1, 1).to(device)
phys_u = torch.tensor(ds.u, dtype=torch.float32).reshape(-1, 1).to(device)

# Preload training data
print("Preloading training data to GPU...")
t_data_gpu, y_real_gpu, i_real_gpu, u_data_gpu = next(iter(data_loader))
t_data_gpu = t_data_gpu.to(device)
y_real_gpu = y_real_gpu.to(device)
i_real_gpu = i_real_gpu.to(device)

# REDUCED MODEL for GPU safety (32 hidden instead of 64, depth 2 instead of 3)
model = KANLevitator(
    y_mu=ds.stats.y_mu, y_std=ds.stats.y_std,
    i_mu=ds.stats.i_mu, i_std=ds.stats.i_std,
    hidden=32,      # Reduced from 64
    depth=2,        # Reduced from 3
    num_knots=6     # Reduced from 8
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"GPU-SAFE KAN model created with {n_params} parameters (reduced for stability)")

# Loss weights
m, g, R = 0.018, 9.81, 2.72
LAMBDA_DATA = 1.0
LAMBDA_PHYSICS = 1e-3


def fitness_physics_only(params_3d: np.ndarray) -> float:
    """
    GPU-SAFE fitness function with memory management.
    """
    # Clear GPU cache periodically
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Update physical parameters
    with torch.no_grad():
        model.log_k0.data = torch.tensor(params_3d[0], device=device)
        model.log_k.data = torch.tensor(params_3d[1], device=device)
        model.log_a.data = torch.tensor(params_3d[2], device=device)
    
    model.eval()
    
    # Data loss
    loss_data = loss_data_from_model(model, t_data_gpu, y_real_gpu, i_real_gpu)
    
    # Physics loss in VERY small batches (50 points for safety)
    batch_size = 50  # Extra small for GTX 1060
    n_points = phys_t.shape[0]
    physics_losses = []
    
    for i in range(0, n_points, batch_size):
        end_idx = min(i + batch_size, n_points)
        t_batch = phys_t[i:end_idx]
        u_batch = phys_u[i:end_idx]
        
        loss_phys_batch = loss_phys_from_model(model, t_batch, u_batch, m=m, g=g, R=R)
        physics_losses.append(loss_phys_batch.detach())  # Detach to save memory
    
    loss_physics = torch.stack(physics_losses).mean()
    loss_total = LAMBDA_DATA * loss_data + LAMBDA_PHYSICS * loss_physics
    
    return loss_total.item()


def phase1_physics_search(
    maxiter: int = 30,    # Reduced from 100
    popsize: int = 15     # Reduced from 30
) -> tuple[np.ndarray, float]:
    """
    Phase 1: Optimize only physical parameters with DE (FAST TEST).
    """
    print("\n" + "="*80)
    print("PHASE 1: PHYSICS-ONLY SEARCH (Differential Evolution, 3D) - GPU SAFE")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Dimensions: 3 (log_k0, log_k, log_a)")
    print(f"  - Max iterations: {maxiter}")
    print(f"  - Population size: {popsize}")
    print(f"  - Model: REDUCED (hidden=32, depth=2)")
    print(f"  - Batch size: 50 (extra small for safety)")
    print()
    
    # Temperature warning
    if torch.cuda.is_available():
        print("⚠️  MONITOR GPU TEMPERATURE with GPU-Z")
        print("    Stop if temperature > 80°C")
        print()
    
    # Bounds
    log_k0_init = np.log(36.3e-3)
    log_k_init = np.log(3.5e-3)
    log_a_init = np.log(5.2e-3)
    
    bounds = [
        (log_k0_init - 3.0, log_k0_init + 3.0),
        (log_k_init - 3.0, log_k_init + 3.0),
        (log_a_init - 3.0, log_a_init + 3.0)
    ]
    
    start_time = time.time()
    
    result = differential_evolution(
        fitness_physics_only,
        bounds,
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=True,
        workers=1,
        polish=False
    )
    
    elapsed = time.time() - start_time
    
    print(f"\nPhase 1 completed in {elapsed:.1f}s")
    print(f"Best loss found: {result.fun:.6e}")
    
    k0_best = np.exp(result.x[0])
    k_best = np.exp(result.x[1])
    a_best = np.exp(result.x[2])
    
    print(f"\nBest Physical Parameters Found:")
    print(f"  k0 = {k0_best:.6e}")
    print(f"  k  = {k_best:.6e}")
    print(f"  a  = {a_best:.6e}")
    
    return result.x, result.fun


def phase2_full_refinement(
    best_physics: np.ndarray,
    num_epochs: int = 500,   # Reduced from 1000
    lr: float = 5e-5
) -> tuple[float, list[float]]:
    """
    Phase 2: Refine ALL parameters with Adam (GPU-SAFE).
    """
    print("\n" + "="*80)
    print("PHASE 2: FULL REFINEMENT (Adam) - GPU SAFE")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Batch size: 50")
    print()
    
    # Load best physics
    with torch.no_grad():
        model.log_k0.data = torch.tensor(best_physics[0], device=device)
        model.log_k.data = torch.tensor(best_physics[1], device=device)
        model.log_a.data = torch.tensor(best_physics[2], device=device)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_history = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Clear cache every 50 epochs
        if epoch % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        optimizer.zero_grad()
        
        loss_data = loss_data_from_model(model, t_data_gpu, y_real_gpu, i_real_gpu)
        
        # Physics in mini-batches
        batch_size = 50
        n_points = phys_t.shape[0]
        physics_losses = []
        
        for i in range(0, n_points, batch_size):
            end_idx = min(i + batch_size, n_points)
            t_batch = phys_t[i:end_idx]
            u_batch = phys_u[i:end_idx]
            loss_phys_batch = loss_phys_from_model(model, t_batch, u_batch, m=m, g=g, R=R)
            physics_losses.append(loss_phys_batch)
        
        loss_physics = torch.stack(physics_losses).mean()
        loss_total = LAMBDA_DATA * loss_data + LAMBDA_PHYSICS * loss_physics
        
        loss_total.backward()
        optimizer.step()
        
        loss_value = loss_total.item()
        loss_history.append(loss_value)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            k0, k, a = model.get_phys_params()
            print(f"Epoch {epoch+1:4d} | Loss: {loss_value:.6e} | "
                  f"k0={k0.item():.3e}, k={k.item():.3e}, a={a.item():.3e}")
    
    elapsed = time.time() - start_time
    final_loss = loss_history[-1]
    
    print(f"\nPhase 2 completed in {elapsed:.1f}s")
    print(f"Final loss: {final_loss:.6e}")
    
    return final_loss, loss_history


def main():
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  KAN-PINN GPU-SAFE OPTIMIZATION".center(78) + "#")
    print("#" + "  (Reduced model for GTX 1060 stability)".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    save_dir = Path("runs_physics_safe")
    save_dir.mkdir(exist_ok=True)
    
    # Phase 1: DE on physics only
    best_physics, loss_phase1 = phase1_physics_search(
        maxiter=30,   # Fast test (increase to 50-100 if stable)
        popsize=15
    )
    
    np.save(save_dir / "phase1_best_physics.npy", best_physics)
    
    # Phase 2: Adam on all
    loss_final, adam_history = phase2_full_refinement(
        best_physics=best_physics,
        num_epochs=500,
        lr=5e-5
    )
    
    np.save(save_dir / "phase2_loss_history.npy", np.array(adam_history))
    
    # Save model
    k0_final, k_final, a_final = model.get_phys_params()
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'loss': loss_final,
        'physical_params': {
            'k0': k0_final.item(),
            'k': k_final.item(),
            'a': a_final.item()
        },
        'stats': {
            'y_mu': ds.stats.y_mu,
            'y_std': ds.stats.y_std,
            'i_mu': ds.stats.i_mu,
            'i_std': ds.stats.i_std
        },
        'architecture': {
            'hidden': 32,
            'depth': 2,
            'num_knots': 6
        }
    }
    
    torch.save(checkpoint, save_dir / "best_model_safe.pt")
    
    print("\n" + "="*80)
    print("GPU-SAFE OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Final loss: {loss_final:.6e}")
    print()
    print("Final Physical Parameters:")
    print(f"  k0 = {k0_final.item():.6e}")
    print(f"  k  = {k_final.item():.6e}")
    print(f"  a  = {a_final.item():.6e}")
    print()
    print(f"Saved to: {save_dir}/")
    print("="*80)
    
    # GPU stats
    if torch.cuda.is_available():
        print(f"\nGPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
