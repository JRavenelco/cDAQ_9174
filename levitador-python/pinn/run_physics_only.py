"""
Physics-Only Hybrid Optimization for KAN-PINN.
Optimizes ONLY physical parameters (k0, k, a) with DE, keeping KAN weights fixed.
Then refines everything with Adam.
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import differential_evolution

from train_meta import (
    model,
    device,
    ds,
    t_data_gpu,
    y_real_gpu,
    i_real_gpu,
    phys_t,
    phys_u,
    LAMBDA_DATA,
    LAMBDA_PHYSICS,
    m, g, R
)
from losses import loss_data_from_model, loss_phys_from_model


def fitness_physics_only(params_3d: np.ndarray) -> float:
    """
    Fitness function for 3D optimization (only physical parameters).
    
    Args:
        params_3d: [log_k0, log_k, log_a]
    
    Returns:
        Total loss
    """
    # Update only the physical parameters
    with torch.no_grad():
        model.log_k0.data = torch.tensor(params_3d[0], device=device)
        model.log_k.data = torch.tensor(params_3d[1], device=device)
        model.log_a.data = torch.tensor(params_3d[2], device=device)
    
    model.eval()
    
    # Calculate data loss
    loss_data = loss_data_from_model(model, t_data_gpu, y_real_gpu, i_real_gpu)
    
    # Calculate physics loss in mini-batches to avoid GPU OOM
    batch_size = 100  # Smaller batches for GTX 1060 (3GB VRAM)
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
    
    return loss_total.item()


def phase1_physics_search(
    maxiter: int = 100,
    popsize: int = 30
) -> tuple[np.ndarray, float]:
    """
    Phase 1: Optimize only physical parameters with DE (3D problem).
    """
    print("\n" + "="*80)
    print("PHASE 1: PHYSICS-ONLY SEARCH (Differential Evolution, 3D)")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Dimensions: 3 (log_k0, log_k, log_a)")
    print(f"  - Max iterations: {maxiter}")
    print(f"  - Population size: {popsize}")
    print(f"  - KAN weights: FROZEN")
    print()
    
    # Define bounds for physical parameters
    log_k0_init = np.log(36.3e-3)
    log_k_init = np.log(3.5e-3)
    log_a_init = np.log(5.2e-3)
    
    bounds = [
        (log_k0_init - 3.0, log_k0_init + 3.0),
        (log_k_init - 3.0, log_k_init + 3.0),
        (log_a_init - 3.0, log_a_init + 3.0)
    ]
    
    print(f"Bounds:")
    print(f"  log_k0: [{bounds[0][0]:.3f}, {bounds[0][1]:.3f}]")
    print(f"  log_k:  [{bounds[1][0]:.3f}, {bounds[1][1]:.3f}]")
    print(f"  log_a:  [{bounds[2][0]:.3f}, {bounds[2][1]:.3f}]")
    print()
    
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
    
    # Show found parameters
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
    num_epochs: int = 1000,
    lr: float = 5e-5
) -> tuple[float, list[float]]:
    """
    Phase 2: Refine ALL parameters (KAN + physics) with Adam.
    """
    print("\n" + "="*80)
    print("PHASE 2: FULL REFINEMENT (Adam on all parameters)")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Parameters: ALL (KAN weights + physics)")
    print()
    
    # Load best physics parameters
    with torch.no_grad():
        model.log_k0.data = torch.tensor(best_physics[0], device=device)
        model.log_k.data = torch.tensor(best_physics[1], device=device)
        model.log_a.data = torch.tensor(best_physics[2], device=device)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_history = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        loss_data = loss_data_from_model(model, t_data_gpu, y_real_gpu, i_real_gpu)
        
        # Physics loss in mini-batches to avoid OOM
        batch_size = 100
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
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            k0, k, a = model.get_phys_params()
            print(f"Epoch {epoch+1:4d} | Loss: {loss_value:.6e} | "
                  f"k0={k0.item():.3e}, k={k.item():.3e}, a={a.item():.3e}")
    
    elapsed = time.time() - start_time
    final_loss = loss_history[-1]
    
    print(f"\nPhase 2 completed in {elapsed:.1f}s")
    print(f"Loss improvement: {loss_history[0]:.6e} → {final_loss:.6e}")
    
    return final_loss, loss_history


def main():
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  KAN-PINN PHYSICS-ONLY HYBRID OPTIMIZATION".center(78) + "#")
    print("#" + "  (DE on physics → Adam on all)".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    save_dir = Path("runs_physics_hybrid")
    save_dir.mkdir(exist_ok=True)
    
    # Phase 1: Optimize only physics with DE
    best_physics, loss_phase1 = phase1_physics_search(
        maxiter=100,
        popsize=30
    )
    
    np.save(save_dir / "phase1_best_physics.npy", best_physics)
    
    # Phase 2: Refine everything with Adam
    loss_final, adam_history = phase2_full_refinement(
        best_physics=best_physics,
        num_epochs=1000,
        lr=5e-5
    )
    
    np.save(save_dir / "phase2_loss_history.npy", np.array(adam_history))
    
    # Save final model
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
            'hidden': 64,
            'depth': 3,
            'num_knots': 8
        }
    }
    
    torch.save(checkpoint, save_dir / "best_model_physics_hybrid.pt")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Phase 1 (DE physics) loss:  {loss_phase1:.6e}")
    print(f"Phase 2 (Adam all) loss:    {loss_final:.6e}")
    print(f"Total improvement:          {loss_phase1 - loss_final:.6e}")
    print()
    print("Final Physical Parameters:")
    print(f"  k0 = {k0_final.item():.6e}")
    print(f"  k  = {k_final.item():.6e}")
    print(f"  a  = {a_final.item():.6e}")
    print()
    print(f"Saved to: {save_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
