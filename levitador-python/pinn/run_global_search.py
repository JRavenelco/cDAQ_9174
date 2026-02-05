"""
Hybrid Global-Local Optimization for KAN-PINN.
Phase 1: Differential Evolution (global search)
Phase 2: Adam refinement (local descent)
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import differential_evolution

# Import from train_meta (uses KAN now)
from train_meta import (
    fitness_function,
    model,
    get_initial_theta,
    set_model_weights_from_theta,
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


def phase1_global_search(
    bounds: list[tuple[float, float]],
    maxiter: int = 50,
    popsize: int = 15,
    workers: int = 1
) -> tuple[np.ndarray, float]:
    """
    Phase 1: Global search using Differential Evolution.
    
    Args:
        bounds: Parameter bounds for each dimension
        maxiter: Number of DE generations
        popsize: Population size per generation
        workers: Number of parallel workers
    
    Returns:
        best_theta: Best parameter vector found
        best_loss: Best loss value achieved
    """
    print("\n" + "="*80)
    print("PHASE 1: GLOBAL SEARCH (Differential Evolution)")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Dimensions: {len(bounds)}")
    print(f"  - Max iterations: {maxiter}")
    print(f"  - Population size: {popsize}")
    print(f"  - Workers: {workers}")
    print(f"  - Strategy: best1bin with high mutation for exploration")
    print()
    
    start_time = time.time()
    
    result = differential_evolution(
        fitness_function,
        bounds,
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        mutation=(0.5, 1.0),  # High mutation for global exploration
        recombination=0.7,
        disp=True,
        workers=workers,
        updating='deferred',  # More thorough exploration
        polish=False  # We'll do our own polishing in Phase 2
    )
    
    elapsed = time.time() - start_time
    
    print(f"\nPhase 1 completed in {elapsed:.1f}s")
    print(f"Best loss found: {result.fun:.6e}")
    print(f"Function evaluations: {result.nfev}")
    
    return result.x, result.fun


def phase2_local_refinement(
    initial_theta: np.ndarray,
    num_epochs: int = 500,
    lr: float = 1e-4
) -> tuple[np.ndarray, float, list[float]]:
    """
    Phase 2: Local refinement using Adam.
    
    Takes the best solution from Phase 1 and fine-tunes it with gradient descent.
    
    Args:
        initial_theta: Starting point (from Phase 1)
        num_epochs: Number of Adam iterations
        lr: Learning rate
    
    Returns:
        final_theta: Refined parameter vector
        final_loss: Final loss value
        loss_history: Loss at each epoch
    """
    print("\n" + "="*80)
    print("PHASE 2: LOCAL REFINEMENT (Adam)")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Starting loss: {fitness_function(initial_theta.tolist()):.6e}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning rate: {lr}")
    print()
    
    # Load initial solution into model
    set_model_weights_from_theta(model, initial_theta.tolist())
    model.train()
    
    # Setup Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_history = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass (with gradients enabled)
        loss_data = loss_data_from_model(model, t_data_gpu, y_real_gpu, i_real_gpu)
        loss_physics = loss_phys_from_model(model, phys_t, phys_u, m=m, g=g, R=R)
        loss_total = LAMBDA_DATA * loss_data + LAMBDA_PHYSICS * loss_physics
        
        # Backward pass
        loss_total.backward()
        optimizer.step()
        
        loss_value = loss_total.item()
        loss_history.append(loss_value)
        
        # Progress reporting
        if (epoch + 1) % 50 == 0 or epoch == 0:
            k0, k, a = model.get_phys_params()
            print(f"Epoch {epoch+1:4d} | Loss: {loss_value:.6e} | "
                  f"k0={k0.item():.3e}, k={k.item():.3e}, a={a.item():.3e}")
    
    elapsed = time.time() - start_time
    
    # Extract final parameters
    final_theta = get_initial_theta(model)
    final_loss = loss_history[-1]
    
    print(f"\nPhase 2 completed in {elapsed:.1f}s")
    print(f"Initial loss: {loss_history[0]:.6e}")
    print(f"Final loss: {final_loss:.6e}")
    print(f"Improvement: {loss_history[0] - final_loss:.6e}")
    
    return np.array(final_theta), final_loss, loss_history


def main():
    """Main hybrid optimization loop."""
    
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  KAN-PINN HYBRID OPTIMIZATION".center(78) + "#")
    print("#" + "  (Global Search → Local Refinement)".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    # Get initial theta and create bounds
    print("\nInitializing...")
    initial_theta = get_initial_theta(model)
    n_params = len(initial_theta)
    print(f"Model has {n_params} trainable parameters")
    
    # Define bounds
    # Generic bounds for KAN spline coefficients and network parameters
    bounds = [(-2.0, 2.0)] * (n_params - 3)
    
    # Specific bounds for physical parameters (last 3 elements: log_k0, log_k, log_a)
    log_k0_init = np.log(36.3e-3)
    log_k_init = np.log(3.5e-3)
    log_a_init = np.log(5.2e-3)
    
    # Tighter bounds around nominal values (±2.0 in log-space)
    bounds.append((log_k0_init - 2.0, log_k0_init + 2.0))
    bounds.append((log_k_init - 2.0, log_k_init + 2.0))
    bounds.append((log_a_init - 2.0, log_a_init + 2.0))
    
    print(f"Bounds defined for {len(bounds)} parameters")
    
    # Create save directory
    save_dir = Path("runs_hybrid")
    save_dir.mkdir(exist_ok=True)
    
    # === PHASE 1: Global Search ===
    best_theta_global, best_loss_global = phase1_global_search(
        bounds=bounds,
        maxiter=50,  # Adjust based on computational budget
        popsize=15,
        workers=1  # Use 1 for Windows compatibility, increase on Linux
    )
    
    # Save Phase 1 results
    np.save(save_dir / "phase1_best_theta.npy", best_theta_global)
    print(f"\nPhase 1 solution saved to: {save_dir / 'phase1_best_theta.npy'}")
    
    # === PHASE 2: Local Refinement ===
    best_theta_final, best_loss_final, adam_history = phase2_local_refinement(
        initial_theta=best_theta_global,
        num_epochs=500,
        lr=1e-4
    )
    
    # Save Phase 2 results
    np.save(save_dir / "phase2_loss_history.npy", np.array(adam_history))
    np.save(save_dir / "hybrid_best_theta.npy", best_theta_final)
    
    # Save final model as PyTorch checkpoint
    set_model_weights_from_theta(model, best_theta_final.tolist())
    k0_final, k_final, a_final = model.get_phys_params()
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'theta': best_theta_final,
        'loss': best_loss_final,
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
    
    torch.save(checkpoint, save_dir / "best_model_hybrid.pt")
    
    # Final summary
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Phase 1 (DE) best loss:  {best_loss_global:.6e}")
    print(f"Phase 2 (Adam) best loss: {best_loss_final:.6e}")
    print(f"Total improvement:        {best_loss_global - best_loss_final:.6e}")
    print()
    print("Final Physical Parameters:")
    print(f"  k0 = {k0_final.item():.6e}")
    print(f"  k  = {k_final.item():.6e}")
    print(f"  a  = {a_final.item():.6e}")
    print()
    print("Saved files:")
    print(f"  - {save_dir / 'phase1_best_theta.npy'}")
    print(f"  - {save_dir / 'phase2_loss_history.npy'}")
    print(f"  - {save_dir / 'hybrid_best_theta.npy'}")
    print(f"  - {save_dir / 'best_model_hybrid.pt'}")
    print("="*80)


if __name__ == "__main__":
    main()
