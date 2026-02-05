"""
Validation script for KAN-PINN hybrid model.
Loads the best model from hybrid optimization and evaluates performance.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset import MonitDataset
from kan_model import KANLevitator


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Calculate RMSE and R²."""
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return rmse, r2


def main():
    parser = argparse.ArgumentParser(description='Validate KAN-PINN hybrid model')
    parser.add_argument('--ckpt', type=str, default='runs_hybrid/best_model_hybrid.pt',
                        help='Path to checkpoint file')
    parser.add_argument('--save-dir', type=str, default='runs_hybrid',
                        help='Directory to save validation plots')
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    
    # Extract architecture info
    arch = ckpt['architecture']
    stats = ckpt['stats']
    phys_params = ckpt['physical_params']
    
    print(f"\nModel Architecture:")
    print(f"  Hidden size: {arch['hidden']}")
    print(f"  Depth: {arch['depth']}")
    print(f"  Num knots: {arch['num_knots']}")
    
    print(f"\nPhysical Parameters:")
    print(f"  k0 = {phys_params['k0']:.6e}")
    print(f"  k  = {phys_params['k']:.6e}")
    print(f"  a  = {phys_params['a']:.6e}")
    
    # Recreate model
    model = KANLevitator(
        y_mu=stats['y_mu'],
        y_std=stats['y_std'],
        i_mu=stats['i_mu'],
        i_std=stats['i_std'],
        hidden=arch['hidden'],
        depth=arch['depth'],
        num_knots=arch['num_knots']
    )
    
    # Load weights
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Load dataset
    data_path = "../../levitador valentin/experimentos/datos_levitador_20251024_162706.txt"
    ds = MonitDataset(data_path)
    
    # Predict on full dataset
    with torch.no_grad():
        t_norm = torch.tensor(ds.t_norm, dtype=torch.float32).reshape(-1, 1)
        y_pred, i_pred = model(t_norm)
        y_pred = y_pred.squeeze().numpy()
        i_pred = i_pred.squeeze().numpy()
    
    # Calculate metrics
    rmse_y, r2_y = calculate_metrics(ds.y, y_pred)
    rmse_i, r2_i = calculate_metrics(ds.i, i_pred)
    
    print(f"\n--- Validation Metrics (Full Dataset) ---")
    print(f"Position (y) | RMSE: {rmse_y:.4f} | R²: {r2_y:.4f}")
    print(f"Current  (i) | RMSE: {rmse_i:.4f} | R²: {r2_i:.4f}")
    print("-" * 40)
    
    # Create validation plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Position prediction
    axes[0, 0].plot(ds.t, ds.y, 'b-', label='Real', alpha=0.7, linewidth=1.5)
    axes[0, 0].plot(ds.t, y_pred, 'r--', label='KAN Prediction', alpha=0.8)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].set_title(f'Position Prediction (R²={r2_y:.4f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Current prediction
    axes[0, 1].plot(ds.t, ds.i, 'b-', label='Real', alpha=0.7, linewidth=1.5)
    axes[0, 1].plot(ds.t, i_pred, 'r--', label='KAN Prediction', alpha=0.8)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Current (A)')
    axes[0, 1].set_title(f'Current Prediction (R²={r2_i:.4f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Position residuals
    residuals_y = ds.y - y_pred
    axes[1, 0].scatter(ds.t, residuals_y, alpha=0.5, s=10)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Residual (m)')
    axes[1, 0].set_title(f'Position Residuals (RMSE={rmse_y:.4f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Current residuals
    residuals_i = ds.i - i_pred
    axes[1, 1].scatter(ds.t, residuals_i, alpha=0.5, s=10)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Residual (A)')
    axes[1, 1].set_title(f'Current Residuals (RMSE={rmse_i:.4f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_dir = Path(args.save_dir)
    save_path = save_dir / 'hybrid_validation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nValidation plot saved to: {save_path}")
    
    # Also plot Phase 2 loss history if available
    loss_history_path = save_dir / 'phase2_loss_history.npy'
    if loss_history_path.exists():
        loss_history = np.load(loss_history_path)
        
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Phase 2 (Adam) Loss History')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        loss_plot_path = save_dir / 'phase2_loss_curve.png'
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        print(f"Loss curve saved to: {loss_plot_path}")


if __name__ == "__main__":
    main()
