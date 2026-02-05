from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import MonitDataset, concat_datasets
from model import PINNLevitator


def build_datasets(ds, val_split: float, seed: int) -> Tuple[Subset, Subset]:
    """Recreate the exact same train/val splits as in training."""
    n = len(ds)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(n * val_split)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return Subset(ds, train_idx.tolist()), Subset(ds, val_idx.tolist())


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a trained model on the validation set and compute metrics.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint (.pt file)")
    p.add_argument("--plot-limit", type=int, default=1000, help="Number of points to plot for the prediction graph.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")

    # --- Load Checkpoint and Training Args ---
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    train_args = argparse.Namespace(**ckpt["args"])
    print(f"Loaded model trained with the following args:\n{train_args}")

    # --- Load Dataset ---
    ds = MonitDataset(train_args.data[0]) if len(train_args.data) == 1 else concat_datasets(train_args.data)
    _, val_ds = build_datasets(ds, train_args.val_split, train_args.seed)
    val_loader = DataLoader(val_ds, batch_size=train_args.batch_size, shuffle=False)

    # --- Load Model ---
    model = PINNLevitator(
        y_mu=ckpt["stats"]["y_mu"], y_std=ckpt["stats"]["y_std"],
        i_mu=ckpt["stats"]["i_mu"], i_std=ckpt["stats"]["i_std"],
        hidden=train_args.hidden, depth=train_args.depth
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # --- Get Predictions ---
    y_preds, i_preds = [], []
    y_true, i_true, t_true = [], [], []
    with torch.no_grad():
        for t, y, i, _ in val_loader:  # Unpack u and ignore it
            t = t.to(device)
            y_hat, i_hat = model(t)
            y_preds.append(y_hat.cpu().numpy())
            i_preds.append(i_hat.cpu().numpy())
            y_true.append(y.cpu().numpy())
            i_true.append(i.cpu().numpy())
            t_true.append(t.cpu().numpy())

    y_pred_np = np.concatenate(y_preds)
    i_pred_np = np.concatenate(i_preds)
    y_true_np = np.concatenate(y_true)
    i_true_np = np.concatenate(i_true)
    t_true_np = np.concatenate(t_true)

    # --- Calculate and Print Metrics ---
    rmse_y = rmse_score(y_true_np, y_pred_np)
    rmse_i = rmse_score(i_true_np, i_pred_np)
    r2_y = r2_score(y_true_np, y_pred_np)
    r2_i = r2_score(i_true_np, i_pred_np)

    print("\n--- Validation Metrics ---")
    print(f"Position (y) | RMSE: {rmse_y:.4f} | R2: {r2_y:.4f}")
    print(f"Current  (i) | RMSE: {rmse_i:.4f} | R2: {r2_i:.4f}")
    print("------------------------\n")

    # --- Plot Results ---
    save_dir = ckpt_path.parent
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Sort by time for clean plotting
    sort_indices = np.argsort(t_true_np.squeeze())
    t_plot = t_true_np[sort_indices][:args.plot_limit]
    y_true_plot = y_true_np[sort_indices][:args.plot_limit]
    y_pred_plot = y_pred_np[sort_indices][:args.plot_limit]
    i_true_plot = i_true_np[sort_indices][:args.plot_limit]
    i_pred_plot = i_pred_np[sort_indices][:args.plot_limit]

    axs[0].plot(t_plot, y_true_plot, label="Real", alpha=0.7)
    axs[0].plot(t_plot, y_pred_plot, label="Predicción", linestyle="--")
    axs[0].set_ylabel("Posición (y)")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t_plot, i_true_plot, label="Real", alpha=0.7)
    axs[1].plot(t_plot, i_pred_plot, label="Predicción", linestyle="--")
    axs[1].set_xlabel("Tiempo (t)")
    axs[1].set_ylabel("Corriente (i)")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    fig.suptitle("Predicciones del Modelo vs. Valores Reales (Validación)")
    fig.tight_layout()
    
    # Create a unique filename for the plot based on the checkpoint name
    plot_filename = ckpt_path.stem + "_predictions.png"
    fig_out = save_dir / plot_filename
    fig.savefig(fig_out, dpi=150)
    print(f"Saved prediction plot to: {fig_out}")
    plt.close(fig)

if __name__ == "__main__":
    main()
