from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MonitDataset, concat_datasets, build_datasets
from model import PINNLevitator
from losses import get_residuals_from_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze the physics residuals of a trained PINN model.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to the PINN model checkpoint (.pt file)")
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

    # --- Load Dataset ---
    ds = MonitDataset(train_args.data[0]) if len(train_args.data) == 1 else concat_datasets(train_args.data)
    _, val_ds = build_datasets(ds, train_args.val_split, train_args.seed)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False) # Use a single batch

    # --- Load Model ---
    model = PINNLevitator(
        y_mu=ckpt["stats"]["y_mu"], y_std=ckpt["stats"]["y_std"],
        i_mu=ckpt["stats"]["i_mu"], i_std=ckpt["stats"]["i_std"],
        hidden=train_args.hidden, depth=train_args.depth
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # --- Get Residuals ---
    t_val, _, _, u_val = next(iter(val_loader))
    t_val, u_val = t_val.to(device), u_val.to(device)

    res_mech, res_elect = get_residuals_from_model(model, t_val, u_val)

    res_mech_np = res_mech.detach().cpu().numpy().squeeze()
    res_elect_np = res_elect.detach().cpu().numpy().squeeze()
    t_val_np = t_val.detach().cpu().numpy().squeeze()

    # --- Plot Results ---
    save_dir = ckpt_path.parent
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Sort by time for clean plotting
    sort_indices = np.argsort(t_val_np)
    t_plot = t_val_np[sort_indices]
    res_mech_plot = res_mech_np[sort_indices]
    res_elect_plot = res_elect_np[sort_indices]

    axs[0].plot(t_plot, res_mech_plot, label="Residual Mecánico", color="C0")
    axs[0].set_ylabel("$R_{\text{mech}}$")
    axs[0].axhline(0, color='k', linestyle='--', alpha=0.5)
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t_plot, res_elect_plot, label="Residual Eléctrico", color="C1")
    axs[1].set_xlabel("Tiempo (t)")
    axs[1].set_ylabel("$R_{\text{elect}}$")
    axs[1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    fig.suptitle("Análisis de Residuales Físicos en el Conjunto de Validación")
    fig.tight_layout()
    plot_filename = ckpt_path.stem + "_residuals.png"
    fig_out = save_dir / plot_filename
    fig.savefig(fig_out, dpi=150)
    print(f"Saved residuals plot to: {fig_out}")
    plt.close(fig)

if __name__ == "__main__":
    main()
