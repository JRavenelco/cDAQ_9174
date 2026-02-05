from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import MonitDataset, concat_datasets, build_datasets
from model import PINNLevitator
from losses import loss_data_from_model, loss_phys_from_model




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a PINN model for the levitator system.")
    # Data and Model
    p.add_argument("--data", nargs="+", required=True, help="One or more MONIT-like txt files to train on.")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    # Training
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    # Losses
    p.add_argument("--w-data", type=float, default=1.0, help="Weight for data loss.")
    p.add_argument("--w-phys", type=float, default=1e-2, help="Weight for physics loss (lambda).")
    p.add_argument("--n-phys", type=int, default=2000, help="Number of collocation points for physics loss.")
    p.add_argument("--standardize", action="store_true", help="Standardize data loss residuals by dataset std.")
    # Output
    p.add_argument("--save-dir", type=str, default=str(Path(__file__).resolve().parent / "runs_pinn"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")

    # --- Datasets and Loaders ---
    ds = MonitDataset(args.data[0]) if len(args.data) == 1 else concat_datasets(args.data)
    train_ds, val_ds = build_datasets(ds, args.val_split, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # --- Model and Optimizer ---
    model = PINNLevitator(
        ds.stats.y_mu, ds.stats.y_std, ds.stats.i_mu, ds.stats.i_std,
        hidden=args.hidden, depth=args.depth
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Loss Params ---
    sigma_y = torch.tensor(ds.stats.y_std, dtype=torch.float32, device=device) if args.standardize else None
    sigma_i = torch.tensor(ds.stats.i_std, dtype=torch.float32, device=device) if args.standardize else None

    os.makedirs(args.save_dir, exist_ok=True)
    train_hist, phys_hist, val_hist = [], [], []

    # --- Training Loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_train_loss, epoch_phys_loss = 0.0, 0.0
        
        for t, y, i, u in train_loader:
            t, y, i, u = t.to(device), y.to(device), i.to(device), u.to(device)

            # Calculate losses
            loss_d = loss_data_from_model(model, t, y, i, w_y=args.w_data, w_i=args.w_data, sigma_y=sigma_y, sigma_i=sigma_i)
            loss_p = loss_phys_from_model(model, t, u) # m, g, R are defaults
            loss = loss_d + args.w_phys * loss_p

            # Optimization step
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_train_loss += loss_d.detach().item()
            epoch_phys_loss += loss_p.detach().item()

        train_hist.append(epoch_train_loss / len(train_loader))
        phys_hist.append(epoch_phys_loss / len(train_loader))

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_loss = sum(loss_data_from_model(model, t.to(device), y.to(device), i.to(device)) for t, y, i, u in val_loader) / len(val_loader)
        val_hist.append(val_loss.item())

        # Log current physical parameters
        k0_curr, k_curr, a_curr = model.get_phys_params()
        print(f"Epoch {epoch:04d} | Train Data: {train_hist[-1]:.4e} | Phys: {phys_hist[-1]:.4e} | Val Data: {val_hist[-1]:.4e}")
        print(f"    Params: k0={k0_curr.item():.4e}, k={k_curr.item():.4e}, a={a_curr.item():.4e}")

    # --- Save Artifacts ---
    # Plot losses
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_hist, label="Train Data Loss")
    ax.plot(val_hist, label="Val Data Loss")
    ax.plot(phys_hist, label=f"Physics Loss (x{args.w_phys})", linestyle="--", alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_yscale("log")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(Path(args.save_dir) / "pinn_loss_curve.png", dpi=150)
    plt.close(fig)

    # Save checkpoint
    ckpt = {
        "state_dict": model.state_dict(),
        "stats": {
            "y_mu": ds.stats.y_mu, "y_std": ds.stats.y_std,
            "i_mu": ds.stats.i_mu, "i_std": ds.stats.i_std,
            "t_mu": ds.stats.t_mu, "t_std": ds.stats.t_std,
        },
        "args": vars(args),
    }
    torch.save(ckpt, Path(args.save_dir) / "pinn_model.pt")
    print(f"Saved model and plots to {args.save_dir}")

if __name__ == "__main__":
    main()
