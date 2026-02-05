from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from dataset import MonitDataset, concat_datasets, build_datasets
from model import TimeMLP
from losses import loss_data_from_model




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NN(t)->(y_hat,i_hat) minimizing Loss_data on MONIT-like TXT")
    p.add_argument("--data", nargs="+", required=True, help="One or more MONIT-like txt files to train on")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wy", type=float, default=1.0)
    p.add_argument("--wi", type=float, default=1.0)
    p.add_argument("--standardize", action="store_true", help="Standardize residuals by dataset std in Loss_data")
    p.add_argument("--save-dir", type=str, default=str(Path(__file__).resolve().parent / "runs"))
    return p.parse_args()


def main():
    args = parse_args()

    paths: List[str] = args.data
    for fp in paths:
        if not Path(fp).exists():
            raise FileNotFoundError(fp)

    ds = MonitDataset(paths[0]) if len(paths) == 1 else concat_datasets(paths)

    train_ds, val_ds = build_datasets(ds, args.val_split, args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cpu")

    model = TimeMLP(ds.stats.y_mu, ds.stats.y_std, ds.stats.i_mu, ds.stats.i_std,
                    hidden=args.hidden, depth=args.depth).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    sigma_y = torch.tensor(ds.stats.y_std, dtype=torch.float32, device=device) if args.standardize else None
    sigma_i = torch.tensor(ds.stats.i_std, dtype=torch.float32, device=device) if args.standardize else None

    os.makedirs(args.save_dir, exist_ok=True)
    train_hist: List[float] = []
    val_hist: List[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for t, y, i in train_loader:
            t = t.to(device)
            y = y.to(device)
            i = i.to(device)

            opt.zero_grad(set_to_none=True)
            loss = loss_data_from_model(model, t, y, i,
                                        w_y=args.wy, w_i=args.wi,
                                        sigma_y=sigma_y, sigma_i=sigma_i,
                                        reduction="mean")
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * t.shape[0]
            count += t.shape[0]
        train_loss = total / max(count, 1)

        model.eval()
        with torch.no_grad():
            total = 0.0
            count = 0
            for t, y, i in val_loader:
                t = t.to(device)
                y = y.to(device)
                i = i.to(device)
                loss = loss_data_from_model(model, t, y, i,
                                            w_y=args.wy, w_i=args.wi,
                                            sigma_y=sigma_y, sigma_i=sigma_i,
                                            reduction="mean")
                total += float(loss.cpu()) * t.shape[0]
                count += t.shape[0]
            val_loss = total / max(count, 1)

        train_hist.append(train_loss)
        val_hist.append(val_loss)
        print(f"epoch {epoch:04d} | train {train_loss:.6e} | val {val_loss:.6e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_hist) + 1), train_hist, label="train")
    ax.plot(range(1, len(val_hist) + 1), val_hist, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss_data")
    ax.legend()
    fig.tight_layout()
    fig_out = Path(args.save_dir) / "loss_curve.png"
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)

    ckpt = {
        "state_dict": model.state_dict(),
        "stats": {
            "y_mu": ds.stats.y_mu, "y_std": ds.stats.y_std,
            "i_mu": ds.stats.i_mu, "i_std": ds.stats.i_std,
            "t_mu": ds.stats.t_mu, "t_std": ds.stats.t_std,
            "t_min": ds.stats.t_min, "t_max": ds.stats.t_max,
        },
        "args": vars(args),
    }
    out = Path(args.save_dir) / "data_only.pt"
    torch.save(ckpt, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
