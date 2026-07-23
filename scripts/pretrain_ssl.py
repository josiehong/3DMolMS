"""Self-supervised 3D pretraining of the 3DMolMS encoder (single GPU).

SSL task — Masked distance reconstruction (MolNet_SSL):
  Randomly mask `mask_ratio` (default 15%) of atoms per molecule by zeroing
  all 21 feature dimensions, including the xyz coordinates in dims 0-2. For
  each (masked, unmasked) atom pair, predict the Euclidean distance between
  the two atoms from per-atom encoder features. Loss: MSE on raw distances (Å).

  Zeroing xyz stops the encoder from reading masked coordinates directly, so
  it must infer their 3D positions from the unmasked chemical context. Because
  MolConv2 is E(3)-invariant and Euclidean distances are too, the encoder-head
  mapping is consistent regardless of molecular orientation.

The input pkls are produced by scripts/chembl2pkl.py (or any pkl of the form
  {"title": str, "smiles": str, "mol": np.ndarray[max_atom_num, 21],
   "mask": np.ndarray[max_atom_num] bool}).

After pretraining, transfer the encoder into a downstream model:
  MolNet(...).train(..., resume_path='./check_point/molnet_pre_ssl.pt', transfer=True)

Usage:
  python scripts/pretrain_ssl.py \\
      --train_data ./data/chembl_ssl_train.pkl \\
      --valid_data ./data/chembl_ssl_valid.pkl \\
      --config     ./molnetpack/config/molnet_pre_ssl.yml \\
      --checkpoint ./check_point/molnet_pre_ssl.pt \\
      --device 0
"""

import os
import argparse

import numpy as np
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from molnetpack import __version__
from molnetpack.model import MolNet_SSL
from molnetpack.dataset import MolSSL_Dataset
from molnetpack.utils import pretrain_ssl_step, eval_ssl_step, get_lr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-supervised 3D pretraining of the 3DMolMS encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_data", type=str, default="./data/chembl_ssl_train.pkl",
                        help="ChEMBL train pkl (produced by chembl2pkl.py).")
    parser.add_argument("--valid_data", type=str, default="./data/chembl_ssl_valid.pkl",
                        help="ChEMBL validation pkl.")
    parser.add_argument("--config", type=str, default="./molnetpack/config/molnet_pre_ssl.yml",
                        help="SSL pretraining config YAML.")
    parser.add_argument("--checkpoint", type=str, default="./check_point/molnet_pre_ssl.pt",
                        help="Path to save the best checkpoint.")
    parser.add_argument("--resume_path", type=str, default="",
                        help="Resume training from an existing checkpoint.")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device index, or 'cpu'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override num_workers from config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    device = torch.device(f"cuda:{int(args.device)}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    print(f"Device: {device}")

    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    model_cfg, train_cfg = config["model"], config["train"]
    print(f"Loaded config from {args.config}")

    batch_size = train_cfg["batch_size"]
    num_points = model_cfg["max_atom_num"]
    mask_ratio = train_cfg.get("mask_ratio", 0.15)
    num_pairs = train_cfg.get("num_pairs", 128)
    early_stop_lim = train_cfg.get("early_stop_patience", 15)
    num_workers = args.num_workers if args.num_workers is not None \
        else train_cfg.get("num_workers", 4)

    # --- data ---
    train_set = MolSSL_Dataset(args.train_data, mask_ratio=mask_ratio, num_pairs=num_pairs)
    valid_set = MolSSL_Dataset(args.valid_data, mask_ratio=mask_ratio, num_pairs=num_pairs)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)

    # --- model ---
    model = MolNet_SSL(model_cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"MolNet_SSL  #params: {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                            weight_decay=train_cfg.get("weight_decay", 0.01))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                     factor=0.5, patience=5)

    best_val_loss = float("inf")
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {args.resume_path}  (best_val_loss={best_val_loss:.4f})")

    if args.checkpoint:
        os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)

    # --- training loop ---
    early_stop_patience = 0
    for epoch in range(1, train_cfg["epochs"] + 1):
        print(f"\n===== Epoch {epoch} =====")
        train_loss, train_mae = pretrain_ssl_step(
            model, device, train_loader, optimizer,
            batch_size=batch_size, num_points=num_points)
        val_loss, val_mae = eval_ssl_step(
            model, device, valid_loader,
            batch_size=batch_size, num_points=num_points)
        print(f"Train  mse={train_loss:.4f}  MAE={train_mae:.3f} Å")
        print(f"Valid  mse={val_loss:.4f}  MAE={val_mae:.3f} Å")

        scheduler.step(val_loss)
        print(f"LR: {get_lr(optimizer):.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_patience = 0
            if args.checkpoint:
                torch.save({
                    "version": __version__,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "num_params": num_params,
                    "config": config,
                }, args.checkpoint)
                print(f"Saved checkpoint → {args.checkpoint}")
        else:
            early_stop_patience += 1
            print(f"Early-stop count: {early_stop_patience}/{early_stop_lim}  "
                  f"(best val_mse={best_val_loss:.4f})")
            if early_stop_patience >= early_stop_lim:
                print("Early stopping triggered.")
                break

    print(f"\nDone.  Best validation loss: {best_val_loss:.4f}")
    print("Transfer into a downstream model with:\n"
          f"  MolNet(...).train(..., resume_path='{args.checkpoint}', transfer=True)")


if __name__ == "__main__":
    main()
