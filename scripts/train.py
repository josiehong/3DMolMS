"""Unified training script for all 3DMolMS tasks.

Usage:
    python scripts/train.py --task msms [options]
    python scripts/train.py --task rt   [options]
    python scripts/train.py --task ccs  [options]
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    init_random_seed, get_lr, setup_device, make_idx_base,
    load_checkpoint, save_checkpoint,
)

from molnetpack import MolNet_MS, MolNet_Oth
from molnetpack import MolMS_Dataset, MolRT_Dataset, MolCCS_Dataset
from molnetpack import __version__


# ---------------------------------------------------------------------------
# Per-task configuration
# ---------------------------------------------------------------------------

TASK_DEFAULTS = {
    "msms": dict(
        train_data="./data/qtof_etkdgv3_train.pkl",
        test_data="./data/qtof_etkdgv3_test.pkl",
        model_config_path="./config/molnet.yml",
        data_config_path="./config/preprocess_etkdgv3.yml",
        checkpoint_path="./check_point/molnet_qtof_etkdgv3.pt",
    ),
    "rt": dict(
        train_data="./data/metlin_etkdgv3_train.pkl",
        test_data="./data/metlin_etkdgv3_test.pkl",
        model_config_path="./config/molnet_rt.yml",
        data_config_path="./config/preprocess_etkdgv3.yml",
        checkpoint_path="./check_point/molnet_rt_etkdgv3.pt",
    ),
    "ccs": dict(
        train_data="./data/allccs_etkdgv3_train.pkl",
        test_data="./data/allccs_etkdgv3_test.pkl",
        model_config_path="./config/molnet_ccs.yml",
        data_config_path="./config/preprocess_etkdgv3.yml",
        checkpoint_path="./check_point/molnet_ccs_etkdgv3.pt",
    ),
}

# Whether each task's batch includes an environment tensor
HAS_ENV = {"msms": True, "rt": False, "ccs": True}

SCHEDULER_MODE      = {"msms": "max", "rt": "min", "ccs": "min"}
SCHEDULER_PATIENCE  = {"msms": 5,     "rt": 20,    "ccs": 20}
EARLY_STOP_PATIENCE = {"msms": 10,    "rt": 60,    "ccs": 60}
BEST_INIT           = {"msms": 0.0,   "rt": float("inf"), "ccs": float("inf")}
BEST_KEY            = {"msms": "best_val_acc", "rt": "best_val_mae", "ccs": "best_val_mae"}


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def _unpack_batch(batch, has_env, device):
    if has_env:
        _, x, mask, y, env = batch
        env = env.to(device=device, dtype=torch.float)
    else:
        _, x, mask, y = batch
        env = None
    x    = x.to(device=device, dtype=torch.float).permute(0, 2, 1)
    mask = mask.to(device=device)
    y    = y.to(device=device, dtype=torch.float)
    return x, mask, y, env


def train_step(model, device, loader, optimizer, batch_size, num_points, task):
    model.train()
    has_env = HAS_ENV[task]
    cos = nn.CosineSimilarity(dim=1)
    metric_sum = 0

    with tqdm(total=len(loader), desc="Train") as bar:
        for step, batch in enumerate(loader):
            x, mask, y, env = _unpack_batch(batch, has_env, device)
            idx_base = make_idx_base(batch_size, num_points, device)

            optimizer.zero_grad()
            pred = model(x, mask, env, idx_base)

            if task == "msms":
                loss = torch.mean(1 - cos(pred, y))
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    metric_sum += F.cosine_similarity(
                        torch.pow(pred, 2), torch.pow(y, 2), dim=1
                    ).mean().item()
            else:
                y_scaled = model.scale(y) if model.scaler is not None else y
                loss = nn.MSELoss()(pred, y_scaled)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    pred_u = model.unscale(pred) if model.scaler is not None else pred
                    metric_sum += torch.abs(pred_u - y).mean().item()

            bar.set_postfix(lr=get_lr(optimizer), loss=f"{loss.item():.4f}")
            bar.update(1)

    return metric_sum / (step + 1)


def eval_step(model, device, loader, batch_size, num_points, task):
    model.eval()
    has_env = HAS_ENV[task]
    metric_sum = 0

    with tqdm(total=len(loader), desc="Eval") as bar:
        for step, batch in enumerate(loader):
            x, mask, y, env = _unpack_batch(batch, has_env, device)
            idx_base = make_idx_base(batch_size, num_points, device)

            with torch.no_grad():
                pred = model(x, mask, env, idx_base)

                if task == "msms":
                    pred = pred / torch.max(pred)
                    pred = pred.cpu().apply_(lambda v: v if v > 0.01 else 0).to(device)
                    metric_sum += F.cosine_similarity(
                        torch.pow(pred, 2), torch.pow(y, 2), dim=1
                    ).mean().item()
                else:
                    pred_u = model.unscale(pred) if model.scaler is not None else pred
                    metric_sum += torch.abs(pred_u - y).mean().item()

            bar.update(1)

    return metric_sum / (step + 1)


def collect_training_targets(loader):
    """Collect all training targets for fitting the output scaler (RT task)."""
    all_targets = []
    for batch in loader:
        _, _, _, y = batch
        all_targets.append(y.cpu().numpy())
    return np.concatenate(all_targets, axis=0).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="3DMolMS unified training script")
    parser.add_argument("--task", required=True, choices=["msms", "rt", "ccs"],
                        help="Training task")
    parser.add_argument("--train_data",        type=str, default="",
                        help="Path to training data (.pkl). Overrides task default.")
    parser.add_argument("--test_data",         type=str, default="",
                        help="Path to test/validation data (.pkl). Overrides task default.")
    parser.add_argument("--model_config_path", type=str, default="",
                        help="Path to model YAML config. Overrides task default.")
    parser.add_argument("--data_config_path",  type=str, default="",
                        help="Path to data YAML config. Overrides task default.")
    parser.add_argument("--checkpoint_path",   type=str, default="",
                        help="Path to save best checkpoint. Overrides task default.")
    parser.add_argument("--resume_path",       type=str, default="",
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--transfer", action="store_true",
                        help="Load only encoder weights from --resume_path (transfer learning).")
    parser.add_argument("--ex_model_path",     type=str, default="",
                        help="Export traced TorchScript model to this path (msms only).")
    # msms-specific
    parser.add_argument("--precursor_type",    type=str, default="All",
                        choices=["All", "[M+H]+", "[M-H]-"],
                        help="Filter training data by precursor type (msms task only).")
    # rt-specific
    parser.add_argument("--use_scaler", action="store_true",
                        help="Fit an output StandardScaler on training targets (rt task only).")
    # general
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--device",   type=int, default=0,
                        help="GPU index to use.")
    parser.add_argument("--no_cuda",  action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply task-specific defaults for any paths not explicitly provided
    defaults = TASK_DEFAULTS[args.task]
    for attr, default_val in defaults.items():
        if not getattr(args, attr):
            setattr(args, attr, default_val)

    init_random_seed(args.seed)
    device = setup_device(args.device, args.no_cuda)
    print(f"Task: {args.task}  |  Device: {device}")

    # Load configs
    with open(args.model_config_path) as f:
        config = yaml.safe_load(f)
    with open(args.data_config_path) as f:
        data_config = yaml.safe_load(f)
    assert config["model"]["batch_size"] == config["train"]["batch_size"], (
        "batch_size must match in model and train sections of the model config"
    )

    batch_size = config["train"]["batch_size"]
    num_points = config["model"]["max_atom_num"]

    # ------------------------------------------------------------------
    # 1. Datasets
    # ------------------------------------------------------------------
    if args.task == "msms":
        # Build a map from precursor-type string → encoded string used by the dataset
        precursor_encoder = {
            k: ",".join(str(int(i)) for i in v)
            for k, v in data_config["encoding"]["precursor_type"].items()
        }
        precursor_encoder["All"] = False
        encoded_type = precursor_encoder[args.precursor_type]
        train_set = MolMS_Dataset(args.train_data, encoded_type)
        valid_set = MolMS_Dataset(args.test_data,  encoded_type)
    elif args.task == "rt":
        train_set = MolRT_Dataset(args.train_data)
        valid_set = MolRT_Dataset(args.test_data)
    else:  # ccs
        train_set = MolCCS_Dataset(args.train_data)
        valid_set = MolCCS_Dataset(args.test_data)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=config["train"]["num_workers"], drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                              num_workers=config["train"]["num_workers"], drop_last=True)

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    ModelCls = MolNet_MS if args.task == "msms" else MolNet_Oth
    model = ModelCls(config["model"]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{model.__class__.__name__}  #params: {num_params:,}")

    # ------------------------------------------------------------------
    # 3. Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=config["train"]["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=SCHEDULER_MODE[args.task],
        factor=0.5,
        patience=SCHEDULER_PATIENCE[args.task],
    )

    # Resume / transfer-learn
    if args.resume_path:
        if args.transfer:
            print("Loading pretrained encoder weights (frozen)...")
            load_checkpoint(model, args.resume_path, device, transfer=True)
            if args.use_scaler and args.task == "rt":
                targets = collect_training_targets(train_loader)
                model.fit_scaler(targets)
        else:
            print("Resuming from checkpoint...")
            load_checkpoint(model, args.resume_path, device, optimizer, scheduler)
            if args.task == "rt":
                ckpt = torch.load(args.resume_path, map_location=device, weights_only=False)
                model.set_scaler(ckpt.get("scaler"))
    elif args.use_scaler and args.task == "rt":
        targets = collect_training_targets(train_loader)
        model.fit_scaler(targets)

    if args.checkpoint_path:
        os.makedirs(os.path.dirname(args.checkpoint_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    best_metric = BEST_INIT[args.task]
    early_stop_patience = 0
    early_stop_limit = EARLY_STOP_PATIENCE[args.task]
    higher_is_better = SCHEDULER_MODE[args.task] == "max"

    for epoch in range(1, config["train"]["epochs"] + 1):
        print(f"\n===== Epoch {epoch}")
        train_metric = train_step(model, device, train_loader, optimizer,
                                  batch_size, num_points, args.task)
        valid_metric = eval_step(model, device, valid_loader,
                                 batch_size, num_points, args.task)

        label = "cosine" if args.task == "msms" else "MAE"
        print(f"Train {label}: {train_metric:.4f}  |  Valid {label}: {valid_metric:.4f}")

        improved = valid_metric > best_metric if higher_is_better else valid_metric < best_metric
        if improved:
            best_metric = valid_metric
            early_stop_patience = 0
            print("Early stop patience reset")
            if args.checkpoint_path:
                print("Saving checkpoint...")
                extra = {"scaler": model.scaler} if args.task == "rt" else {}
                save_checkpoint(
                    args.checkpoint_path, model, optimizer, scheduler,
                    epoch, __version__, num_params,
                    **{BEST_KEY[args.task]: best_metric},
                    **extra,
                )
        else:
            early_stop_patience += 1
            print(f"Early stop count: {early_stop_patience}/{early_stop_limit}")

        scheduler.step(valid_metric)
        print(f"Best {label} so far: {best_metric:.4f}")

        if early_stop_patience >= early_stop_limit:
            print("Early stop!")
            break

    # ------------------------------------------------------------------
    # 5. Optional: export TorchScript model
    # ------------------------------------------------------------------
    if args.ex_model_path:
        if args.task != "msms":
            raise NotImplementedError("TorchScript export is only implemented for the msms task.")
        print("Exporting traced TorchScript model...")
        x       = torch.randn(batch_size, 21, num_points, device=device)
        mask    = torch.ones(batch_size, num_points, device=device, dtype=torch.bool)
        env     = torch.randn(batch_size, 6, device=device)
        idx_base = make_idx_base(batch_size, num_points, device)
        model.eval()
        traced = torch.jit.trace(model, (x, mask, env, idx_base))
        torch.jit.save(traced, args.ex_model_path)
        print(f"Saved traced model to {args.ex_model_path}")


if __name__ == "__main__":
    main()
