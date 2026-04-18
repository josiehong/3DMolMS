from decimal import Decimal

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def make_idx_base(batch_size, num_points, device):
    return torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points


# ---------------------------------------------------------------------------
# Inference steps  (batch_size == 1, used by MolNet.pred_*)
# ---------------------------------------------------------------------------

def pred_step(model, device, loader, batch_size, num_points):
    assert batch_size == 1, "batch_size should be 1 for prediction"
    model.eval()
    id_list, pred_list = [], []

    with tqdm(total=len(loader), desc="Predict") as bar:
        for batch in loader:
            ids, x, mask, env = batch
            x    = x.to(device=device, dtype=torch.float).permute(0, 2, 1)
            mask = mask.to(device=device)
            env  = env.to(device=device, dtype=torch.float)
            idx_base = make_idx_base(batch_size, num_points, device)

            with torch.no_grad():
                pred = model(x, mask, env, idx_base)
                pred = pred / torch.max(pred)

            pred = torch.pow(pred, 2)
            pred = pred.detach().cpu().apply_(lambda v: v if v > 0.01 else 0)

            id_list += list(ids)
            pred_list.append(pred)
            bar.update(1)

    return id_list, torch.cat(pred_list, dim=0)


def eval_step_oth(model, device, loader, batch_size, num_points):
    assert batch_size == 1, "batch_size should be 1 for prediction"
    model.eval()
    id_list, pred_list = [], []

    with tqdm(total=len(loader), desc="Predict") as bar:
        for batch in loader:
            ids, x, mask, env = batch
            x    = x.to(device=device, dtype=torch.float).permute(0, 2, 1)
            mask = mask.to(device=device)
            env  = env.to(device=device, dtype=torch.float)
            env  = env[:, 1:]  # remove collision energy
            idx_base = make_idx_base(batch_size, num_points, device)

            with torch.no_grad():
                pred = model(x, mask, env, idx_base)

            id_list += list(ids)
            pred_list.append(pred)
            bar.update(1)

    return id_list, torch.cat(pred_list, dim=0)


def pred_feat(model, device, loader, batch_size, num_points):
    assert batch_size == 1, "batch_size should be 1 for prediction"
    model.eval()
    id_list, pred_list = [], []

    with tqdm(total=len(loader), desc="Features") as bar:
        for batch in loader:
            ids, x, mask, _ = batch
            x    = x.to(device=device, dtype=torch.float).permute(0, 2, 1)
            mask = mask.to(device=device)
            idx_base = make_idx_base(batch_size, num_points, device)

            with torch.no_grad():
                pred = model(x, mask, idx_base)

            id_list += list(ids)
            pred_list.append(pred)
            bar.update(1)

    return id_list, torch.cat(pred_list, dim=0)


# ---------------------------------------------------------------------------
# Training steps  (used by MolNet.train)
# ---------------------------------------------------------------------------

def _unpack_train_batch(batch, has_env, device):
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
    """One full training epoch. Returns the average metric for the epoch."""
    model.train()
    has_env = task in ("msms", "ccs")
    cos = nn.CosineSimilarity(dim=1)
    metric_sum = 0.0

    with tqdm(total=len(loader), desc="Train") as bar:
        for step, batch in enumerate(loader):
            x, mask, y, env = _unpack_train_batch(batch, has_env, device)
            idx_base = make_idx_base(batch_size, num_points, device)

            optimizer.zero_grad()
            pred = model(x, mask, env, idx_base)

            if task == "msms":
                loss = torch.mean(1 - cos(pred, y))
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    pred_max = pred.max(dim=1, keepdim=True).values.clamp(min=1e-8)
                    pred_m = pred / pred_max
                    pred_m = pred_m.cpu().apply_(lambda v: v if v > 0.01 else 0).to(device)
                    metric_sum += F.cosine_similarity(
                        torch.pow(pred_m, 2), torch.pow(y, 2), dim=1
                    ).mean().item()
            else:
                y_scaled = model.scale(y) if model.scaler is not None else y
                loss = nn.MSELoss()(pred, y_scaled)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    pred_u = model.unscale(pred) if model.scaler is not None else pred
                    metric_sum += torch.abs(pred_u - y).mean().item()

            bar.set_postfix(lr=f"{get_lr(optimizer):.2e}", loss=f"{loss.item():.4f}")
            bar.update(1)

    return metric_sum / (step + 1)


def eval_step(model, device, loader, batch_size, num_points, task):
    """One full validation epoch. Returns the average metric for the epoch."""
    model.eval()
    has_env = task in ("msms", "ccs")
    metric_sum = 0.0

    with tqdm(total=len(loader), desc="Eval") as bar:
        for step, batch in enumerate(loader):
            x, mask, y, env = _unpack_train_batch(batch, has_env, device)
            idx_base = make_idx_base(batch_size, num_points, device)

            with torch.no_grad():
                pred = model(x, mask, env, idx_base)

                if task == "msms":
                    pred_max = pred.max(dim=1, keepdim=True).values.clamp(min=1e-8)
                    pred = pred / pred_max
                    pred = pred.cpu().apply_(lambda v: v if v > 0.01 else 0).to(device)
                    metric_sum += F.cosine_similarity(
                        torch.pow(pred, 2), torch.pow(y, 2), dim=1
                    ).mean().item()
                else:
                    pred_u = model.unscale(pred) if model.scaler is not None else pred
                    metric_sum += torch.abs(pred_u - y).mean().item()

            bar.update(1)

    return metric_sum / (step + 1)


def collect_targets(loader):
    """Collect all training targets; used to fit the output scaler for RT."""
    all_targets = []
    for batch in loader:
        _, _, _, y = batch
        all_targets.append(y.cpu().numpy())
    return np.concatenate(all_targets, axis=0).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Evaluation helpers  (used by MolNet.evaluate)
# ---------------------------------------------------------------------------

def bin_spectrum(mz_array, intensity_array, resolution=0.2, max_mz=1500):
    """Bin a sparse (mz, intensity) spectrum into a fixed-length vector."""
    res    = Decimal(str(resolution))
    max_mz = Decimal(str(max_mz))
    n_bins = int(max_mz // res)
    binned = [0.0] * n_bins

    for mz, intensity in zip(mz_array, intensity_array):
        idx = int(round(Decimal(str(mz)) // res))
        if 0 <= idx < n_bins:
            binned[idx] += intensity

    return binned if sum(binned) > 0 else None


def cosine_similarity(vec1, vec2):
    a, b  = np.array(vec1), np.array(vec2)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else None
