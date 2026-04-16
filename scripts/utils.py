"""Shared utilities for training and prediction scripts."""

import numpy as np
import torch


def init_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def setup_device(device_id, no_cuda):
    if torch.cuda.is_available() and not no_cuda:
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def make_idx_base(batch_size, num_points, device):
    return torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points


def load_checkpoint(model, resume_path, device, optimizer=None, scheduler=None, transfer=False):
    """Load weights from a checkpoint file.

    If transfer=True, loads only the encoder weights and freezes them.
    Otherwise loads full model + optimizer + scheduler state.
    Returns the best validation metric stored in the checkpoint, or None.
    """
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    if transfer:
        encoder_dict = {
            k: v for k, v in ckpt["model_state_dict"].items()
            if not k.startswith("decoder")
        }
        for v in encoder_dict.values():
            v.requires_grad = False
        model.load_state_dict(encoder_dict, strict=False)
        return None
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt.get("best_val_acc") or ckpt.get("best_val_mae")


def save_checkpoint(path, model, optimizer, scheduler, epoch, version, num_params, **metrics):
    """Save a training checkpoint."""
    torch.save(
        {
            "version": version,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "num_params": num_params,
            **metrics,
        },
        path,
    )
