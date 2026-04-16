"""Unified training script for all 3DMolMS tasks.

Usage:
    python scripts/train.py --task msms [options]
    python scripts/train.py --task rt   [options]
    python scripts/train.py --task ccs  [options]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import setup_device

from molnetpack import MolNet


def parse_args():
    parser = argparse.ArgumentParser(description="3DMolMS unified training script")
    parser.add_argument("--task", required=True, choices=["msms", "rt", "ccs"],
                        help="Training task")
    parser.add_argument("--train_data",      type=str, required=True,
                        help="Path to training data (.pkl)")
    parser.add_argument("--test_data",       type=str, required=True,
                        help="Path to validation data (.pkl)")
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="Path to save the best checkpoint")
    parser.add_argument("--resume_path",     type=str, default="",
                        help="Path to a checkpoint to resume from")
    parser.add_argument("--transfer", action="store_true",
                        help="Load only encoder weights from --resume_path (transfer learning)")
    parser.add_argument("--precursor_type",  type=str, default="All",
                        choices=["All", "[M+H]+", "[M-H]-"],
                        help="Filter training data by precursor type (msms task only)")
    parser.add_argument("--use_scaler", action="store_true",
                        help="Fit a StandardScaler on training targets (rt task only)")
    # TorchScript export (msms only, runs after training)
    parser.add_argument("--ex_model_path",   type=str, default="",
                        help="Export traced TorchScript model to this path (msms only)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--device",  type=int, default=0, help="GPU index to use")
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = setup_device(args.device, args.no_cuda)
    print(f"Task: {args.task}  |  Device: {device}")

    molnet = MolNet(device=device, seed=args.seed)
    molnet.train(
        task=args.task,
        train_data=args.train_data,
        valid_data=args.test_data,
        checkpoint_path=args.checkpoint_path,
        resume_path=args.resume_path,
        transfer=args.transfer,
        precursor_type=args.precursor_type,
        use_scaler=args.use_scaler,
    )

    # Optional TorchScript export (msms only)
    if args.ex_model_path:
        if args.task != "msms":
            raise NotImplementedError("TorchScript export is only implemented for the msms task.")
        config = molnet.msms_config
        batch_size = config["train"]["batch_size"]
        num_points = config["model"]["max_atom_num"]
        model = molnet.msms_model
        model.eval()
        x        = torch.randn(batch_size, 21, num_points, device=device)
        mask     = torch.ones(batch_size, num_points, device=device, dtype=torch.bool)
        env      = torch.randn(batch_size, 6, device=device)
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        traced   = torch.jit.trace(model, (x, mask, env, idx_base))
        torch.jit.save(traced, args.ex_model_path)
        print(f"Saved traced model to {args.ex_model_path}")


if __name__ == "__main__":
    main()
