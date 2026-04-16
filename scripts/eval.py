"""Evaluate predicted MS/MS spectra against ground-truth spectra.

Usage:
    python scripts/eval.py <ground_truth.pkl> <predictions.mgf> \\
        [--result_path results.csv] [--plot_path similarity_hist.png]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from molnetpack import MolNet


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predicted MS/MS spectra")
    parser.add_argument("pkl_file", help="Ground-truth spectra (.pkl from preprocessing)")
    parser.add_argument("mgf_file", help="Predicted spectra (.mgf from predict.py)")
    parser.add_argument("--result_path", type=str, default="",
                        help="Save per-spectrum results to this CSV file")
    parser.add_argument("--plot_path",   type=str, default="",
                        help="Save cosine similarity histogram to this PNG file")
    return parser.parse_args()


def main():
    args = parse_args()
    molnet = MolNet(device=torch.device("cpu"), seed=42)
    molnet.evaluate(
        test_pkl=args.pkl_file,
        pred_mgf=args.mgf_file,
        result_path=args.result_path,
        plot_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
