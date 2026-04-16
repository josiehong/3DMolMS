"""Evaluate predicted MS/MS spectra against ground-truth spectra.

Compares predictions in an MGF file against ground-truth spectra stored in a
PKL file (as produced by the preprocessing pipeline), computing cosine similarity
per spectrum and reporting mean similarity grouped by precursor type.

Usage:
    python scripts/eval.py <ground_truth.pkl> <predictions.mgf> \\
        [--result_path results.csv] [--plot_path similarity_hist.png]
"""

import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from decimal import Decimal
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyteomics import mgf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bin_spectrum(mz_array, intensity_array, resolution=0.2, max_mz=1500):
    """Bin a sparse spectrum into a fixed-length intensity vector."""
    res     = Decimal(str(resolution))
    max_mz  = Decimal(str(max_mz))
    n_bins  = int(max_mz // res)
    binned  = [0.0] * n_bins

    for mz, intensity in zip(mz_array, intensity_array):
        idx = int(round(Decimal(str(mz)) // res))
        if 0 <= idx < n_bins:
            binned[idx] += intensity

    total = sum(binned)
    if total == 0:
        return False, binned
    return True, binned


def _cosine_similarity(vec1, vec2):
    a, b = np.array(vec1), np.array(vec2)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return None
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(pkl_path, mgf_path):
    with open(pkl_path, "rb") as f:
        pkl_spectra = pickle.load(f)
    mgf_spectra = list(mgf.read(mgf_path))

    pkl_by_title = {s["title"]: s["spec"]        for s in pkl_spectra}
    mgf_by_title = {s["params"]["title"]: s       for s in mgf_spectra}

    results = []
    for title, gt_spec in tqdm(pkl_by_title.items(), desc="Evaluating"):
        if title not in mgf_by_title:
            continue
        pred_spec = mgf_by_title[title]
        ok, binned = _bin_spectrum(pred_spec["m/z array"], pred_spec["intensity array"])
        if not ok or len(gt_spec) != len(binned):
            continue
        sim = _cosine_similarity(gt_spec, binned)
        if sim is None:
            continue
        results.append({
            "title":          title,
            "smiles":         pred_spec["params"].get("smiles", ""),
            "collision_energy": pred_spec["params"].get("collision_energy", ""),
            "precursor_type": pred_spec["params"].get("precursor_type", "Unknown"),
            "cosine_similarity": sim,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predicted MS/MS spectra")
    parser.add_argument("pkl_file",  help="Ground-truth spectra (.pkl from preprocessing)")
    parser.add_argument("mgf_file",  help="Predicted spectra (.mgf from predict.py)")
    parser.add_argument("--result_path", type=str, default="",
                        help="Save per-spectrum results to this CSV file")
    parser.add_argument("--plot_path",   type=str, default="",
                        help="Save cosine similarity histogram to this PNG file")
    return parser.parse_args()


def main():
    args = parse_args()
    df = evaluate(args.pkl_file, args.mgf_file)

    print(f"\nEvaluated {len(df)} matched spectra")
    print(f"Overall mean cosine similarity: {df['cosine_similarity'].mean():.4f}")
    print("\nMean cosine similarity by precursor type:")
    print(df.groupby("precursor_type")["cosine_similarity"].mean().to_string())

    if args.result_path:
        df.to_csv(args.result_path, index=False)
        print(f"\nSaved per-spectrum results to {args.result_path}")

    if args.plot_path:
        plt.figure(figsize=(8, 6))
        plt.hist(df["cosine_similarity"].tolist(), bins=50, edgecolor="black")
        plt.title("Cosine Similarity Distribution")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.plot_path)
        plt.close()
        print(f"Saved histogram to {args.plot_path}")


if __name__ == "__main__":
    main()
