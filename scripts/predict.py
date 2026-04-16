"""Unified prediction script for all 3DMolMS tasks.

Usage:
    python scripts/predict.py --task msms --test_data input.csv --result_path out.mgf [options]
    python scripts/predict.py --task prop --test_data input.csv --result_path out.csv  [options]

Supported input formats for msms: .csv, .mgf, .pkl
Supported input formats for prop: .csv, .pkl
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from pyteomics import mgf

import torch
from torch.utils.data import DataLoader

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import Descriptors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import init_random_seed, setup_device, make_idx_base

from molnetpack import MolNet_MS, MolNet_Oth
from molnetpack import Mol_Dataset
from molnetpack import csv2pkl_wfilter, nce2ce, precursor_calculator
from molnetpack import filter_spec, mgf2pkl, ms_vec2dict


# ---------------------------------------------------------------------------
# Prediction steps
# ---------------------------------------------------------------------------

def pred_step_msms(model, device, loader, batch_size, num_points):
    """Run MS/MS prediction; returns (id_list, pred_list) where pred_list is
    a list of sparse {m/z: ..., intensity: ...} dicts."""
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
            pred = pred.cpu().apply_(lambda v: v if v > 0.01 else 0)

            id_list += list(ids)
            pred_list.append(pred)
            bar.update(1)

    pred_tensor = torch.cat(pred_list, dim=0)
    return id_list, pred_tensor


def pred_step_prop(model, device, loader, batch_size, num_points):
    """Run property (RT / CCS) prediction; returns (id_list, scalar_list)."""
    model.eval()
    id_list, pred_list = [], []

    with tqdm(total=len(loader), desc="Predict") as bar:
        for batch in loader:
            ids, x, mask = batch
            x    = x.to(device=device, dtype=torch.float).permute(0, 2, 1)
            mask = mask.to(device=device)
            idx_base = make_idx_base(batch_size, num_points, device)

            with torch.no_grad():
                pred = model(x, mask, None, idx_base).squeeze()

            if batch_size == 1:
                id_list.append(ids[0])
                pred_list.append(pred.item())
            else:
                id_list += list(ids)
                pred_list += pred.tolist()
            bar.update(1)

    return id_list, pred_list


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _decode_env(pkl_dict, data_config):
    """Recover precursor_type and collision energy from encoded env vectors."""
    decoding = {
        ",".join(map(str, map(int, v))): k
        for k, v in data_config["encoding"]["precursor_type"].items()
    }
    ce_list, adduct_list, smiles_list = [], [], []
    for d in pkl_dict:
        adduct = decoding[",".join(map(str, map(int, d["env"][1:])))]
        smiles = d["smiles"]
        mass   = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
        charge = int(data_config["encoding"]["type2charge"][adduct])
        ce_list.append(nce2ce(d["env"][0], precursor_calculator(adduct, mass), charge))
        adduct_list.append(adduct)
        smiles_list.append(smiles)
    return smiles_list, ce_list, adduct_list


def save_msms_results(result_path, id_list, pred_tensor, pkl_dict, config, data_config):
    smiles_list, ce_list, adduct_list = _decode_env(pkl_dict, data_config)
    resolution = float(config["model"]["resolution"])
    pred_dicts = [ms_vec2dict(spec, resolution) for spec in pred_tensor.tolist()]

    res_df = pd.DataFrame({
        "ID":             id_list,
        "SMILES":         smiles_list,
        "Collision Energy": ce_list,
        "Precursor Type": adduct_list,
        "Pred M/Z":       [p["m/z"]       for p in pred_dicts],
        "Pred Intensity": [p["intensity"] for p in pred_dicts],
    })

    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    if result_path.endswith(".csv"):
        res_df.to_csv(result_path, sep="\t")
    elif result_path.endswith(".mgf"):
        spectra = []
        for idx, row in res_df.iterrows():
            spectra.append({
                "params": {
                    "title":            row["ID"],
                    "mslevel":          "2",
                    "organism":         f"3DMolMS_v{config.get('version', '1')}",
                    "spectrumid":       f"pred_{idx}",
                    "smiles":           row["SMILES"],
                    "collision_energy": row["Collision Energy"],
                    "precursor_type":   row["Precursor Type"],
                },
                "m/z array":       np.array([float(v) for v in row["Pred M/Z"].split(",")]),
                "intensity array": np.array([float(v) * 1000 for v in row["Pred Intensity"].split(",")]),
            })
        mgf.write(spectra, result_path, file_mode="w", write_charges=False)
    else:
        raise ValueError("Unsupported result format. Use .csv or .mgf")

    print(f"Saved predictions to {result_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="3DMolMS unified prediction script")
    parser.add_argument("--task", required=True, choices=["msms", "prop"],
                        help="Prediction task: msms or prop (RT/CCS)")
    parser.add_argument("--test_data",         type=str, required=True,
                        help="Input file (.csv / .mgf / .pkl)")
    parser.add_argument("--result_path",       type=str, required=True,
                        help="Output file (.csv or .mgf for msms; .csv for prop)")
    parser.add_argument("--model_config_path", type=str, default="./config/molnet.yml")
    parser.add_argument("--data_config_path",  type=str, default="./config/preprocess_etkdgv3.yml")
    parser.add_argument("--resume_path",       type=str, default="./check_point/molnet_qtof_etkdgv3.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--save_pkl", action="store_true",
                        help="Cache the converted .pkl alongside the input file")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--device",  type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    init_random_seed(args.seed)
    device = setup_device(args.device, args.no_cuda)
    print(f"Task: {args.task}  |  Device: {device}")

    with open(args.model_config_path) as f:
        config = yaml.safe_load(f)
    with open(args.data_config_path) as f:
        data_config = yaml.safe_load(f)

    batch_size = 1  # keep at 1 for prediction to avoid drop_last issues
    num_points = config["model"]["max_atom_num"]

    # ------------------------------------------------------------------
    # 1. Load / convert data
    # ------------------------------------------------------------------
    fmt = args.test_data.rsplit(".", 1)[-1].lower()

    if fmt == "pkl":
        with open(args.test_data, "rb") as f:
            pkl_dict = pickle.load(f)

    elif fmt == "csv":
        pkl_dict = csv2pkl_wfilter(args.test_data, data_config["encoding"])

    elif fmt == "mgf" and args.task == "msms":
        origin_spectra = list(mgf.read(args.test_data))
        filtered, _ = filter_spec(
            origin_spectra, config.get("all", {}),
            type2charge=data_config["encoding"]["type2charge"],
        )
        pkl_dict = mgf2pkl(filtered, data_config["encoding"])

    else:
        raise ValueError(f"Unsupported input format '.{fmt}' for task '{args.task}'")

    print(f"Loaded {len(pkl_dict)} records from {args.test_data}")

    if args.save_pkl:
        pkl_path = args.test_data.rsplit(".", 1)[0] + ".pkl"
        if not os.path.exists(pkl_path):
            with open(pkl_path, "wb") as f:
                pickle.dump(pkl_dict, f)
            print(f"Cached pkl to {pkl_path}")

    loader = DataLoader(
        Mol_Dataset(pkl_dict),
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    ModelCls = MolNet_MS if args.task == "msms" else MolNet_Oth
    model = ModelCls(config["model"]).to(device)

    if not os.path.exists(args.resume_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.resume_path}\n"
            "Download pre-trained weights from the GitHub releases page."
        )
    ckpt = torch.load(args.resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from {args.resume_path}")

    # ------------------------------------------------------------------
    # 3. Predict & save
    # ------------------------------------------------------------------
    if args.task == "msms":
        id_list, pred_tensor = pred_step_msms(model, device, loader, batch_size, num_points)
        save_msms_results(args.result_path, id_list, pred_tensor, pkl_dict, config, data_config)
    else:
        id_list, pred_list = pred_step_prop(model, device, loader, batch_size, num_points)
        os.makedirs(os.path.dirname(args.result_path) or ".", exist_ok=True)
        pd.DataFrame({"ID": id_list, "Pred": pred_list}).to_csv(args.result_path)
        print(f"Saved predictions to {args.result_path}")


if __name__ == "__main__":
    main()
