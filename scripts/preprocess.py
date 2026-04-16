"""Unified preprocessing script for all 3DMolMS tasks.

Usage:
    # MS/MS: convert raw spectral libraries to training-ready pkl files
    python scripts/preprocess.py --task msms \\
        --dataset nist mona gnps --instrument_type qtof orbitrap \\
        --raw_dir ./data/origin --pkl_dir ./data

    # RT: process METLIN SMRT dataset (SDF)
    python scripts/preprocess.py --task rt \\
        --raw_dir ./data/origin --pkl_dir ./data

    # CCS: process AllCCS dataset (CSV)
    python scripts/preprocess.py --task ccs \\
        --raw_dir ./data/origin --pkl_dir ./data
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from pyteomics import mgf

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

from molnetpack import sdf2mgf, filter_spec, mgf2pkl
from molnetpack import conformation_array, filter_mol, check_atom


# ---------------------------------------------------------------------------
# MS/MS helpers (was preprocess.py)
# ---------------------------------------------------------------------------

def load_msms_raw(raw_dir, datasets):
    """Read raw spectra from each source into a dict keyed by dataset name."""
    origin_spectra = {}

    if "agilent" in datasets:
        assert os.path.exists(os.path.join(raw_dir, "Agilent_Combined.sdf"))
        assert os.path.exists(os.path.join(raw_dir, "Agilent_Metlin.sdf"))
        origin_spectra["agilent"] = (
            sdf2mgf(os.path.join(raw_dir, "Agilent_Combined.sdf"), prefix="agilent_combine")
            + sdf2mgf(os.path.join(raw_dir, "Agilent_Metlin.sdf"),  prefix="agilent_metlin")
        )

    if "nist" in datasets:
        assert os.path.exists(os.path.join(raw_dir, "hr_msms_nist.SDF"))
        origin_spectra["nist"] = sdf2mgf(os.path.join(raw_dir, "hr_msms_nist.SDF"), prefix="nist20")

    if "mona" in datasets:
        assert os.path.exists(os.path.join(raw_dir, "MoNA-export-All_LC-MS-MS_QTOF.sdf"))
        assert os.path.exists(os.path.join(raw_dir, "MoNA-export-All_LC-MS-MS_Orbitrap.sdf"))
        origin_spectra["mona"] = (
            sdf2mgf(os.path.join(raw_dir, "MoNA-export-All_LC-MS-MS_QTOF.sdf"),      prefix="mona_qtof")
            + sdf2mgf(os.path.join(raw_dir, "MoNA-export-All_LC-MS-MS_Orbitrap.sdf"), prefix="mona_orbitrap")
        )

    if "waters" in datasets:
        assert os.path.exists(os.path.join(raw_dir, "waters_qtof.mgf"))
        origin_spectra["waters"] = list(mgf.read(os.path.join(raw_dir, "waters_qtof.mgf")))
        print(f"Loaded {len(origin_spectra['waters'])} Waters spectra")

    if "gnps" in datasets:
        assert os.path.exists(os.path.join(raw_dir, "ALL_GNPS_cleaned.mgf"))
        assert os.path.exists(os.path.join(raw_dir, "ALL_GNPS_cleaned.csv"))
        raw_spectra = list(mgf.read(os.path.join(raw_dir, "ALL_GNPS_cleaned.mgf")))
        meta_df = pd.read_csv(os.path.join(raw_dir, "ALL_GNPS_cleaned.csv"))[
            ["spectrum_id", "Adduct", "Precursor_MZ", "ExactMass", "Ion_Mode",
             "msMassAnalyzer", "msDissociationMethod", "Smiles", "InChIKey_smiles", "collision_energy"]
        ]
        meta_df["collision_energy"] = meta_df["collision_energy"].fillna("Unknown").astype(str)
        meta_df = meta_df.dropna()
        meta_df["Adduct"]   = meta_df["Adduct"].apply(lambda x: x[:-2] + x[-1:])
        meta_df["Ion_Mode"] = meta_df["Ion_Mode"].str.upper()
        meta_df = meta_df.rename(columns={
            "spectrum_id": "title", "Adduct": "precursor_type", "Precursor_MZ": "precursor_mz",
            "ExactMass": "molmass", "Ion_Mode": "ionmode", "msMassAnalyzer": "source_instrument",
            "msDissociationMethod": "instrument_type", "Smiles": "smiles", "InChIKey_smiles": "inchi_key",
        }).set_index("title").to_dict("index")

        tmp = []
        for idx, spec in enumerate(tqdm(raw_spectra, desc="GNPS")):
            title = spec["params"]["title"]
            if title in meta_df:
                spec["params"] = {**meta_df[title], "title": f"gnps_{idx}"}
                tmp.append(spec)
        origin_spectra["gnps"] = tmp
        print(f"Loaded {len(tmp)} GNPS spectra after metadata join")

    return origin_spectra


def preprocess_msms(args, config):
    os.makedirs(args.pkl_dir, exist_ok=True)
    if args.mgf_dir:
        os.makedirs(args.mgf_dir, exist_ok=True)

    print("\n>>> Step 1: load raw spectra")
    origin_spectra = load_msms_raw(args.raw_dir, args.dataset)

    print("\n>>> Steps 2-4: filter → split → encode")
    for ins in args.instrument_type:
        spectra, smiles_list = [], []
        for ds in args.dataset:
            key = f"{ds}_{ins}"
            if key not in config:
                continue
            print(f"  Filtering {key}...")
            filt, filt_smiles = filter_spec(
                origin_spectra[ds], config[key],
                type2charge=config["encoding"]["type2charge"],
            )
            spectra    += filt
            smiles_list += list(set(filt_smiles))
        smiles_list = list(set(smiles_list))

        if args.mgf_dir:
            out = os.path.join(args.mgf_dir, f"{ins}_{'_'.join(args.dataset)}.mgf")
            mgf.write(spectra, output=out)

        if args.maxmin_pick:
            print(f"  ({ins}) MaxMin split...")
            fpgen   = rdFingerprintGenerator.GetMorganGenerator(radius=3)
            fp_list = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]
            picker  = MaxMinPicker()
            train_idx = list(picker.LazyBitVectorPick(
                fp_list, len(fp_list), int(len(fp_list) * args.train_ratio), seed=42,
            ))
        else:
            print(f"  ({ins}) Random split...")
            train_idx = list(np.random.choice(
                len(smiles_list), int(len(smiles_list) * args.train_ratio), replace=False,
            ))

        train_smiles = {smiles_list[i] for i in train_idx}
        train_spectra = [s for s in spectra if s["params"]["smiles"] in train_smiles]
        test_spectra  = [s for s in spectra if s["params"]["smiles"] not in train_smiles]
        print(f"  ({ins}) {len(train_spectra)} train / {len(test_spectra)} test spectra")

        conf = config["encoding"]["conf_type"]
        for split, subset in [("train", train_spectra), ("test", test_spectra)]:
            data     = mgf2pkl(subset, config["encoding"])
            out_path = os.path.join(args.pkl_dir, f"{ins}_{conf}_{split}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(data, f)
            print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# RT helpers (was preprocess_oth.py — METLIN branch)
# ---------------------------------------------------------------------------

def _sdf2arr(suppl, encoder):
    data = []
    for mol in tqdm(suppl, desc="Encoding"):
        good, xyz, atom_type = conformation_array(Chem.MolToSmiles(mol), encoder["conf_type"])
        if not good:
            continue
        one_hot = np.array([encoder["atom_type"][a] for a in atom_type])
        mol_arr = np.concatenate([xyz, one_hot], axis=1)
        mol_arr = np.pad(mol_arr, ((0, encoder["max_atom_num"] - xyz.shape[0]), (0, 0)))
        data.append({
            "title": mol.GetProp("PUBCHEM_COMPOUND_CID"),
            "mol":   mol_arr,
            "rt":    np.array([mol.GetProp("RETENTION_TIME")], dtype=np.float64),
        })
    return data


def _random_split(suppl, smiles_list, test_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    test_smiles = set(rng.choice(smiles_list, int(len(smiles_list) * test_ratio), replace=False))
    train_mol, test_mol = [], []
    for mol in suppl:
        (test_mol if Chem.MolToSmiles(mol) in test_smiles else train_mol).append(mol)
    return train_mol, test_mol


def preprocess_rt(args, config):
    sdf_path = os.path.join(args.raw_dir, "SMRT_dataset.sdf")
    assert os.path.exists(sdf_path), f"Missing {sdf_path}"
    os.makedirs(args.pkl_dir, exist_ok=True)

    print("\n>>> Step 1: load METLIN SDF")
    suppl = [m for m in Chem.SDMolSupplier(sdf_path)
             if m is not None and m.HasProp("PUBCHEM_COMPOUND_CID") and m.HasProp("RETENTION_TIME")]
    print(f"Loaded {len(suppl)} molecules")

    print("\n>>> Step 2: filter & split")
    suppl, smiles_list = filter_mol(suppl, config["metlin_rt"])
    train_mol, test_mol = _random_split(suppl, list(set(smiles_list)))
    print(f"{len(train_mol)} train / {len(test_mol)} test molecules")

    conf = config["encoding"]["conf_type"]
    for split, subset in [("train", train_mol), ("test", test_mol)]:
        data     = _sdf2arr(subset, config["encoding"])
        out_path = os.path.join(args.pkl_dir, f"metlin_{conf}_{split}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# CCS helpers (was preprocess_oth.py — AllCCS branch)
# ---------------------------------------------------------------------------

def _df2arr(df, encoder):
    data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding"):
        good, xyz, atom_type = conformation_array(row["Structure"], encoder["conf_type"])
        if not good:
            continue
        one_hot = np.array([encoder["atom_type"][a] for a in atom_type])
        mol_arr = np.concatenate([xyz, one_hot], axis=1)
        mol_arr = np.pad(mol_arr, ((0, encoder["max_atom_num"] - xyz.shape[0]), (0, 0)))
        env_arr = np.array(encoder["precursor_type"][row["Adduct"]])
        data.append({
            "title": f"{row['AllCCS ID']}_{_}",
            "mol":   mol_arr,
            "ccs":   np.array([row["CCS"]], dtype=np.float64),
            "env":   env_arr,
        })
    return data


def preprocess_ccs(args, config):
    csv_path = os.path.join(args.raw_dir, "allccs_download.csv")
    assert os.path.exists(csv_path), f"Missing {csv_path}"
    os.makedirs(args.pkl_dir, exist_ok=True)

    print("\n>>> Step 1: load AllCCS CSV")
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df = df[df["Type"] == "Experimental CCS"].dropna(subset=["Structure", "Adduct", "CCS"])
    print(f"Loaded {len(df)} CCS entries")

    print("\n>>> Step 2: filter & split")
    df = df[df["Structure"].apply(lambda s: check_atom(s, config["allccs"], in_type="smiles"))]
    df = df[df["Adduct"].isin(config["allccs"]["precursor_type"])]

    smiles_list = list(set(df["Structure"]))
    rng         = np.random.default_rng(42)
    test_smiles = set(rng.choice(smiles_list, int(len(smiles_list) * 0.1), replace=False))
    train_df    = df[~df["Structure"].isin(test_smiles)]
    test_df     = df[df["Structure"].isin(test_smiles)]
    print(f"{len(train_df)} train / {len(test_df)} test entries")

    conf = config["encoding"]["conf_type"]
    for split, subset in [("train", train_df), ("test", test_df)]:
        data     = _df2arr(subset, config["encoding"])
        out_path = os.path.join(args.pkl_dir, f"allccs_{conf}_{split}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="3DMolMS unified preprocessing script")
    parser.add_argument("--task", required=True, choices=["msms", "rt", "ccs"],
                        help="Preprocessing task")
    parser.add_argument("--raw_dir",  type=str, default="./data/origin",
                        help="Directory containing raw input files")
    parser.add_argument("--pkl_dir",  type=str, default="./data",
                        help="Directory for output pkl files")
    parser.add_argument("--data_config_path", type=str,
                        default="./config/preprocess_etkdgv3.yml")

    # msms-specific
    parser.add_argument("--dataset", type=str, nargs="+",
                        choices=["agilent", "nist", "mona", "waters", "gnps"],
                        help="Source datasets to include (msms task only)")
    parser.add_argument("--instrument_type", type=str, nargs="+",
                        choices=["qtof", "orbitrap"],
                        help="Instrument types to process (msms task only)")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--maxmin_pick", action="store_true",
                        help="Use MaxMin diversity picker for train split (msms task only)")
    parser.add_argument("--mgf_dir", type=str, default="",
                        help="Save intermediate MGF files here for debugging (msms task only)")

    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(42)

    with open(args.data_config_path) as f:
        config = yaml.safe_load(f)

    if args.task == "msms":
        if not args.dataset or not args.instrument_type:
            raise ValueError("--dataset and --instrument_type are required for the msms task")
        assert args.train_ratio < 1.0
        preprocess_msms(args, config)
    elif args.task == "rt":
        preprocess_rt(args, config)
    elif args.task == "ccs":
        preprocess_ccs(args, config)

    print("\nDone!")


if __name__ == "__main__":
    main()
