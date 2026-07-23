"""Convert ChEMBL SMILES to a 3DMolMS self-supervised pretraining pkl.

Produces the exact format that MolSSL_Dataset / scripts/pretrain_ssl.py read:
    {"title": str, "smiles": str,
     "mol": np.ndarray[max_atom_num, 21], "mask": np.ndarray[max_atom_num] bool}

The 21-d atom features match the rest of 3DMolMS (see conformation_array +
the `encoding` section of a preprocess config): dims 0-2 are the centered xyz
coordinates, 3-9 are seven atom attributes, and 10-20 are the atom-type
one-hot. Using the config's `atom_type` map (rather than a hardcoded one)
guarantees the pretraining featurization is identical to the downstream tasks.

Data source (either):
  --sdf_path    a local ChEMBL SDF / SDF.GZ, or
  --chembl_url  a ChEMBL chemreps TSV.GZ (URL or local path; defaults to the
                EBI FTP bulk download, using only the standard library).

Filters: parseable SMILES; molecular weight in [150, 900] Da; all atoms in the
config's supported element set; total atom count (incl. H) in [min, max]_atoms.

Usage:
  # quick test from the EBI FTP (a few thousand molecules)
  python scripts/chembl2pkl.py --output ./data/chembl_ssl.pkl --limit 5000

  # from a pre-downloaded SDF, more workers
  python scripts/chembl2pkl.py --sdf_path ./data/chembl_34.sdf.gz \\
      --output ./data/chembl_ssl.pkl --n_jobs 8

With the default --train_frac 0.95 the output is split into
<output>_train.pkl / <output>_valid.pkl, ready for pretrain_ssl.py.
"""

import argparse
import gzip
import io
import os
import pickle
import urllib.request
from multiprocessing import Pool

import numpy as np
import yaml
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from molnetpack import conformation_array

RDLogger.DisableLog("rdApp.*")

MIN_MW, MAX_MW = 150.0, 900.0
_CHEMBL_VERSION = 34
_CHEMBL_CHEMREPS_URL = (
    f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
    f"chembl_{_CHEMBL_VERSION}/chembl_{_CHEMBL_VERSION}_chemreps.txt.gz"
)

# Set once per worker process via the Pool initializer.
_ENC: dict = {}


def _init_worker(enc: dict) -> None:
    global _ENC
    _ENC = enc


def _process_molecule(args):
    """(chembl_id, smiles) -> pkl entry, or None on failure/filter."""
    chembl_id, smiles = args
    atom_map = _ENC["atom_type"]           # symbol -> one-hot list
    conf_type = _ENC["conf_type"]
    max_atoms = _ENC["max_atom_num"]
    min_atoms = _ENC["min_atom_num"]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if not (MIN_MW <= Descriptors.ExactMolWt(mol) <= MAX_MW):
            return None
        if any(a.GetSymbol() not in atom_map for a in mol.GetAtoms()):
            return None

        good, xyz_arr, atom_type = conformation_array(smiles, conf_type)
        if not good:
            return None
        n_atoms = xyz_arr.shape[0]          # total atoms incl. explicit H
        if n_atoms < min_atoms or n_atoms > max_atoms:
            return None
        if any(a not in atom_map for a in atom_type):
            return None

        one_hot = np.array([atom_map[a] for a in atom_type], dtype=np.float32)
        mol_arr = np.concatenate([xyz_arr, one_hot], axis=1).astype(np.float32)  # [n, 21]
        mol_arr = np.pad(mol_arr, ((0, max_atoms - n_atoms), (0, 0)), constant_values=0.0)
        mask = np.zeros(max_atoms, dtype=bool)
        mask[:n_atoms] = True

        return {"title": chembl_id, "smiles": smiles, "mol": mol_arr, "mask": mask}
    except Exception:
        return None


def _iter_from_sdf(sdf_path):
    opener = gzip.open if sdf_path.endswith(".gz") else open
    with opener(sdf_path, "rt", errors="ignore") as fh:
        for mol in Chem.ForwardSDMolSupplier(fh, removeHs=True, sanitize=True):
            if mol is None:
                continue
            props = mol.GetPropsAsDict()
            chembl_id = props.get("chembl_id") or props.get("CHEMBL_ID") or \
                props.get("_Name", "unknown")
            smiles = Chem.MolToSmiles(mol, canonical=True)
            if smiles:
                yield str(chembl_id), smiles


def _iter_from_chemreps(source, limit):
    """Yield (chembl_id, smiles) from a chemreps TSV/TSV.GZ (path or URL).

    Format: chembl_id \\t canonical_smiles \\t standard_inchi \\t inchi_key
    """
    if source.startswith(("http://", "https://", "ftp://")):
        print(f"Downloading ChEMBL chemreps from {source} ...")
        with urllib.request.urlopen(source) as resp:
            buf = io.BytesIO(resp.read())
    else:
        buf = open(source, "rb")
    try:
        fh = gzip.open(buf, "rt") if source.endswith(".gz") \
            else io.TextIOWrapper(buf, encoding="utf-8")
        count = 0
        for line in fh:
            if line.startswith("chembl_id") or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[1]:
                continue
            yield parts[0], parts[1]
            count += 1
            if limit is not None and count >= limit:
                return
    finally:
        buf.close()


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert ChEMBL compounds to a 3DMolMS SSL pretraining pkl.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output", type=str, default="./data/chembl_ssl.pkl",
                   help="Output pkl path (split into _train/_valid unless train_frac=1).")
    p.add_argument("--sdf_path", type=str, default=None,
                   help="Local ChEMBL SDF/SDF.GZ; takes priority over --chembl_url.")
    p.add_argument("--chembl_url", type=str, default=_CHEMBL_CHEMREPS_URL,
                   help="ChEMBL chemreps TSV.GZ URL or local path (used if no --sdf_path).")
    p.add_argument("--config_path", type=str,
                   default="./molnetpack/config/preprocess_etkdgv3.yml",
                   help="Preprocess config supplying the `encoding` (atom_type / conf_type).")
    p.add_argument("--conf_type", type=str, default=None,
                   choices=["etkdgv3", "mmff", "2d", "etkdg"],
                   help="Override the conformation method (default: config's encoding.conf_type).")
    p.add_argument("--max_atoms", type=int, default=None,
                   help="Max total atoms incl. H (default: config's encoding.max_atom_num).")
    p.add_argument("--min_atoms", type=int, default=5, help="Min total atoms incl. H.")
    p.add_argument("--n_jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="Parallel worker processes.")
    p.add_argument("--chunk_size", type=int, default=64, help="Rows per worker chunk.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of SMILES attempted (quick tests).")
    p.add_argument("--train_frac", type=float, default=0.95,
                   help="Train split fraction; the rest is validation. 1.0 = single file.")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed for the split.")
    return p.parse_args()


def main():
    args = parse_args()

    encoding = yaml.safe_load(open(args.config_path))["encoding"]
    enc = {
        "atom_type": encoding["atom_type"],
        "conf_type": args.conf_type or encoding["conf_type"],
        "max_atom_num": args.max_atoms or encoding["max_atom_num"],
        "min_atom_num": args.min_atoms,
    }
    print(f"conf_type={enc['conf_type']}  max_atoms={enc['max_atom_num']}  "
          f"elements={list(enc['atom_type'].keys())}")

    print("Collecting SMILES ...")
    if args.sdf_path:
        pairs = list(tqdm(_iter_from_sdf(args.sdf_path), desc="SDF"))
        if args.limit:
            pairs = pairs[: args.limit]
    else:
        pairs = list(tqdm(_iter_from_chemreps(args.chembl_url, args.limit), desc="chemreps"))
    print(f"Collected {len(pairs):,} SMILES")

    data, skipped = [], 0
    with Pool(processes=args.n_jobs, initializer=_init_worker, initargs=(enc,)) as pool:
        for rec in tqdm(
            pool.imap_unordered(_process_molecule, pairs, chunksize=args.chunk_size),
            total=len(pairs), desc="Generating conformers",
        ):
            (data.append(rec) if rec is not None else None)
            skipped += rec is None
    print(f"Accepted {len(data):,} / {len(pairs):,} molecules ({skipped:,} skipped)")
    if not data:
        print("No data generated. Exiting.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.train_frac >= 1.0:
        with open(args.output, "wb") as fh:
            pickle.dump(data, fh)
        print(f"Saved {len(data):,} entries -> {args.output}")
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(data))
        split = int(len(data) * args.train_frac)
        base, ext = os.path.splitext(args.output)
        for name, sel in [("_train", idx[:split]), ("_valid", idx[split:])]:
            path = base + name + ext
            with open(path, "wb") as fh:
                pickle.dump([data[i] for i in sel], fh)
            print(f"Saved {len(sel):,} entries -> {path}")
    print("Done.")


if __name__ == "__main__":
    main()
