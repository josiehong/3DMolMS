"""End-to-end smoke tests for the SSL pretraining pipeline (MolNet_SSL).

Masked distance reconstruction: MolSSL_Dataset masks atoms and samples
(masked, unmasked) pairs; MolNet_SSL predicts their Euclidean distances from
per-atom encoder features; pretrain_ssl_step / eval_ssl_step run one epoch.

These tests exercise the real classes and the shipped config, but use a small
model for the forward passes so they stay fast (the production config has
max_atom_num=300, which makes a forward pass slow).
"""
import importlib.util
import os
import pickle
import tempfile

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from molnetpack.model import MolNet_SSL
from molnetpack.dataset import MolSSL_Dataset
from molnetpack.utils import pretrain_ssl_step, eval_ssl_step

HERE = os.path.dirname(__file__)
SSL_CONFIG = os.path.join(HERE, "..", "molnetpack", "config", "molnet_pre_ssl.yml")
PREPROCESS_CONFIG = os.path.join(HERE, "..", "molnetpack", "config", "preprocess_etkdgv3.yml")


def _load_chembl2pkl():
    """Load scripts/chembl2pkl.py by path (scripts/ is not an importable package)."""
    path = os.path.join(HERE, "..", "scripts", "chembl2pkl.py")
    spec = importlib.util.spec_from_file_location("chembl2pkl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Small model: emb_dim MUST equal sum(encode_layers) (the encoder concatenates
# every layer's output before the final conv).
SMALL_MODEL = {
    "in_dim": 21,
    "max_atom_num": 20,
    "emb_dim": 24,               # == sum([8, 16])
    "k": 3,
    "encode_layers": [8, 16],
    "encoder_version": 2,
    "chirality": False,
}
NUM_PAIRS = 16


def _synthetic_pkl(path, n_mols=8, max_atom_num=20):
    """Write a ChEMBL-style pkl: {title, smiles, mol[N,21], mask[N]}."""
    rng = np.random.default_rng(0)
    data = []
    for i in range(n_mols):
        n = int(rng.integers(6, max_atom_num - 2))
        mol = np.zeros((max_atom_num, 21), dtype=np.float32)
        mol[:n, :3] = rng.normal(size=(n, 3)) * 3.0     # xyz
        mol[:n, 3:] = rng.normal(size=(n, 18))          # atom features
        data.append({"title": f"m{i}", "smiles": "C", "mol": mol,
                     "mask": ~np.all(mol == 0, axis=1)})
    with open(path, "wb") as fh:
        pickle.dump(data, fh)


def test_ssl_config_builds():
    """The shipped molnet_pre_ssl.yml builds a valid MolNet_SSL."""
    cfg = yaml.safe_load(open(SSL_CONFIG))
    assert cfg["model"]["emb_dim"] == sum(cfg["model"]["encode_layers"]), \
        "emb_dim must equal sum(encode_layers)"
    model = MolNet_SSL(cfg["model"])
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
    assert int(cfg["model"].get("encoder_version", 2)) == 2  # pretrains the E(3) encoder


def test_ssl_pipeline_end_to_end():
    """Dataset -> model -> one train step + eval step run and give finite losses."""
    with tempfile.TemporaryDirectory() as tmp:
        pkl = os.path.join(tmp, "ssl.pkl")
        _synthetic_pkl(pkl, n_mols=8, max_atom_num=SMALL_MODEL["max_atom_num"])

        ds = MolSSL_Dataset(pkl, mask_ratio=0.15, num_pairs=NUM_PAIRS)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

        device = torch.device("cpu")
        model = MolNet_SSL(SMALL_MODEL).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        num_points = SMALL_MODEL["max_atom_num"]
        train_loss, train_mae = pretrain_ssl_step(
            model, device, loader, opt, batch_size=4, num_points=num_points)
        val_loss, val_mae = eval_ssl_step(
            model, device, loader, batch_size=4, num_points=num_points)

        for name, v in [("train_loss", train_loss), ("train_mae", train_mae),
                        ("val_loss", val_loss), ("val_mae", val_mae)]:
            assert np.isfinite(v), f"{name} is not finite: {v}"
            assert v >= 0.0, f"{name} should be non-negative: {v}"


def test_ssl_dataset_item_shapes():
    """MolSSL_Dataset yields the documented (title, mol, mask, pairs, dists) tuple."""
    with tempfile.TemporaryDirectory() as tmp:
        pkl = os.path.join(tmp, "ssl.pkl")
        N = SMALL_MODEL["max_atom_num"]
        _synthetic_pkl(pkl, n_mols=4, max_atom_num=N)
        ds = MolSSL_Dataset(pkl, mask_ratio=0.15, num_pairs=NUM_PAIRS)

        title, mol, mask, pairs, dists = ds[0]
        assert isinstance(title, str)
        assert mol.shape == (N, 21) and mol.dtype == np.float32
        assert mask.shape == (N,) and mask.dtype == bool
        assert pairs.shape == (NUM_PAIRS, 2) and pairs.dtype == np.int64
        assert dists.shape == (NUM_PAIRS,) and dists.dtype == np.float32
        assert np.all(dists >= 0.0)


def test_ssl_checkpoint_roundtrips():
    """The saved checkpoint's state_dict loads back into a fresh MolNet_SSL
    (the format transferred into downstream MolNet models)."""
    with tempfile.TemporaryDirectory() as tmp:
        model = MolNet_SSL(SMALL_MODEL)
        ckpt_path = os.path.join(tmp, "molnet_pre_ssl.pt")
        torch.save({"version": "test", "model_state_dict": model.state_dict(),
                    "config": {"model": SMALL_MODEL}}, ckpt_path)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        fresh = MolNet_SSL(SMALL_MODEL)
        missing, unexpected = fresh.load_state_dict(ckpt["model_state_dict"], strict=True)
        assert not missing and not unexpected


def test_chembl2pkl_featurization():
    """chembl2pkl emits the exact MolSSL_Dataset format ([max_atom_num, 21],
    4 keys, no 'bonds') using the config's atom_type map, and filters correctly."""
    c2p = _load_chembl2pkl()
    encoding = yaml.safe_load(open(PREPROCESS_CONFIG))["encoding"]
    max_atoms = encoding["max_atom_num"]
    c2p._init_worker({
        "atom_type": encoding["atom_type"],
        "conf_type": encoding["conf_type"],
        "max_atom_num": max_atoms,
        "min_atom_num": 5,
    })

    rec = c2p._process_molecule(("CHEMBL25", "CC(=O)Oc1ccccc1C(=O)O"))  # aspirin
    assert rec is not None
    assert set(rec.keys()) == {"title", "smiles", "mol", "mask"}  # no 'bonds'
    assert rec["mol"].shape == (max_atoms, 21) and rec["mol"].dtype == np.float32
    assert rec["mask"].dtype == bool
    n = int(rec["mask"].sum())
    assert n > 0
    # dims 0-2 are xyz coordinates — not all zero for real atoms
    assert np.any(rec["mol"][:n, :3] != 0.0)
    # padding rows past the real atoms are zeroed
    assert np.all(rec["mol"][n:] == 0.0)

    # a molecule below the MW floor (benzene, 78 Da < 150) is filtered out
    assert c2p._process_molecule(("X", "c1ccccc1")) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
