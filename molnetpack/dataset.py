"""
Date: 2023-10-02 20:24:27
LastEditors: yuhhong
LastEditTime: 2023-10-20 17:01:37
"""

import pickle
import numpy as np

from torch.utils.data import Dataset


class MolMS_Dataset(Dataset):
    def __init__(self, x, data_augmentation=True, precursor_type=False, mode="path"):
        if mode == "path":
            path = x
            with open(path, "rb") as file:
                data = pickle.load(file)
        elif mode == "data":
            data = x
            path = "unknown"
        else:
            raise ValueError("Unsupported mode:", mode)

        if precursor_type:
            data = self.filter_precursor_type(data, precursor_type)

        # generate mask
        for idx in range(len(data)):
            mask = ~np.all(data[idx]["mol"] == 0, axis=1)
            data[idx]["mask"] = mask.astype(bool)

        # data augmentation by flipping the x,y,z-coordinates
        if data_augmentation:
            flipping_data = []
            for d in data:
                flipping_mol_arr = np.copy(d["mol"])
                flipping_mol_arr[:, 0] *= -1
                flipping_data.append(
                    {
                        "title": d["title"] + "_f",
                        "mol": flipping_mol_arr,
                        "mask": d["mask"],        # mask is identical for flipped mol
                        "spec": d["spec"],
                        "env": d["env"],
                    }
                )

            self.data = data + flipping_data
            print(
                "Load {} data (with data augmentation by flipping coordinates)".format(
                    len(self.data)
                )
            )
        else:
            self.data = data
            print("Load {} data".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx]["title"],
            self.data[idx]["mol"],
            self.data[idx]["mask"],
            self.data[idx]["spec"],
            self.data[idx]["env"],
        )

    def filter_precursor_type(self, data, precursor_type):
        filtered_data = []
        for d in data:
            d_precursor_type = ",".join([str(int(i)) for i in d["env"][1:]])
            if d_precursor_type == precursor_type:
                filtered_data.append(d)
        return filtered_data


class Mol_Dataset(Dataset):
    def __init__(self, data, precursor_type=False):
        if precursor_type:
            data = self.filter_precursor_type(data, precursor_type)

        # generate mask
        for idx in range(len(data)):
            mask = ~np.all(data[idx]["mol"] == 0, axis=1)
            data[idx]["mask"] = mask.astype(bool)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx]["title"],
            self.data[idx]["mol"],
            self.data[idx]["mask"],
            self.data[idx]["env"],
        )

    def filter_precursor_type(self, data, precursor_type):
        filtered_data = []
        for d in data:
            d_precursor_type = ",".join([str(int(i)) for i in d["env"][1:]])
            if d_precursor_type == precursor_type:
                filtered_data.append(d)
        return filtered_data


class MolRT_Dataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as file:
            self.data = pickle.load(file)
        print("Load {} data from {}".format(len(self.data), path))

        # generate mask
        for idx in range(len(self.data)):
            mask = ~np.all(self.data[idx]["mol"] == 0, axis=1)
            self.data[idx]["mask"] = mask.astype(bool)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx]["title"],
            self.data[idx]["mol"],
            self.data[idx]["mask"],
            self.data[idx]["rt"],
        )


class MolCCS_Dataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as file:
            self.data = pickle.load(file)
        print("Load {} data from {}".format(len(self.data), path))

        # generate mask
        for idx in range(len(self.data)):
            mask = ~np.all(self.data[idx]["mol"] == 0, axis=1)
            self.data[idx]["mask"] = mask.astype(bool)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx]["title"],
            self.data[idx]["mol"],
            self.data[idx]["mask"],
            self.data[idx]["ccs"],
            self.data[idx]["env"],
        )


class MolPRE_Dataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as file:
            data = pickle.load(file)

        self.data = []
        for d in data:
            if "mol" in d.keys():
                self.data.append(d)
        print("Load {} data from {}".format(len(self.data), path))

        # generate mask
        for idx in range(len(self.data)):
            mask = ~np.all(self.data[idx]["mol"] == 0, axis=1)
            self.data[idx]["mask"] = mask.astype(bool)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx]["title"],
            self.data[idx]["mol"],
            self.data[idx]["mask"],
            self.data[idx]["y"],
        )


class MolCSV_Dataset(Dataset):
    def __init__(self, x, mode="path"):
        assert mode in ["path", "data"]
        if mode == "path":
            with open(x, "rb") as file:
                self.data = pickle.load(file)
            print("Load {} data from {}".format(len(self.data), x))
        elif mode == "data":
            self.data = x

        # generate mask
        for idx in range(len(self.data)):
            mask = ~np.all(self.data[idx]["mol"] == 0, axis=1)
            self.data[idx]["mask"] = mask.astype(bool)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx]["title"],
            self.data[idx]["mol"],
            self.data[idx]["mask"],
            self.data[idx]["prop"],
        )


class MolCSV_Test_Dataset(Dataset):
    def __init__(self, x, mode="path"):
        assert mode in ["path", "data"]
        if mode == "path":
            with open(x, "rb") as file:
                self.data = pickle.load(file)
            print("Load {} data from {}".format(len(self.data), x))
        elif mode == "data":
            self.data = x

        # generate mask
        for idx in range(len(self.data)):
            mask = ~np.all(self.data[idx]["mol"] == 0, axis=1)
            self.data[idx]["mask"] = mask.astype(bool)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["title"], self.data[idx]["mol"], self.data[idx]["mask"]


class MolSSL_Dataset(Dataset):
    """Dataset for masked distance reconstruction pretraining.

    SSL task — Masked distance reconstruction:
        Randomly mask mask_ratio of valid atoms per molecule.  All 21
        feature dimensions (including xyz in dims 0–2) are zeroed for
        masked atoms.  For each masked atom, pair it with every unmasked
        valid atom, then sample num_pairs such (masked, unmasked) pairs.
        Predict the Euclidean distance between the two atoms using the
        original (unmasked) coordinates as ground truth.
        Loss: MSELoss on raw distances (Å).

    Zeroing xyz of masked atoms is essential: the encoder cannot read
    the coordinates directly and must infer the masked atom's location
    from its unmasked chemical context.  Because MolConv4 is SE(3)-
    invariant and distances are also SE(3)-invariant, the encoder–head
    mapping is consistent regardless of molecular orientation.

    Expected pkl format (produced by chembl2pkl.py):
        {"title": str, "smiles": str, "mol": np.ndarray [max_atom_num, 21],
         "mask": np.ndarray [max_atom_num] bool}

    Returns:
        title        : str
        mol_masked   : float32 [max_atom_num, 21]  — all dims zeroed for masked atoms
        valid_mask   : bool    [max_atom_num]
        pair_indices : int64   [num_pairs, 2]       — (masked_i, unmasked_j)
        distances    : float32 [num_pairs]          — true Euclidean distances (Å)
    """

    def __init__(self, x, mask_ratio=0.15, num_pairs=128, mode="path"):
        if mode == "path":
            with open(x, "rb") as f:
                data = pickle.load(f)
            print("Load {} data from {}".format(len(data), x))
        elif mode == "data":
            data = x
        else:
            raise ValueError("Unsupported mode: {}".format(mode))

        for idx in range(len(data)):
            data[idx]["mask"] = ~np.all(data[idx]["mol"] == 0, axis=1)

        self.data       = data
        self.mask_ratio = mask_ratio
        self.num_pairs  = num_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry      = self.data[idx]
        mol_orig   = entry["mol"]   # [max_atom_num, 21], read-only
        valid_mask = entry["mask"]  # [max_atom_num] bool

        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)

        if n_valid >= 2:
            # ---------------------------------------------------------------- #
            # Select masked atoms: at least 1, at most n_valid-1 so there is  #
            # always at least one unmasked atom to pair with.                  #
            # ---------------------------------------------------------------- #
            n_mask = max(1, min(int(n_valid * self.mask_ratio), n_valid - 1))
            masked_local = np.random.choice(n_valid, n_mask, replace=False)
            masked_global = valid_indices[masked_local]

            # Zero all 21 dims (including xyz) for masked atoms
            mol_masked = mol_orig.copy()
            mol_masked[masked_global, :] = 0.0

            # ---------------------------------------------------------------- #
            # Build (masked, unmasked) pair pool                               #
            # ---------------------------------------------------------------- #
            masked_set       = set(masked_global.tolist())
            unmasked_global  = np.array(
                [v for v in valid_indices if v not in masked_set], dtype=np.int64
            )

            # All combinations: each masked atom × each unmasked atom
            mg, ug = np.meshgrid(masked_global, unmasked_global, indexing="ij")
            pool = np.stack([mg.ravel(), ug.ravel()], axis=1)  # [n_mask * n_unmasked, 2]

            chosen = np.random.choice(
                len(pool), self.num_pairs,
                replace=(len(pool) < self.num_pairs),
            )
            pair_indices = pool[chosen].astype(np.int64)  # [num_pairs, 2]

            # True Euclidean distances from original (unmasked) coordinates
            xyz = mol_orig[:, :3]
            diffs = xyz[pair_indices[:, 0]] - xyz[pair_indices[:, 1]]
            distances = np.linalg.norm(diffs, axis=1).astype(np.float32)

        else:
            mol_masked   = mol_orig.copy()
            pair_indices = np.zeros((self.num_pairs, 2), dtype=np.int64)
            distances    = np.zeros(self.num_pairs, dtype=np.float32)

        return (
            entry["title"],
            mol_masked.astype(np.float32),  # [max_atom_num, 21]
            valid_mask.astype(bool),        # [max_atom_num]
            pair_indices,                   # [num_pairs, 2] int64
            distances,                      # [num_pairs] float32 (Å)
        )
