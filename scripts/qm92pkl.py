"""
Date: 2023-10-20 14:24:33
LastEditors: yuhhong
LastEditTime: 2023-10-20 17:09:24
"""

import os
import argparse
import yaml
import pickle
from tqdm import tqdm
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors

# ignore the warning
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from molnetpack import conformation_array


def xyz2dict(xyz_dir):
    pkl_list = []
    xyz_list = os.listdir(xyz_dir)
    for _, file_name in enumerate(tqdm(xyz_list)):
        xyz_path = os.path.join(xyz_dir, file_name)

        with open(xyz_path, "r") as f:
            data = f.read().split("\n")[:-1]  # remove the last empty line

        # parse the data according to https://www.nature.com/articles/sdata201422/tables/3
        # and https://www.nature.com/articles/sdata201422/tables/4
        atom_num = int(data[0])
        assert (
            len(data) == atom_num + 5
        ), "something goes wrong with the xyz file\n{}".format(data)
        scalar_prop = data[1].split("\t")
        # atoms = data[2:2+atom_num]
        smiles = data[-2].split("\t")[
            0
        ]  # SMILES strings from GDB-17 and from B3LYP relaxation
        # inchi = data[-1].split('\t')[0] # InChI strings for Corina and B3LYP geometries

        # josie: since we need the atomic attributes, we need to calculate the xyz-coordinates by ourselves
        # mol_arr = []
        # for atom in atoms:
        # 	atom_type = encoder['atom_type'][atom.split('\t')[0]]
        # 	atom_xyz = atom.split('\t')[1:4]
        # 	# atom_charge = atom.split('\t')[1:-1]
        # 	mol_arr.append(atom_type+atom_xyz)
        # mol_arr = np.array(mol_arr, dtype=np.float)
        pkl_list.append(
            {
                "title": file_name,
                "smiles": smiles,
                "y": np.array(scalar_prop[3:16], dtype=np.float),
            }
        )
    print("Load {} data from {}".format(len(pkl_list), xyz_dir))
    return pkl_list


def filter_mol(pkl_list, uncharact_smiles, config):
    clean_pkl_list = []
    for _, data in enumerate(tqdm(pkl_list)):
        smiles = data["smiles"]

        # Remove the uncharacterized molecules
        if smiles in uncharact_smiles:
            continue

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if mol == None:
            continue

        # Filter by atom number and atom type
        if (
            len(mol.GetAtoms()) > config["max_atom_num"]
            or len(mol.GetAtoms()) < config["min_atom_num"]
        ):
            continue
        is_compound_countain_rare_atom = False
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in config["atom_type"]:
                is_compound_countain_rare_atom = True
                break
        if is_compound_countain_rare_atom:
            continue

        # Filter by molecular mass
        molmass = Descriptors.ExactMolWt(mol)
        if molmass > config["max_molmass"] or molmass < config["min_molmass"]:
            continue

        clean_pkl_list.append(data)
    return clean_pkl_list


def random_split(pkl_list, test_ratio=0.1):
    test_idx = np.random.choice(
        range(len(pkl_list)), int(len(pkl_list) * test_ratio), replace=False
    )

    train_pkl = []
    test_pkl = []
    for idx, pkl in enumerate(pkl_list):
        if idx in test_idx:
            test_pkl.append(pkl)
        else:
            train_pkl.append(pkl)
    return test_pkl, train_pkl


def add_mol_arr(pkl_list, encoder):
    clean_pkl_list = []
    for _, data in enumerate(tqdm(pkl_list)):
        # mol array
        good_conf, xyz_arr, atom_type = conformation_array(
            smiles=data["smiles"], conf_type=encoder["conf_type"]
        )
        # There are some limitations of conformation generation methods.
        # e.g. https://github.com/rdkit/rdkit/issues/5145
        # Let's skip the unsolvable molecules.
        if not good_conf:
            continue
        atom_type_one_hot = np.array([encoder["atom_type"][atom] for atom in atom_type])
        assert xyz_arr.shape[0] == atom_type_one_hot.shape[0]

        mol_arr = np.concatenate([xyz_arr, atom_type_one_hot], axis=1)
        mol_arr = np.pad(
            mol_arr,
            ((0, encoder["max_atom_num"] - xyz_arr.shape[0]), (0, 0)),
            constant_values=0,
        )

        data["mol"] = mol_arr
        clean_pkl_list.append(data)
    return clean_pkl_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the Data")
    parser.add_argument(
        "--raw_dir", type=str, default="./data/qm9/", help="path to raw data"
    )
    parser.add_argument(
        "--pkl_dir", type=str, default="./data/", help="path to mgf data"
    )
    parser.add_argument(
        "--data_config_path",
        type=str,
        default="./config/preprocess_etkdg.yml",
        help="path to configuration",
    )
    args = parser.parse_args()

    assert os.path.exists(os.path.join(args.raw_dir, "dsC7O2H10nsd"))
    assert os.path.exists(os.path.join(args.raw_dir, "dsgdb9nsd"))
    assert os.path.exists(os.path.join(args.raw_dir, "uncharacterized.txt"))

    # load the configurations
    with open(args.data_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 1. read all mol_arr from dsC7O2H10nsd and dsgdb9nsd
    print("\n>>> Step 1: convert original format to pkl;")
    pkl_list1 = xyz2dict(os.path.join(args.raw_dir, "dsC7O2H10nsd"))
    pkl_list2 = xyz2dict(os.path.join(args.raw_dir, "dsgdb9nsd"))

    # 2. remove uncharacterized molecules & filter our by rules
    print(
        "\n>>> Step 2: remove uncharacterized molecules; filter out spectra by certain rules;"
    )
    with open(os.path.join(args.raw_dir, "uncharacterized.txt"), "r") as f:
        data = f.read().split("\n")[9:-2]
        uncharact_smiles = [d.split()[1] for d in data]
    print("Load {} uncharact SMILES".format(len(uncharact_smiles)))

    print("Filter QM9 molecules...")
    qm9_pkl_list = filter_mol(
        pkl_list1 + pkl_list2, uncharact_smiles, config=config["qm9"]
    )
    print("Get {} data after filtering".format(len(qm9_pkl_list)))

    # 3. split data into training and test sets
    print("\n>>> Step 3: randomly split SMILES into training set and test set;")
    qm9_test_pkl, qm9_train_pkl = random_split(qm9_pkl_list, test_ratio=0.1)
    print(
        "Get {} training data and {} test data".format(
            len(qm9_train_pkl), len(qm9_test_pkl)
        )
    )

    # 4. generate 3d conformattions & encoding data into arrays
    print("\n>>> Step 4: generate molecular array;")
    qm9_test_pkl = add_mol_arr(qm9_test_pkl, config["encoding"])
    out_path = os.path.join(
        args.pkl_dir, "qm9_{}_test.pkl".format(config["encoding"]["conf_type"])
    )
    with open(out_path, "wb") as f:
        pickle.dump(qm9_test_pkl, f)
        print("Save {}".format(out_path))

    qm9_train_pkl = add_mol_arr(qm9_train_pkl, config["encoding"])
    out_path = os.path.join(
        args.pkl_dir, "qm9_{}_train.pkl".format(config["encoding"]["conf_type"])
    )
    with open(out_path, "wb") as f:
        pickle.dump(qm9_train_pkl, f)
        print("Save {}".format(out_path))

    print("Done!")
