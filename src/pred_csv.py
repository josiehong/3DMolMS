"""
Date: 2023-10-03 21:09:14
LastEditors: yuhhong
LastEditTime: 2023-10-20 17:16:17
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from molnetpack.model import MolNet_Oth
from molnetpack.data_utils import conformation_array
from molnetpack.dataset import MolCSV_Test_Dataset
from molnetpack import __version__


def csv2pkl_wfilter(csv_path, encoder):
    """
    This function is only used in prediction, so by default, the spectra are not contained.
    """
    df = pd.read_csv(csv_path)
    data = []
    for idx, row in df.iterrows():
        # mol array
        good_conf, xyz_arr, atom_type = conformation_array(
            smiles=row["smiles"], conf_type=encoder["conf_type"]
        )
        # There are some limitations of conformation generation methods.
        # e.g. https://github.com/rdkit/rdkit/issues/5145
        # Let's skip the unsolvable molecules.
        if not good_conf:  # filter 1
            print(
                "Can not generate correct conformation: {} {}".format(
                    row["smiles"], row["id"]
                )
            )
            continue
        if xyz_arr.shape[0] > encoder["max_atom_num"]:  # filter 2
            print(
                "Atomic number ({}) exceed the limitation ({})".format(
                    encoder["max_atom_num"], xyz_arr.shape[0]
                )
            )
            continue
        # filter 3
        rare_atom_flag = False
        rare_atom = ""
        for atom in list(set(atom_type)):
            if atom not in encoder["atom_type"].keys():
                rare_atom_flag = True
                rare_atom = atom
                break
        if rare_atom_flag:
            print("Unsupported atom type: {} {}".format(rare_atom, row["id"]))
            continue

        atom_type_one_hot = np.array([encoder["atom_type"][atom] for atom in atom_type])
        assert xyz_arr.shape[0] == atom_type_one_hot.shape[0]

        mol_arr = np.concatenate([xyz_arr, atom_type_one_hot], axis=1)
        mol_arr = np.pad(
            mol_arr,
            ((0, encoder["max_atom_num"] - xyz_arr.shape[0]), (0, 0)),
            constant_values=0,
        )

        data.append({"title": row["id"], "smiles": row["smiles"], "mol": mol_arr})
    return data


def test_step(model, device, loader, batch_size, num_points):
    model.eval()
    id_list = []
    pred_list = []
    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            ids, x, mask = batch
            x = x.to(device=device, dtype=torch.float)
            x = x.permute(0, 2, 1)
            mask = mask.to(device=device)
            idx_base = (
                torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
            )

            with torch.no_grad():
                pred = model(x, mask, None, idx_base)
                # pred = nn.functional.relu(pred) # ReLU
                pred = pred.squeeze()

            bar.set_description("Eval")
            bar.update(1)

            if batch_size == 1:
                id_list.append(ids[0])
                pred_list.append(pred.item())
            else:
                id_list += ids
                pred_list += pred.tolist()

    return id_list, pred_list


def init_random_seed(seed):
    random.seed(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Molecular Retention Time Prediction (Train)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./retention_index_data/dist_0/group_0.csv",
        help="path to data (csv)",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="./molnet_sc.yml",
        help="path to model and training configuration",
    )
    parser.add_argument(
        "--data_config_path",
        type=str,
        default="./src/molnetpack/config/preprocess_etkdgv3.yml",
        help="path to configuration",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="", help="Path to save checkpoint"
    )
    parser.add_argument(
        "--result_path", type=str, default="", help="Path to save the results"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random functions"
    )
    parser.add_argument("--device", type=int, default=0, help="Which gpu to use if any")
    parser.add_argument(
        "--no_cuda", type=bool, default=False, help="Enables CUDA training"
    )
    args = parser.parse_args()

    batch_size = 1  # batch size in evaluation

    init_random_seed(args.seed)
    with open(args.model_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(
        "Load the model & training configuration from {}".format(args.model_config_path)
    )
    with open(args.data_config_path, "r") as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    print("Load the data configuration from {}".format(args.data_config_path))

    # 0. Data preprocessing
    if args.data.endswith(".csv"):
        pkl_dict = csv2pkl_wfilter(args.data, data_config["encoding"])
        pkl_path = args.data.replace(".csv", ".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(pkl_dict, f)
            print("Save converted pkl file to {}".format(pkl_path))
    elif not args.data.endswith(".pkl"):
        raise ValueError("Unsupported data format:", args.data)
    else:
        pkl_path = args.data
        pkl_dict = pickle.load(open(pkl_path, "rb"))
        print("Load the data from {}".format(pkl_path))

    # 1. Data
    all_set = MolCSV_Test_Dataset(pkl_path)
    valid_loader = DataLoader(
        all_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        drop_last=True,
    )

    # 2. Model
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available() and not args.no_cuda
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    model = MolNet_Oth(config["model"]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{str(model)} #Params: {num_params}")

    # 3. Test
    result_dir = "/".join(args.result_path.split("/")[:-1])
    os.makedirs(result_dir, exist_ok=True)

    model.load_state_dict(
        torch.load(args.checkpoint_path, map_location=device)["model_state_dict"]
    )
    id_list, pred_list = test_step(
        model,
        device,
        valid_loader,
        batch_size=batch_size,
        num_points=config["model"]["max_atom_num"],
    )
    res_df = pd.DataFrame({"ID": id_list, "Pred Prop": pred_list})
    res_df.to_csv(args.result_path)
    print("Save the results to {}".format(args.result_path))
