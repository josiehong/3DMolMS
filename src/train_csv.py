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
from molnetpack.dataset import MolCSV_Dataset
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

        data.append(
            {
                "title": row["id"],
                "smiles": row["smiles"],
                "mol": mol_arr,
                "prop": row["prop"],
            }
        )
    return data


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_step(model, device, loader, optimizer, batch_size, num_points):
    accuracy = 0
    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            _, x, mask, y = batch
            x = x.to(device=device, dtype=torch.float)
            x = x.permute(0, 2, 1)
            mask = mask.to(device=device)
            y = y.to(device=device, dtype=torch.float)
            idx_base = (
                torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
            )

            optimizer.zero_grad()
            model.train()
            pred = model(x, mask, None, idx_base)
            # pred = nn.functional.relu(pred) # ReLU
            pred = pred.squeeze()
            loss = nn.MSELoss()(pred, y)
            loss.backward()

            bar.set_description("Train")
            bar.set_postfix(lr=get_lr(optimizer), loss=loss.item())
            bar.update(1)

            optimizer.step()

            accuracy += torch.abs(pred - y).mean().item()
    return accuracy / (step + 1)


def eval_step(model, device, loader, batch_size, num_points):
    model.eval()
    accuracy = 0
    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            _, x, mask, y = batch
            x = x.to(device=device, dtype=torch.float)
            x = x.permute(0, 2, 1)
            mask = mask.to(device=device)
            y = y.to(device=device, dtype=torch.float)
            idx_base = (
                torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
            )

            with torch.no_grad():
                pred = model(x, mask, None, idx_base)
                # pred = nn.functional.relu(pred) # ReLU
                pred = pred.squeeze()

            bar.set_description("Eval")
            bar.update(1)

            accuracy += torch.abs(pred - y).mean().item()
    return accuracy / (step + 1)


def test_step(model, device, loader, batch_size, num_points):
    model.eval()
    id_list = []
    pred_list = []
    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            ids, x, mask, _ = batch
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
        default="./molnetpack/test/input_ccs.csv",
        help="path to data (csv)",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="./src/config/molnet_rt.yml",
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
        "--resume_path", type=str, default="", help="Path to pretrained model"
    )
    parser.add_argument(
        "--transfer", action="store_true", help="Whether to load the pretrained encoder"
    )
    parser.add_argument(
        "--ex_model_path",
        type=str,
        default="",
        help="Path to export the whole model (structure & weights)",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random functions"
    )
    parser.add_argument("--device", type=int, default=0, help="Which gpu to use if any")
    parser.add_argument(
        "--no_cuda", type=bool, default=False, help="Enables CUDA training"
    )
    args = parser.parse_args()

    batch_size = 2  # batch size in evaluation

    init_random_seed(args.seed)
    with open(args.model_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(
        "Load the model & training configuration from {}".format(args.model_config_path)
    )
    with open(args.data_config_path, "r") as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    print("Load the data configuration from {}".format(args.data_config_path))
    # configuration check
    assert (
        config["model"]["batch_size"] == config["train"]["batch_size"]
    ), "Batch size should be the same in model and training configuration"

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

    # spliting the dataset
    unique_train_smiles = set()
    unique_test_smiles = set()
    train_indices = []
    test_indices = []
    for idx in range(len(pkl_dict)):
        smiles = pkl_dict[idx]["smiles"]
        if (
            smiles in unique_train_smiles
        ):  # if compond already in train set then add to train set
            train_indices.append(idx)
        if smiles in unique_test_smiles:
            test_indices.append(idx)
        else:
            if random.random() < 0.9:  # Adjust the split ratio as needed
                train_indices.append(idx)
                unique_train_smiles.add(smiles)
            else:
                test_indices.append(idx)
                unique_test_smiles.add(smiles)
    train_pkl = [pkl_dict[idx] for idx in train_indices]
    valid_pkl = [pkl_dict[idx] for idx in test_indices]

    # 1. Data
    # all_set = MolCSV_Dataset(pkl_path)
    # randomly split the dataset (data leaky)
    # train_set, valid_set = torch.utils.data.random_split(all_set, [0.9, 0.1])
    train_set = MolCSV_Dataset(train_pkl, mode="data")
    valid_set = MolCSV_Dataset(valid_pkl, mode="data")
    train_loader = DataLoader(
        train_set,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=True,
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

    # 3. Train
    optimizer = optim.AdamW(model.parameters(), lr=config["train"]["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    if args.transfer and args.resume_path != "":
        print("Load the pretrained encoder (freeze the encoder)...")
        state_dict = torch.load(
            args.resume_path, map_location=device, weights_only=False
        )["model_state_dict"]
        encoder_dict = {}
        for name, param in state_dict.items():
            if not name.startswith("decoder"):
                param.requires_grad = False  # freeze the encoder
                encoder_dict[name] = param
        model.load_state_dict(encoder_dict, strict=False)
    elif args.resume_path != "":
        print("Load the checkpoints...")
        model.load_state_dict(
            torch.load(args.resume_path, map_location=device, weights_only=True)[
                "model_state_dict"
            ]
        )
        optimizer.load_state_dict(
            torch.load(args.resume_path, map_location=device, weights_only=True)[
                "optimizer_state_dict"
            ]
        )
        scheduler.load_state_dict(
            torch.load(args.resume_path, map_location=device, weights_only=True)[
                "scheduler_state_dict"
            ]
        )
        best_valid_mae = torch.load(args.resume_path, weights_only=True)["best_val_mae"]

    if args.checkpoint_path != "":
        checkpoint_dir = "/".join(args.checkpoint_path.split("/")[:-1])
        os.makedirs(checkpoint_dir, exist_ok=True)

    best_valid_mae = 999999
    early_stop_step = 30
    early_stop_patience = 0
    for epoch in range(1, config["train"]["epochs"] + 1):
        print("\n=====Epoch {}".format(epoch))
        train_mae = train_step(
            model,
            device,
            train_loader,
            optimizer,
            batch_size=config["train"]["batch_size"],
            num_points=config["model"]["max_atom_num"],
        )
        valid_mae = eval_step(
            model,
            device,
            valid_loader,
            batch_size=batch_size,
            num_points=config["model"]["max_atom_num"],
        )
        print("Train: MAE: {}, \nValidation: MAE: {}".format(train_mae, valid_mae))

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae

            if args.checkpoint_path != "":
                print("Saving checkpoint...")
                checkpoint = {
                    "version": __version__,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_mae": best_valid_mae,
                    "num_params": num_params,
                }
                torch.save(checkpoint, args.checkpoint_path)

            early_stop_patience = 0
            print("Early stop patience reset")
        else:
            early_stop_patience += 1
            print(
                "Early stop count: {}/{}".format(early_stop_patience, early_stop_step)
            )

        # scheduler.step()
        scheduler.step(valid_mae)  # ReduceLROnPlateau
        print(f"Best MAE so far: {best_valid_mae}")

        if early_stop_patience == early_stop_step:
            print("Early stop!")
            break

    if args.ex_model_path != "":  # export the model
        raise NotImplementedError("Exporting the model is not implemented yet!")

    if args.result_path != "":
        result_dir = "/".join(args.result_path.split("/")[:-1])
        os.makedirs(result_dir, exist_ok=True)

        model.load_state_dict(
            torch.load(args.checkpoint_path, map_location=device, weights_only=True)[
                "model_state_dict"
            ]
        )
        id_list, pred_list = test_step(
            model,
            device,
            valid_loader,
            batch_size=batch_size,
            num_points=config["model"]["max_atom_num"],
        )
        res_df = pd.DataFrame({"ID": id_list, "Pred Time": pred_list})
        res_df.to_csv(args.result_path)
        print("Save the results to {}".format(args.result_path))
