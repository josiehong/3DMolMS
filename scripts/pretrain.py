import os
import argparse
import numpy as np
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from molnetpack import MolNet_Oth
from molnetpack import MolPRE_Dataset
from molnetpack import __version__


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_step(model, device, loader, optimizer, batch_size, num_points, prop_index):
    accuracy = 0
    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            _, x, mask, y = batch
            x = x.to(device=device, dtype=torch.float)
            x = x.permute(0, 2, 1)
            mask = mask.to(device=device)
            y = y[:, prop_index].to(device=device, dtype=torch.float).unsqueeze(1)
            idx_base = (
                torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
            )

            optimizer.zero_grad()
            model.train()
            pred = model(x, mask, None, idx_base)
            loss = nn.MSELoss()(pred, y)
            loss.backward()

            bar.set_description("Train")
            bar.set_postfix(lr=get_lr(optimizer), loss=loss.item())
            bar.update(1)

            optimizer.step()

            accuracy += torch.abs(pred - y).mean().item()
    return accuracy / (step + 1)


def eval_step(model, device, loader, batch_size, num_points, prop_index):
    model.eval()
    accuracy = 0
    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            _, x, mask, y = batch
            x = x.to(device=device, dtype=torch.float)
            x = x.permute(0, 2, 1)
            mask = mask.to(device=device)
            y = y[:, prop_index].to(device=device, dtype=torch.float).unsqueeze(1)
            idx_base = (
                torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
            )

            with torch.no_grad():
                pred = model(x, mask, None, idx_base)

            bar.set_description("Eval")
            bar.update(1)

            accuracy += torch.abs(pred - y).mean().item()
    return accuracy / (step + 1)


def init_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Molecular Mass Spectra Prediction (Pre-train)"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="./data/qm9_etkdg_train.pkl",
        help="path to training data (pkl)",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/qm9_etkdg_test.pkl",
        help="path to test data (pkl)",
    )
    parser.add_argument("--prop_index", type=int, default=3, help="Index of property")
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="./config/molnet.yml",
        help="path to model and training configuration",
    )
    parser.add_argument(
        "--data_config_path",
        type=str,
        default="./config/preprocess_etkdg.yml",
        help="path to configuration",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./check_point/molnet_pre_etkdg.pt",
        help="Path to save checkpoint",
    )
    parser.add_argument(
        "--resume_path", type=str, default="", help="Path to pretrained model"
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

    init_random_seed(args.seed)
    with open(args.model_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(
        "Load the model & training configuration from {}".format(args.model_config_path)
    )

    # 1. Data
    train_set = MolPRE_Dataset(args.train_data)
    train_loader = DataLoader(
        train_set,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        drop_last=True,
    )
    valid_set = MolPRE_Dataset(args.test_data)
    valid_loader = DataLoader(
        valid_set,
        batch_size=config["train"]["batch_size"],
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
    if args.resume_path != "":
        print("Load the checkpoints...")
        model.load_state_dict(
            torch.load(args.resume_path, map_location=device)["model_state_dict"]
        )
        optimizer.load_state_dict(
            torch.load(args.resume_path, map_location=device)["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            torch.load(args.resume_path, map_location=device)["scheduler_state_dict"]
        )
        best_valid_mae = torch.load(args.resume_path)["best_val_mae"]

    if args.checkpoint_path != "":
        checkpoint_dir = "/".join(args.checkpoint_path.split("/")[:-1])
        os.makedirs(checkpoint_dir, exist_ok=True)

    best_valid_mae = 999999
    early_stop_step = 10
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
            prop_index=args.prop_index,
        )
        valid_mae = eval_step(
            model,
            device,
            valid_loader,
            batch_size=config["train"]["batch_size"],
            num_points=config["model"]["max_atom_num"],
            prop_index=args.prop_index,
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
