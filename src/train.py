import os
import argparse
import numpy as np
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from molnetpack import MolNet_MS
from molnetpack import MolMS_Dataset
from molnetpack import __version__


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def reg_criterion(outputs, targets):
    # cosine similarity
    t = nn.CosineSimilarity(dim=1)
    spec_cosi = torch.mean(1 - t(outputs, targets))
    return spec_cosi


def train_step(model, device, loader, optimizer, batch_size, num_points):
    accuracy = 0
    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            _, x, mask, y, env = batch
            x = x.to(device=device, dtype=torch.float).permute(0, 2, 1)
            mask = mask.to(device=device)
            y = y.to(device=device, dtype=torch.float)
            env = env.to(device=device, dtype=torch.float)
            idx_base = (
                torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
            )

            optimizer.zero_grad()
            model.train()
            pred = model(x, mask, env, idx_base)
            loss = reg_criterion(pred, y)
            loss.backward()

            bar.set_description("Train")
            bar.set_postfix(lr=get_lr(optimizer), loss=loss.item())
            bar.update(1)

            optimizer.step()

            # recover sqrt spectra to original spectra
            y = torch.pow(y, 2)
            pred = torch.pow(pred, 2)
            accuracy += F.cosine_similarity(pred, y, dim=1).mean().item()
    return accuracy / (step + 1)


def eval_step(model, device, loader, batch_size, num_points):
    model.eval()
    accuracy = 0
    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            _, x, mask, y, env = batch
            x = x.to(device=device, dtype=torch.float)
            x = x.permute(0, 2, 1)
            mask = mask.to(device=device)
            y = y.to(device=device, dtype=torch.float)
            env = env.to(device=device, dtype=torch.float)
            idx_base = (
                torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
            )

            with torch.no_grad():
                pred = model(x, mask, env, idx_base)
                pred = pred / torch.max(pred)  # normalize the output

            bar.set_description("Eval")
            bar.update(1)

            # recover sqrt spectra to original spectra
            y = torch.pow(y, 2)
            pred = torch.pow(pred, 2)
            # post process
            pred = pred.detach().cpu().apply_(lambda x: x if x > 0.01 else 0).to(device)

            accuracy += F.cosine_similarity(pred, y, dim=1).mean().item()
    return accuracy / (step + 1)


def init_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Molecular Mass Spectra Prediction (Train)"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="./data/qtof_etkdg_train.pkl",
        help="path to training data (pkl)",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/qtof_etkdg_test.pkl",
        help="path to test data (pkl)",
    )
    parser.add_argument(
        "--precursor_type",
        type=str,
        default="All",
        choices=["All", "[M+H]+", "[M-H]-"],
        help="Precursor type",
    )
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
        default="./check_point/molnet_qtof_etkdg.pt",
        help="Path to save checkpoint",
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
    parser.add_argument("--no_cuda", action="store_true", help="Enables CUDA training")
    args = parser.parse_args()

    init_random_seed(args.seed)
    with open(args.model_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(
        "Load the model & training configuration from {}".format(args.model_config_path)
    )
    # configuration check
    assert (
        config["model"]["batch_size"] == config["train"]["batch_size"]
    ), "Batch size should be the same in model and training configuration"

    # 1. Data
    # convert precursor type to encoded precursor type for filtering
    with open(args.data_config_path, "r") as f:
        tmp = yaml.load(f, Loader=yaml.FullLoader)
        precursor_encoder = {}
        for k, v in tmp["encoding"]["precursor_type"].items():
            precursor_encoder[k] = ",".join([str(int(i)) for i in v])
        precursor_encoder["All"] = False
        del tmp

    train_set = MolMS_Dataset(args.train_data, precursor_encoder[args.precursor_type])
    train_loader = DataLoader(
        train_set,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        drop_last=True,
    )
    valid_set = MolMS_Dataset(args.test_data, precursor_encoder[args.precursor_type])
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

    model = MolNet_MS(config["model"]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{str(model)} #Params: {num_params}")

    # 3. Train
    optimizer = optim.AdamW(model.parameters(), lr=config["train"]["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    if args.transfer and args.resume_path != "":
        print("Load the pretrained encoder (freeze the encoder)...")
        state_dict = torch.load(
            args.resume_path, map_location=device, weights_only=True
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
        best_valid_acc = torch.load(args.resume_path, weights_only=True)["best_val_acc"]

    if args.checkpoint_path != "":
        checkpoint_dir = "/".join(args.checkpoint_path.split("/")[:-1])
        os.makedirs(checkpoint_dir, exist_ok=True)

    best_valid_acc = 0
    early_stop_step = 10
    early_stop_patience = 0
    for epoch in range(1, config["train"]["epochs"] + 1):
        print("\n=====Epoch {}".format(epoch))
        train_acc = train_step(
            model,
            device,
            train_loader,
            optimizer,
            batch_size=config["train"]["batch_size"],
            num_points=config["model"]["max_atom_num"],
        )
        valid_acc = eval_step(
            model,
            device,
            valid_loader,
            batch_size=config["train"]["batch_size"],
            num_points=config["model"]["max_atom_num"],
        )
        print("Train: Acc: {}, \nValidation: Acc: {}".format(train_acc, valid_acc))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc

            if args.checkpoint_path != "":
                print("Saving checkpoint...")
                checkpoint = {
                    "version": __version__,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_valid_acc,
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
        scheduler.step(valid_acc)  # ReduceLROnPlateau
        print(f"Best cosine similarity so far: {best_valid_acc}")

        if early_stop_patience == early_stop_step:
            print("Early stop!")
            break

    if args.ex_model_path != "":  # export the model
        print("Export the model...")
        # model_scripted = torch.jit.script(model) # Export to TorchScript
        # model_scripted.save(args.ex_model_path) # Save

        print("Export the traced model...")
        batch_size = config["train"]["batch_size"]  # Set to the desired batch size
        num_points = 300  # Replace with your actual num_points value

        # Create example inputs with the same data types and shapes as expected by your model
        x = torch.randn(batch_size, 21, num_points, device=device, dtype=torch.float)
        mask = torch.ones(batch_size, num_points, device=device, dtype=torch.bool)
        env = torch.randn(batch_size, 6, device=device, dtype=torch.float)
        idx_base = (
            torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        )
        example_inputs = (x, mask, env, idx_base)

        model.eval()
        traced_model = torch.jit.trace(model, example_inputs)
        torch.jit.save(traced_model, args.ex_model_path)
