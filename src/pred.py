import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import pickle
from pyteomics import mgf

import torch
from torch.utils.data import DataLoader

from rdkit import Chem

# ignore the warning
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import Descriptors

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from PIL import ImageFilter
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

from molnetpack import MolNet_MS
from molnetpack import Mol_Dataset
from molnetpack import csv2pkl_wfilter, nce2ce, precursor_calculator
from molnetpack import filter_spec, mgf2pkl, ms_vec2dict
from molnetpack import __version__

global batch_size
batch_size = 1


def pred_step(model, device, loader, batch_size, num_points):
    model.eval()
    id_list = []
    pred_list = []
    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            ids, x, mask, env = batch
            x = x.to(device=device, dtype=torch.float)
            x = x.permute(0, 2, 1)
            mask = mask.to(device=device)
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
            pred = torch.pow(pred, 2)
            # post process
            pred = pred.detach().cpu().apply_(lambda x: x if x > 0.01 else 0)

            id_list += ids
            pred_list.append(pred)

    pred_list = torch.cat(pred_list, dim=0)
    return id_list, pred_list


def init_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Molecular Mass Spectra Prediction (Pred)"
    )
    parser.add_argument(
        "--test_data", type=str, default="", help="path to test data (csv/mgf/pkl)"
    )
    parser.add_argument(
        "--save_pkl", action="store_true", help="save converted pkl file"
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
        default="./config/preprocess_etkdgv3.yml",
        help="path to data configuration",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="./check_point/molnet_qtof_etkdgv3.pt",
        help="path to pretrained model",
    )
    parser.add_argument(
        "--result_path", type=str, required=True, help="path to saving results"
    )
    parser.add_argument(
        "--save_img_dir", type=str, default="", help="path to saving images"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="seed for random functions"
    )
    parser.add_argument("--device", type=int, default=0, help="which gpu to use if any")
    parser.add_argument("--no_cuda", action="store_true", help="enables CUDA training")
    args = parser.parse_args()

    init_random_seed(args.seed)
    with open(args.model_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(
        "Load the model & training configuration from {}".format(args.model_config_path)
    )
    with open(args.data_config_path, "r") as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    print("Load the data configuration from {}".format(args.data_config_path))

    # 1. Data
    test_format = args.test_data.split(".")[-1]
    if test_format == "csv":  # convert csv file into pkl
        pkl_dict = csv2pkl_wfilter(args.test_data, data_config["encoding"])

    elif test_format == "mgf":  # convert mgf file into pkl
        origin_spectra = mgf.read(args.test_data)

        print("Filter spectra...")
        filter_spectra, _ = filter_spec(
            origin_spectra, config["all"], type2charge=config["encoding"]["type2charge"]
        )
        pkl_dict = mgf2pkl(filter_spectra, config["encoding"])

    elif test_format == "pkl":  # load pkl directly
        with open(args.test_data, "rb") as file:
            pkl_dict = pickle.load(file)

    else:
        raise ValueError("Unsupported format: {}".format(test_format))
    print("Load {} data from {}".format(len(pkl_dict), args.test_data))

    # same the pkl, so do not need to convert it again next time
    pkl_path = args.test_data.replace("." + test_format, ".pkl")
    if args.save_pkl:
        if not os.path.exists(pkl_path):
            raise OSError("The pkl file exists. Do not need to save it again. ")

        with open(pkl_path, "wb") as f:
            pickle.dump(pkl_dict, f)
            print("Save converted pkl file to {}".format(pkl_path))

    valid_set = Mol_Dataset(pkl_dict)
    valid_loader = DataLoader(
        valid_set,
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

    model = MolNet_MS(config["model"]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{str(model)} #Params: {num_params}")

    # 3. Evaluation
    print("Load the checkpoints...")
    if not os.path.exists(args.resume_path):
        raise OSError(
            "The checkpoint file does not exist. Please download it from release."
        )

    model.load_state_dict(
        torch.load(args.resume_path, map_location=device)["model_state_dict"]
    )
    best_valid_acc = torch.load(args.resume_path, map_location=device)["best_val_acc"]

    id_list, pred_list = pred_step(
        model,
        device,
        valid_loader,
        batch_size=batch_size,
        num_points=config["model"]["max_atom_num"],
    )
    pred_list = [
        ms_vec2dict(spec, float(config["model"]["resolution"]))
        for spec in pred_list.tolist()
    ]
    pred_mz = [pred["m/z"] for pred in pred_list]
    pred_intensity = [pred["intensity"] for pred in pred_list]

    # 4. Output the results
    result_dir = "".join(args.result_path.split("/")[:-1])
    os.makedirs(result_dir, exist_ok=True)

    # convert 'env' in pkl_dict back to experimental conditions
    decoding_precursor_type = {
        ",".join(map(str, v)): k
        for k, v in data_config["encoding"]["precursor_type"].items()
    }
    ce_list = []
    add_list = []
    smiles_list = []
    for d in pkl_dict:
        precursor_type = decoding_precursor_type[
            ",".join(map(str, map(int, d["env"][1:])))
        ]
        smiles = d["smiles"]
        mass = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
        precurosr_mz = precursor_calculator(precursor_type, mass)
        ce_list.append(
            nce2ce(
                d["env"][0],
                precurosr_mz,
                int(data_config["encoding"]["type2charge"][precursor_type]),
            )
        )
        add_list.append(precursor_type)
        smiles_list.append(smiles)

    # save the final results
    res_df = pd.DataFrame(
        {
            "ID": id_list,
            "SMILES": smiles_list,
            "Collision Energy": ce_list,
            "Precursor Type": add_list,
            "Pred M/Z": pred_mz,
            "Pred Intensity": pred_intensity,
        }
    )
    if args.result_path[-3:] == "csv":  # save results to csv file
        res_df.to_csv(args.result_path, sep="\t")
    elif args.result_path[-3:] == "mgf":  # save results to mgf file
        spectra = []
        prefix = "pred"
        for idx, row in res_df.iterrows():
            spectrum = {
                "params": {
                    "title": row["ID"],
                    "mslevel": "2",
                    "organism": "3DMolMS_v1.1",
                    "spectrumid": prefix + "_" + str(idx),
                    "smiles": row["SMILES"],
                    "collision_energy": row["Collision Energy"],
                    "precursor_type": row["Precursor Type"],
                },
                "m/z array": np.array([float(i) for i in row["Pred M/Z"].split(",")]),
                "intensity array": np.array(
                    [float(i) * 1000 for i in row["Pred Intensity"].split(",")]
                ),
            }
            spectra.append(spectrum)
        mgf.write(spectra, args.result_path, file_mode="w", write_charges=False)
    else:
        raise Exception(
            "Not implemented output format. Please choose `.csv` or `.mgf`."
        )
    print("Save the predicted MS/MS to {}".format(args.result_path))

    # plot
    if args.save_img_dir != "":
        img_dpi = 300
        y_max = 1
        x_max = None  # varies in different MS/MS
        bin_width = 0.4  # please adujst it for good looking
        figsize = (9, 4)

        for idx, row in res_df.iterrows():
            fig, ax = plt.subplots(figsize=figsize)
            mz_values = np.array([float(i) for i in row["Pred M/Z"].split(",")])
            x_max = np.max(mz_values)
            plt.bar(
                mz_values,
                np.array([float(i) * y_max for i in row["Pred Intensity"].split(",")]),
                width=bin_width,
                color="k",
            )
            plt.xlim(0, x_max)
            plt.title("ID: " + row["ID"])
            plt.xlabel("M/Z")
            plt.ylabel("Relative intensity")

            # plot the molecules
            mol = Chem.MolFromSmiles(row["SMILES"])
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            mol_img = Draw.MolToImage(mol, size=(800, 800))
            # make the backgrounf transparent
            alpha_img = mol_img.convert("L")
            alpha_img = Image.fromarray(255 - np.array(alpha_img))
            mol_img.putalpha(alpha_img)
            imagebox = OffsetImage(
                mol_img, zoom=72.0 / img_dpi
            )  # https://stackoverflow.com/questions/48639369/does-adding-images-in-pyplot-lowers-their-resolution
            mol_ab = AnnotationBbox(
                imagebox, (x_max * 0.28, y_max * 0.64), frameon=False, xycoords="data"
            )
            ax.add_artist(mol_ab)

            plt.savefig(
                os.path.join(args.save_img_dir, row["ID"]),
                dpi=img_dpi,
                bbox_inches="tight",
            )
            plt.close()

        print("Save the plotted MS/MS to {}".format(args.save_img_dir))
