import os
import argparse
import yaml
import pandas as pd
import pickle

from rdkit import Chem

# ignore the warning
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from molnetpack import filter_mol, sdf2pkl_with_cond


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the Data")
    parser.add_argument(
        "--raw_dir", type=str, default="./data/refmet/", help="path to raw data"
    )
    parser.add_argument(
        "--pkl_dir", type=str, default="./data/refmet/", help="path to pkl data"
    )
    parser.add_argument(
        "--data_config_path",
        type=str,
        default="./config/preprocess_etkdgv3.yml",
        help="path to configuration",
    )
    args = parser.parse_args()

    assert os.path.exists(os.path.join(args.raw_dir, "refmet.csv"))

    # load the configurations
    with open(args.data_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load the data in pd and convert to sdf
    df = pd.read_csv(os.path.join(args.raw_dir, "refmet.csv"))
    df = df.dropna(subset=["smiles"])
    supp = []
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol != None:
            mol.SetProp("DATABASE_ID", "REFMET_" + str(idx))
            supp.append(mol)
    print(
        "Read {} data from {}".format(
            len(supp), os.path.join(args.raw_dir, "refmet.csv")
        )
    )

    # filter the molecules
    supp, _ = filter_mol(supp, config["hmdb"])
    print("Filter to get {} molecules".format(len(supp)))

    # convert data to pkl (with multiple meta data)
    # ID,SMILES,Precursor_Type,Source_Instrument,Collision_Energy
    collision_energies = ["20 eV", "40 eV"]
    precursor_types = ["[M+H]+", "[M-H]-"]
    hmdb_pkl = sdf2pkl_with_cond(
        supp, config["encoding"], collision_energies, precursor_types
    )

    # save
    out_path = os.path.join(
        args.pkl_dir, "refmet_{}.pkl".format(config["encoding"]["conf_type"])
    )
    with open(out_path, "wb") as f:
        pickle.dump(hmdb_pkl, f)
        print("Save {}".format(out_path))

    print("Done!")
