import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from pyteomics import mgf

from rdkit import Chem

# ignore the warning
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

from molnetpack import sdf2mgf, filter_spec, mgf2pkl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the Data")
    parser.add_argument(
        "--raw_dir", type=str, default="./data/origin/", help="path to raw data"
    )
    parser.add_argument(
        "--pkl_dir", type=str, default="./data/", help="path to pkl data"
    )
    parser.add_argument(
        "--mgf_dir", type=str, default="", help="path to mgf data"
    )  # output mgf file for debuging and data analysis
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        required=True,
        choices=["agilent", "nist", "mona", "waters", "gnps"],
        help="dataset name",
    )
    parser.add_argument(
        "--instrument_type",
        type=str,
        nargs="+",
        required=True,
        choices=["qtof", "orbitrap"],
        help="dataset name",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.9, help="Ratio for train set"
    )
    parser.add_argument(
        "--maxmin_pick",
        action="store_true",
        help="If using MaxMin algorithm to pick training molecules",
    )
    parser.add_argument(
        "--data_config_path",
        type=str,
        default="./config/preprocess_etkdgv3.yml",
        help="path to configuration",
    )
    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)

    # create the directory if not exists
    if not os.path.exists(args.pkl_dir):
        os.makedirs(args.pkl_dir)
    if args.mgf_dir != "" and not os.path.exists(args.mgf_dir):
        os.makedirs(args.mgf_dir)

    assert args.train_ratio < 1.0

    if "agilent" in args.dataset:
        assert os.path.exists(os.path.join(args.raw_dir, "Agilent_Combined.sdf"))
        assert os.path.exists(os.path.join(args.raw_dir, "Agilent_Metlin.sdf"))
        assert os.path.exists(os.path.join(args.raw_dir, "hr_msms_nist.SDF"))
    if "nist" in args.dataset:
        assert os.path.exists(os.path.join(args.raw_dir, "hr_msms_nist.SDF"))
    if "mona" in args.dataset:
        assert os.path.exists(
            os.path.join(args.raw_dir, "MoNA-export-All_LC-MS-MS_QTOF.sdf")
        )
        assert os.path.exists(
            os.path.join(args.raw_dir, "MoNA-export-All_LC-MS-MS_Orbitrap.sdf")
        )
    if "waters" in args.dataset:
        assert os.path.exists(os.path.join(args.raw_dir, "waters_qtof.mgf"))
    if "gnps" in args.dataset:
        assert os.path.exists(os.path.join(args.raw_dir, "ALL_GNPS_cleaned.mgf"))
        assert os.path.exists(os.path.join(args.raw_dir, "ALL_GNPS_cleaned.csv"))

    # load the configurations
    with open(args.data_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 1. convert original format to mgf
    print("\n>>> Step 1: convert original format to mgf;")
    origin_spectra = {}
    if "agilent" in args.dataset:
        spectra1 = sdf2mgf(
            path=os.path.join(args.raw_dir, "Agilent_Combined.sdf"),
            prefix="agilent_combine",
        )
        spectra2 = sdf2mgf(
            path=os.path.join(args.raw_dir, "Agilent_Metlin.sdf"),
            prefix="agilent_metlin",
        )
        origin_spectra["agilent"] = spectra1 + spectra2
    if "nist" in args.dataset:
        origin_spectra["nist"] = sdf2mgf(
            path=os.path.join(args.raw_dir, "hr_msms_nist.SDF"), prefix="nist20"
        )
    if "mona" in args.dataset:
        spectra1 = sdf2mgf(
            path=os.path.join(args.raw_dir, "MoNA-export-All_LC-MS-MS_QTOF.sdf"),
            prefix="mona_qtof",
        )
        spectra2 = sdf2mgf(
            path=os.path.join(args.raw_dir, "MoNA-export-All_LC-MS-MS_Orbitrap.sdf"),
            prefix="mona_orbitrap",
        )
        origin_spectra["mona"] = spectra1 + spectra2
    if "waters" in args.dataset:
        origin_spectra["waters"] = mgf.read(
            os.path.join(args.raw_dir, "waters_qtof.mgf")
        )
        print(
            "Load {} data from {}".format(
                len(origin_spectra["waters"]),
                os.path.join(args.raw_dir, "waters_qtof.mgf"),
            )
        )
    if "gnps" in args.dataset:
        raw_spectra = mgf.read(os.path.join(args.raw_dir, "ALL_GNPS_cleaned.mgf"))
        all_metadata = pd.read_csv(os.path.join(args.raw_dir, "ALL_GNPS_cleaned.csv"))
        all_metadata = all_metadata[
            [
                "spectrum_id",
                "Adduct",
                "Precursor_MZ",
                "ExactMass",
                "Ion_Mode",
                "msMassAnalyzer",
                "msDissociationMethod",
                "Smiles",
                "InChIKey_smiles",
                "collision_energy",
            ]
        ]
        all_metadata["collision_energy"] = all_metadata["collision_energy"].fillna(
            "Unknown"
        )
        all_metadata = all_metadata.dropna()
        all_metadata["Adduct"] = all_metadata["Adduct"].apply(lambda x: x[:-2] + x[-1:])
        all_metadata["collision_energy"] = all_metadata["collision_energy"].astype(str)
        all_metadata["Ion_Mode"] = all_metadata["Ion_Mode"].apply(lambda x: x.upper())
        all_metadata = (
            all_metadata.rename(
                columns={
                    "spectrum_id": "title",
                    "Adduct": "precursor_type",
                    "Precursor_MZ": "precursor_mz",
                    "ExactMass": "molmass",
                    "Ion_Mode": "ionmode",
                    "msMassAnalyzer": "source_instrument",
                    "msDissociationMethod": "instrument_type",
                    "Smiles": "smiles",
                    "InChIKey_smiles": "inchi_key",
                }
            )
            .set_index("title")
            .to_dict("index")
        )
        # add meta-data into the spectra
        tmp_spectra = []
        for idx, spec in enumerate(tqdm(raw_spectra)):
            title = spec["params"]["title"]
            if title in all_metadata.keys():
                metadata = all_metadata[title]
                spec["params"] = metadata
                spec["params"]["title"] = "gnps_" + str(idx)
                tmp_spectra.append(spec)
        origin_spectra["gnps"] = tmp_spectra
        print(
            "Read {} data from {}".format(
                len(origin_spectra["gnps"]),
                os.path.join(args.raw_dir, "ALL_GNPS_cleaned.mgf"),
            )
        )

    # 2. filter the spectra
    # 3. split spectra into training and test set according to smiles
    # Note that there is not overlapped molecules between training set and tes set.
    # 4. generate 3d conformattions & encoding data into arrays
    print(
        "\n>>> Step 2: filter out spectra by certain rules; \n\
\tStep 3: split SMILES into training set and test set; \n\
\tStep 4: encode all the data into pkl format;"
    )
    for ins in args.instrument_type:
        spectra = []
        smiles_list = []
        for ds in args.dataset:
            config_name = ds + "_" + ins
            if config_name not in config.keys():
                continue
            print("({}) Filter {} spectra...".format(ins, config_name))
            filter_spectra, filter_smiles_list = filter_spec(
                origin_spectra[ds],
                config[config_name],
                type2charge=config["encoding"]["type2charge"],
            )
            # mgf.write(filter_spectra, './data/mgf_debug/{}.mgf'.format(config_name), file_mode="w") # save mgf debug
            filter_smiles_list = list(set(filter_smiles_list))
            spectra += filter_spectra
            smiles_list += filter_smiles_list
            del filter_spectra, filter_smiles_list
        smiles_list = list(set(smiles_list))

        # save mgf for debuging
        if args.mgf_dir != "":
            mgf.write(
                spectra,
                output=os.path.join(
                    args.mgf_dir, "{}_{}.mgf".format(ins, "_".join(args.dataset))
                ),
            )

        if args.maxmin_pick:
            print("({}) Split training and test set by MaxMin algorithm...".format(ins))
            # maxmin algorithm picking training smiles (Since the training set so far is not large enough,
            # we'd like to make a full utilizing of our datasets. It worth noting that MaxMin picker is
            # for application, and the experiments in our paper spliting the molecules randomly according
            # to their smiles. )
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3)
            fp_list = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]
            picker = MaxMinPicker()
            train_indices = picker.LazyBitVectorPick(
                fp_list, len(fp_list), int(len(fp_list) * args.train_ratio), seed=42
            )
            train_indices = list(train_indices)
        else:
            print("({}) Split training and test set by randomly choose...".format(ins))
            train_indices = np.random.choice(
                len(smiles_list),
                int(len(smiles_list) * args.train_ratio),
                replace=False,
            )

        train_smiles_list = [
            smiles_list[i] for i in range(len(smiles_list)) if i in train_indices
        ]
        test_smiles_list = [s for s in smiles_list if s not in train_smiles_list]
        print(
            "({}) Get {} training compounds and {} test compounds".format(
                ins, len(train_smiles_list), len(test_smiles_list)
            )
        )

        train_spectra = []
        test_spectra = []
        for _, spectrum in enumerate(tqdm(spectra)):
            smiles = spectrum["params"]["smiles"]
            if smiles in train_smiles_list:
                train_spectra.append(spectrum)
            else:
                test_spectra.append(spectrum)
        del spectra, smiles_list, test_smiles_list
        print(
            "({}) Get {} training spectra and {} test spectra".format(
                ins, len(train_spectra), len(test_spectra)
            )
        )

        print("({}) Convert spectra and molecules data into arrays...".format(ins))
        # test
        test_data = mgf2pkl(test_spectra, config["encoding"])
        out_path = os.path.join(
            args.pkl_dir, "{}_{}_test.pkl".format(ins, config["encoding"]["conf_type"])
        )
        with open(out_path, "wb") as f:
            pickle.dump(test_data, f)
            print("Save {}".format(out_path))
        # train
        train_data = mgf2pkl(train_spectra, config["encoding"])
        out_path = os.path.join(
            args.pkl_dir, "{}_{}_train.pkl".format(ins, config["encoding"]["conf_type"])
        )
        with open(out_path, "wb") as f:
            pickle.dump(train_data, f)
            print("Save {}".format(out_path))

    print("Done!")
