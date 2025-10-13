import os
import argparse
import yaml
import pickle

from rdkit import Chem

# ignore the warning
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from molnetpack import filter_mol, sdf2pkl_with_cond


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the Data")
    parser.add_argument(
        "--raw_dir", type=str, default="./data/hmdb/", help="path to raw data"
    )
    parser.add_argument(
        "--sdf_file", type=str, default="structures.sdf", help="name of the sdf file"
    )
    parser.add_argument(
        "--pkl_dir", type=str, default="./data/hmdb/", help="path to pkl data"
    )
    parser.add_argument(
        "--data_config_path",
        type=str,
        default="./config/preprocess_hmdb.yml",
        help="path to configuration",
    )
    args = parser.parse_args()

    sdf_path = os.path.join(args.raw_dir, args.sdf_file)
    assert os.path.exists(sdf_path)

    # load the configurations
    with open(args.data_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load the data in sdf
    supp = Chem.SDMolSupplier(sdf_path)
    print("Read {} data from {}".format(len(supp), sdf_path))

    # split the HMDB by chunks
    chunk_size = 10000
    chunk_num = len(supp) // chunk_size
    chunk_num = chunk_num + 1 if len(supp) % chunk_size > 0 else chunk_num
    print(
        "Processing HMDB by chunks: # chunk size: {}, # chunk number: {}".format(
            chunk_size, chunk_num
        )
    )

    for i in range(chunk_num):
        out_path = os.path.join(
            args.pkl_dir, "hmdb_{}_{}.pkl".format(config["encoding"]["conf_type"], i)
        )
        if os.path.exists(out_path):
            print("skip chunk {}".format(i))
            continue

        if (i + 1) * chunk_size <= len(supp):
            supp_tmp = [supp[j] for j in range(i * chunk_size, (i + 1) * chunk_size)]
        else:
            supp_tmp = [supp[j] for j in range(i * chunk_size, len(supp))]

        # filter the molecules
        supp_tmp, _ = filter_mol(supp_tmp, config["hmdb"])
        print("(Chunk {}) Filter to get {} molecules".format(i, len(supp_tmp)))

        # convert data to pkl (with multiple meta data)
        collision_energies = ["20 eV", "40 eV"]
        precursor_types = ["[M+H]+", "[M-H]-"]
        hmdb_pkl_tmp = sdf2pkl_with_cond(
            supp_tmp, config["encoding"], collision_energies, precursor_types
        )

        # save
        with open(out_path, "wb") as f:
            pickle.dump(hmdb_pkl_tmp, f)
            print("Save {}".format(out_path))

    print("Done!")
