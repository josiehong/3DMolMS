import os
import requests
import numpy as np
import pandas as pd
import yaml
import pickle
from pathlib import Path
from pyteomics import mgf
import zipfile

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    print("PyTorch is not installed. Please install it to use this module.")
    raise

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Draw, AllChem
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from .model import MolNet_MS, MolNet_Oth, Encoder
from .dataset import Mol_Dataset
from .data_utils import (
    csv2pkl_wfilter,
    nce2ce,
    precursor_calculator,
    filter_spec,
    mgf2pkl,
    ms_vec2dict,
)
from .utils import pred_step, eval_step_oth, pred_feat
from ._version import __version__

# Disable RDKit warnings
RDLogger.DisableLog("rdApp.*")


class MolNet:
    def __init__(self, device, seed):
        self.version = __version__
        print("MolNetPack version:", self.version)

        self.device = device
        self.current_path = Path(__file__).parent

        # Load configurations
        self.data_config = self._load_config("preprocess_etkdgv3.yml")
        self.msms_config = self._load_config("molnet.yml")
        self.ccs_config = self._load_config("molnet_ccs_tl.yml")
        self.rt_config = self._load_config("molnet_rt_tl.yml")

        # Initialize variables
        self.pkl_dict = None
        self.valid_loader = None
        self.batch_size = None

        self.msms_model = None
        self.qtof_msms_res_df = None
        self.orbitrap_msms_res_df = None

        self.ccs_model = None
        self.ccs_res_df = None

        self.rt_model = None
        self.rt_res_df = None

        self.encoder = None

        self._init_random_seed(seed)

    def get_data(self):
        return self.pkl_dict

    def _load_config(self, filename):
        config_path = self.current_path / f"./config/{filename}"
        with open(config_path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def _init_random_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def load_data(self, path_to_test_data): 
        """Load data from the specified path.

        :param path_to_test_data: Path to the test data file. Supported formats are 'csv', 'mgf', and 'pkl'.
		:type path_to_test_data: str
		:return: None
		:rtype: None

        This method loads molecular data from the given file path using appropriate loaders
        based on file extension. For CSV files, it applies filters according to the configuration.
        For MGF files, it filters spectra and converts them to the internal format.
        For PKL files, it loads the pickled data directly.
        """
        loaders = {
            "csv": lambda p: csv2pkl_wfilter(p, self.data_config["encoding"]),
            "mgf": lambda p: mgf2pkl(
                filter_spec(
                    mgf.read(p),
                    self.data_config["all"],
                    self.data_config["encoding"]["type2charge"],
                )[0],
                self.data_config["encoding"],
            ),
            "pkl": lambda p: pickle.load(open(p, "rb")),
        }

        ext = path_to_test_data.split(".")[-1]
        if ext in loaders:
            self.pkl_dict = loaders[ext](path_to_test_data)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        print(f"\nLoad {len(self.pkl_dict)} data from {path_to_test_data}")
        self.valid_loader = DataLoader(
            Mol_Dataset(self.pkl_dict),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

    def load_checkpoint(self, task_name, path_to_checkpoint=None, instrument=None):
        checkpoint_path = str(
            self.current_path / self._get_checkpoint_path(task_name, instrument)
        )
        if path_to_checkpoint:
            self._load_model(path_to_checkpoint, task_name)
        else:
            self._download_and_load_checkpoint(checkpoint_path, task_name, instrument)

    def _get_checkpoint_path(self, task_name, instrument):
        task_map = {
            "msms": (
                self.msms_config["test"]["local_path_qtof"]
                if instrument == "qtof"
                else self.msms_config["test"]["local_path_orbitrap"]
            ),
            "ccs": self.ccs_config["test"]["local_path"],
            "rt": self.rt_config["test"]["local_path"],
            "save_feat": (
                self.msms_config["test"]["local_path_qtof"]
                if instrument == "qtof"
                else self.msms_config["test"]["local_path_orbitrap"]
            ),
        }
        return task_map.get(task_name)

    def _download_and_load_checkpoint(
        self, checkpoint_path, task_name, instrument=None
    ):
        if not os.path.exists(checkpoint_path):
            checkpoint_dir = os.path.dirname(checkpoint_path)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_zip_path = f"{checkpoint_path}.zip"

            # URL of the GitHub Release asset
            if task_name == "ccs":
                github_url = self.ccs_config["test"]["github_release_url"]
            elif task_name == "rt":
                github_url = self.rt_config["test"]["github_release_url"]
            else:
                if instrument == "qtof":
                    github_url = self.msms_config["test"]["github_release_url_qtof"]
                else:
                    github_url = self.msms_config["test"]["github_release_url_orbitrap"]
            print(
                f"Downloading the checkpoints from GitHub Release to {checkpoint_zip_path}"
            )

            # Download the file
            response = requests.get(github_url)
            with open(checkpoint_zip_path, "wb") as f:
                f.write(response.content)

            print(f"Unzipping {checkpoint_zip_path}")
            with zipfile.ZipFile(checkpoint_zip_path, "r") as zip_ref:
                zip_ref.extractall(checkpoint_dir)

        self._load_model(checkpoint_path, task_name)

    def _load_model(self, checkpoint_path, task_name):
        model_map = {
            "msms": self.msms_model,
            "ccs": self.ccs_model,
            "rt": self.rt_model,
            "save_feat": self.encoder,
        }
        model = model_map.get(task_name)
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)[
                "model_state_dict"
            ]
        )

    def save_features(self, checkpoint_path=None, instrument="qtof"):
        """Extract molecular feature embeddings from the encoder model.

        :param checkpoint_path: Optional path to a custom encoder model checkpoint file.
        :type checkpoint_path: str, optional
        :param instrument: Instrument type to use for feature extraction, either 'qtof' or 'orbitrap'.
        :type instrument: str, default='qtof'
        :return: Tuple containing molecule IDs and corresponding feature embeddings.
        :rtype: tuple(list, numpy.ndarray)

        This method extracts latent feature embeddings from the encoder component of the model
        for the loaded molecules. These features can be used for downstream tasks such as
        clustering, similarity search, or transfer learning.
        """
        self.encoder = Encoder(**self.msms_config["model"]).to(self.device)
        self.load_checkpoint("save_feat", checkpoint_path, instrument)
        ids, features = pred_feat(
            self.encoder,
            self.device,
            self.valid_loader,
            batch_size=1,
            num_points=self.msms_config["model"]["max_atom_num"],
        )
        return ids, features.cpu().detach().numpy()

    def pred_msms(
        self, path_to_results=None, path_to_checkpoint=None, instrument="qtof"
    ):
        """Predict MS/MS spectra for loaded molecules.

        :param path_to_results: Optional path to save prediction results. Supports '.mgf' and '.csv' formats.
        :type path_to_results: str, optional
        :param path_to_checkpoint: Optional path to a custom model checkpoint file.
        :type path_to_checkpoint: str, optional
        :param instrument: Instrument type to use for prediction, either 'qtof' or 'orbitrap'.
        :type instrument: str, default='qtof'
        :return: DataFrame containing MS/MS prediction results.
        :rtype: pandas.DataFrame

        This method predicts MS/MS spectra for previously loaded molecules using the specified
        instrument type. Results include m/z values, intensities, and associated metadata.
        If a path is provided, results are saved in either MGF or CSV format.
        """
        assert instrument in [
            "qtof",
            "orbitrap",
        ], 'Instrument should be either "qtof" or "orbitrap".'

        self.msms_model = MolNet_MS(self.msms_config["model"]).to(self.device)
        self.load_checkpoint("msms", path_to_checkpoint, instrument)
        id_list, pred_list = pred_step(
            self.msms_model,
            self.device,
            self.valid_loader,
            batch_size=1,
            num_points=self.msms_config["model"]["max_atom_num"],
        )
        pred_list = [
            ms_vec2dict(spec, float(self.msms_config["model"]["resolution"]))
            for spec in pred_list.tolist()
        ]
        pred_msms_df = self._assemble_msms_results(id_list, pred_list, instrument)
        if path_to_results:
            result_dir = os.path.dirname(path_to_results)
            if result_dir:
                os.makedirs(result_dir, exist_ok=True)

            if path_to_results.endswith(".mgf"):
                spectra = self.generate_spectra_from_df(pred_msms_df, instrument)
                mgf.write(spectra, path_to_results, file_mode="w", write_charges=False)
                print(f"\nSaved the results to {path_to_results}")
            elif path_to_results.endswith(".csv"):
                pred_msms_df.to_csv(path_to_results, index=False)
                print(f"\nSaved the results to {path_to_results}")
        return pred_msms_df

    def _assemble_msms_results(self, id_list, pred_list, instrument):
        ce_list, add_list, smiles_list = [], [], []
        decoding_precursor_type = {
            ",".join(map(str, v)): k
            for k, v in self.data_config["encoding"]["precursor_type"].items()
        }
        for d in self.pkl_dict:
            precursor_type = decoding_precursor_type[
                ",".join(map(str, map(int, d["env"][1:])))
            ]
            smiles = d["smiles"]
            mass = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
            precursor_mz = precursor_calculator(precursor_type, mass)
            ce_list.append(
                nce2ce(
                    d["env"][0],
                    precursor_mz,
                    int(self.data_config["encoding"]["type2charge"][precursor_type]),
                )
            )
            add_list.append(precursor_type)
            smiles_list.append(smiles)

        pred_mz, pred_intensity = zip(*[(p["m/z"], p["intensity"]) for p in pred_list])
        if instrument == "qtof":
            self.qtof_msms_res_df = pd.DataFrame(
                {
                    "ID": id_list,
                    "SMILES": smiles_list,
                    "Collision Energy": ce_list,
                    "Precursor Type": add_list,
                    "Pred M/Z": pred_mz,
                    "Pred Intensity": pred_intensity,
                }
            )
            return self.qtof_msms_res_df
        else:
            self.orbitrap_msms_res_df = pd.DataFrame(
                {
                    "ID": id_list,
                    "SMILES": smiles_list,
                    "Collision Energy": ce_list,
                    "Precursor Type": add_list,
                    "Pred M/Z": pred_mz,
                    "Pred Intensity": pred_intensity,
                }
            )
            return self.orbitrap_msms_res_df

    def generate_spectra_from_df(self, df, instrument=None):
        spectra = []
        for idx, row in df.iterrows():
            spectrum = {
                "params": {
                    "title": row["ID"],
                    "mslevel": "2",
                    "organism": "3DMolMS_{}".format(self.version),
                    "spectrumid": f"pred_{idx}",
                    "smiles": row["SMILES"],
                    "collision_energy": row["Collision Energy"],
                    "precursor_type": row["Precursor Type"],
                    "instrument_type": instrument,
                },
                "m/z array": np.array(
                    [float(i) for i in row["Pred M/Z"].split(",") if i]
                ),
                "intensity array": np.array(
                    [float(i) * 1000 for i in row["Pred Intensity"].split(",") if i]
                ),
            }
            spectra.append(spectrum)
        return spectra

    def pred_ccs(self, path_to_results=None, path_to_checkpoint=None):
        """Predict collision cross section (CCS) values for loaded molecules.

        :param path_to_results: Optional path to save prediction results as CSV.
        :type path_to_results: str, optional
        :param path_to_checkpoint: Optional path to a custom model checkpoint file.
        :type path_to_checkpoint: str, optional
        :return: DataFrame containing CCS prediction results.
        :rtype: pandas.DataFrame

        This method predicts collision cross section values for previously loaded molecules
        using a trained deep learning model. Results include molecule IDs, SMILES structures,
        precursor types, and predicted CCS values in a pandas DataFrame.
        """
        self.ccs_model = MolNet_Oth(self.ccs_config["model"]).to(self.device)
        self.load_checkpoint("ccs", path_to_checkpoint)
        id_list, pred_list = eval_step_oth(
            self.ccs_model,
            self.device,
            self.valid_loader,
            batch_size=1,
            num_points=self.ccs_config["model"]["max_atom_num"],
        )

        decoding_precursor_type = {
            ",".join(map(str, v)): k
            for k, v in self.data_config["encoding"]["precursor_type"].items()
        }
        add_list, smiles_list = [], []
        for d in self.pkl_dict:
            precursor_type = decoding_precursor_type[
                ",".join(map(str, map(int, d["env"][1:])))
            ]
            add_list.append(precursor_type)
            smiles_list.append(d["smiles"])

        self.ccs_res_df = pd.DataFrame(
            {
                "ID": id_list,
                "SMILES": smiles_list,
                "Precursor Type": add_list,
                "Pred CCS": pred_list.squeeze(),
            }
        )

        if path_to_results:
            result_dir = os.path.dirname(path_to_results)
            if result_dir:
                os.makedirs(result_dir, exist_ok=True)

            self.ccs_res_df.to_csv(path_to_results, index=False)
            print(f"\nSaved the results to {path_to_results}")
        return self.ccs_res_df

    def pred_rt(self, path_to_results=None, path_to_checkpoint=None):
        """Predict retention times for loaded molecules.

        :param path_to_results: Optional path to save prediction results as CSV.
        :type path_to_results: str, optional
        :param path_to_checkpoint: Optional path to a custom model checkpoint file.
        :type path_to_checkpoint: str, optional
        :return: DataFrame containing retention time prediction results.
        :rtype: pandas.DataFrame

        This method predicts chromatographic retention times for previously loaded molecules
        using a trained deep learning model. Results include molecule IDs, SMILES structures,
        and predicted retention times in a pandas DataFrame.
        """
        self.rt_model = MolNet_Oth(self.rt_config["model"]).to(self.device)
        self.load_checkpoint("rt", path_to_checkpoint)
        id_list, pred_list = eval_step_oth(
            self.rt_model,
            self.device,
            self.valid_loader,
            batch_size=1,
            num_points=self.rt_config["model"]["max_atom_num"],
        )

        smiles_list = []
        for d in self.pkl_dict:
            smiles_list.append(d["smiles"])

        self.rt_res_df = pd.DataFrame(
            {"ID": id_list, "SMILES": smiles_list, "Pred RT": pred_list.squeeze()}
        )

        if path_to_results:
            result_dir = os.path.dirname(path_to_results)
            if result_dir:
                os.makedirs(result_dir, exist_ok=True)

            self.rt_res_df.to_csv(path_to_results, index=False)
            print(f"\nSaved the results to {path_to_results}")
        return self.rt_res_df


def plot_msms(msms_res_df, dir_to_img):
    """Plot MS/MS spectra with molecular structures.

    :param msms_res_df: DataFrame containing MS/MS prediction results.
    :type msms_res_df: pandas.DataFrame
    :param dir_to_img: Directory path to save the generated spectrum plots.
    :type dir_to_img: str
    :return: None
    :rtype: None

    This function creates visualization plots for MS/MS spectra, displaying m/z values
    versus relative intensities as bar charts. Each plot includes the 2D structure
    of the corresponding molecule. Images are saved to the specified directory.
    """
    os.makedirs(dir_to_img, exist_ok=True)

    img_dpi = 300
    y_max = 1
    x_max = None  # varies in different MS/MS
    bin_width = 0.4  # please adujst it for good looking
    figsize = (9, 4)

    for idx, row in msms_res_df.iterrows():
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
            os.path.join(dir_to_img, row["ID"]), dpi=img_dpi, bbox_inches="tight"
        )
        plt.close()
    print("\nSaved the plotted MS/MS to {}".format(dir_to_img))
    return
