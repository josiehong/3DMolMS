import os
import pickle
import requests
import zipfile
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyteomics import mgf

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
except ImportError:
    print("PyTorch is not installed. Please install it to use this module.")
    raise

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Draw
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from .model import MolNet_MS, MolNet_Oth, Encoder
from .dataset import MolMS_Dataset, MolRT_Dataset, MolCCS_Dataset, Mol_Dataset
from .data_utils import (
    csv2pkl_wfilter,
    filter_spec,
    mgf2pkl,
    ms_vec2dict,
    nce2ce,
    precursor_calculator,
)
from .utils import (
    pred_step, eval_step_oth, pred_feat,
    train_step, eval_step, collect_targets,
    bin_spectrum, cosine_similarity,
)
from ._version import __version__

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Per-task constants
# ---------------------------------------------------------------------------

_SCHEDULER_MODE      = {"msms": "max", "rt": "min", "ccs": "min"}
_SCHEDULER_PATIENCE  = {"msms": 5,     "rt": 20,    "ccs": 20}
_EARLY_STOP_PATIENCE = {"msms": 10,    "rt": 60,    "ccs": 60}
_BEST_INIT           = {"msms": 0.0,   "rt": float("inf"), "ccs": float("inf")}
_BEST_KEY            = {"msms": "best_val_acc", "rt": "best_val_mae", "ccs": "best_val_mae"}
_METRIC_LABEL        = {"msms": "cosine",       "rt": "MAE",          "ccs": "MAE"}


class MolNet:
    def __init__(self, device, seed):
        self.version = __version__
        print("MolNetPack version:", self.version)

        self.device = device
        self.current_path = Path(__file__).parent

        # Configs (shared across inference and training)
        self.data_config  = self._load_config("preprocess_etkdgv3.yml")
        self.msms_config  = self._load_config("molnet.yml")
        self.ccs_config   = self._load_config("molnet_ccs_tl.yml")
        self.rt_config    = self._load_config("molnet_rt_tl.yml")

        # Inference state
        self.pkl_dict     = None
        self.valid_loader = None

        # Models (set by pred_* or train)
        self.msms_model   = None
        self.ccs_model    = None
        self.rt_model     = None
        self.encoder      = None

        # Cached result DataFrames
        self.qtof_msms_res_df     = None
        self.orbitrap_msms_res_df = None
        self.ccs_res_df           = None
        self.rt_res_df            = None

        self._init_random_seed(seed)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_config(self, filename):
        path = self.current_path / "config" / filename
        with open(path) as f:
            return yaml.safe_load(f)

    def _init_random_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _task_config(self, task):
        return {"msms": self.msms_config, "rt": self.rt_config, "ccs": self.ccs_config}[task]

    def _task_model_attr(self, task):
        return {"msms": "msms_model", "rt": "rt_model", "ccs": "ccs_model"}[task]

    def _build_model(self, task, config):
        ModelCls = MolNet_MS if task == "msms" else MolNet_Oth
        return ModelCls(config["model"]).to(self.device)

    def _load_weights(self, model, checkpoint_path, optimizer=None, scheduler=None, transfer=False):
        """Load a checkpoint into model (and optionally optimizer/scheduler).

        Returns the best validation metric stored in the checkpoint, or None.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if transfer:
            encoder_dict = {
                k: v for k, v in ckpt["model_state_dict"].items()
                if not k.startswith("decoder")
            }
            for v in encoder_dict.values():
                v.requires_grad = False
            model.load_state_dict(encoder_dict, strict=False)
            return None
        else:
            model.load_state_dict(ckpt["model_state_dict"])
            if optimizer is not None and "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            return ckpt.get("best_val_acc") or ckpt.get("best_val_mae")

    # ------------------------------------------------------------------
    # Checkpoint download (used by pred_* methods)
    # ------------------------------------------------------------------

    def _get_checkpoint_path(self, task_name, instrument=None):
        task_map = {
            "msms": (
                self.msms_config["test"]["local_path_qtof"]
                if instrument == "qtof"
                else self.msms_config["test"]["local_path_orbitrap"]
            ),
            "ccs":       self.ccs_config["test"]["local_path"],
            "rt":        self.rt_config["test"]["local_path"],
            "save_feat": (
                self.msms_config["test"]["local_path_qtof"]
                if instrument == "qtof"
                else self.msms_config["test"]["local_path_orbitrap"]
            ),
        }
        return str(self.current_path / task_map[task_name])

    def _ensure_checkpoint(self, checkpoint_path, task_name, instrument=None):
        if os.path.exists(checkpoint_path):
            return
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        zip_path = checkpoint_path + ".zip"

        if task_name == "ccs":
            url = self.ccs_config["test"]["github_release_url"]
        elif task_name == "rt":
            url = self.rt_config["test"]["github_release_url"]
        elif instrument == "qtof":
            url = self.msms_config["test"]["github_release_url_qtof"]
        else:
            url = self.msms_config["test"]["github_release_url_orbitrap"]

        print(f"Downloading checkpoint from {url}")
        response = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.dirname(checkpoint_path))

    def load_checkpoint(self, task_name, path_to_checkpoint=None, instrument=None):
        checkpoint_path = path_to_checkpoint or self._get_checkpoint_path(task_name, instrument)
        self._ensure_checkpoint(checkpoint_path, task_name, instrument)
        model = getattr(self, self._task_model_attr(task_name) if task_name != "save_feat" else "encoder")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)["model_state_dict"]
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, path_to_test_data):
        """Load input molecules from a CSV, MGF, or PKL file.

        :param path_to_test_data: Path to the input file. Supported formats: ``csv``, ``mgf``, ``pkl``.
        :type path_to_test_data: str
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
        ext = path_to_test_data.rsplit(".", 1)[-1].lower()
        if ext not in loaders:
            raise ValueError(f"Unsupported format: .{ext}")

        self.pkl_dict = loaders[ext](path_to_test_data)
        print(f"\nLoaded {len(self.pkl_dict)} records from {path_to_test_data}")
        self.valid_loader = DataLoader(
            Mol_Dataset(self.pkl_dict),
            batch_size=1, shuffle=False, num_workers=0, drop_last=False,
        )

    def get_data(self):
        return self.pkl_dict

    # ------------------------------------------------------------------
    # Inference  (pred_msms / pred_ccs / pred_rt / save_features)
    # ------------------------------------------------------------------

    def save_features(self, checkpoint_path=None, instrument="qtof"):
        """Extract encoder embeddings for loaded molecules.

        :param checkpoint_path: Optional path to a custom checkpoint.
        :type checkpoint_path: str, optional
        :param instrument: ``'qtof'`` or ``'orbitrap'``.
        :type instrument: str
        :return: ``(id_list, features)`` where features is a numpy array of shape ``(N, emb_dim)``.
        :rtype: tuple
        """
        self.encoder = Encoder(**self.msms_config["model"]).to(self.device)
        self.load_checkpoint("save_feat", checkpoint_path, instrument)
        ids, features = pred_feat(
            self.encoder, self.device, self.valid_loader,
            batch_size=1, num_points=self.msms_config["model"]["max_atom_num"],
        )
        return ids, features.cpu().detach().numpy()

    def pred_msms(self, path_to_results=None, path_to_checkpoint=None, instrument="qtof"):
        """Predict MS/MS spectra for loaded molecules.

        :param path_to_results: Optional path to save results (``.mgf`` or ``.csv``).
        :type path_to_results: str, optional
        :param path_to_checkpoint: Optional path to a custom checkpoint.
        :type path_to_checkpoint: str, optional
        :param instrument: ``'qtof'`` or ``'orbitrap'``.
        :type instrument: str
        :return: DataFrame with columns ID, SMILES, Collision Energy, Precursor Type, Pred M/Z, Pred Intensity.
        :rtype: pandas.DataFrame
        """
        assert instrument in ("qtof", "orbitrap"), 'instrument must be "qtof" or "orbitrap"'

        if self.msms_model is None:
            self.msms_model = MolNet_MS(self.msms_config["model"]).to(self.device)
            self.load_checkpoint("msms", path_to_checkpoint, instrument)

        id_list, pred_tensor = pred_step(
            self.msms_model, self.device, self.valid_loader,
            batch_size=1, num_points=self.msms_config["model"]["max_atom_num"],
        )
        pred_dicts = [
            ms_vec2dict(spec, float(self.msms_config["model"]["resolution"]))
            for spec in pred_tensor.tolist()
        ]
        res_df = self._assemble_msms_results(id_list, pred_dicts, instrument)

        if path_to_results:
            self._save_msms_results(res_df, path_to_results, instrument)
        return res_df

    def pred_ccs(self, path_to_results=None, path_to_checkpoint=None):
        """Predict CCS values for loaded molecules.

        :param path_to_results: Optional path to save results as CSV.
        :type path_to_results: str, optional
        :param path_to_checkpoint: Optional path to a custom checkpoint.
        :type path_to_checkpoint: str, optional
        :return: DataFrame with columns ID, SMILES, Precursor Type, Pred CCS.
        :rtype: pandas.DataFrame
        """
        if self.ccs_model is None:
            self.ccs_model = MolNet_Oth(self.ccs_config["model"]).to(self.device)
            self.load_checkpoint("ccs", path_to_checkpoint)

        id_list, pred_tensor = eval_step_oth(
            self.ccs_model, self.device, self.valid_loader,
            batch_size=1, num_points=self.ccs_config["model"]["max_atom_num"],
        )
        decoding = self._precursor_decoder()
        add_list    = [decoding[",".join(map(str, map(int, d["env"][1:])))] for d in self.pkl_dict]
        smiles_list = [d["smiles"] for d in self.pkl_dict]

        self.ccs_res_df = pd.DataFrame({
            "ID": id_list, "SMILES": smiles_list,
            "Precursor Type": add_list, "Pred CCS": pred_tensor.squeeze().tolist(),
        })
        if path_to_results:
            self._save_csv(self.ccs_res_df, path_to_results)
        return self.ccs_res_df

    def pred_rt(self, path_to_results=None, path_to_checkpoint=None):
        """Predict retention times for loaded molecules.

        :param path_to_results: Optional path to save results as CSV.
        :type path_to_results: str, optional
        :param path_to_checkpoint: Optional path to a custom checkpoint.
        :type path_to_checkpoint: str, optional
        :return: DataFrame with columns ID, SMILES, Pred RT.
        :rtype: pandas.DataFrame
        """
        if self.rt_model is None:
            self.rt_model = MolNet_Oth(self.rt_config["model"]).to(self.device)
            self.load_checkpoint("rt", path_to_checkpoint)

        id_list, pred_tensor = eval_step_oth(
            self.rt_model, self.device, self.valid_loader,
            batch_size=1, num_points=self.rt_config["model"]["max_atom_num"],
        )
        smiles_list = [d["smiles"] for d in self.pkl_dict]

        self.rt_res_df = pd.DataFrame({
            "ID": id_list, "SMILES": smiles_list,
            "Pred RT": pred_tensor.squeeze().tolist(),
        })
        if path_to_results:
            self._save_csv(self.rt_res_df, path_to_results)
        return self.rt_res_df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        task,
        train_data,
        valid_data,
        checkpoint_path="",
        resume_path="",
        transfer=False,
        precursor_type="All",
        use_scaler=False,
    ):
        """Train a model and store it on this MolNet instance.

        After training the model is ready for immediate use via ``pred_msms``,
        ``pred_rt``, or ``pred_ccs`` — no checkpoint reload needed.

        :param task: One of ``'msms'``, ``'rt'``, ``'ccs'``.
        :type task: str
        :param train_data: Path to training PKL file.
        :type train_data: str
        :param valid_data: Path to validation PKL file.
        :type valid_data: str
        :param checkpoint_path: Where to save the best checkpoint. Empty string disables saving.
        :type checkpoint_path: str
        :param resume_path: Resume from or transfer-learn from this checkpoint.
        :type resume_path: str
        :param transfer: If ``True``, load only encoder weights from ``resume_path`` and freeze them.
        :type transfer: bool
        :param precursor_type: Filter training data by precursor type (``msms`` task only).
            One of ``'All'``, ``'[M+H]+'``, ``'[M-H]-'``.
        :type precursor_type: str
        :param use_scaler: Fit a StandardScaler on training targets (``rt`` task only).
        :type use_scaler: bool
        :return: Best validation metric achieved during training.
        :rtype: float
        """
        assert task in ("msms", "rt", "ccs"), f"Unknown task: {task}"
        config = self._task_config(task)
        batch_size = config["train"]["batch_size"]
        num_points = config["model"]["max_atom_num"]

        # --- Datasets & loaders ---
        train_loader, valid_loader = self._build_loaders(
            task, train_data, valid_data, config, precursor_type,
        )

        # --- Model ---
        model = self._build_model(task, config)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{model.__class__.__name__}  #params: {num_params:,}")

        # --- Optimizer & scheduler ---
        optimizer = optim.AdamW(model.parameters(), lr=config["train"]["lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=_SCHEDULER_MODE[task],
            factor=0.5,
            patience=_SCHEDULER_PATIENCE[task],
        )

        # --- Resume / transfer ---
        if resume_path:
            if transfer:
                print("Loading encoder weights (frozen) for transfer learning...")
                self._load_weights(model, resume_path, transfer=True)
                if use_scaler and task == "rt":
                    model.fit_scaler(collect_targets(train_loader))
            else:
                print(f"Resuming from {resume_path} ...")
                self._load_weights(model, resume_path, optimizer, scheduler)
                if task == "rt":
                    ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
                    model.set_scaler(ckpt.get("scaler"))
        elif use_scaler and task == "rt":
            model.fit_scaler(collect_targets(train_loader))

        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

        # --- Training loop ---
        best_metric      = _BEST_INIT[task]
        higher_is_better = _SCHEDULER_MODE[task] == "max"
        early_patience   = 0
        early_limit      = _EARLY_STOP_PATIENCE[task]
        label            = _METRIC_LABEL[task]

        for epoch in range(1, config["train"]["epochs"] + 1):
            print(f"\n===== Epoch {epoch}")
            train_metric = train_step(model, self.device, train_loader, optimizer,
                                      batch_size, num_points, task)
            valid_metric = eval_step(model, self.device, valid_loader,
                                     batch_size, num_points, task)
            print(f"Train {label}: {train_metric:.4f}  |  Valid {label}: {valid_metric:.4f}")

            improved = valid_metric > best_metric if higher_is_better else valid_metric < best_metric
            if improved:
                best_metric    = valid_metric
                early_patience = 0
                print("Early stop patience reset")
                if checkpoint_path:
                    print("Saving checkpoint...")
                    extra = {"scaler": model.scaler} if task == "rt" else {}
                    torch.save(
                        {
                            "version":            __version__,
                            "epoch":              epoch,
                            "model_state_dict":   model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "num_params":         num_params,
                            _BEST_KEY[task]:      best_metric,
                            **extra,
                        },
                        checkpoint_path,
                    )
            else:
                early_patience += 1
                print(f"Early stop count: {early_patience}/{early_limit}")

            scheduler.step(valid_metric)
            print(f"Best {label} so far: {best_metric:.4f}")

            if early_patience >= early_limit:
                print("Early stop!")
                break

        # Store the trained model so pred_* methods can use it immediately
        setattr(self, self._task_model_attr(task), model)
        print(f"\nTraining complete. Best {label}: {best_metric:.4f}")
        return best_metric

    # ------------------------------------------------------------------
    # Evaluation (ground truth vs. predictions)
    # ------------------------------------------------------------------

    def evaluate(self, test_pkl, pred_mgf, result_path="", plot_path=""):
        """Compare predicted MS/MS spectra against ground-truth spectra.

        :param test_pkl: Path to the ground-truth PKL file (from preprocessing).
        :type test_pkl: str
        :param pred_mgf: Path to the predicted spectra MGF file (from ``pred_msms``).
        :type pred_mgf: str
        :param result_path: Optional path to save per-spectrum results as CSV.
        :type result_path: str
        :param plot_path: Optional path to save a cosine similarity histogram PNG.
        :type plot_path: str
        :return: DataFrame with per-spectrum cosine similarity and metadata.
        :rtype: pandas.DataFrame
        """
        with open(test_pkl, "rb") as f:
            gt_spectra = pickle.load(f)
        pred_spectra = list(mgf.read(pred_mgf))

        gt_by_title   = {s["title"]: s["spec"]  for s in gt_spectra}
        pred_by_title = {s["params"]["title"]: s for s in pred_spectra}

        rows = []
        for title, gt_vec in gt_by_title.items():
            if title not in pred_by_title:
                continue
            pred = pred_by_title[title]
            binned = bin_spectrum(pred["m/z array"], pred["intensity array"])
            if binned is None or len(gt_vec) != len(binned):
                continue
            sim = cosine_similarity(gt_vec, binned)
            if sim is None:
                continue
            rows.append({
                "title":              title,
                "smiles":             pred["params"].get("smiles", ""),
                "collision_energy":   pred["params"].get("collision_energy", ""),
                "precursor_type":     pred["params"].get("precursor_type", "Unknown"),
                "cosine_similarity":  sim,
            })

        df = pd.DataFrame(rows)
        print(f"\nEvaluated {len(df)} matched spectra")
        print(f"Overall mean cosine similarity: {df['cosine_similarity'].mean():.4f}")
        print("\nMean by precursor type:")
        print(df.groupby("precursor_type")["cosine_similarity"].mean().to_string())

        if result_path:
            self._save_csv(df, result_path)
        if plot_path:
            self._plot_similarity_hist(df["cosine_similarity"].tolist(), plot_path)
        return df

    # ------------------------------------------------------------------
    # Private: result assembly & saving
    # ------------------------------------------------------------------

    def _precursor_decoder(self):
        return {
            ",".join(map(str, v)): k
            for k, v in self.data_config["encoding"]["precursor_type"].items()
        }

    def _assemble_msms_results(self, id_list, pred_dicts, instrument):
        decoding    = self._precursor_decoder()
        ce_list, add_list, smiles_list = [], [], []
        for d in self.pkl_dict:
            adduct = decoding[",".join(map(str, map(int, d["env"][1:])))]
            smiles = d["smiles"]
            mass   = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
            charge = int(self.data_config["encoding"]["type2charge"][adduct])
            ce_list.append(nce2ce(d["env"][0], precursor_calculator(adduct, mass), charge))
            add_list.append(adduct)
            smiles_list.append(smiles)

        res_df = pd.DataFrame({
            "ID":               id_list,
            "SMILES":           smiles_list,
            "Collision Energy": ce_list,
            "Precursor Type":   add_list,
            "Pred M/Z":         [p["m/z"]       for p in pred_dicts],
            "Pred Intensity":   [p["intensity"] for p in pred_dicts],
        })
        if instrument == "qtof":
            self.qtof_msms_res_df = res_df
        else:
            self.orbitrap_msms_res_df = res_df
        return res_df

    def generate_spectra_from_df(self, df, instrument=None):
        spectra = []
        for idx, row in df.iterrows():
            spectra.append({
                "params": {
                    "title":            row["ID"],
                    "mslevel":          "2",
                    "organism":         f"3DMolMS_{self.version}",
                    "spectrumid":       f"pred_{idx}",
                    "smiles":           row["SMILES"],
                    "collision_energy": row["Collision Energy"],
                    "precursor_type":   row["Precursor Type"],
                    "instrument_type":  instrument,
                },
                "m/z array":       np.array([float(v) for v in row["Pred M/Z"].split(",") if v]),
                "intensity array": np.array([float(v) * 1000 for v in row["Pred Intensity"].split(",") if v]),
            })
        return spectra

    def _save_msms_results(self, res_df, path, instrument):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if path.endswith(".mgf"):
            mgf.write(self.generate_spectra_from_df(res_df, instrument),
                      path, file_mode="w", write_charges=False)
        elif path.endswith(".csv"):
            res_df.to_csv(path, index=False)
        else:
            raise ValueError("result path must end with .mgf or .csv")
        print(f"\nSaved results to {path}")

    def _save_csv(self, df, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)
        print(f"\nSaved results to {path}")

    @staticmethod
    def _plot_similarity_hist(similarities, path):
        plt.figure(figsize=(8, 6))
        plt.hist(similarities, bins=50, edgecolor="black")
        plt.title("Cosine Similarity Distribution")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"\nSaved histogram to {path}")

    # ------------------------------------------------------------------
    # Private: training helpers
    # ------------------------------------------------------------------

    def _build_loaders(self, task, train_path, valid_path, config, precursor_type="All"):
        batch_size = config["train"]["batch_size"]
        num_workers = config["train"]["num_workers"]

        if task == "msms":
            # Build the encoded precursor-type filter string
            encoder = {
                k: ",".join(str(int(i)) for i in v)
                for k, v in self.data_config["encoding"]["precursor_type"].items()
            }
            encoder["All"] = False
            encoded = encoder[precursor_type]
            train_set = MolMS_Dataset(train_path, encoded)
            valid_set = MolMS_Dataset(valid_path, encoded)
        elif task == "rt":
            train_set = MolRT_Dataset(train_path)
            valid_set = MolRT_Dataset(valid_path)
        else:  # ccs
            train_set = MolCCS_Dataset(train_path)
            valid_set = MolCCS_Dataset(valid_path)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, drop_last=True)
        return train_loader, valid_loader


# ---------------------------------------------------------------------------
# Standalone visualization helper
# ---------------------------------------------------------------------------

def plot_msms(msms_res_df, dir_to_img):
    """Plot MS/MS spectra with inset 2-D molecular structures.

    :param msms_res_df: DataFrame returned by :meth:`MolNet.pred_msms`.
    :type msms_res_df: pandas.DataFrame
    :param dir_to_img: Directory where PNG files will be saved (one per spectrum).
    :type dir_to_img: str
    """
    os.makedirs(dir_to_img, exist_ok=True)
    img_dpi, y_max, bin_width = 300, 1, 0.4

    for _, row in msms_res_df.iterrows():
        mz_values  = np.array([float(v) for v in row["Pred M/Z"].split(",")])
        intensities = np.array([float(v) * y_max for v in row["Pred Intensity"].split(",")])

        fig, ax = plt.subplots(figsize=(9, 4))
        plt.bar(mz_values, intensities, width=bin_width, color="k")
        plt.xlim(0, np.max(mz_values))
        plt.title("ID: " + row["ID"])
        plt.xlabel("M/Z")
        plt.ylabel("Relative intensity")

        mol = Chem.MolFromSmiles(row["SMILES"])
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        mol_img = Draw.MolToImage(mol, size=(800, 800))
        alpha   = Image.fromarray(255 - np.array(mol_img.convert("L")))
        mol_img.putalpha(alpha)
        imagebox = OffsetImage(mol_img, zoom=72.0 / img_dpi)
        ax.add_artist(AnnotationBbox(
            imagebox, (np.max(mz_values) * 0.28, y_max * 0.64),
            frameon=False, xycoords="data",
        ))

        plt.savefig(os.path.join(dir_to_img, row["ID"]), dpi=img_dpi, bbox_inches="tight")
        plt.close()

    print(f"\nSaved plots to {dir_to_img}")
