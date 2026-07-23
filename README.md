# 3DMolMS

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] (free for academic use)

**3D** **Mol**ecular Network for **M**ass **S**pectra Prediction (3DMolMS) is a deep neural network model to predict the MS/MS spectra of compounds from their 3D conformations. This model's molecular representation, learned through MS/MS prediction tasks, can be further applied to enhance performance in other molecular-related tasks, such as predicting retention times (RT) and collision cross sections (CCS).

[Paper](https://academic.oup.com/bioinformatics/article/39/6/btad354/7186501) | [Document](https://3dmolms.readthedocs.io/en/latest/) | [Workflow on Koina](https://koina.wilhelmlab.org/docs#post-/3dmolms_qtof/infer) | [PyPI package](https://pypi.org/project/molnetpack/)

## Latest release

3DMolMS v1.3.2 is now available on PyPI!

This release makes the 3D molecular encoder correctly **E(3)-invariant** (to rotation, reflection, and translation) — the previous encoder's output depended on a molecule's absolute position in space. All checkpoints (MS/MS for QTOF and Orbitrap, RT, and CCS) have been retrained with the corrected encoder. This release also adds batched MS/MS inference, in-memory input, per-user checkpoint caching, and optional self-supervised pretraining.

The full change log is at [CHANGE_LOG.md](https://github.com/JosieHong/3DMolMS/blob/main/CHANGE_LOG.md).

## Installation

3DMolMS is available on PyPI:

```bash
pip install molnetpack
```

PyTorch must be installed separately. Choose the build that matches your system (see the [official PyTorch site](https://pytorch.org/get-started/locally/) for other options):

```bash
# CUDA 11.6
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

# CUDA 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

# CPU only
pip install torch==1.13.0+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

Or install from source:

```bash
git clone https://github.com/JosieHong/3DMolMS.git
cd 3DMolMS
pip install .
```

Prefer no installation? Use the [Koina web service](https://koina.wilhelmlab.org/docs#post-/3dmolms_qtof/infer) for inference with API support.

## Usage

**Predict MS/MS spectra:**

```python
import torch
from molnetpack import MolNet, plot_msms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
molnet_engine = MolNet(device, seed=42)

# Supports CSV, MGF, and PKL input
molnet_engine.load_data("./examples/input_msms.csv")

# Predict and save to MGF (checkpoints are downloaded automatically)
pred_df = molnet_engine.pred_msms(
    path_to_results="./output_msms.mgf",
    instrument="qtof",  # or "orbitrap"
)

# Plot the predicted spectra alongside their 3D conformations
plot_msms(pred_df, dir_to_img="./img/")
```

**Predict retention time (RT) and CCS:**

```python
molnet_engine.load_data("./examples/input_ccs.csv")
rt_df  = molnet_engine.pred_rt(path_to_results="./output_rt.csv")
ccs_df = molnet_engine.pred_ccs(path_to_results="./output_ccs.csv")
```

**Save molecular embeddings:**

```python
molnet_engine.load_data("./examples/input_savefeat.csv")
ids, features = molnet_engine.save_features()
print("Feature shape:", features.shape)
```

**Train your own model:**

```python
# Fine-tune from a pretrained checkpoint
molnet_engine.train(
    task="msms",                                          # or "rt" / "ccs"
    train_data="./data/qtof_etkdgv3_train.pkl",
    valid_data="./data/qtof_etkdgv3_test.pkl",
    checkpoint_path="./check_point/molnet_qtof_tl.pt",
    resume_path="./check_point/molnet_pre_etkdgv3.pt",   # pretrained encoder
    transfer=True,
)

# The trained model is immediately ready — no reload needed
molnet_engine.load_data("./data/qtof_etkdgv3_test.pkl")
pred_df = molnet_engine.pred_msms(instrument="qtof")

# Evaluate against ground truth
results_df = molnet_engine.evaluate(
    test_pkl="./data/qtof_etkdgv3_test.pkl",
    pred_mgf="./output_msms.mgf",
    result_path="./eval_results.csv",
    plot_path="./eval_similarity_hist.png",
)
```

Sample input files are in [examples/](https://github.com/JosieHong/3DMolMS/tree/main/examples). For CCS-only input you may assign an arbitrary value to the `Collision_Energy` field. Unsupported formats are automatically excluded during loading. Supported inputs:

| Item             | Supported input                                    |
|------------------|----------------------------------------------------|
| Atom number      | ≤ 300                                              |
| Atom types       | C, O, N, H, P, S, F, Cl, B, Br, I                  |
| Precursor types  | [M+H]+, [M-H]-, [M+H-H2O]+, [M+Na]+, [M+2H]2+      |
| Collision energy | any number                                         |

See the [full documentation](https://3dmolms.readthedocs.io/en/latest/) for dataset preparation, preprocessing, and advanced usage, and the [source-code docs](https://3dmolms.readthedocs.io/en/latest/sourcecode.html) for script-based workflows.

## Citation

If you use 3DMolMS in your research, please cite:

1. Hong, Y., Li, S., Welch, C.J., Tichy, S., Ye, Y. and Tang, H., 2023. 3DMolMS: prediction of tandem mass spectra from 3D molecular conformations. *Bioinformatics*, 39(6), p.btad354.
2. Hong, Y., Welch, C.J., Piras, P. and Tang, H., 2024. Enhanced structure-based prediction of chiral stationary phases for chromatographic enantioseparation from 3D molecular conformations. *Analytical Chemistry*, 96(6), pp.2351-2359.

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
