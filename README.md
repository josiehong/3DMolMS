# 3DMolMS

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] (free for academic use) 

**3D** **Mol**ecular Network for **M**ass **S**pectra Prediction (3DMolMS) is a deep neural network model to predict the MS/MS spectra of compounds from their 3D conformations. This model's molecular representation, learned through MS/MS prediction tasks, can be further applied to enhance performance in other molecular-related tasks, such as predicting retention times (RT) and collision cross sections (CCS). 

[Paper](https://academic.oup.com/bioinformatics/article/39/6/btad354/7186501) | [Document](https://3dmolms.readthedocs.io/en/latest/) | [Workflow on Konia](https://koina.wilhelmlab.org/docs#post-/3dmolms_qtof/infer) | [PyPI package](https://pypi.org/project/molnetpack/)

## Latest release

3DMolMS v1.2.2 is now available on PyPI!

This release adds `train()` and `evaluate()` methods directly to `MolNet`, making it possible to train and evaluate models entirely from Python without CLI scripts. The source code has also been reorganized for clarity. Pretrained checkpoints from v1.2.0 remain fully compatible.

The changes log can be found at [./CHANGE_LOG.md](./CHANGE_LOG.md). 

## Getting started

### ☁️ Web service

Access the no-installation web service via [Koina](https://koina.wilhelmlab.org/docs#post-/3dmolms_qtof/infer) for quick inference with API support.

### 📦 PyPI package

```bash
pip install molnetpack
```

PyTorch must be installed separately — see the [official PyTorch site](https://pytorch.org/get-started/locally/) for the right command for your system.

**Predict MS/MS spectra:**

```python
import torch
from molnetpack import MolNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
molnet_engine = MolNet(device, seed=42)

# Supports CSV, MGF, and PKL input
molnet_engine.load_data("./examples/demo_input.csv")

# Predict and save to MGF (checkpoints are downloaded automatically)
pred_df = molnet_engine.pred_msms(
    path_to_results="./output_msms.mgf",
    instrument="qtof",  # or "orbitrap"
)
```

**Predict retention time (RT) and CCS:**

```python
molnet_engine.load_data("./examples/demo_input.csv")
rt_df  = molnet_engine.pred_rt(path_to_results="./output_rt.csv")
ccs_df = molnet_engine.pred_ccs(path_to_results="./output_ccs.csv")
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

See [examples/](./examples/) for more complete scripts and the [full documentation](https://3dmolms.readthedocs.io/en/latest/) for dataset preparation, preprocessing, and advanced usage.

### 🧪 Source code

Clone the repository for script-based training and advanced customization. See the [source code documentation](https://3dmolms.readthedocs.io/en/latest/sourcecode.html) for setup instructions.


## Citation

```
@article{hong20233dmolms,
  title={3DMolMS: prediction of tandem mass spectra from 3D molecular conformations},
  author={Hong, Yuhui and Li, Sujun and Welch, Christopher J and Tichy, Shane and Ye, Yuzhen and Tang, Haixu},
  journal={Bioinformatics},
  volume={39},
  number={6},
  pages={btad354},
  year={2023},
  publisher={Oxford University Press}
}
@article{hong2024enhanced,
  title={Enhanced structure-based prediction of chiral stationary phases for chromatographic enantioseparation from 3D molecular conformations},
  author={Hong, Yuhui and Welch, Christopher J and Piras, Patrick and Tang, Haixu},
  journal={Analytical Chemistry},
  volume={96},
  number={6},
  pages={2351--2359},
  year={2024},
  publisher={ACS Publications}
}
```

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg