* 2026.04.15 (v1.2.2): reorganize source code (scripts consolidated under `scripts/`, package moved to top-level `molnetpack/`); add `train()` and `evaluate()` methods directly to `MolNet` so training and evaluation are available from the Python API without using CLI scripts; update all documentation and examples; add `scikit-learn` and `tqdm` to declared dependencies.
* 2025.10.12 (v1.2.1): change the build-backend to "setuptools.build_meta"; fix the error of unmatched inputs in v1.2.0; apply `black` for better code format. 
* 2025.4.28 (v1.2.0): update `MolConv` from version 2 to version 3; add online documentation; add scaler option for RT prediction but using scaler makes convergence harder. 
* 2024.12.20 (v1.1.10.post1): fix the bug of importing `plot_msms`. 
* 2024.08.30 (v1.1.10): modify the arguments and return values in `MolNet` for better usage; update the documents.
* 2024.08.29 (v1.1.9): fix the GitHub release address (using `v1.1.8` release until update the weights). 
* 2024.08.28 (v1.1.8): release retention time prediction model. 
* 2024.08.27 (v1.1.7): update `MolConv` to `MolConv 2.0`, which is introduced in "Hong, Y., Welch, C.J., Piras, P. and Tang, H., 2024. Enhanced structure-based prediction of chiral stationary phases for chromatographic enantioseparation from 3D molecular conformations. Analytical Chemistry, 96(6), pp.2351-2359.". 
* 2024.08.19 (v1.1.6): release orbitrap ms/ms model, and codes for fine-tuning our model on customized data; move the model weights on google drive to GitHub release. 
* 2024.07.09 (v1.1.5): enable the local checkpoint path; add evaluation (cosine similarity) step. 
* 2024.07.08 (v1.1.4): ~~enable batched prediction~~. 
* 2024.03.08 (v1.1.3): add the function `save_features`, which can be used to save embedded features; these features can then be used in downstream tasks.
* 2024.02.27 (v1.1.2): fix the independences; update ccs prediction. 
* 2024.02.27 (v1.1.1): PyPI package release. 
* 2023.10.30 (v1.1.0): enlarging training set by MoNA and Waters QTOF datasets. 
* 2023.10.22 (v1.0.2): pretraining on QM9-mu dataset + ETKDG algorithm. We establish a dataset from QM9-mu (dipole moment) with the generated conformations using ETKDG for pretraining 3DMolMS. It helps the model learning knowledge of molecular 3D conformations and pretraining enhances the performance on MS/MS slightly (~0.01 cosine similarity). 
* 2023.09.14 (v1.0.1): data augmentation by flipping atomic coordinates. Notably, this model is sensitive to the geometric structure of molecules. For tasks insensitive to geometric structure, e.g. mass spectrometry is chirally blind, please use data augmentation. However, for the tasks sensitive to geometric structure, e.g. different enantiomers with varying retention times, avoid data augmentation. 
* 2023.06.30 (v1.0.0): initial version. 