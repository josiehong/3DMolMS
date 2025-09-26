# 3DMolMS

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] (free for academic use) 

**3D** **Mol**ecular Network for **M**ass **S**pectra Prediction (3DMolMS) is a deep neural network model to predict the MS/MS spectra of compounds from their 3D conformations. This model's molecular representation, learned through MS/MS prediction tasks, can be further applied to enhance performance in other molecular-related tasks, such as predicting retention times (RT) and collision cross sections (CCS). 

[Paper](https://academic.oup.com/bioinformatics/article/39/6/btad354/7186501) | [Document](https://3dmolms.readthedocs.io/en/latest/) | [Workflow on Konia](https://koina.wilhelmlab.org/docs#post-/3dmolms_qtof/infer) | [PyPI package](https://pypi.org/project/molnetpack/)

🔔 Latest Release: 3DMolMS v1.2.1 is now available!

The changes log can be found at [./CHANGE_LOG.md](./CHANGE_LOG.md). 

**Accessed in 3 ways**: 

1. 🧪 **Source Code**: Clone the repository for both training and inference functionality ([document for source code](https://3dmolms.readthedocs.io/en/latest/sourcecode.html)). 

2. 📦 **PyPI Package**: Install `molnetpack` for easy inference with pre-trained models ([document for PyPI package](https://3dmolms.readthedocs.io/en/latest/molnetpack.html)). 

3. ☁️ **Web Service**: Access our no-installation web service with API support for inference ([workflow on Konia](https://koina.wilhelmlab.org/docs#post-/3dmolms_qtof/infer)). 

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