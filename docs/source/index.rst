3DMolMS
=======

*Version* |release|

3DMolMS (*3D Molecular Network for Mass Spectra Prediction*) is a deep-learning model that predicts the tandem mass (MS/MS) spectra of small molecules from their 3D conformations. Its learned molecular representation also transfers to related tasks such as retention time (RT) and collision cross section (CCS) prediction.

Use 3DMolMS to:

* Predict MS/MS spectra for small molecules
* Predict molecular properties — retention time (RT) and collision cross section (CCS)
* Generate reference MS/MS libraries for compound identification
* Pretrain models on 3D molecular datasets

There are two ways to run it: the :doc:`molnetpack Python package <molnetpack>` for minimal-code inference and training, or the :doc:`command-line source code <sourcecode>` for full control. New to the input files? Start with :doc:`supported_formats`.

.. toctree::
   :maxdepth: 2
   :caption: PyPI package

   molnetpack
   molnetpack_training

.. toctree::
   :maxdepth: 1
   :caption: Background

   encoder
   supported_formats

.. toctree::
   :maxdepth: 2
   :caption: Source code

   sourcecode
   usage/index
   advanced_usage/index

References
----------

[1] Hong, Y., Li, S., Welch, C.J., Tichy, S., Ye, Y. and Tang, H., 2023. 3DMolMS: prediction of tandem mass spectra from 3D molecular conformations. *Bioinformatics*, 39(6), p.btad354.

[2] Hong, Y., Welch, C.J., Piras, P. and Tang, H., 2024. Enhanced structure-based prediction of chiral stationary phases for chromatographic enantioseparation from 3D molecular conformations. *Analytical Chemistry*, 96(6), pp.2351-2359.
