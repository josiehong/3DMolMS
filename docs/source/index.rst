Welcome to 3DMolMS Documentation
================================

3D Molecular Network for Mass Spectra Prediction (3DMolMS) is a deep neural network model to predict the tandem mass (MS/MS) spectra of compounds from their 3D conformations. It can be used to:

* Predict MS/MS spectra for small molecules
* Generate reference libraries of small molecule MS/MS spectra for identification
* Predict molecular properties like retention time (RT) and collision cross section (CCS)
* Pretrain models on molecular datasets

This document provides a guide to using 3DMolMS for inference through the ``molnetpack`` package and comprehensive usages of the source code for training and inference. 

.. toctree::
   :maxdepth: 2
   :caption: PyPI package

   molnetpack

.. toctree::
   :maxdepth: 2
   :caption: Source code

   sourcecode
   usage/index
   advanced_usage/index

.. toctree::
   :maxdepth: 1
   :caption: Addtional information

   supported_formats
   encoder

References
----------

.. code-block:: bibtex

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