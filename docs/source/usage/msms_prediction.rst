Tandem mass spectra prediction
==============================

This guide explains how to use 3DMolMS for tandem mass spectra (MS/MS) prediction.

Setup
-----

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Input preparation
-----------------------------

Prepare the test set as a CSV, MGF, or PKL file. A minimal CSV needs ``ID``, ``SMILES``, ``Precursor_Type``, and ``Collision_Energy``:

.. code-block:: text

   ID,SMILES,Precursor_Type,Collision_Energy
   demo_0,C/C(=C\CNc1nc[nH]c2ncnc1-2)CO,[M+H]+,40 V

See :doc:`../supported_formats` for the MGF layout, the PKL structure, and the supported atom and precursor types. Unsupported molecules are skipped automatically on load.

**Step 2**: Running prediction
------------------------------

Predict the MS/MS spectra using the following command:

.. code-block:: bash

  python scripts/predict.py --task msms \
  --test_data ./examples/input_msms.csv \
  --model_config_path ./molnetpack/config/molnet.yml \
  --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
  --resume_path ./check_point/molnet_qtof_etkdgv3.pt \
  --result_path ./examples/output_msms.mgf

Arguments
~~~~~~~~~

* ``--resume_path``: model checkpoint. On the first run it downloads automatically from the `GitHub release <https://github.com/JosieHong/3DMolMS/releases>`_; you can also point it at your own model.
* ``--result_path``: where to save the prediction. Use ``.mgf`` (recommended for MS/MS) or ``.csv``.
