Tandem mass spectra prediction
==============================

This guide explains how to use 3DMolMS for tandem mass spectra (MS/MS) prediction.

Setup
-----

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Input preparation
-----------------------------

Prepare the test set. The following formats are supported: csv, mgf, or `customed pkl <https://github.com/JosieHong/3DMolMS/blob/main/molnetpack/data_utils/all2pkl.py>`_.

CSV Format Example
~~~~~~~~~~~~~~~~~~

Here is an input example from MoNA of csv format:

.. code-block:: text

   ID,SMILES,Precursor_Type,Collision_Energy
   demo_0,C/C(=C\CNc1nc[nH]c2ncnc1-2)CO,[M+H]+,40 V
   demo_1,C/C(=C\CNc1nc[nH]c2ncnc1-2)CO,[M+H]+,20 V
   demo_2,C/C(=C\CNc1nc[nH]c2ncnc1-2)CO,[M+H]+,10 V

MGF Format Example
~~~~~~~~~~~~~~~~~~

Here is an input example from MoNA of mgf format:

.. code-block:: text

   BEGIN IONS
   TITLE=demo_0
   CHARGE=1+
   PRECURSOR_TYPE=[M+H]+
   PRECURSOR_MZ=220.1193
   MOLMASS=219.11201003600002
   MS_LEVEL=MS2
   IONMODE=P
   SOURCE_INSTRUMENT=Agilent 6530 Q-TOF
   INSTRUMENT_TYPE=LC-ESI-QTOF
   COLLISION_ENERGY=40 V
   SMILES=C/C(=C\CNc1nc[nH]c2ncnc1-2)CO
   INCHI_KEY=UZKQTCBAMSWPJD-FARCUNLSSA-N
   41.0399 6.207207
   43.0192 49.711712
   43.0766 1.986987
   ...

Supported inputs
~~~~~~~~~~~~~~~~

The unsupported input will be filtered out automatically when loading the dataset. The supported inputs are:

.. list-table::
   :header-rows: 1

   * - Item
     - Supported input
   * - Atom number
     - ≤300
   * - Atom types
     - 'C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I', 'Na'
   * - Precursor types
     - '[M+H]+', '[M-H]-', '[M+H-H2O]+', '[M+Na]+'
   * - Collision energy
     - any number

**Step 2**: Running prediction
------------------------------

Predict the MS/MS spectra using the following command:

.. code-block:: bash

  python scripts/predict.py --task msms \
  --test_data ./examples/demo_input.csv \
  --model_config_path ./molnetpack/config/molnet.yml \
  --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
  --resume_path ./check_point/molnet_qtof_etkdgv3.pt \
  --result_path ./examples/demo_output.mgf

Arguments
~~~~~~~~~

* ``--resume_path``: Path of model's checkpoint. In the first running, the checkpoint (``./checkpoint/molnet_qtof_etkdgv3.pt``) will be downloaded from `Google Drive <https://drive.google.com/drive/folders/1fWx3d8vCPQi-U-obJ3kVL3XiRh75x5Ce?usp=drive_link>`_. You can also set the resume path to your own model.
* ``--result_path``: Path to save the predicted MS/MS. It should end with ``.mgf`` or ``.csv``. We recommend using ``.mgf`` because MGF is a more common format for MS/MS.
