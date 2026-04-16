Pretraining 3DMolMS on QM9
==========================

This guide explains how to pretrain the 3DMolMS model on the QM9 dataset.

Setup
-----

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Data preparation
----------------------------

Download the QM9 dataset from `Figshare <https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904>`_. The expected data directory structure is:

.. code-block:: text

   |- data
     |- qm9
       |- dsgdb9nsd.xyz.tar.bz2
       |- dsC7O2H10nsd.xyz.tar.bz2
       |- uncharacterized.txt

**Step 2**: Preprocessing
-------------------------

Use the following commands to preprocess the datasets. The dataset configuration is stored in ``./molnetpack/config/preprocess_etkdgv3.yml``.

.. code-block:: bash

   python scripts/qm92pkl.py --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml

**Step 3**: Pretraining
-----------------------

Use the following commands to pretrain the model. The model and training settings are in ``./molnetpack/config/molnet_pre.yml``.

.. code-block:: bash

   python scripts/pretrain.py \
   --train_data ./data/qm9_etkdgv3_train.pkl \
   --test_data ./data/qm9_etkdgv3_test.pkl \
   --model_config_path ./molnetpack/config/molnet_pre.yml \
   --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
   --checkpoint_path ./check_point/molnet_pre_etkdgv3.pt
