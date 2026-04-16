Retention time prediction
=========================

3DMolMS can be used to predict MS/MS-related properties, such as retention time (RT) and collision cross section (CCS). This guide shows how to train a model for RT prediction and CCS prediction, and how to transfer these models to your own RT and CCS dataset.

All models mentioned can be downloaded from `release v1.2.0 <https://github.com/JosieHong/3DMolMS/releases/tag/v1.2.0>`_.

Setup
-----

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Data preparation
----------------------------

Download the retention time dataset, `METLIN <https://figshare.com/articles/dataset/The_METLIN_small_molecule_dataset_for_machine_learning-based_retention_time_prediction/8038913?file=18130625>`_. The structure of data directory is:

.. code-block:: text

   |- data
     |- origin
       |- SMRT_dataset.sdf

**Step 2**: Preprocessing
-------------------------

Use the following commands to preprocess the datasets. The settings of datasets are in ``./molnetpack/config/preprocess_etkdgv3.yml``.

.. code-block:: bash

   python scripts/preprocess.py --task rt \
   --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml

**Step 3**: Training
--------------------

Use the following commands to train the model. The settings of model and training are in ``./molnetpack/config/molnet_rt.yml``.

*Using the command-line script:*

Learning from scratch:

.. code-block:: bash

   python scripts/train.py --task rt \
   --train_data ./data/metlin_etkdgv3_train.pkl \
   --test_data ./data/metlin_etkdgv3_test.pkl \
   --checkpoint_path ./check_point/molnet_rt_etkdgv3.pt

If you'd like to train this model from the pre-trained model on MS/MS prediction, please download the pre-trained model from `release v1.2.0 <https://github.com/JosieHong/3DMolMS/releases/tag/v1.2.0>`_.

Learning from MS/MS model:

.. code-block:: bash

   python scripts/train.py --task rt \
   --train_data ./data/metlin_etkdgv3_train.pkl \
   --test_data ./data/metlin_etkdgv3_test.pkl \
   --checkpoint_path ./check_point/molnet_rt_etkdgv3_tl.pt \
   --transfer \
   --resume_path ./check_point/molnet_qtof_etkdgv3.pt

*Using the Python API:*

.. code-block:: python

   import torch
   from molnetpack import MolNet

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   molnet_engine = MolNet(device, seed=42)

   # Learning from scratch:
   molnet_engine.train(
       task='rt',
       train_data='./data/metlin_etkdgv3_train.pkl',
       valid_data='./data/metlin_etkdgv3_test.pkl',
       checkpoint_path='./check_point/molnet_rt_etkdgv3.pt',
   )

   # Learning from MS/MS model (transfer learning):
   molnet_engine.train(
       task='rt',
       train_data='./data/metlin_etkdgv3_train.pkl',
       valid_data='./data/metlin_etkdgv3_test.pkl',
       checkpoint_path='./check_point/molnet_rt_etkdgv3_tl.pt',
       resume_path='./check_point/molnet_qtof_etkdgv3.pt',
       transfer=True,
       use_scaler=True,
   )
