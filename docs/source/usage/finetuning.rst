Fine-tune on your own data
==========================

This section introduces how to fine-tune the model for regression tasks, such as retention time prediction, on your own data.

Setup
-----

Please set up the environment as shown in the :doc:`../sourcecode` page.

**Step 1**: Data preparation
----------------------------

Please prepare the data of molecular properties as:

.. code-block:: text

   ,id,smiles,prop
   0,0382_00004,NC(=O)N1c2ccccc2[C@H](O)[C@@H](O)c2ccccc21,5.79
   1,0382_00005,CN(C)[C@@H]1C(=O)C(C(N)=O)=C(O)[C@@]2(O)C(=O)C3=C(O)c4c(O)ccc(Cl)c4[C@@](C)(O)[C@H]3C[C@@H]12,4.5
   2,0382_00008,Cc1onc(-c2c(Cl)cccc2Cl)c1C(=O)N[C@@H]1C(=O)N2[C@@H](C(=O)O)C(C)(C)S[C@H]12,7.8
   3,0382_00009,C[C@H]1c2cccc(O)c2C(O)=C2C(=O)[C@]3(O)C(O)=C(C(N)=O)C(=O)[C@@H](N(C)C)[C@@H]3[C@@H](O)[C@@H]21,6.2
   4,0382_00010,C#C[C@]1(O)CC[C@H]2[C@@H]3CCc4cc(O)ccc4[C@H]3CC[C@@]21C,9.46
   5,0382_00012,Cc1onc(-c2ccccc2)c1C(=O)N[C@@H]1C(=O)N2[C@@H](C(=O)O)C(C)(C)S[C@H]12,6.9

where ``prop`` column is the RT or CCS values. Split your data into train and test CSV files, then convert each to pkl using:

.. code-block:: python

   from molnetpack import csv2pkl_wfilter
   import yaml, pickle

   with open('./molnetpack/config/preprocess_etkdgv3.yml') as f:
       cfg = yaml.safe_load(f)

   for split in ['train', 'test']:
       data = csv2pkl_wfilter(f'<path_to_{split}.csv>', cfg['encoding'])
       pickle.dump(data, open(f'<path_to_{split}.pkl>', 'wb'))

**Step 2**: Training
--------------------

Fine-tune the model:

.. code-block:: bash

   python scripts/train.py --task rt \
   --train_data <path_to_train.pkl> \
   --test_data <path_to_test.pkl> \
   --model_config_path ./molnetpack/config/molnet_rt.yml \
   --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
   --checkpoint_path <path_to_save_checkpoint> \
   --transfer \
   --resume_path <path_to_pretrained_model> \
   --seed 42

**Step 3**: Running prediction
------------------------------

Predict the unlabeled data:

.. code-block:: bash

   python scripts/predict.py --task prop \
   --test_data <path_to_csv_or_pkl> \
   --model_config_path ./molnetpack/config/molnet_rt.yml \
   --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
   --resume_path <path_to_checkpoint> \
   --result_path <path_to_results.csv> \
   --seed 42
