# Pretrain 3DMolMS on QM9

Please set up the environment as shown in step 0 from `README.md`. 

Step 1: Download the QM9 dataset [[here]](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904). The structure of data directory is: 

```bash
|- data
  |- qm9
    |- dsgdb9nsd.xyz.tar.bz2
    |- dsC7O2H10nsd.xyz.tar.bz2
    |- uncharacterized.txt
```

Step 2: Use the following commands to preprocess the datasets. The settings of datasets are in `./molnetpack/config/preprocess_etkdgv3.yml`. 

```bash
python scripts/qm92pkl.py --data_config_path ./molnetpack/config/preprocess_etkdgv3.yml
```

Step 3: Use the following commands to pretrain the model. The settings of model and training are in `./molnetpack/config/molnet_pre.yml`. 

```bash
python scripts/pretrain.py \
--train_data ./data/qm9_etkdgv3_train.pkl \
--test_data ./data/qm9_etkdgv3_test.pkl \
--model_config_path ./molnetpack/config/molnet_pre.yml \
--data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnet_pre_etkdgv3.pt
```
