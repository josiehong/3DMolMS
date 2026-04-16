# Molecular Properties Prediction using 3DMolMS

3DMolMS can be used to predict MS/MS-related properties, such as retention time (RT) and collision cross section (CCS). This file shows how to train a model for [[RT prediction]](#retention-time-prediction) and [[CCS prediction]](#cross-collision-section-prediction), and how to [[transfer these models to your own RT and CCS dataset]](#fine-tune-on-your-own-data). All the following models can be downloaded from [[release v1.1.6]](https://github.com/JosieHong/3DMolMS/releases/tag/v1.1.6). 

## Retention time prediction

Please set up the environment as shown in step 0 from `README.md`. 

Step 1: Download the retention time dataset, [[METLIN]](https://figshare.com/articles/dataset/The_METLIN_small_molecule_dataset_for_machine_learning-based_retention_time_prediction/8038913?file=18130625). The structure of data directory is: 

```bash
|- data
  |- origin
    |- SMRT_dataset.sdf
```

Step 2: Use the following commands to preprocess the datasets. The settings of datasets are in `./molnetpack/config/preprocess_etkdgv3.yml`. 

```bash
python scripts/preprocess.py --task rt \
--data_config_path ./molnetpack/config/preprocess_etkdgv3.yml
```

Step 3: Use the following commands to train the model. The settings of model and training are in `./molnetpack/config/molnet_rt.yml`. If you'd like to train this model from the pre-trained model on MS/MS prediction, please download the pre-trained model from [[Google Drive]](https://drive.google.com/drive/folders/1fWx3d8vCPQi-U-obJ3kVL3XiRh75x5Ce?usp=drive_link). 

```bash
# learn from scratch
python scripts/train.py --task rt \
--train_data ./data/metlin_etkdgv3_train.pkl \
--test_data ./data/metlin_etkdgv3_test.pkl \
--model_config_path ./molnetpack/config/molnet_rt.yml \
--data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnet_rt_etkdgv3.pt

# learn from pretrained model
python scripts/train.py --task rt \
--train_data ./data/metlin_etkdgv3_train.pkl \
--test_data ./data/metlin_etkdgv3_test.pkl \
--model_config_path ./molnetpack/config/molnet_rt.yml \
--data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnet_rt_etkdgv3_tl.pt \
--transfer \
--resume_path ./check_point/molnet_qtof_etkdgv3.pt
```



## Cross-collision section prediction

Please set up the environment as shown in step 0 from `README.md`. 

Step 1: Download the cross-collision section dataset, [[AllCCS]](http://allccs.zhulab.cn/), manually or using `download_allccs.py`:

```bash
python scripts/download_allccs.py --user <user_name> --passw <passwords> --output ./data/origin/allccs_download.csv
```

The structure of data directory is: 

```bash
|- data
  |- origin
    |- allccs_download.csv
```

Step 2: Use the following commands to preprocess the datasets. The settings of datasets are in `./molnetpack/config/preprocess_etkdgv3.yml`. 

```bash
python scripts/preprocess.py --task ccs \
--data_config_path ./molnetpack/config/preprocess_etkdgv3.yml
```

Step 3: Use the following commands to train the model. The settings of model and training are in `./molnetpack/config/molnet_ccs.yml`. If you'd like to train this model from the pre-trained model on MS/MS prediction, please download the pre-trained model from [[Google Drive]](https://drive.google.com/drive/folders/1fWx3d8vCPQi-U-obJ3kVL3XiRh75x5Ce?usp=drive_link). 

```bash
# learn from scratch
python scripts/train.py --task ccs \
--train_data ./data/allccs_etkdgv3_train.pkl \
--test_data ./data/allccs_etkdgv3_test.pkl \
--model_config_path ./molnetpack/config/molnet_ccs.yml \
--data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnet_ccs_etkdgv3.pt

# learn from pretrained model
python scripts/train.py --task ccs \
--train_data ./data/allccs_etkdgv3_train.pkl \
--test_data ./data/allccs_etkdgv3_test.pkl \
--model_config_path ./molnetpack/config/molnet_ccs.yml \
--data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
--checkpoint_path ./check_point/molnet_ccs_etkdgv3_tl.pt \
--transfer \
--resume_path ./check_point/molnet_qtof_etkdgv3.pt
```

## Fine-tune on your own data

Step 1: Please prepare your data as two pkl files (train and test). If starting from a CSV, split it into train/test CSVs first, then convert each with: 

```bash
python -c "
from molnetpack import csv2pkl_wfilter
import yaml, pickle
with open('./molnetpack/config/preprocess_etkdgv3.yml') as f:
    cfg = yaml.safe_load(f)
data = csv2pkl_wfilter('<path_to_train.csv>', cfg['encoding'])
pickle.dump(data, open('<path_to_train.pkl>', 'wb'))
"
```

The CSV should have at minimum `id`, `smiles`, and `prop` columns, where `prop` is the RT or CCS value:

```csv
id,smiles,prop
0382_00004,NC(=O)N1c2ccccc2[C@H](O)[C@@H](O)c2ccccc21,5.79
0382_00005,CN(C)[C@@H]1C(=O)C(C(N)=O)=C(O)[C@@]2(O)...,4.5
```

Step 2: Fine-tune the model:

```bash
python scripts/train.py --task rt \
--train_data <path_to_train.pkl> \
--test_data <path_to_test.pkl> \
--model_config_path ./molnetpack/config/molnet_rt.yml \
--data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
--checkpoint_path <path_to_save_checkpoint> \
--transfer \
--resume_path <path_to_pretrained_model>
```

Step 3: Predict on unlabeled data:

```bash
python scripts/predict.py --task prop \
--test_data <path_to_csv_or_pkl> \
--model_config_path ./molnetpack/config/molnet_rt.yml \
--data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
--resume_path <path_to_checkpoint> \
--result_path <path_to_results.csv>
```
