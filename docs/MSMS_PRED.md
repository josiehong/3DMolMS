# MS/MS Prediction using 3DMolMS

Step 0: Clone this repository and setup the anaconda environment by the following commands: 

```bash
git clone https://github.com/JosieHong/3DMolMS.git
cd 3DMolMS

pip install .
```

Step 1: Prepare the test set. The following formats are supported: csv, mgf, or [customed pkl](molnetpack/data_utils/all2pkl.py). 

Here is an input example from MoNA of csv format (see the whole file at `./examples/demo_input.csv`): 

```
ID,SMILES,Precursor_Type,Collision_Energy
demo_0,C/C(=C\CNc1nc[nH]c2ncnc1-2)CO,[M+H]+,40 V
demo_1,C/C(=C\CNc1nc[nH]c2ncnc1-2)CO,[M+H]+,20 V
demo_2,C/C(=C\CNc1nc[nH]c2ncnc1-2)CO,[M+H]+,10 V
```

Here is an input example from MoNA of mgf format (see the whole file at `./examples/demo_input.mgf`): 

```
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
......
```

Please notice that the unsupported input will be filtered out automatically when loading the dataset. The supported inputs are shown in the following table. 

| Item             | Supported input                                               |
|------------------|---------------------------------------------------------------|
| Atom number      | <=300                                                         |
| Atom types       | 'C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I', 'Na' |
| Precursor types  | '[M+H]+', '[M-H]-', '[M+H-H2O]+', '[M+Na]+'                   |
| Collision energy | any number                                                    |

Step 2: Predict the MS/MS spectra using the following command: 

```bash
python scripts/predict.py --task msms \
--test_data ./examples/demo_input.csv \
--model_config_path ./molnetpack/config/molnet.yml \
--data_config_path ./molnetpack/config/preprocess_etkdgv3.yml \
--resume_path ./check_point/molnet_qtof_etkdgv3.pt \
--result_path ./examples/demo_output.mgf
```

Arguments: 

- `--resume_path` is the path of model's checkpoint. In the first running, the checkpoint (`./checkpoint/molnet_qtof_etkdgv3.pt`) will be downloaded from [Google Drive](https://drive.google.com/drive/folders/1fWx3d8vCPQi-U-obJ3kVL3XiRh75x5Ce?usp=drive_link). You can also set the resume path as the path to your own model. 
- `--result_path` is the path to save the predicted MS/MS. It should end with `.mgf` or `.csv`. We recommend you use `.mgf` because mgf is a more common format for MS/MS.  

<p align="center">
  <img src='img/demo_0.png' width='600'>
</p>
