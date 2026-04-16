# Usage of `molnetpack` for Molecular Properties Prediction

## MolNet Instantiation

Before doing any prediction, please intantiate `MolNet`:

```python
import torch
from molnetpack import MolNet

# Set the device to CPU for CPU-only usage:
device = torch.device("cpu")

# For GPU usage, set the device as follows (replace '0' with your desired GPU index):
# gpu_index = 0
# device = torch.device(f"cuda:{gpu_index}")

# Instantiate a MolNet object
molnet_engine = MolNet(device, seed=42) # The random seed can be any integer. 
```

## CCS Prediction

For CCS prediction, please use `pred_ccs` as shown in the following codes after instantiating a MolNet object. 

```python
# Load input data
molnet_engine.load_data(path_to_test_data='./examples/input_ccs.csv')

# Pred CCS
ccs_df = molnet_engine.pred_ccs()
"""Predict Collision Cross Section (CCS) values.
Args:
    path_to_results (Optional[str]): Path to save the prediction results. The file will be saved in '.csv' format. If None, the results won't be saved. 
    path_to_checkpoint (Optional[str]): Path to the model checkpoint. If None, the model will be downloaded from a default URL.
Returns:
    pd.DataFrame: DataFrame containing the predicted CCS values.
"""
```

## RT Prediction

For RT prediction, please use `pred_rt` as shown in the following codes after instantiating a MolNet object. Please note that since this model is trained on the METLIN-SMRT dataset, the predicted retention time is under the same experimental conditions as the METLIN-SMRT set.

```python
# Load input data
molnet_engine.load_data(path_to_test_data='./examples/input_rt.csv')

# Pred RT
rt_df = molnet_engine.pred_rt()
"""Predict Retention Time (RT) values.
Args:
    path_to_results (Optional[str]): Path to save the prediction results. The file will be saved in '.csv' format. If None, the results won't be saved. 
    path_to_checkpoint (Optional[str]): Path to the model checkpoint. If None, the model will be downloaded from a default URL.
Returns:
    pd.DataFrame: DataFrame containing the predicted RT values.
"""
```

## Small Molecular Feature Embedding

For saving the molecular embeddings, please use the following codes after instantiating a MolNet object. 

```python
# Load input data
molnet_engine.load_data(path_to_test_data='./examples/input_savefeat.csv')

# Inference to get the features
ids, features = molnet_engine.save_features()

print('Titles:', ids)
print('Features shape:', features.shape)
```