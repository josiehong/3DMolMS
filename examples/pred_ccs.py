import torch
from molnetpack import MolNet

# CPU
device = torch.device("cpu")

# GPU
# gpu_index = 0 # please set this into the index of GPU you plan to use
# device = torch.device("cuda:" + str(gpu_index))

molnet_engine = MolNet(device, seed=42)

molnet_engine.load_data(path_to_test_data="./examples/demo_input.csv")

# Pred CCS
# ccs_df = molnet_engine.pred_ccs(path_to_results='./examples/output_ccs.csv', path_to_checkpoint='./check_point/molnet_ccs_etkdgv3_tl.pt')
ccs_df = molnet_engine.pred_ccs(
    path_to_results="./examples/output_ccs.csv"
)  # Download checkpoint from GitHub release page.
