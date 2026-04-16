import torch
import molnetpack
from molnetpack import MolNet

# Set the device to CPU for CPU-only usage:
device = torch.device("cpu")

# For GPU usage, set the device as follows (replace '0' with your desired GPU index):
# gpu_index = 0
# device = torch.device(f"cuda:{gpu_index}")

# Instantiate a MolNet object
molnet_engine = MolNet(device, seed=42)  # The random seed can be any integer.

# Load input data (here we use a CSV file as an example)
molnet_engine.load_data(
    path_to_test_data="./examples/demo_input.csv"
)  # Increasing the batch size if you wanna speed up.
# molnet_engine.load_data(path_to_test_data='./examples/demo_input.mgf') # MGF file is also supported
# molnet_engine.load_data(path_to_test_data='./examples/demo_input.pkl') # PKL file is faster.

# Predict MS/MS
# spectra = molnet_engine.pred_msms(path_to_results='./examples/output_msms.mgf', path_to_checkpoint='./check_point/molnet_qtof_etkdgv3.pt', instrument='qtof')
spectra = molnet_engine.pred_msms(
    path_to_results="./examples/output_msms.mgf", instrument="qtof"
)  # Download checkpoint from GitHub release page.
msms_res_df = molnet_engine.pred_msms(
    path_to_results="./examples/output_msms.csv", instrument="qtof"
)  # Download checkpoint from GitHub release page.

# Plot the predicted MS/MS with 3D molecular conformation
molnetpack.plot_msms(msms_res_df=msms_res_df, dir_to_img="./img")
