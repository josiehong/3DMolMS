import torch
from molnetpack import MolNet

# Set the device to CPU for CPU-only usage:
device = torch.device("cpu")

# For GPU usage, set the device as follows (replace '0' with your desired GPU index):
# gpu_index = 0
# device = torch.device(f"cuda:{gpu_index}")

# Instantiate a MolNet object
molnet_engine = MolNet(device, seed=42)  # The random seed can be any integer.

# ---------------------------------------------------------------------------
# Option 1: Fine-tune from a pretrained checkpoint (recommended)
# Download molnet_pre_etkdgv3.pt from https://github.com/JosieHong/3DMolMS/releases
# ---------------------------------------------------------------------------
molnet_engine.train(
    task="msms",
    train_data="./data/qtof_etkdgv3_train.pkl",
    valid_data="./data/qtof_etkdgv3_test.pkl",
    checkpoint_path="./check_point/molnet_qtof_etkdgv3_tl.pt",
    resume_path="./check_point/molnet_pre_etkdgv3.pt",  # pretrained encoder
    transfer=True,  # freeze encoder weights, train decoder only
)

# ---------------------------------------------------------------------------
# Option 2: Train from scratch (no pretrained weights)
# ---------------------------------------------------------------------------
# molnet_engine.train(
#     task="msms",
#     train_data="./data/qtof_etkdgv3_train.pkl",
#     valid_data="./data/qtof_etkdgv3_test.pkl",
#     checkpoint_path="./check_point/molnet_qtof_etkdgv3.pt",
# )

# ---------------------------------------------------------------------------
# After training, the model is immediately ready for inference — no reload needed.
# ---------------------------------------------------------------------------
molnet_engine.load_data(path_to_test_data="./data/qtof_etkdgv3_test.pkl")
pred_df = molnet_engine.pred_msms(
    path_to_results="./result/pred_qtof_etkdgv3_test.mgf",
    instrument="qtof",
)

# ---------------------------------------------------------------------------
# Evaluate predicted spectra against ground truth
# ---------------------------------------------------------------------------
results_df = molnet_engine.evaluate(
    test_pkl="./data/qtof_etkdgv3_test.pkl",
    pred_mgf="./result/pred_qtof_etkdgv3_test.mgf",
    result_path="./result/eval_qtof_etkdgv3_test.csv",
    plot_path="./result/eval_qtof_etkdgv3_test.png",
)
