import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import torch

# Always run relative to the code/ directory, regardless of where the script is launched
os.chdir(Path(__file__).resolve().parent)

from utils import list_csvs, dataframe_from_csvs, normalize
from preprocess import HaiDataset
from modules import build_model, device
from training import train

# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train an anomaly detection model.")
parser.add_argument("--model", type=str, default="Bi-GRU+GAT",
                    help="Model variant name (e.g. 'GRU', 'Bi-GRU+GNN', 'Bi-GRU+GAT', 'LSTM+GCN')")
args = parser.parse_args()
MODEL_NAME = args.model

# ─── Logging (console + file) ────────────────────────────────────────────────
RESULT_DIR = f"../result/{MODEL_NAME}"
os.makedirs(RESULT_DIR, exist_ok=True)

class _Tee:
    def __init__(self, filepath):
        self._file = open(filepath, "w")
        self._stdout = sys.__stdout__
    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)
    def flush(self):
        self._stdout.flush()
        self._file.flush()

log_path = f"{RESULT_DIR}/train_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
sys.stdout = _Tee(log_path)
print(f"Log: {log_path}")
print(f"Device: {device}")

# ─── Hyperparameters ──────────────────────────────────────────────────────────
STRIDE = 1
WINDOW_SIZE = 40
N_HIDDENS = 100
N_LAYERS = 3
BATCH_SIZE = 512
EPOCH = 70

#Get Dataset
#==============================================================================
train_dataset = list_csvs("../data/train/")
if not train_dataset:
    sys.exit("No training files found in data/train/. Run `python download_data.py` first.")
train_df_raw = dataframe_from_csvs(train_dataset).drop(['attack', 'attack_P1', 'attack_P2', 'attack_P3'], axis = 1)
valid_columns_in_train_dataset = train_df_raw.columns.drop(["time"])

#Normalization
#==============================================================================
tag_min = train_df_raw[valid_columns_in_train_dataset].min()
tag_max = train_df_raw[valid_columns_in_train_dataset].max()
train_df = normalize(train_df_raw[valid_columns_in_train_dataset], tag_min, tag_max).ewm(alpha=0.9).mean()

#Define data input & output
#==============================================================================
print("train_df_raw length:", len(train_df_raw))
hai_dataset_train = HaiDataset(train_df_raw["time"], WINDOW_SIZE, train_df, stride=STRIDE)

#Model
#==============================================================================
print(f"Building model: {MODEL_NAME}")
model = build_model(MODEL_NAME, n_tags=train_df.shape[1], n_hiddens=N_HIDDENS,
                    n_layers=N_LAYERS, window_size=WINDOW_SIZE).to(device)
print("train_df.shape:", train_df.shape[1])
model.train()
t_train_start = time.time()
best_model, loss_history, epoch_times = train(hai_dataset_train, model, BATCH_SIZE, EPOCH)
total_train_s = time.time() - t_train_start

# For GAT models, print the attention weights
if (hasattr(model, 'graph_layer') and model.graph_layer is not None
        and hasattr(model.graph_layer, 'attention_weights')
        and model.graph_layer.attention_weights is not None):
    attention_weights = model.graph_layer.attention_weights
    print("attention_weights:")
    print(attention_weights)
    print(attention_weights[0].shape)
    print(attention_weights[1].shape)
    print("attention_weights[1]:")
    print(attention_weights[1].t())
    print(attention_weights[1].t().shape)

# ─── Save ─────────────────────────────────────────────────────────────────────
save_dir = "../model"
os.makedirs(save_dir, exist_ok=True)
save_path = f"{save_dir}/saved_model_{MODEL_NAME}.pt"
torch.save({"state": best_model["state"], "best_epoch": best_model["epoch"]}, save_path)
print(f"Model saved to {save_path}")

# ─── Efficiency metrics ───────────────────────────────────────────────────────
n_params = sum(p.numel() for p in model.parameters())
efficiency = {
    "model": MODEL_NAME,
    "n_params": n_params,
    "total_training_time_min": round(total_train_s / 60, 4),
    "avg_time_per_epoch_s": round(sum(epoch_times) / len(epoch_times), 4),
    "epoch_times_s": [round(t, 4) for t in epoch_times],
}
eff_path = f"{RESULT_DIR}/efficiency.json"
with open(eff_path, "w") as f:
    json.dump(efficiency, f, indent=2)
print(f"Efficiency metrics saved to {eff_path}")
print(f"  Parameters : {n_params:,}")
print(f"  Total train: {efficiency['total_training_time_min']:.2f} min")
print(f"  Avg/epoch  : {efficiency['avg_time_per_epoch_s']:.2f} s")
