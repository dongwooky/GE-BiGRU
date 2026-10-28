import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import dateutil.parser
from sklearn.metrics import confusion_matrix
from scipy import signal
from tqdm import tqdm

# Always run relative to the code/ directory, regardless of where the script is launched
os.chdir(Path(__file__).resolve().parent)

from utils import list_csvs, dataframe_from_csvs, normalize
from preprocess import HaiDataset
from modules import build_model, device, GATLayer
from training import inference
from plotting import plot_anomaly_scores, plot_channel_heatmap, plot_topk_channel_heatmap
from prediction import put_labels, fill_blank
from TaPR_pkg import etapr
import pandas as pd

# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate an anomaly detection model.")
parser.add_argument("--model", type=str, default="Bi-GRU+GAT",
                    help="Model variant name (e.g. 'GRU', 'Bi-GRU+GNN', 'Bi-GRU+GAT', 'LSTM+GCN')")
args = parser.parse_args()
MODEL_NAME = args.model

RESULT_DIR = f"../result/{MODEL_NAME}"
os.makedirs(RESULT_DIR, exist_ok=True)

# ─── Logging (console + file) ────────────────────────────────────────────────
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

log_path = f"{RESULT_DIR}/predict_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
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
torch.set_printoptions(threshold=50000)

# The first 43,201 rows of the concatenated test files (= test1 of HAI 21.03, 5 attacks)
# are used as the validation split for threshold selection; the remaining rows
# (test2-test5, 45 attacks) form the held-out test split.
VALIDATION_ROWS = 43201

#Get Dataset
#==============================================================================
train_dataset = list_csvs("../data/train/")
test_dataset = list_csvs("../data/test/")
if not train_dataset or not test_dataset:
    sys.exit("Data not found in data/train/ and data/test/. Run `python download_data.py` first.")
train_df_raw = dataframe_from_csvs(train_dataset).drop(['attack', 'attack_P1', 'attack_P2', 'attack_P3'], axis = 1)
valid_columns = train_df_raw.columns.drop(["time"])

#Normalization
#==============================================================================
tag_min = train_df_raw[valid_columns].min()
tag_max = train_df_raw[valid_columns].max()
train_df = normalize(train_df_raw[valid_columns], tag_min, tag_max).ewm(alpha=0.9).mean()

#Model
#==============================================================================
print(f"Loading model: {MODEL_NAME}")
model = build_model(MODEL_NAME, n_tags=train_df.shape[1], n_hiddens=N_HIDDENS,
                    n_layers=N_LAYERS, window_size=WINDOW_SIZE)
model.to(device)
saved_model = torch.load(f"../model/saved_model_{MODEL_NAME}.pt", map_location=device)
model.load_state_dict(saved_model["state"])

#Validation: threshold selection
#==============================================================================
validation_df_raw = dataframe_from_csvs(test_dataset).iloc[:VALIDATION_ROWS]
validation_df = normalize(validation_df_raw[valid_columns], tag_min, tag_max).ewm(alpha=0.9).mean()

hai_dataset_validation = HaiDataset(validation_df_raw["time"], WINDOW_SIZE, validation_df, attacks=validation_df_raw["attack"])

model.eval()
check_ts, check_dist, check_att = inference(hai_dataset_validation, model, BATCH_SIZE)
per_channel_score_val = check_dist                  # [T, 79], kept for per-channel analysis
anomaly_score = np.mean(check_dist, axis=1)
anomaly_score_smooth = signal.filtfilt(*signal.butter(1, 0.1, btype='lowpass'), anomaly_score)

# Validation TaPR F1 maximization
attack_labels_val = put_labels(np.array(validation_df_raw["attack"]), threshold=0.5)
val_total_ts = np.array(validation_df_raw["time"])

# Parse timestamps once (fill_blank would re-parse them on every call)
parsed_check_ts = [dateutil.parser.parse(t) for t in check_ts]
parsed_total_ts = [dateutil.parser.parse(t) for t in val_total_ts]

thresholds = np.linspace(anomaly_score_smooth.min(), anomaly_score_smooth.max(), 200)
f1_scores = []
for t in tqdm(thresholds, desc="Threshold sweep"):
    preds = put_labels(anomaly_score_smooth, t)
    label_dict = dict(zip(parsed_check_ts, preds))
    final = np.array([label_dict.get(ts, 0) for ts in parsed_total_ts], dtype=np.int8)
    tapr = etapr.evaluate(anomalies=attack_labels_val, predictions=final)
    f1_scores.append(float(tapr['f1']))

TOLERANCE = 0.01  # allowed gap from the maximum validation F1
max_f1 = max(f1_scores)
best_idx = max(i for i, f1 in enumerate(f1_scores) if f1 >= max_f1 - TOLERANCE)
THRESHOLD = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
print(f"Best Threshold (Val TaPR F1 Max): {THRESHOLD:.6f}, Val TaPR F1: {best_f1:.3f}")

plot_anomaly_scores(anomaly_score_smooth, check_att, num_segments=3, THRESHOLD=THRESHOLD,
                    filename=f'{RESULT_DIR}/valid_result.png')
plot_channel_heatmap(
    per_channel_score_val, check_att,
    sensor_names=valid_columns,
    filename=f'{RESULT_DIR}/valid_channel_heatmap.png',
)

# Channels with high GAT attention weights (Section 4.3, Fig. 6 of the paper)
GAT_NODES = ['P1_FCV03D', 'P1_B3004', 'P1_FT03', 'P1_LCV01D', 'P1_LIT01']
attack_labels_val_aligned = attack_labels_val[-len(check_ts):]
plot_topk_channel_heatmap(
    per_channel_score_val,
    attack_labels=attack_labels_val_aligned,
    sensor_names=valid_columns,
    gat_nodes=GAT_NODES,
    top_k=20,
    filename=f'{RESULT_DIR}/valid_topk_heatmap.png',
)

# Save per-channel anomaly scores (validation)
val_score_df = pd.DataFrame(per_channel_score_val, index=check_ts, columns=valid_columns)
val_score_df.index.name = 'time'
val_score_df.to_csv(f'{RESULT_DIR}/valid_channel_score.csv')

#Test
#==============================================================================
test_df_raw = dataframe_from_csvs(test_dataset).iloc[VALIDATION_ROWS:]
test_df = normalize(test_df_raw[valid_columns], tag_min, tag_max).ewm(alpha=0.9).mean()

hai_dataset_test = HaiDataset(test_df_raw["time"], WINDOW_SIZE, test_df, attacks=test_df_raw["attack"])

model.eval()
t_infer_start = time.time()
check_ts, check_dist, check_att = inference(hai_dataset_test, model, BATCH_SIZE)
inference_total_s = time.time() - t_infer_start
per_channel_score_test = check_dist                 # [T, 79], kept for per-channel analysis
anomaly_score = np.mean(check_dist, axis=1)
anomaly_score_smooth = signal.filtfilt(*signal.butter(1, 0.1, btype='lowpass'), anomaly_score)
labels = put_labels(anomaly_score_smooth, THRESHOLD)
plot_anomaly_scores(anomaly_score_smooth, labels, num_segments=4, THRESHOLD=THRESHOLD,
                    filename=f'{RESULT_DIR}/test_result.png')
plot_channel_heatmap(
    per_channel_score_test, check_att,
    sensor_names=valid_columns,
    filename=f'{RESULT_DIR}/test_channel_heatmap.png',
)

attack_labels_test = put_labels(np.array(test_df_raw["attack"]), threshold=0.5)
attack_labels_test_aligned = attack_labels_test[-len(check_ts):]
plot_topk_channel_heatmap(
    per_channel_score_test,
    attack_labels=attack_labels_test_aligned,
    sensor_names=valid_columns,
    gat_nodes=GAT_NODES,
    top_k=20,
    filename=f'{RESULT_DIR}/test_topk_heatmap.png',
)

# Save per-channel anomaly scores (test)
test_score_df = pd.DataFrame(per_channel_score_test, index=check_ts, columns=valid_columns)
test_score_df.index.name = 'time'
test_score_df.to_csv(f'{RESULT_DIR}/test_channel_score.csv')

print("==================Test Result==================")
attack_labels = put_labels(np.array(test_df_raw["attack"]), threshold=0.5)
final_labels = fill_blank(check_ts, labels, np.array(test_df_raw["time"]))
TaPR = etapr.evaluate(anomalies=attack_labels, predictions=final_labels)
print(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")

cm = confusion_matrix(attack_labels, final_labels)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
pointwise_f1 = 2 * precision * sensitivity / (precision + sensitivity)
print(f"Confusion Matrix:\n{cm}\nSensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}\nPoint-wise F1: {pointwise_f1:.3f}")

# ─── Efficiency metrics (inference) ──────────────────────────────────────────
n_test_samples = len(check_ts)
inference_ms_per_sample = inference_total_s * 1000 / n_test_samples if n_test_samples > 0 else 0.0
print(f"Inference time: {inference_total_s:.3f} s  ({inference_ms_per_sample:.4f} ms/sample, {n_test_samples} samples)")

eff_path = f"{RESULT_DIR}/efficiency.json"
try:
    with open(eff_path) as f:
        efficiency = json.load(f)
except FileNotFoundError:
    efficiency = {"model": MODEL_NAME}
efficiency["n_test_samples"] = n_test_samples
efficiency["inference_total_s"] = round(inference_total_s, 4)
efficiency["inference_ms_per_sample"] = round(inference_ms_per_sample, 6)
with open(eff_path, "w") as f:
    json.dump(efficiency, f, indent=2)
print(f"Efficiency metrics updated: {eff_path}")

# For GAT models, save the attention weights as a 79x79 matrix
if hasattr(model, 'graph_layer') and isinstance(model.graph_layer, GATLayer):
    attention_weights = model.graph_layer.attention_weights
    print(attention_weights[0])
    print(attention_weights[0].shape)
    print(attention_weights[1].t())
    print(attention_weights[1].t().shape)

    num_nodes = train_df.shape[1]
    adj_matrix_out = torch.zeros((num_nodes, num_nodes))
    for i in range(attention_weights[0].size(1)):
        src = attention_weights[0][0, i].item()
        dst = attention_weights[0][1, i].item()
        weight = attention_weights[1].t()[0, i].item()
        adj_matrix_out[src, dst] = weight
    adj_matrix_out = adj_matrix_out.numpy()
    pd.DataFrame(adj_matrix_out).to_csv(f'{RESULT_DIR}/adj_matrix.csv', index=False)
