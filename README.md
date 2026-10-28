# GE-BiGRU: Graph-Enhanced Bidirectional GRU for Anomaly Detection in Power Generation Environments

Official code for the paper:

> Dongwook Kwon, Youngshin Kang, Jiwoon Lee, Yusang Nam, Jihwan Won, Ko Keun Kim, Dan Dongseong Kim, Cheolsoo Park.
> **Graph-enhanced bidirectional GRU for anomaly detection in power generation environments.**
> *Data Mining and Knowledge Discovery* 40, 102 (2026). https://doi.org/10.1007/s10618-026-01262-3

![GE-BiGRU architecture](figures/architecture.png)

*Figure: prediction-based anomaly detection framework (Fig. 3 of the paper, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).*

## Overview

GE-BiGRU is a prediction-based anomaly detector for multivariate time series from industrial control systems:

- The model is trained on normal operation only. From a 39-step window it predicts the next value of all 79 sensors and actuators, and the anomaly score is the mean absolute prediction error.
- A graph layer models the relations between sensors using a binary adjacency matrix derived from the piping and instrumentation diagrams (P&ID) of the HAI testbed (`data/graph/graph.csv`). Its output feeds a 3-layer bidirectional GRU with a residual connection.
- The detection threshold is chosen on a validation split from the eTaPR F1 score (see [Implementation notes](#implementation-notes)) and then fixed for the test split.

The code covers all 16 variants in Table 6 of the paper (Tables 4–5 report detection results for 14 of them): LSTM or GRU, unidirectional or bidirectional, with no graph layer or with a GNN, GCN, or GAT layer.

| Graph layer | Implementation |
|---|---|
| GNN | Aggregation with the fixed, row-normalized adjacency (D⁻¹A), followed by a learnable linear map |
| GCN | `torch_geometric.nn.GCNConv` |
| GAT | `torch_geometric.nn.GATConv` (single head); the attention coefficients of the last test window are saved |

## Results

HAI 21.03, 45 test attacks. Bidirectional models from Table 5 of the paper:

| Model | Sensitivity | Specificity | TaP | TaR | F1 |
|---|---|---|---|---|---|
| Bi-GRU | 0.720 | 0.995 | 0.933 | 0.854 | 0.892 |
| **Bi-GRU+GNN** | 0.752 | 0.994 | 0.940 | 0.909 | **0.924** |
| Bi-GRU+GCN | 0.724 | 0.994 | 0.921 | 0.893 | 0.907 |
| Bi-GRU+GAT | 0.694 | 0.996 | 0.972 | 0.843 | 0.903 |

F1 is the harmonic mean of TaP and TaR. Unidirectional and LSTM variants are reported in the paper. No random seed is fixed, so retrained results can differ slightly from these numbers.

## Setup

```bash
git clone https://github.com/dongwooky/GE-BiGRU.git
cd GE-BiGRU
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version first ([instructions](https://pytorch.org/get-started/locally/)). The paper's experiments used PyTorch 2.1.0 with CUDA 12.1 on one NVIDIA RTX 6000 Ada GPU. The code also runs on CPU, but training is slow.

## Data

The HAI 21.03 dataset ([Shin et al., 2021](https://doi.org/10.1145/3474718.3474719)) is available from the official repository: https://github.com/icsdataset/hai

```bash
cd code
python download_data.py
```

This downloads `train1–3.csv.gz` into `data/train/` and `test1–5.csv.gz` into `data/test/`. Uncompressed `.csv` files work as well.

| Split | Files | Attacks | Use |
|---|---|---|---|
| Train | train1–3 | none | model training (normal operation only) |
| Validation | test1 (first 43,201 rows of the test files) | 5 | threshold selection only |
| Test | test2–5 | 45 | evaluation |

`data/graph/graph.csv` is the 79×79 binary adjacency matrix. The first line is a header of node indices 0–78, and rows and columns follow the sensor column order of the HAI 21.03 files (Tables 1–2 in the paper). Entry (i, j) = 1 denotes an edge i → j, and the diagonal is 1. Some node names in Tables 1–2 are descriptive aliases of the CSV headers (e.g., P2_AutoSD = P2_ASD).

## Usage

Run all scripts from `code/`:

```bash
cd code

# Quick check without data: builds all 16 models and compares parameter counts with Table 6
python test_models.py

# Train and evaluate one model
python train.py --model Bi-GRU+GNN
python predict.py --model Bi-GRU+GNN

# All 16 variants, or a subset
python run_experiments.py
python run_experiments.py --models Bi-GRU Bi-GRU+GNN Bi-GRU+GAT
```

Model names: `LSTM`, `GRU`, `LSTM+GNN`, `GRU+GNN`, `LSTM+GCN`, `GRU+GCN`, `LSTM+GAT`, `GRU+GAT`, and the same names with a `Bi-` prefix.

Outputs:

- `model/saved_model_<MODEL>.pt`: trained weights
- `result/<MODEL>/`: logs, anomaly score plots, per-channel heatmaps and scores, `efficiency.json`, and for GAT models the attention coefficients of the last test window as a 79×79 matrix (`adj_matrix.csv`)
- `logs/<run_id>/summary.md`: result tables in the format of the paper (from `run_experiments.py`)

Training time per model on one RTX 6000 Ada ranges from about 20 minutes (GRU) to about 16–17 hours (GAT variants); see Table 6 of the paper.

## Hyperparameters

Defined at the top of `train.py` and `predict.py`, identical for all variants:

| Parameter | Value |
|---|---|
| Window | 40 steps (39 input steps → next step), stride 1 |
| RNN | 3 layers, hidden size 100 |
| Training | 70 epochs, batch size 512, AdamW (default lr = 1e-3), MSE loss |
| Preprocessing | min–max scaling with training statistics, EMA smoothing (α = 0.9) |
| Score smoothing | first-order low-pass Butterworth filter (normalized cutoff 0.1, i.e., 0.05 Hz), applied offline with `scipy.signal.filtfilt` |
| Threshold | 200 evenly spaced candidates over the smoothed validation scores; the largest threshold within 0.01 of the best validation eTaPR F1 is used |

## Implementation notes

The results in the paper were produced with this code, and the parameter counts in Table 6 of the paper match these implementations exactly (`python test_models.py` checks them). Where the description in the paper differs from or is less specific than the code, the code reflects what was used in the experiments:

- **Graph layers**: GNN, GCN, and GAT are alternative variants; each model uses at most one graph layer, placed before the RNN. The layer acts on each node's 39-step input window and outputs 39 features per node, which the RNN reads as its input sequence.
- **GNN layer**: aggregation with the fixed, row-normalized P&ID adjacency (D⁻¹A), followed by a learnable linear map over the 39 features. The adjacency matrix itself is not trained. In this layer node *i* aggregates from the nodes *j* with A[i, j] = 1, whereas in the GCN and GAT layers messages flow along *i* → *j*.
- **GAT layer**: a single attention head (`heads=1`), with no additional activation after the layer. `predict.py` saves the attention coefficients of the last test window as a 79×79 matrix; they depend on the input window.
- **Output**: the prediction is read from the last RNN output and added to the first element of the (graph-transformed) input window, as in the HAICon 2020 baseline.
- **Threshold selection**: among the 200 candidates, the largest threshold whose validation F1 is within 0.01 of the best one is used, which favors fewer false alarms when several thresholds perform almost equally. Only the validation split (test1) is used for this; the test split is never used for tuning.
- **Score smoothing**: the Butterworth filter is applied with zero-phase filtering (`scipy.signal.filtfilt`) over the full score series, i.e., offline. An online deployment would use a causal filter instead.
- **Checkpoints**: `saved_model_<MODEL>.pt` holds the weights after the final epoch (70); the `best_epoch` value stored with it is not used.

## Citation

If you use this code, please cite:

```bibtex
@article{kwon2026gebigru,
  title   = {Graph-enhanced bidirectional {GRU} for anomaly detection in power generation environments},
  author  = {Kwon, Dongwook and Kang, Youngshin and Lee, Jiwoon and Nam, Yusang and Won, Jihwan and Kim, Ko Keun and Kim, Dan Dongseong and Park, Cheolsoo},
  journal = {Data Mining and Knowledge Discovery},
  volume  = {40},
  pages   = {102},
  year    = {2026},
  doi     = {10.1007/s10618-026-01262-3}
}
```

Please also cite the HAI dataset ([Shin et al., CSET '21](https://doi.org/10.1145/3474718.3474719)) and the TaPR/eTaPR metrics ([Hwang et al., CIKM '19](https://doi.org/10.1145/3357384.3358118); [Hwang et al., SAC '22](https://doi.org/10.1145/3477314.3507024)) if you use them.

## Acknowledgements

The data loading pipeline and the stacked bidirectional GRU backbone are adapted from the baseline code provided for the [HAICon 2020](https://dacon.io/competitions/official/235624/overview/description) anomaly detection competition on HAI 21.03. Evaluation uses [eTaPR](https://github.com/wshw4ng/eTaPR). We thank the authors of the HAI dataset for making it publicly available.

## License

- **Code**: [MIT License](LICENSE).
- **`data/graph/graph.csv`**: derived from the P&ID documentation of the HAI testbed and shared under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), the license of the HAI dataset.
- **`figures/architecture.png`**: Fig. 3 of the paper, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **HAI dataset**: not included in this repository. `code/download_data.py` downloads it from the [official repository](https://github.com/icsdataset/hai), where it is distributed under CC BY-SA 4.0.
- **eTaPR**: installed from its [official repository](https://github.com/wshw4ng/eTaPR) (MIT License).
