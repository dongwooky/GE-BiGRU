from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Binary 79x79 adjacency matrix derived from the HAI testbed P&ID diagrams.
# Node order follows the sensor/actuator column order of the HAI 21.03 CSV files.
GRAPH_CSV = Path(__file__).resolve().parent.parent / "data" / "graph" / "graph.csv"
tensor = torch.tensor(pd.read_csv(GRAPH_CSV).values, dtype=torch.float).to(device)
edge_index = tensor.nonzero().t()  # the adjacency matrix is binary
edge_index = edge_index.to(device)
print(f"tensor shape: {tensor.shape}")
print(f"Original edge_index shape: {edge_index.shape}")

# Row-normalized adjacency for GNNLayer: D^{-1} A
row_sum = tensor.sum(dim=1, keepdim=True).clamp(min=1e-6)
adj_norm = (tensor / row_sum).to(device)


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=2):
        super(GATLayer, self).__init__()
        self.gat = GATConv(in_features, out_features, heads=heads, add_self_loops=True).to(device)
        self.attention_weights = None

    def forward(self, x, edge_index):
        self.attention_weights = None
        out, attention_weights = self.gat(x=x, edge_index=edge_index, return_attention_weights=True)
        self.attention_weights = attention_weights
        return out


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.gcn = GCNConv(in_features, out_features, add_self_loops=True).to(device)
        self.attention_weights = None  # stub for uniform attribute access

    def forward(self, x, edge_index):
        return self.gcn(x, edge_index)


class GNNLayer(nn.Module):
    """Simple graph aggregation: H' = W (D^{-1} A H)

    Uses the row-normalized P&ID adjacency matrix for neighbor aggregation,
    followed by a learnable linear map over the time features.
    """
    def __init__(self, in_features, out_features, adj_norm):
        super(GNNLayer, self).__init__()
        self.register_buffer("adj_norm", adj_norm)  # (F, F), not trainable
        self.W = nn.Linear(in_features, out_features, bias=False).to(device)
        self.attention_weights = None  # stub for uniform attribute access

    def forward(self, x, edge_index=None):
        # x: (F, T-1) - F sensor nodes, each with T-1 time features
        # adj_norm: (F, F)
        return self.W(self.adj_norm @ x)  # (F, T-1)


# ─── Model variants (Tables 4-6 of the paper) ────────────────────────────────

MODEL_REGISTRY = {
    # name: (rnn_type, graph_type, bidirectional)
    "LSTM":        ("LSTM", None,  False),
    "GRU":         ("GRU",  None,  False),
    "LSTM+GNN":    ("LSTM", "GNN", False),
    "GRU+GNN":     ("GRU",  "GNN", False),
    "LSTM+GCN":    ("LSTM", "GCN", False),
    "GRU+GCN":     ("GRU",  "GCN", False),
    "LSTM+GAT":    ("LSTM", "GAT", False),
    "GRU+GAT":     ("GRU",  "GAT", False),
    "Bi-LSTM":     ("LSTM", None,  True),
    "Bi-GRU":      ("GRU",  None,  True),
    "Bi-LSTM+GNN": ("LSTM", "GNN", True),
    "Bi-GRU+GNN":  ("GRU",  "GNN", True),
    "Bi-LSTM+GCN": ("LSTM", "GCN", True),
    "Bi-GRU+GCN":  ("GRU",  "GCN", True),
    "Bi-LSTM+GAT": ("LSTM", "GAT", True),
    "Bi-GRU+GAT":  ("GRU",  "GAT", True),
}


class AnomalyModel(nn.Module):
    def __init__(self, n_tags, n_hiddens, n_layers, window_size,
                 rnn_type="GRU", graph_type=None, bidirectional=False,
                 edge_index=edge_index):
        super().__init__()
        self.graph_type = graph_type
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.edge_index = edge_index
        n_features = window_size - 1  # each node carries the T-1 input time steps

        # Graph layer
        if graph_type == "GAT":
            self.graph_layer = GATLayer(in_features=n_features, out_features=n_features, heads=1)
        elif graph_type == "GCN":
            self.graph_layer = GCNLayer(in_features=n_features, out_features=n_features)
        elif graph_type == "GNN":
            self.graph_layer = GNNLayer(in_features=n_features, out_features=n_features, adj_norm=adj_norm)
        else:
            self.graph_layer = None

        # RNN
        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_tags,
            hidden_size=n_hiddens,
            num_layers=n_layers,
            bidirectional=bidirectional,
        ).to(device)

        # FC: input size doubles for bidirectional
        rnn_out_size = n_hiddens * (2 if bidirectional else 1)
        self.fc = nn.Linear(rnn_out_size, n_tags).to(device)

    def forward(self, x):
        x = x.transpose(1, 2).to(device)  # (B, T-1, F) -> (B, F, T-1)

        if self.graph_layer is not None:
            x = torch.stack(
                [self.graph_layer(graph.to(device), edge_index=self.edge_index)
                 for graph in x],
                dim=0,
            )  # (B, F, T-1)

        x = x.permute(2, 0, 1)  # (T-1, B, F)

        if self.rnn_type == "LSTM":
            outs, (h_n, c_n) = self.rnn(x)
        else:
            outs, _ = self.rnn(x)

        out = self.fc(outs[-1])
        return x[0] + out  # residual: first timestep + prediction


def build_model(model_name, n_tags, n_hiddens, n_layers, window_size):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    rnn_type, graph_type, bidirectional = MODEL_REGISTRY[model_name]
    return AnomalyModel(
        n_tags=n_tags,
        n_hiddens=n_hiddens,
        n_layers=n_layers,
        window_size=window_size,
        rnn_type=rnn_type,
        graph_type=graph_type,
        bidirectional=bidirectional,
    ).to(device)
