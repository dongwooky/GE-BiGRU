"""
test_models.py - smoke test for all model variants.

Builds every model and runs a forward pass on synthetic input (no dataset needed),
then checks the parameter count against Table 6 of the paper.

Usage:
  cd code
  python test_models.py
"""

import sys
import torch
from modules import build_model, MODEL_REGISTRY

# ─── Settings ─────────────────────────────────────────────────────────────────
BATCH_SIZE  = 4
WINDOW_SIZE = 40       # same as the paper
N_TAGS      = 79       # number of HAI 21.03 sensors/actuators
N_HIDDENS   = 100
N_LAYERS    = 3

# Number of parameters reported in Table 6 of the paper
PAPER_PARAMS = {
    "LSTM": 241_979,        "GRU": 183_479,
    "LSTM+GNN": 243_500,    "GRU+GNN": 185_000,
    "LSTM+GCN": 243_539,    "GRU+GCN": 185_039,
    "LSTM+GAT": 243_617,    "GRU+GAT": 185_117,
    "Bi-LSTM": 643_879,     "Bi-GRU": 486_879,
    "Bi-LSTM+GNN": 645_400, "Bi-GRU+GNN": 488_400,
    "Bi-LSTM+GCN": 645_439, "Bi-GRU+GCN": 488_439,
    "Bi-LSTM+GAT": 645_517, "Bi-GRU+GAT": 488_517,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# forward pass input: (B, T-1, F)
dummy_input = torch.randn(BATCH_SIZE, WINDOW_SIZE - 1, N_TAGS).to(device)
expected_shape = (BATCH_SIZE, N_TAGS)

# ─── Test ─────────────────────────────────────────────────────────────────────
passed = []
failed = []

print(f"Device: {device}")
print(f"Input shape: {tuple(dummy_input.shape)}")
print(f"Expected output shape: {expected_shape}")
print("=" * 60)

for model_name in MODEL_REGISTRY:
    try:
        # 1. Build the model
        model = build_model(model_name, n_tags=N_TAGS, n_hiddens=N_HIDDENS,
                            n_layers=N_LAYERS, window_size=WINDOW_SIZE)
        model.eval()

        # 2. Forward pass
        with torch.no_grad():
            output = model(dummy_input)

        # 3. Check output shape and parameter count
        assert output.shape == expected_shape, \
            f"shape mismatch: got {tuple(output.shape)}, expected {expected_shape}"
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == PAPER_PARAMS[model_name], \
            f"parameter count {n_params:,} differs from Table 6 ({PAPER_PARAMS[model_name]:,})"

        print(f"  [PASS] {model_name:12s}  output: {tuple(output.shape)}  params: {n_params:,}")
        passed.append(model_name)

    except Exception as e:
        print(f"  [FAIL] {model_name:12s}  ERROR: {e}")
        failed.append((model_name, str(e)))

# ─── Summary ──────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"Results: {len(passed)}/{len(MODEL_REGISTRY)} passed")

if failed:
    print("\nFailed models:")
    for name, err in failed:
        print(f"  {name}: {err}")
    sys.exit(1)
else:
    print("All models passed!")
