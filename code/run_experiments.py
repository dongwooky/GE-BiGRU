"""
run_experiments.py - train and evaluate all (or selected) model variants.

Per-model logs are written automatically by train.py / predict.py to
../result/{MODEL_NAME}/.

This script additionally writes an overall summary (results.csv, summary.md)
to ../logs/{run_id}/.

  GE-BiGRU/
  ├── result/
  │   ├── GRU/
  │   │   ├── valid_result.png
  │   │   ├── test_result.png
  │   │   ├── ...
  │   │   ├── train_YYYYMMDD_HHMM.log    <- written by train.py
  │   │   └── predict_YYYYMMDD_HHMM.log  <- written by predict.py
  │   └── Bi-GRU+GAT/
  │       └── ...
  └── logs/
      └── YYYYMMDD_HHMMSS/
          ├── results.csv    <- metrics for all models
          └── summary.md     <- tables in the format of the paper

Usage:
  # all 16 variants
  python run_experiments.py

  # selected models only
  python run_experiments.py --models GRU Bi-GRU+GAT

  # training only (skip evaluation)
  python run_experiments.py --skip-predict
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Always run relative to the code/ directory, regardless of where the script is launched
os.chdir(Path(__file__).resolve().parent)

from modules import MODEL_REGISTRY

PYTHON = sys.executable
ALL_MODELS = list(MODEL_REGISTRY.keys())

UNIDIRECTIONAL = ["LSTM", "GRU", "LSTM+GNN", "GRU+GNN", "LSTM+GCN", "GRU+GCN", "LSTM+GAT", "GRU+GAT"]
BIDIRECTIONAL  = ["Bi-LSTM", "Bi-GRU", "Bi-LSTM+GNN", "Bi-GRU+GNN", "Bi-LSTM+GCN", "Bi-GRU+GCN", "Bi-LSTM+GAT", "Bi-GRU+GAT"]


# ─── Run ──────────────────────────────────────────────────────────────────────

def run(script, model_name):
    """Run a script as a subprocess. Logs are written by train.py / predict.py themselves."""
    cmd = [PYTHON, script, "--model", model_name]
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[WARNING] {script} --model {model_name} exited with code {result.returncode}")
    return result.returncode


# ─── Metric parsing ───────────────────────────────────────────────────────────

def parse_metrics(output):
    """Parse performance metrics from a predict.py log."""
    m = {}

    # F1: 0.924 (TaP: 0.940, TaR: 0.909)
    hit = re.search(r"F1:\s*([\d.]+)\s*\(TaP:\s*([\d.]+),\s*TaR:\s*([\d.]+)\)", output)
    if hit:
        m["F1_TaPR"] = float(hit.group(1))
        m["TaP"]     = float(hit.group(2))
        m["TaR"]     = float(hit.group(3))

    # # of detected anomalies: 44
    hit = re.search(r"# of detected anomalies:\s*(\d+)", output)
    if hit:
        m["detected"] = int(hit.group(1))

    # Sensitivity: 0.752  Specificity: 0.994  Point-wise F1: ...
    hit = re.search(
        r"Sensitivity:\s*([\d.]+)\s*Specificity:\s*([\d.]+)\s*Point-wise F1:\s*([\d.]+)", output
    )
    if hit:
        m["sensitivity"] = float(hit.group(1))
        m["specificity"] = float(hit.group(2))
        m["F1_pointwise"] = float(hit.group(3))

    return m


def read_latest_predict_log(model_name):
    """Return the most recent predict log in ../result/{model_name}/."""
    pattern = f"../result/{model_name}/predict_*.log"
    files = sorted(glob.glob(pattern))
    if not files:
        return ""
    with open(files[-1]) as f:
        return f.read()


def read_efficiency_json(model_name):
    """Return ../result/{model_name}/efficiency.json, or an empty dict if missing."""
    path = f"../result/{model_name}/efficiency.json"
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# ─── Save results ─────────────────────────────────────────────────────────────

def save_csv(metrics_map, path, eff_map=None):
    eff_map = eff_map or {}
    cols = ["model", "sensitivity", "specificity", "TaP", "TaR", "F1_TaPR", "F1_pointwise", "detected",
            "n_params", "train_time_min", "avg_epoch_s", "inference_ms_per_sample"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for model_name, m in metrics_map.items():
            e = eff_map.get(model_name, {})
            row = [
                model_name,
                m.get("sensitivity", ""),
                m.get("specificity", ""),
                m.get("TaP", ""),
                m.get("TaR", ""),
                m.get("F1_TaPR", ""),
                m.get("F1_pointwise", ""),
                m.get("detected", ""),
                e.get("n_params", ""),
                e.get("total_training_time_min", ""),
                e.get("avg_time_per_epoch_s", ""),
                e.get("inference_ms_per_sample", ""),
            ]
            f.write(",".join(str(v) for v in row) + "\n")


def _md_table(models, metrics_map):
    header = "| Model | Sensitivity | Specificity | TaP | TaR | F1 (TaPR) | Detected |\n"
    sep    = "|-------|-------------|-------------|-----|-----|-----------|----------|\n"
    rows = []
    for name in models:
        if name not in metrics_map:
            continue
        m = metrics_map[name]
        rows.append(
            f"| {name} "
            f"| {m.get('sensitivity', '-'):.3f} "
            f"| {m.get('specificity', '-'):.3f} "
            f"| {m.get('TaP', '-'):.3f} "
            f"| {m.get('TaR', '-'):.3f} "
            f"| {m.get('F1_TaPR', '-'):.3f} "
            f"| {m.get('detected', '-')} |"
        )
    return header + sep + "\n".join(rows)


def _eff_table(models, eff_map):
    header = "| Model | Params | Train (min) | Avg/epoch (s) | Infer (ms/sample) |\n"
    sep    = "|-------|--------|-------------|---------------|-------------------|\n"
    rows = []
    for name in models:
        e = eff_map.get(name)
        if not e:
            continue
        rows.append(
            f"| {name} "
            f"| {e.get('n_params', '-'):,} "
            f"| {e.get('total_training_time_min', '-'):.2f} "
            f"| {e.get('avg_time_per_epoch_s', '-'):.2f} "
            f"| {e.get('inference_ms_per_sample', '-'):.4f} |"
        )
    return header + sep + "\n".join(rows)


def save_summary_md(metrics_map, rc_map, path, run_id, eff_map=None):
    eff_map = eff_map or {}
    uni_metrics = {k: v for k, v in metrics_map.items() if k in UNIDIRECTIONAL}
    bi_metrics  = {k: v for k, v in metrics_map.items() if k in BIDIRECTIONAL}

    best_name, best_f1 = None, -1.0
    for name, m in metrics_map.items():
        if m.get("F1_TaPR", -1) > best_f1:
            best_f1   = m["F1_TaPR"]
            best_name = name

    lines = [
        "# Experiment Results",
        "",
        f"**Run ID**: `{run_id}`  ",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    if uni_metrics:
        lines += ["## Unidirectional Models", "", _md_table(UNIDIRECTIONAL, uni_metrics), ""]

    if bi_metrics:
        lines += ["## Bidirectional Models", "", _md_table(BIDIRECTIONAL, bi_metrics), ""]

    all_models_ordered = UNIDIRECTIONAL + BIDIRECTIONAL
    eff_present = {k: v for k, v in eff_map.items() if k in all_models_ordered}
    if eff_present:
        lines += ["## Computational Efficiency", "", _eff_table(all_models_ordered, eff_present), ""]

    if best_name:
        lines += ["## Best Model", "", f"**{best_name}** - F1 (TaPR) = {best_f1:.3f}", ""]

    lines += ["## Run Status", ""]
    lines.append("| Model | Train | Predict |")
    lines.append("|-------|-------|---------|")
    for name, (t_rc, p_rc) in rc_map.items():
        t_s = "OK" if t_rc == 0 else f"FAILED({t_rc})"
        p_s = "OK" if p_rc == 0 else ("SKIP" if p_rc is None else f"FAILED({p_rc})")
        lines.append(f"| {name} | {t_s} | {p_s} |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training and evaluation for multiple models.")
    parser.add_argument("--models", nargs="*", default=ALL_MODELS,
                        help=f"Subset of models to run. Default: all {len(ALL_MODELS)}.")
    parser.add_argument("--skip-predict", action="store_true",
                        help="Only run training, skip prediction/evaluation.")
    args = parser.parse_args()

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"../logs/{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Summary will be saved to: {log_dir}/")

    rc_map      = {}   # model_name -> (train_rc, predict_rc | None)
    metrics_map = {}   # model_name -> parsed metrics dict
    eff_map     = {}   # model_name -> efficiency dict

    for model_name in args.models:
        if model_name not in MODEL_REGISTRY:
            print(f"[ERROR] Unknown model '{model_name}'. Skipping.")
            rc_map[model_name] = (-1, None)
            continue

        train_rc = run("train.py", model_name)

        predict_rc = None
        if not args.skip_predict:
            predict_rc = run("predict.py", model_name)
            if predict_rc == 0:
                output = read_latest_predict_log(model_name)
                metrics_map[model_name] = parse_metrics(output)

        eff = read_efficiency_json(model_name)
        if eff:
            eff_map[model_name] = eff

        rc_map[model_name] = (train_rc, predict_rc)

    # Console summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_name, (t_rc, p_rc) in rc_map.items():
        t_s = "OK" if t_rc == 0 else f"FAILED({t_rc})"
        p_s = "OK" if p_rc == 0 else ("SKIP" if p_rc is None else f"FAILED({p_rc})")
        status = "OK" if t_rc == 0 and (p_rc is None or p_rc == 0) else "FAILED"
        print(f"  {model_name:20s}  train={t_s:12s}  predict={p_s:12s}  [{status}]")

    # Save results.csv + summary.md
    if metrics_map:
        csv_path = f"{log_dir}/results.csv"
        md_path  = f"{log_dir}/summary.md"
        save_csv(metrics_map, csv_path, eff_map)
        save_summary_md(metrics_map, rc_map, md_path, run_id, eff_map)
        print(f"\nResults saved:")
        print(f"  {csv_path}")
        print(f"  {md_path}")
