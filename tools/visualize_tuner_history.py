#!/usr/bin/env python3
"""
visualize_tuner_history.py

Reads 'tuner_results.json' (which must contain the "history" field) 
and generates visualizations of the parameter space and loss landscape.

Outputs:
    tuner_logs/scatter_matrix.png
    tuner_logs/parallel_coordinates.png
    tuner_logs/correlation_matrix.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates, scatter_matrix

# Determine project root
ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "tuner_results.json"


def main():
    if not RESULTS_FILE.exists():
        print(f"❌ Error: {RESULTS_FILE} not found.")
        print("   Run 'python tools/auto_tuner.py --quick' first.")
        sys.exit(1)

    print(f"📖 Reading {RESULTS_FILE}...")
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    if "history" not in data:
        print("❌ Error: 'history' not found in results.")
        print("   Your auto_tuner.py might be old. Re-run with the updated tuner.")
        sys.exit(1)

    # Convert history list to a DataFrame
    records = data["history"]
    if not records:
        print("❌ Error: History is empty.")
        sys.exit(1)

    # Structure: [{'params': [p1, p2...], 'loss': X, ...}, ...]
    # We need to map the list 'params' to named columns.
    # The order is defined in auto_tuner.py: params_to_vec
    param_names = [
        "global_coupling",
        "I_ext",
        "c_ee",
        "c_ie",
        "noise_strength",
        "theta_e",
        "slow_drive_sigma",
        "delay_jitter_pct",
    ]

    rows = []
    for r in records:
        row = {}
        # Unpack params
        p_vals = r["params"]
        for name, val in zip(param_names, p_vals):
            row[name] = val
        
        row["loss"] = r["loss"]
        # Include metrics too?
        for m_name, m_val in r["metrics"].items():
            row[f"metric_{m_name}"] = m_val
            
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"✅ Loaded {len(df)} trials.")
    
    # Create output directory
    log_dir = Path(data.get("log_dir", "tuner_logs"))
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Scatter Matrix (Pair Plot)
    # -------------------------------------------------------------------------
    print("📊 Generating Scatter Matrix...")
    # Select only params + loss
    plot_cols = param_names + ["loss"]
    
    # Subsample if too large (matplotlib is slow with 1000+ points in scatter matrix)
    df_plot = df[plot_cols]
    if len(df) > 500:
        df_plot = df_plot.sample(500, random_state=42)

    # Color mapping: we can't easily do continuous color in pandas scatter_matrix
    # But we can sort by loss so better points are on top? 
    # Actually, pandas scatter_matrix is basic. 
    # Let's do a trick: we define a 'Quality' category for coloring.
    loss_quantiles = df["loss"].quantile([0.33, 0.66])
    def categorize(l):
        if l <= loss_quantiles[0.33]: return "Best"
        elif l <= loss_quantiles[0.66]: return "Mid"
        else: return "Poor"
    
    df_plot = df_plot.copy()
    df_plot["Quality"] = df["loss"].apply(categorize)
    
    colors = {"Best": "green", "Mid": "blue", "Poor": "red"}
    
    plt.figure(figsize=(16, 16))
    scatter_matrix(
        df_plot[param_names], 
        figsize=(16, 16), 
        diagonal='kde', 
        alpha=0.6,
        c=df_plot["Quality"].map(colors)
    )
    plt.suptitle("Parameter Scatter Matrix (Green=Low Loss)", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(log_dir / "scatter_matrix.png", dpi=100)
    plt.close()
    
    # -------------------------------------------------------------------------
    # 2. Parallel Coordinates
    # -------------------------------------------------------------------------
    print("📈 Generating Parallel Coordinates...")
    plt.figure(figsize=(14, 6))
    
    # Normalize data for better parallel plot visualization
    df_norm = df[param_names].copy()
    for col in df_norm.columns:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    
    df_norm["Quality"] = df_plot["Quality"]
    
    # We want Best to be plotted LAST so it's on top
    df_norm["sort_key"] = df_norm["Quality"].map({"Poor":0, "Mid":1, "Best":2})
    df_norm = df_norm.sort_values("sort_key")
    
    parallel_coordinates(df_norm.drop(columns=["sort_key"]), "Quality", color=["red", "blue", "green"], alpha=0.4)
    plt.title("Parallel Coordinates (Normalized Parameters)")
    plt.ylabel("Normalized Value (0-1)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(log_dir / "parallel_coordinates.png", dpi=100)
    plt.close()

    # -------------------------------------------------------------------------
    # 3. Correlation Heatmap
    # -------------------------------------------------------------------------
    print("🔥 Generating Correlation Heatmap...")
    corr = df[param_names + ["loss"]].corr()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation Coefficient")
    plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr)), corr.columns)
    
    # Annotate
    for i in range(len(corr)):
        for j in range(len(corr)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
            
    plt.title("Parameter-Loss Correlation")
    plt.tight_layout()
    plt.savefig(log_dir / "correlation_matrix.png", dpi=100)
    plt.close()

    print("\n✅ Visualizations saved to:")
    print(f"   - {log_dir}/scatter_matrix.png")
    print(f"   - {log_dir}/parallel_coordinates.png")
    print(f"   - {log_dir}/correlation_matrix.png")


if __name__ == "__main__":
    main()
