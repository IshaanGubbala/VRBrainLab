
import json
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    results_path = Path("tuner_results.json")
    if not results_path.exists():
        print("tuner_results.json not found.")
        return

    with open(results_path) as f:
        data = json.load(f)

    if "history" not in data:
        print("No history found in tuner_results.json")
        return

    records = data["history"]
    
    # Map params in order
    param_names = [
        "global_coupling", "I_ext", "c_ee", "c_ie", 
        "noise_strength", "theta_e", "slow_drive_sigma", "delay_jitter_pct"
    ]
    
    rows = []
    for r in records:
        row = dict(zip(param_names, r["params"]))
        row["loss"] = r["loss"]
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    print(f"Total trials: {len(df)}")
    
    # 1. Correlation Analysis
    print("\n--- Correlation with Loss (Negative = Higher param is better) ---")
    correlations = df.corr()["loss"].sort_values()
    print(correlations.drop("loss"))
    
    # 2. Elite Analysis (Top 15 runs)
    elite_n = 15
    elite = df.nsmallest(elite_n, "loss")
    
    print(f"\n--- Analysis of Top {elite_n} Configurations (Mean Loss: {elite['loss'].mean():.3f}) ---")
    
    for param in param_names:
        full_mean = df[param].mean()
        full_std = df[param].std()
        
        elite_mean = elite[param].mean()
        elite_std = elite[param].std()
        
        # Calculate how "constricted" this param is in the elite set vs full set
        # A low ratio means this parameter is CRITICAL (must be in a specific range)
        ratio = elite_std / full_std if full_std > 0 else 1.0
        
        print(f"\n{param.upper()}:")
        print(f"  Range:  {elite[param].min():.3f} - {elite[param].max():.3f}")
        print(f"  Mean:   {elite_mean:.3f} (vs global {full_mean:.3f})")
        print(f"  Tightness: {ratio:.2f} (Lower = more sensitive)")
        
        if ratio < 0.6:
            print("  >>> CRITICAL PARAMETER (Tight range required)")
        elif abs(full_mean - elite_mean) > full_std * 0.5:
             print("  >>> IMPORTANT SHIFT (Optimal is far from average)")

if __name__ == "__main__":
    main()
