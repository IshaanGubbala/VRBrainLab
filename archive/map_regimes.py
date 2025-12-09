
"""
map_regimes.py - Dynamical Regime Mapping Tool

Performs a 2D parameter sweep to classify brain dynamics into regimes:
- Quiet (Low activity)
- Stable (High activity, no oscillation)
- Oscillatory (Periodic, synchronous)
- Saturated (Extreme activity)
- Metastable (Fluctuating synchrony)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.data_loader import create_default_brain
from core.simulator_fast import BrainNetworkSimulator, SimulationConfig
from core.analysis import BrainActivityAnalyzer

def classify_regime(metrics):
    mean = metrics['mean_activity']
    std = metrics['std_activity']
    meta = metrics['metastability']
    
    if mean < 0.1:
        return 0, "Quiet"  # Blue
    elif mean > 0.95:
        return 4, "Saturated"  # Black
    
    if meta > 0.05:
        return 3, "Metastable"  # Red
    
    # Check for oscillations vs stable fixed point
    # We use std of activity as a proxy for oscillation amplitude
    if std > 0.05:
        return 2, "Oscillatory"  # Yellow/Orange
    else:
        return 1, "Stable"  # Green

def main():
    print("="*60)
    print("DYNAMICAL REGIME MAPPER")
    print("="*60)
    
    brain = create_default_brain()
    
    # Define sweep grid
    coupling_vals = np.linspace(0.0, 1.5, 10)
    input_vals = np.linspace(0.0, 1.5, 10)
    
    results_grid = np.zeros((len(input_vals), len(coupling_vals)))
    
    print(f"Sweeping {len(coupling_vals)*len(input_vals)} points...")
    
    for i, I_ext in enumerate(input_vals):
        for j, G in enumerate(coupling_vals):
            print(f"  Simulating G={G:.2f}, I={I_ext:.2f}...", end="", flush=True)
            
            # Create config
            cfg = SimulationConfig(
                duration=1000, 
                dt=0.5,  # Faster dt for sweep
                global_coupling=G, 
                I_ext=I_ext,
                noise_strength=0.05 # Lower noise to see intrinsic regimes
            )
            
            sim = BrainNetworkSimulator(brain, cfg, verbose=False)
            res = sim.run_simulation(suppress_output=True)
            
            # Analyze
            # Quick custom metrics to avoid full analysis overhead
            activity = res['E'][200:] # discard transient (raw indices)
            metrics = {
                'mean_activity': float(np.mean(activity)),
                'std_activity': float(np.std(activity)),
                'metastability': float(np.std(np.std(activity, axis=1))) # quick proxy
            }
            
            regime_code, label = classify_regime(metrics)
            results_grid[i, j] = regime_code
            print(f" -> {label}")

    # Plot
    plt.figure(figsize=(10, 8))
    
    # Custom discrete colormap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#000000'])
    classes = ['Quiet', 'Stable', 'Oscillatory', 'Metastable', 'Saturated']
    
    im = plt.imshow(results_grid, origin='lower', 
                    extent=[coupling_vals.min(), coupling_vals.max(), input_vals.min(), input_vals.max()],
                    aspect='auto', cmap=cmap, vmin=0, vmax=4)
    
    cbar = plt.colorbar(im, ticks=[0,1,2,3,4])
    cbar.ax.set_yticklabels(classes)
    
    plt.xlabel('Global Coupling (G)')
    plt.ylabel('External Input (I_ext)')
    plt.title('Dynamical Regime Map')
    
    output_path = ROOT / "regime_map.png"
    plt.savefig(output_path)
    print(f"\n✓ Classification complete. Map saved to {output_path}")

if __name__ == "__main__":
    main()
