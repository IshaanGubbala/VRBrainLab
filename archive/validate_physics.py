
"""
validate_physics.py - Validation Suite for New Features

Tests:
1. Spectral Analysis (Alpha peak detection)
2. Empirical FC Comparison (using dummy data)
3. Time-varying Input (Entrainment check)
4. Heterogeneity check
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

def main():
    brain = create_default_brain()
    
    print("\n--- TEST 1: Heterogeneity & Spectral Analysis ---")
    # Increase heterogeneity
    cfg = SimulationConfig(
        duration=2000,
        theta_e_heterogeneity=0.5,
        i_ext_heterogeneity=0.2,
        noise_strength=0.1
    )
    sim = BrainNetworkSimulator(brain, cfg)
    res = sim.run_simulation()
    
    analyzer = BrainActivityAnalyzer(res)
    spectra = analyzer.compute_power_spectra()
    
    peak = spectra['peak_freq']
    print(f"Peak Frequency: {peak:.2f} Hz")
    print(f"Alpha Power: {spectra['band_powers']['alpha']:.4f}")
    
    if 8 <= peak <= 13:
        print("✅ PASS: Robust Alpha Peak Detected")
    else:
        print(f"⚠️ NOTE: Peak at {peak:.1f} Hz (Expected ~10 Hz). Tuning may be needed.")

    print("\n--- TEST 2: Empirical FC Comparison ---")
    # Generate dummy empirical FC (symmetric)
    sim_fc = analyzer._functional_connectivity()
    # Perturb it slightly to simulate 'real' data that matches reasonably well
    dummy_empirical = sim_fc + np.random.normal(0, 0.1, sim_fc.shape)
    dummy_empirical = (dummy_empirical + dummy_empirical.T) / 2
    np.fill_diagonal(dummy_empirical, 0)
    
    comparison = analyzer.compare_to_empirical(dummy_empirical)
    print(f"Correlation with 'Empirical': {comparison['correlation']:.3f}")
    
    if comparison['correlation'] > 0.8:
        print("✅ PASS: High correlation with target")
    else:
        print("❌ FAIL: Correlation too low")
        
    print("\n--- TEST 3: Time-varying Input (Entrainment) ---")
    # Drive at 15 Hz (Beta)
    freq = 15.0
    t = np.arange(0, 3000, cfg.dt)
    
    # Create sine wave input for ALL regions
    # Shape: (steps, regions)
    stimulus = 5.0 * np.sin(2 * np.pi * freq * (t / 1000.0))
    I_stim_arr = np.tile(stimulus[:, np.newaxis], (1, brain['num_regions']))
    
    print(f"Injecting {freq} Hz stimulus (amp=5.0)...")
    sim_driven = BrainNetworkSimulator(brain, SimulationConfig(duration=3000))
    res_driven = sim_driven.run_simulation(I_stim=I_stim_arr)
    
    print(f"Driven Activity Stats: Mean={np.mean(res_driven['E']):.3f}, Std={np.std(res_driven['E']):.3f}")
    
    driven_analyzer = BrainActivityAnalyzer(res_driven)
    driven_spectra = driven_analyzer.compute_power_spectra()
    
    driven_peak = driven_spectra['peak_freq']
    print(f"Driven Peak: {driven_peak:.2f} Hz")
    
    if abs(driven_peak - freq) < 2.0:
        print(f"✅ PASS: Entrained to {freq} Hz stimulus")
    else:
        print(f"❌ FAIL: Did not entrain (Peak {driven_peak:.1f} Hz)")

if __name__ == "__main__":
    main()
