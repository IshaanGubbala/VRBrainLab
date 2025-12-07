#!/usr/bin/env python3
"""
diagnose_dynamics.py - Advanced Brain Dynamics Diagnostics

Addresses scientific concerns about model realism:
- Spatial heterogeneity of activity
- Temporal structure and oscillations
- Synchrony patterns and metastability
- Spectral content across regions
- Functional connectivity dynamics

Usage:
    python diagnose_dynamics.py
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from data_loader import create_default_brain
from simulator_fast import BrainNetworkSimulator, SimulationConfig
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def compute_power_spectrum(timeseries, dt):
    """Compute power spectrum of a time series."""
    n = len(timeseries)
    freqs = fftfreq(n, dt)[:n//2]
    fft_vals = fft(timeseries - timeseries.mean())
    power = np.abs(fft_vals[:n//2])**2
    return freqs, power


def compute_functional_connectivity(activity, method='correlation'):
    """Compute functional connectivity matrix."""
    if method == 'correlation':
        # Pearson correlation between all region pairs
        return np.corrcoef(activity.T)
    elif method == 'covariance':
        return np.cov(activity.T)


def compute_metastability(activity):
    """
    Compute metastability (Cabral et al. 2017).
    Variance of synchrony over time.
    """
    # Synchrony at each timepoint = std of Kuramoto order parameter
    sync_timeseries = np.std(activity, axis=1)
    metastability = np.std(sync_timeseries)
    return metastability, sync_timeseries


def analyze_spatial_heterogeneity(activity):
    """Analyze how activity varies across regions."""
    region_means = np.mean(activity, axis=0)
    region_stds = np.std(activity, axis=0)

    return {
        'mean_across_regions': region_means,
        'std_across_regions': region_stds,
        'heterogeneity_index': np.std(region_means) / np.mean(region_means),
        'min_region_mean': np.min(region_means),
        'max_region_mean': np.max(region_means),
        'region_activity_range': np.max(region_means) - np.min(region_means)
    }


def analyze_temporal_structure(activity, dt):
    """Analyze temporal dynamics and oscillatory content."""
    # Global average timeseries
    global_signal = np.mean(activity, axis=1)

    # Compute autocorrelation
    autocorr = np.correlate(global_signal - global_signal.mean(),
                           global_signal - global_signal.mean(),
                           mode='same')
    autocorr = autocorr / autocorr.max()

    # Power spectrum
    freqs, power = compute_power_spectrum(global_signal, dt)

    # Find dominant frequency
    peak_idx = np.argmax(power[freqs > 1])  # Ignore DC
    dominant_freq = freqs[freqs > 1][peak_idx]

    # Spectral bands (Hz)
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 80)
    }

    band_power = {}
    for band_name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        band_power[band_name] = np.sum(power[mask])

    return {
        'autocorrelation': autocorr,
        'power_spectrum': (freqs, power),
        'dominant_frequency': dominant_freq,
        'band_power': band_power,
        'total_power': np.sum(power)
    }


def run_comprehensive_diagnostics():
    """Run complete diagnostic analysis."""
    print("=" * 70)
    print("BRAIN DYNAMICS COMPREHENSIVE DIAGNOSTICS")
    print("=" * 70)

    # Create brain and run simulation
    print("\n1. Generating brain network...")
    brain = create_default_brain(68)

    print("2. Running 3-second simulation...")
    config = SimulationConfig(duration=3000.0, transient=200.0)
    sim = BrainNetworkSimulator(brain, config)
    results = sim.run_simulation()

    activity = results['E']  # Shape: (timepoints, regions)
    dt = config.dt  # ms

    print(f"   Activity shape: {activity.shape}")
    print(f"   Timestep: {dt} ms")

    # ==================================================================
    # GLOBAL STATISTICS
    # ==================================================================
    print("\n" + "=" * 70)
    print("GLOBAL STATISTICS")
    print("=" * 70)

    global_mean = np.mean(activity)
    global_std = np.std(activity)
    global_min = np.min(activity)
    global_max = np.max(activity)

    print(f"\nGlobal Activity:")
    print(f"  Mean: {global_mean:.3f}")
    print(f"  Std:  {global_std:.3f}")
    print(f"  Range: [{global_min:.3f}, {global_max:.3f}]")
    print(f"  Coefficient of Variation: {global_std / global_mean:.3f}")

    # ==================================================================
    # SPATIAL HETEROGENEITY
    # ==================================================================
    print("\n" + "=" * 70)
    print("SPATIAL HETEROGENEITY")
    print("=" * 70)

    spatial = analyze_spatial_heterogeneity(activity)

    print(f"\nRegion-Specific Activity:")
    print(f"  Most active region: {spatial['max_region_mean']:.3f}")
    print(f"  Least active region: {spatial['min_region_mean']:.3f}")
    print(f"  Activity range across regions: {spatial['region_activity_range']:.3f}")
    print(f"  Heterogeneity index: {spatial['heterogeneity_index']:.3f}")

    # Identify most/least active regions
    region_means = spatial['mean_across_regions']
    most_active_idx = np.argmax(region_means)
    least_active_idx = np.argmin(region_means)

    print(f"\n  Most active: Region {most_active_idx} ({results['region_labels'][most_active_idx]})")
    print(f"  Least active: Region {least_active_idx} ({results['region_labels'][least_active_idx]})")

    # ==================================================================
    # TEMPORAL STRUCTURE & OSCILLATIONS
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEMPORAL STRUCTURE & OSCILLATIONS")
    print("=" * 70)

    temporal = analyze_temporal_structure(activity, dt)

    print(f"\nOscillatory Content:")
    print(f"  Dominant frequency: {temporal['dominant_frequency']:.1f} Hz")

    print(f"\n  Band Power (relative):")
    total_power = temporal['total_power']
    for band, power in temporal['band_power'].items():
        rel_power = power / total_power * 100
        print(f"    {band:8s}: {rel_power:5.1f}%")

    # ==================================================================
    # SYNCHRONY & METASTABILITY
    # ==================================================================
    print("\n" + "=" * 70)
    print("SYNCHRONY & METASTABILITY")
    print("=" * 70)

    # Compute pairwise synchrony (correlation coefficient)
    fc_matrix = compute_functional_connectivity(activity, method='correlation')

    # Get upper triangle (exclude diagonal)
    upper_tri = fc_matrix[np.triu_indices(fc_matrix.shape[0], k=1)]
    mean_fc = np.mean(upper_tri)
    std_fc = np.std(upper_tri)

    print(f"\nFunctional Connectivity:")
    print(f"  Mean correlation: {mean_fc:.3f}")
    print(f"  Std correlation: {std_fc:.3f}")
    print(f"  Range: [{np.min(upper_tri):.3f}, {np.max(upper_tri):.3f}]")

    # Metastability
    metastability, sync_timeseries = compute_metastability(activity)
    mean_sync = np.mean(sync_timeseries)

    print(f"\nMetastability Analysis:")
    print(f"  Mean synchrony (temporal std): {mean_sync:.3f}")
    print(f"  Metastability (std of sync): {metastability:.3f}")

    if metastability < 0.01:
        print("  ⚠️  Low metastability - network may be over-synchronized")
    elif metastability > 0.10:
        print("  ✓ Good metastability - dynamic state switching")

    # ==================================================================
    # DIAGNOSTIC SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    issues = []
    good = []

    # Check global activity
    if global_mean < 0.3:
        issues.append("Mean activity too low (subcritical regime)")
    elif global_mean > 0.7:
        issues.append("Mean activity too high (supercritical regime)")
    else:
        good.append(f"Mean activity healthy ({global_mean:.2f})")

    # Check variance
    if global_std < 0.05:
        issues.append("Low variance - network may be in fixed point")
    else:
        good.append(f"Good variance ({global_std:.2f})")

    # Check spatial heterogeneity
    if spatial['heterogeneity_index'] < 0.1:
        issues.append("Low spatial heterogeneity - all regions similar")
    else:
        good.append(f"Spatial heterogeneity present ({spatial['heterogeneity_index']:.2f})")

    # Check functional connectivity
    if mean_fc > 0.9:
        issues.append("Very high FC - possible pathological synchrony")
    elif mean_fc < 0.1:
        issues.append("Very low FC - regions may be decoupled")
    else:
        good.append(f"Functional connectivity reasonable ({mean_fc:.2f})")

    # Check spectral diversity
    dominant_band = max(temporal['band_power'], key=temporal['band_power'].get)
    dominant_power_pct = temporal['band_power'][dominant_band] / total_power * 100

    if dominant_power_pct > 80:
        issues.append(f"Spectral power concentrated in {dominant_band} ({dominant_power_pct:.0f}%)")
    else:
        good.append("Spectral diversity present across bands")

    print("\n✓ Strengths:")
    for item in good:
        print(f"  - {item}")

    if issues:
        print("\n⚠️  Concerns:")
        for item in issues:
            print(f"  - {item}")
    else:
        print("\n✅ No major concerns detected!")

    # ==================================================================
    # RECOMMENDATIONS
    # ==================================================================
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    print("\nNext steps to improve model realism:")
    print("  1. Add regional heterogeneity (vary I_ext, thresholds per region)")
    print("  2. Test different coupling strengths and delay distributions")
    print("  3. Introduce time-varying inputs or noise patterns")
    print("  4. Compare FC matrix to empirical resting-state data")
    print("  5. Run parameter sweeps to map dynamical regimes")
    print("  6. Validate spectral content against real EEG/MEG")

    print("\n" + "=" * 70)

    return {
        'activity': activity,
        'spatial': spatial,
        'temporal': temporal,
        'fc_matrix': fc_matrix,
        'metastability': metastability
    }


if __name__ == "__main__":
    diagnostics = run_comprehensive_diagnostics()
    print("\nDiagnostics complete!")
