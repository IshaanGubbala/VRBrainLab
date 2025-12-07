#!/usr/bin/env python3
"""
auto_tuner.py - Automatic Parameter Tuner for Brain Simulation

Searches parameter space to find optimal brain dynamics.

Usage:
    python auto_tuner.py              # Search only
    python auto_tuner.py --apply      # Search and auto-apply
    python auto_tuner.py --quick      # Quick search (fewer combinations)
"""

import sys
import time
import numpy as np
from data_loader import create_default_brain
from simulator_fast import BrainNetworkSimulator, SimulationConfig
from intervention import BrainIntervention


def score_dynamics(results, lesion_results=None):
    """
    Score brain dynamics health (0-100).

    Target:
    - Mean activity: 0.35-0.65 (30 points)
    - Variance: 0.05-0.20 (25 points)
    - No saturation: max < 0.85 (20 points)
    - Dynamic range: > 0.30 (15 points)
    - Lesion response: 15-40% (10 points)
    """
    E = results['E']
    mean_act = E.mean()
    std_act = E.std()
    max_act = E.max()
    min_act = E.min()

    score = 0

    # 1. Mean activity (30 points)
    if 0.35 <= mean_act <= 0.65:
        score += 30
    elif 0.25 <= mean_act <= 0.75:
        score += 20
    elif mean_act < 0.15 or mean_act > 0.85:
        score += 0
    else:
        score += 10

    # 2. Variance (25 points)
    if 0.05 <= std_act <= 0.20:
        score += 25
    elif 0.02 <= std_act <= 0.30:
        score += 15
    elif std_act < 0.01:
        score += 0
    else:
        score += 5

    # 3. No saturation (20 points)
    if max_act < 0.85:
        score += 20
    elif max_act < 0.95:
        score += 10
    else:
        score += 0

    # 4. Dynamic range (15 points)
    activity_range = max_act - min_act
    if activity_range > 0.30:
        score += 15
    elif activity_range > 0.15:
        score += 10
    else:
        score += 0

    # 5. Lesion response (10 points)
    if lesion_results:
        lesion_change = abs((lesion_results['E'].mean() - mean_act) / mean_act * 100)
        if 15 <= lesion_change <= 40:
            score += 10
        elif 5 <= lesion_change <= 50:
            score += 5

    return score, {
        'mean': mean_act,
        'std': std_act,
        'max': max_act,
        'range': activity_range
    }


def test_params(brain, I_ext, coupling, noise, theta_e, test_lesion=True):
    """Test a parameter combination."""
    try:
        config = SimulationConfig(
            duration=800.0,
            transient=100.0,
            I_ext=I_ext,
            global_coupling=coupling,
            noise_strength=noise,
            theta_e=theta_e,
            theta_i=theta_e - 0.5
        )

        # Baseline simulation
        sim = BrainNetworkSimulator(brain, config)
        results = sim.run_simulation()

        # Optional lesion test
        lesion_results = None
        if test_lesion:
            intervention = BrainIntervention(brain, config)
            intervention.apply_region_lesion(10, severity=0.9)
            sim_lesion = BrainNetworkSimulator(intervention.current_data, config)
            lesion_results = sim_lesion.run_simulation()

        score, metrics = score_dynamics(results, lesion_results)
        return score, metrics

    except Exception as e:
        return 0, {'error': str(e)}


def search_parameters(brain, quick=False):
    """Search parameter space."""
    print("\n" + "="*70)
    print("AUTO-TUNER: Searching for optimal parameters")
    print("="*70)

    # Define search space
    if quick:
        print("\nMode: QUICK (fewer combinations)")
        I_ext_values = [0.9, 1.1, 1.3, 1.5]
        coupling_values = [0.9, 1.1, 1.3]
        noise_values = [0.02, 0.03, 0.04]
        theta_values = [3.3, 3.5, 3.7]
    else:
        print("\nMode: THOROUGH (more combinations)")
        I_ext_values = np.arange(0.8, 1.8, 0.2)
        coupling_values = np.arange(0.8, 1.6, 0.2)
        noise_values = [0.02, 0.03, 0.04, 0.05]
        theta_values = [3.0, 3.3, 3.5, 3.7, 4.0]

    total = len(I_ext_values) * len(coupling_values) * len(noise_values) * len(theta_values)

    print(f"\nSearch space:")
    print(f"  I_ext: {len(I_ext_values)} values")
    print(f"  coupling: {len(coupling_values)} values")
    print(f"  noise: {len(noise_values)} values")
    print(f"  theta: {len(theta_values)} values")
    print(f"  Total: {total} combinations")

    print("\nSearching...\n")

    best_score = 0
    best_params = None
    best_metrics = None
    tested = 0
    start_time = time.time()

    for I_ext in I_ext_values:
        for coupling in coupling_values:
            for noise in noise_values:
                for theta in theta_values:
                    tested += 1

                    score, metrics = test_params(brain, I_ext, coupling, noise, theta)

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'I_ext': I_ext,
                            'global_coupling': coupling,
                            'noise_strength': noise,
                            'theta_e': theta,
                            'theta_i': theta - 0.5
                        }
                        best_metrics = metrics

                        print(f"✓ New best! Score: {score}/100")
                        print(f"  I_ext={I_ext:.2f}, coupling={coupling:.2f}, "
                              f"noise={noise:.2f}, theta={theta:.2f}")
                        print(f"  Mean: {metrics['mean']:.3f}, Std: {metrics['std']:.3f}\n")

                    if tested % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = tested / elapsed
                        remaining = (total - tested) / rate
                        print(f"  Progress: {tested}/{total} ({100*tested/total:.0f}%) "
                              f"ETA: {remaining:.0f}s")

    elapsed = time.time() - start_time

    print("\n" + "="*70)
    print("SEARCH COMPLETE")
    print("="*70)
    print(f"\nTime: {elapsed:.1f} seconds")
    print(f"Tested: {total} combinations")

    if best_params:
        print(f"\n✅ BEST PARAMETERS (Score: {best_score}/100)")
        print("-"*70)
        for key, val in best_params.items():
            print(f"  {key}: {val}")

        print(f"\nMetrics:")
        print(f"  Mean activity: {best_metrics['mean']:.3f}")
        print(f"  Activity std: {best_metrics['std']:.3f}")
        print(f"  Max activity: {best_metrics['max']:.3f}")
        print(f"  Range: {best_metrics['range']:.3f}")

        return best_params
    else:
        print("\n❌ No good parameters found!")
        return None


def apply_params(params, filepath='simulator_fast.py'):
    """Apply parameters to simulator file."""
    print(f"\n" + "="*70)
    print(f"Applying parameters to {filepath}...")
    print("="*70)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    updates = 0
    for i, line in enumerate(lines):
        if 'global_coupling: float =' in line and '#' in line:
            lines[i] = f"    global_coupling: float = {params['global_coupling']}  # Auto-tuned\n"
            print(f"✓ Updated global_coupling = {params['global_coupling']}")
            updates += 1
        elif 'I_ext: float =' in line and '#' in line:
            lines[i] = f"    I_ext: float = {params['I_ext']}  # Auto-tuned\n"
            print(f"✓ Updated I_ext = {params['I_ext']}")
            updates += 1
        elif 'noise_strength: float =' in line and '#' in line:
            lines[i] = f"    noise_strength: float = {params['noise_strength']}  # Auto-tuned\n"
            print(f"✓ Updated noise_strength = {params['noise_strength']}")
            updates += 1
        elif 'theta_e: float =' in line and '#' in line:
            lines[i] = f"    theta_e: float = {params['theta_e']}  # Auto-tuned\n"
            print(f"✓ Updated theta_e = {params['theta_e']}")
            updates += 1
        elif 'theta_i: float =' in line and '#' in line:
            lines[i] = f"    theta_i: float = {params['theta_i']}  # Auto-tuned\n"
            print(f"✓ Updated theta_i = {params['theta_i']}")
            updates += 1

    with open(filepath, 'w') as f:
        f.writelines(lines)

    print(f"\n✅ Applied {updates} parameters to {filepath}")
    print("="*70)


def main():
    print("\n╔" + "═"*68 + "╗")
    print("║" + " "*18 + "BRAIN PARAMETER AUTO-TUNER" + " "*24 + "║")
    print("╚" + "═"*68 + "╝")

    # Parse args
    auto_apply = '--apply' in sys.argv
    quick_mode = '--quick' in sys.argv

    # Create brain
    print("\nCreating brain model...")
    brain = create_default_brain(68)

    # Search
    best_params = search_parameters(brain, quick=quick_mode)

    if best_params:
        if auto_apply:
            apply_params(best_params)
            print("\n✅ Parameters applied! Run 'python test.py' to verify.")
        else:
            print("\n💡 To apply these parameters automatically:")
            print("   python auto_tuner.py --apply")
    else:
        print("\n❌ Could not find good parameters.")
        print("   Try expanding search space or manual tuning.")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
