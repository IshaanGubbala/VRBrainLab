#!/usr/bin/env python3
"""
Test script comparing default (low-activity) vs tuned (realistic) brain dynamics
"""

from data_loader import create_default_brain
from simulator import BrainNetworkSimulator, SimulationConfig
from intervention import BrainIntervention
import numpy as np


def print_comparison(label, results):
    """Print summary statistics."""
    E = results['E']
    print(f"{label}:")
    print(f"  Mean activity: {E.mean():.3f}")
    print(f"  Activity std:  {E.std():.3f}")
    print(f"  Activity range: [{E.min():.3f}, {E.max():.3f}]")


print("=" * 70)
print("Testing: Default vs Tuned Brain Parameters")
print("=" * 70)

# Create brain
brain = create_default_brain(68)

# ===== TEST 1: Default (current) parameters =====
print("\n1. DEFAULT PARAMETERS (current - too stable)")
print("-" * 70)

config_default = SimulationConfig(
    duration=2000.0,
    global_coupling=0.5,   # Default
    I_ext=0.5,             # Default
    noise_strength=0.01    # Default
)

sim_default = BrainNetworkSimulator(brain, config_default)
results_default = sim_default.run_simulation()
print_comparison("Default", results_default)


# ===== TEST 2: Tuned parameters =====
print("\n2. TUNED PARAMETERS (realistic dynamics)")
print("-" * 70)

config_tuned = SimulationConfig(
    duration=2000.0,
    global_coupling=1.8,   # Increased!
    I_ext=3.0,             # Increased!
    noise_strength=0.05,   # Increased!
    theta_e=3.0,           # Decreased (more excitable)
    theta_i=2.5            # Decreased
)

sim_tuned = BrainNetworkSimulator(brain, config_tuned)
results_tuned = sim_tuned.run_simulation()
print_comparison("Tuned", results_tuned)


# ===== TEST 3: Lesion Response Comparison =====
print("\n3. LESION RESPONSE TEST")
print("-" * 70)

# Default brain lesion
print("\nDefault brain + lesion:")
intervention_default = BrainIntervention(brain, config_default)
intervention_default.apply_region_lesion(10, severity=0.9)
sim_lesion_default = BrainNetworkSimulator(intervention_default.current_data, config_default)
results_lesion_default = sim_lesion_default.run_simulation()

change_default = ((results_lesion_default['E'].mean() - results_default['E'].mean()) /
                  results_default['E'].mean() * 100)
print(f"  Activity change: {change_default:+.1f}%")

# Tuned brain lesion
print("\nTuned brain + lesion:")
intervention_tuned = BrainIntervention(brain, config_tuned)
intervention_tuned.apply_region_lesion(10, severity=0.9)
sim_lesion_tuned = BrainNetworkSimulator(intervention_tuned.current_data, config_tuned)
results_lesion_tuned = sim_lesion_tuned.run_simulation()

change_tuned = ((results_lesion_tuned['E'].mean() - results_tuned['E'].mean()) /
                results_tuned['E'].mean() * 100)
print(f"  Activity change: {change_tuned:+.1f}%")


# ===== SUMMARY =====
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nActivity Levels:")
print(f"  Default: {results_default['E'].mean():.3f} (std: {results_default['E'].std():.3f})")
print(f"  Tuned:   {results_tuned['E'].mean():.3f} (std: {results_tuned['E'].std():.3f})")

print("\nDynamics:")
if results_default['E'].std() < 0.01:
    print("  Default: ❌ TOO STABLE (nearly frozen)")
else:
    print("  Default: ✓ Good variance")

if 0.05 < results_tuned['E'].std() < 0.3:
    print("  Tuned:   ✓ HEALTHY FLUCTUATIONS")
else:
    print("  Tuned:   ⚠️  Check parameters")

print("\nLesion Responsiveness:")
if abs(change_default) < 5:
    print(f"  Default: ❌ UNRESPONSIVE ({change_default:+.1f}%)")
else:
    print(f"  Default: ✓ Responsive ({change_default:+.1f}%)")

if 10 < abs(change_tuned) < 50:
    print(f"  Tuned:   ✓ REALISTIC RESPONSE ({change_tuned:+.1f}%)")
else:
    print(f"  Tuned:   ⚠️  Response: {change_tuned:+.1f}%")

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)

if results_default['E'].std() < 0.01 and abs(change_default) < 5:
    print("❌ Default parameters produce unrealistic, frozen dynamics.")
    print("✓  Use TUNED parameters for realistic brain simulation.")
    print("\nTo apply permanently:")
    print("   Edit simulator.py, SimulationConfig (line ~30)")
    print("   Change: I_ext=3.0, global_coupling=1.8, noise_strength=0.05")
else:
    print("✓ Default parameters are working well!")

print("=" * 70)
