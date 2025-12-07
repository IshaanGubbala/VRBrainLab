#!/usr/bin/env python3
"""
Quick diagnostic test - checks if brain dynamics are in healthy range
Run this after changing parameters to verify you're in the Goldilocks zone
"""

from data_loader import create_default_brain
from simulator import BrainNetworkSimulator, SimulationConfig
from intervention import BrainIntervention
import numpy as np

print("=" * 70)
print("QUICK BRAIN DYNAMICS TEST")
print("=" * 70)

# Create small brain for fast testing
print("\n1. Creating brain model...")
brain = create_default_brain(68)

# Run short baseline simulation
print("\n2. Running baseline simulation (1 second)...")
config = SimulationConfig(duration=1000.0, transient=100.0)
sim = BrainNetworkSimulator(brain, config)
results = sim.run_simulation()

# Extract metrics
E = results['E']
mean_activity = E.mean()
std_activity = E.std()
min_activity = E.min()
max_activity = E.max()

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nActivity Metrics:")
print(f"  Mean:  {mean_activity:.3f}")
print(f"  Std:   {std_activity:.3f}")
print(f"  Range: [{min_activity:.3f}, {max_activity:.3f}]")

# Test lesion response
print("\n3. Testing lesion response...")
intervention = BrainIntervention(brain, config)
intervention.apply_region_lesion(10, severity=0.9)
sim_lesion = BrainNetworkSimulator(intervention.current_data, config)
results_lesion = sim_lesion.run_simulation()
lesion_change = ((results_lesion['E'].mean() - mean_activity) / mean_activity) * 100

print(f"  Lesion effect: {lesion_change:+.1f}%")

# Diagnosis
print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

healthy = True
warnings = []
errors = []

# Check activity level
if mean_activity < 0.1:
    errors.append("❌ Activity TOO LOW (subcritical regime)")
    errors.append("   → Increase I_ext or global_coupling")
    healthy = False
elif mean_activity > 0.9:
    errors.append("❌ Activity TOO HIGH (saturated regime)")
    errors.append("   → Decrease I_ext or increase theta values")
    healthy = False
elif 0.3 <= mean_activity <= 0.7:
    print("✓ Activity level: HEALTHY (mid-range)")
else:
    warnings.append(f"⚠️  Activity level: {mean_activity:.3f} (workable but not optimal)")
    warnings.append("   → Target: 0.3-0.7 for most realistic dynamics")

# Check variance
if std_activity < 0.01:
    errors.append("❌ Variance TOO LOW (frozen dynamics)")
    errors.append("   → Increase noise_strength or adjust coupling")
    healthy = False
elif std_activity > 0.3:
    warnings.append("⚠️  Variance high (may be too noisy/chaotic)")
    warnings.append(f"   → Current: {std_activity:.3f}, target: 0.05-0.2")
elif 0.05 <= std_activity <= 0.2:
    print("✓ Activity variance: HEALTHY (dynamic fluctuations)")
else:
    warnings.append(f"⚠️  Variance: {std_activity:.3f} (low but acceptable)")

# Check saturation
if max_activity > 0.95:
    errors.append("❌ SATURATION detected (hitting ceiling)")
    errors.append("   → Neurons stuck at maximum, decrease I_ext")
    healthy = False
elif max_activity > 0.85:
    warnings.append(f"⚠️  Close to saturation (max: {max_activity:.3f})")
elif min_activity < 0.05 and mean_activity > 0.3:
    print("✓ Dynamic range: HEALTHY (not saturated)")

# Check responsiveness
if abs(lesion_change) < 5:
    errors.append("❌ UNRESPONSIVE to lesions")
    errors.append("   → Increase global_coupling (network too independent)")
    healthy = False
elif 15 <= abs(lesion_change) <= 40:
    print("✓ Lesion response: HEALTHY (realistic disruption)")
else:
    warnings.append(f"⚠️  Lesion response: {lesion_change:+.1f}% (workable)")
    warnings.append("   → Target: 15-40% change for realistic network effects")

# Print diagnostics
if errors:
    print("\nERRORS (need fixing):")
    for err in errors:
        print(err)

if warnings:
    print("\nWARNINGS (could improve):")
    for warn in warnings:
        print(warn)

# Final verdict
print("\n" + "=" * 70)
if healthy:
    print("✅ VERDICT: Parameters are in HEALTHY range!")
    print("   → Your brain simulation is realistic and responsive")
    print("   → Ready for experiments and VR visualization")
elif len(errors) == 1:
    print("⚠️  VERDICT: Nearly there, one issue to fix")
else:
    print("❌ VERDICT: Parameters need adjustment")
    print("   → See errors above for specific fixes")

print("=" * 70)

# Parameter recommendations
print("\nCURRENT PARAMETER VALUES:")
print(f"  I_ext: {config.I_ext}")
print(f"  global_coupling: {config.global_coupling}")
print(f"  noise_strength: {config.noise_strength}")
print(f"  theta_e: {config.theta_e}")
print(f"  theta_i: {config.theta_i}")

if not healthy:
    print("\nRECOMMENDED ADJUSTMENTS:")
    if mean_activity < 0.1:
        print("  I_ext: increase to 2.0-3.0")
        print("  global_coupling: increase to 1.2-1.8")
    elif mean_activity > 0.9:
        print("  I_ext: decrease to 1.5-2.5")
        print("  theta_e: increase to 3.5-4.0")

    if std_activity < 0.01:
        print("  noise_strength: increase to 0.03-0.08")

    if abs(lesion_change) < 5:
        print("  global_coupling: increase to 1.5-2.5")

print("\n" + "=" * 70)
