# VR Brain Lab - Parameter Tuning Guide

## Problem: Low-Activity, Over-Stable Dynamics

If your simulations show very low activity (< 0.05) with minimal variance, the model is in a **subcritical regime**. Real brain dynamics should be rich, fluctuating, and responsive to perturbations.

---

## Quick Fix: Increase Drive and Coupling

### Option 1: Boost External Input (Easiest)

Edit `simulator.py`, line ~37:

```python
# Before
I_ext: float = 0.5  # Too low

# After (try these progressively)
I_ext: float = 2.0   # Moderate increase
I_ext: float = 3.5   # Strong increase
I_ext: float = 5.0   # Very strong (may oscillate)
```

### Option 2: Increase Network Coupling

Edit `simulator.py`, line ~34:

```python
# Before
global_coupling: float = 0.5  # Too weak

# After
global_coupling: float = 1.5   # Moderate
global_coupling: float = 2.5   # Strong
```

### Option 3: Add More Noise (for Stochastic Fluctuations)

Edit `simulator.py`, line ~38:

```python
# Before
noise_strength: float = 0.01  # Too weak

# After
noise_strength: float = 0.05  # Moderate
noise_strength: float = 0.1   # Strong
```

### Option 4: Adjust Neural Excitability

Edit sigmoid thresholds in `simulator.py`, line ~42-45:

```python
# Lower thresholds = easier to activate

# Before
theta_e: float = 4.0   # Excitatory threshold
theta_i: float = 3.7   # Inhibitory threshold

# After (more excitable)
theta_e: float = 2.5
theta_i: float = 2.0
```

---

## Recommended Starting Point

Use this configuration for more realistic dynamics:

```python
# In SimulationConfig (simulator.py, ~line 30)
@dataclass
class SimulationConfig:
    dt: float = 0.1
    duration: float = 2000.0
    transient: float = 200.0

    global_coupling: float = 1.8      # ← INCREASED
    conduction_velocity: float = 3.0

    tau_e: float = 10.0
    tau_i: float = 20.0
    c_ee: float = 16.0
    c_ei: float = 12.0
    c_ie: float = 15.0
    c_ii: float = 3.0
    I_ext: float = 3.0                # ← INCREASED
    noise_strength: float = 0.05      # ← INCREASED

    a_e: float = 1.3
    theta_e: float = 3.0              # ← DECREASED
    a_i: float = 2.0
    theta_i: float = 2.5              # ← DECREASED
```

---

## Target Metrics for Healthy Dynamics

After tuning, you should see:

| Metric | Target Range | What It Means |
|--------|--------------|---------------|
| **Mean activity** | 0.3 - 0.7 | Moderate activity level |
| **Activity std** | 0.05 - 0.2 | Dynamic fluctuations |
| **Global synchrony** | 0.2 - 0.6 | Functional coupling |
| **Metastability** | 0.1 - 0.3 | Switching between states |
| **Dominant freq** | 8-40 Hz | Realistic oscillations |
| **Lesion response** | 15-40% change | Network disruption |
| **Stimulation response** | 20-60% change | Responsiveness |

---

## Warning Signs

**Too Low Activity (current problem):**
- Mean < 0.1, std < 0.01
- No response to interventions
- → Increase I_ext, coupling, noise

**Too High Activity (runaway):**
- Mean > 0.9, exploding values
- NaN or inf errors
- → Decrease I_ext, coupling, or increase inhibition (c_ie, c_ii)

**Unresponsive to Lesions:**
- < 5% change after major lesion
- → Increase coupling (network too independent)

**Synchronized "Seizure-like":**
- Synchrony > 0.9, all regions identical
- → Decrease coupling or increase noise

---

## Test Your Tuning

Run this quick test:

```python
from data_loader import create_default_brain
from simulator import BrainNetworkSimulator, SimulationConfig

brain = create_default_brain(68)

# Test configuration
config = SimulationConfig(
    duration=2000.0,
    global_coupling=1.8,  # New value
    I_ext=3.0,            # New value
    noise_strength=0.05   # New value
)

sim = BrainNetworkSimulator(brain, config)
results = sim.run_simulation()

print(f"Mean activity: {results['E'].mean():.3f}")
print(f"Activity std: {results['E'].std():.3f}")
print(f"Activity range: [{results['E'].min():.3f}, {results['E'].max():.3f}]")

# Target: mean ~0.4-0.6, std ~0.1-0.2
```

---

## Advanced Tuning: Bifurcation Analysis

For systematic exploration, vary one parameter and track activity:

```python
import numpy as np
import matplotlib.pyplot as plt

I_ext_values = np.linspace(0.5, 6.0, 20)
mean_activities = []

for I_ext in I_ext_values:
    config = SimulationConfig(I_ext=I_ext, duration=1000.0)
    sim = BrainNetworkSimulator(brain, config)
    results = sim.run_simulation()
    mean_activities.append(results['E'].mean())

plt.plot(I_ext_values, mean_activities, 'o-')
plt.xlabel('External Input (I_ext)')
plt.ylabel('Mean Activity')
plt.title('Bifurcation Diagram')
plt.axhline(0.5, color='r', linestyle='--', label='Target')
plt.legend()
plt.show()
```

---

## Region-Specific Tuning (Advanced)

For heterogeneous brains (some regions more excitable):

Currently, all regions use the same parameters. To make specific regions (e.g., sensory areas) more responsive:

1. Modify `simulator.py` to accept per-region parameter arrays
2. Set higher `I_ext` for sensory regions, lower for prefrontal
3. This requires extending the `NeuralMassModel` class

Example:
```python
# Instead of scalar I_ext, use array:
I_ext: np.ndarray = np.ones(num_regions) * base_I_ext
I_ext[sensory_regions] *= 1.5  # Boost sensory areas
```

---

## Summary Checklist

- [ ] Increase `I_ext` from 0.5 to 2.5-4.0
- [ ] Increase `global_coupling` from 0.5 to 1.5-2.5
- [ ] Increase `noise_strength` from 0.01 to 0.05-0.1
- [ ] Lower activation thresholds (`theta_e`, `theta_i`) by ~30%
- [ ] Re-run `demo_brain_lab.py`
- [ ] Check mean activity ~0.4-0.6, std ~0.1-0.2
- [ ] Verify lesions cause 15-30% activity changes
- [ ] Verify stimulation increases activity 20-50%

---

**After tuning, your brain will be dynamic, responsive, and biologically realistic!**
