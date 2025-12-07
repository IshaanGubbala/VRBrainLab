# Brain Simulation Parameter Guide

## 🎯 The Three Regimes

Your brain simulation can be in one of three states:

```
SUBCRITICAL          CRITICAL            SUPERCRITICAL
(Too quiet)       (Just right!)        (Too excited)

  ●━━━━━━━━━━━━━━━━━━━━●━━━━━━━━━━━━━━━━━━━━●
  0.0              0.4-0.6              1.0

  ❌ Coma           ✅ Healthy          ❌ Seizure
  No response       Dynamic             Saturated
  Frozen            Responsive          Maxed out
```

---

## 📊 How to Identify Each Regime

### ❌ SUBCRITICAL (Your First Results)
```
Mean activity: 0.01-0.15
Activity std:  < 0.01
Lesion effect: < 5%
```
**Diagnosis:** Brain too quiet, no network effects
**Symptoms:** Frozen dynamics, unresponsive to interventions

### ❌ SUPERCRITICAL / SATURATED (Your Second Results)
```
Mean activity: 0.90-1.00
Activity std:  < 0.01
Lesion effect: < 5%
Max activity:  > 0.95
```
**Diagnosis:** Brain stuck at ceiling, saturated
**Symptoms:** All neurons firing max, no room to respond

### ✅ CRITICAL / HEALTHY (Target)
```
Mean activity: 0.35-0.65
Activity std:  0.05-0.20
Lesion effect: 15-40%
Range:         0.20-0.85
```
**Diagnosis:** Balanced, dynamic, responsive
**Symptoms:** Fluctuations, functional connectivity, realistic

---

## 🔧 Parameter Effects Guide

### **I_ext** (External Drive)
Controls baseline excitation level - like "how awake" the brain is.

```
I_ext = 0.5   → Subcritical (too low)
I_ext = 1.0   → Low activity
I_ext = 2.0   → Healthy mid-range ✓
I_ext = 3.0   → High activity (risk of saturation)
I_ext = 5.0   → Supercritical (saturated)
```

**Rule of thumb:** Start at 2.0, adjust by ±0.5 increments

---

### **global_coupling** (Network Strength)
Controls how much regions influence each other.

```
coupling = 0.3   → Independent (no network effects)
coupling = 0.8   → Weak coupling
coupling = 1.2   → Moderate coupling ✓
coupling = 1.8   → Strong coupling
coupling = 3.0   → Over-coupled (synchronized)
```

**Rule of thumb:** Start at 1.2, increase if lesions have no effect

---

### **noise_strength** (Stochasticity)
Controls random fluctuations - gives "life" to dynamics.

```
noise = 0.01    → Too deterministic (frozen)
noise = 0.03    → Moderate fluctuations ✓
noise = 0.08    → Strong fluctuations
noise = 0.15    → Too chaotic
```

**Rule of thumb:** Start at 0.03, increase if std too low

---

### **theta_e, theta_i** (Activation Thresholds)
Controls how easily neurons activate. Lower = more excitable.

```
theta_e = 5.0   → Hard to activate (subcritical)
theta_e = 3.5   → Moderate ✓
theta_e = 2.5   → Easy to activate
theta_e = 1.0   → Too easy (supercritical)
```

**Rule of thumb:** Keep theta_e around 3.5, theta_i around 3.0

---

## 🎓 Parameter Tuning Workflow

### Step 1: Quick Test
```bash
python quick_test.py
```

### Step 2: Interpret Results

**If mean activity < 0.2 (subcritical):**
1. Increase I_ext by +0.5
2. Or increase global_coupling by +0.3
3. Re-test

**If mean activity > 0.8 (supercritical):**
1. Decrease I_ext by -0.5
2. Or increase theta_e by +0.5
3. Re-test

**If std < 0.01 (frozen):**
1. Increase noise_strength by +0.02
2. Re-test

**If lesion effect < 5% (unresponsive):**
1. Increase global_coupling by +0.5
2. Re-test

### Step 3: Fine-Tune

Once you're in the healthy range (0.3-0.6 activity), tweak for specific goals:

**For more oscillatory dynamics:**
- Increase tau_i (inhibitory time constant)
- Creates rhythm-generating feedback loops

**For more synchrony:**
- Increase global_coupling
- Decrease conduction_velocity (faster communication)

**For more metastability (switching states):**
- Moderate noise (0.04-0.06)
- Balanced E-I coupling

---

## 📋 Recommended Starting Points

### **Default Balanced (Good for most uses)**
```python
SimulationConfig(
    I_ext=2.0,
    global_coupling=1.2,
    noise_strength=0.03,
    theta_e=3.5,
    theta_i=3.0
)
```

### **High Activity (Alert brain)**
```python
SimulationConfig(
    I_ext=2.5,
    global_coupling=1.4,
    noise_strength=0.04,
    theta_e=3.0,
    theta_i=2.8
)
```

### **Oscillatory (Rhythmic dynamics)**
```python
SimulationConfig(
    I_ext=2.0,
    global_coupling=1.5,
    noise_strength=0.02,
    tau_e=8.0,
    tau_i=25.0,  # Longer inhibition
    theta_e=3.5,
    theta_i=3.0
)
```

### **Low Activity (Resting state)**
```python
SimulationConfig(
    I_ext=1.5,
    global_coupling=1.0,
    noise_strength=0.03,
    theta_e=3.8,
    theta_i=3.2
)
```

---

## 🚨 Common Problems & Solutions

### Problem: "Activity stuck at 1.0"
**Cause:** Sigmoid saturation
**Solution:**
- Decrease I_ext to 1.5-2.0
- Increase theta_e to 3.5-4.0

### Problem: "Activity stuck at 0.01"
**Cause:** Subthreshold regime
**Solution:**
- Increase I_ext to 2.0-2.5
- Decrease theta_e to 3.0-3.5

### Problem: "No variance (std < 0.01)"
**Cause:** Stuck in fixed point
**Solution:**
- Increase noise to 0.03-0.06
- Adjust I_ext to mid-range (2.0)

### Problem: "Lesions have no effect"
**Cause:** Regions too independent
**Solution:**
- Increase global_coupling to 1.5-2.5

### Problem: "Numerical instability / NaN"
**Cause:** Runaway excitation
**Solution:**
- Decrease I_ext
- Decrease global_coupling
- Increase inhibition (c_ie, c_ii)

### Problem: "All regions identical (over-synchronized)"
**Cause:** Too much coupling, not enough noise
**Solution:**
- Decrease global_coupling
- Increase noise_strength
- Increase tract_length heterogeneity

---

## 🧪 Advanced: Parameter Search

For systematic exploration, use this snippet:

```python
import numpy as np
from data_loader import create_default_brain
from simulator import BrainNetworkSimulator, SimulationConfig

brain = create_default_brain(68)

# Grid search
I_ext_values = np.arange(1.0, 4.0, 0.5)
coupling_values = np.arange(0.5, 2.5, 0.5)

results = []

for I_ext in I_ext_values:
    for coupling in coupling_values:
        config = SimulationConfig(
            I_ext=I_ext,
            global_coupling=coupling,
            duration=1000.0
        )

        sim = BrainNetworkSimulator(brain, config)
        res = sim.run_simulation()

        mean_act = res['E'].mean()
        std_act = res['E'].std()

        results.append({
            'I_ext': I_ext,
            'coupling': coupling,
            'mean': mean_act,
            'std': std_act,
            'healthy': 0.3 < mean_act < 0.7 and std_act > 0.05
        })

        if results[-1]['healthy']:
            print(f"✓ Found healthy: I_ext={I_ext}, coupling={coupling}")
            print(f"  mean={mean_act:.3f}, std={std_act:.3f}")

# Show all healthy combinations
healthy_params = [r for r in results if r['healthy']]
print(f"\nFound {len(healthy_params)} healthy parameter sets")
```

---

## 🎯 Target Metrics Summary

| Metric | Healthy Range | Indicates |
|--------|---------------|-----------|
| Mean activity | 0.35-0.65 | Mid-range dynamics |
| Activity std | 0.05-0.20 | Temporal fluctuations |
| Activity range | 0.15-0.85 | Not saturated |
| Global synchrony | 0.2-0.6 | Functional connectivity |
| Metastability | 0.1-0.3 | State switching |
| Lesion response | 15-40% | Network dependence |
| Stim response | 20-60% | Responsiveness |

---

## 📚 Understanding the Math

The Wilson-Cowan model has a **sigmoidal activation**:

```
σ(x) = 1 / (1 + exp(-a * (x - theta)))
```

When **input >> theta**: σ → 1.0 (saturated)
When **input << theta**: σ → 0.0 (silent)
When **input ≈ theta**: σ ≈ 0.5 (sensitive, dynamic)

**For healthy dynamics, you want neurons operating near the steep part of the sigmoid** (around theta), where they're most sensitive to inputs.

If I_ext is too high relative to theta:
→ neurons always above threshold → saturation

If I_ext is too low:
→ neurons always below threshold → silence

**The sweet spot: I_ext ≈ theta ± 1**

---

## ✅ Quick Reference Card

Copy this for quick parameter adjustments:

```
TOO LOW ACTIVITY?        → Increase I_ext by +0.5
TOO HIGH ACTIVITY?       → Decrease I_ext by -0.5
TOO STABLE (std low)?    → Increase noise by +0.02
NOT RESPONSIVE?          → Increase coupling by +0.5
SATURATED (max > 0.95)?  → Decrease I_ext OR increase theta
```

---

**Remember**: There's no single "correct" parameter set. Different sets produce different dynamics. Choose based on what you want to model (alert vs resting, healthy vs pathological, etc.).
