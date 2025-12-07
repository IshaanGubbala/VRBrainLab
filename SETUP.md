# VR Brain Lab - Setup & Configuration

## 🚀 Installation

### Requirements
- Python 3.8+
- NumPy, SciPy, Flask, Flask-CORS

### Install

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy scipy flask flask-cors matplotlib pandas
```

---

## ✅ Verify Installation

```bash
python test.py
```

**Expected output:**
```
TEST 1: Simulator Functionality
  ✓ Simulation completed

TEST 2: Brain Dynamics Health Check
  ✓ Mean activity: HEALTHY
  ✓ Activity variance: HEALTHY
  ✓ No saturation: HEALTHY

TEST 3: Intervention Functionality
  ✓ Lesion response: HEALTHY

✅ ALL TESTS PASSED - System ready!
```

---

## 🔧 Configuration

### Main Parameters

Edit `simulator_fast.py` (line ~30) to tune parameters:

```python
@dataclass
class SimulationConfig:
    # Time
    dt: float = 0.2              # Timestep (ms) - larger = faster
    duration: float = 2000.0     # Simulation length (ms)
    transient: float = 100.0     # Warmup period (ms)

    # Network
    global_coupling: float = 1.0  # How strongly regions interact
    conduction_velocity: float = 3.0  # Signal speed (mm/ms)

    # Neural dynamics
    tau_e: float = 10.0  # Excitatory time constant
    tau_i: float = 20.0  # Inhibitory time constant
    I_ext: float = 1.5   # External drive (KEY PARAMETER)
    noise_strength: float = 0.04  # Random fluctuations

    # Activation thresholds
    theta_e: float = 3.5  # Excitatory threshold
    theta_i: float = 3.0  # Inhibitory threshold
```

---

## 🎯 Parameter Tuning

### Target Metrics

Healthy brain dynamics:
- **Mean activity**: 0.35-0.60
- **Activity std**: 0.05-0.20
- **Max activity**: < 0.85 (no saturation)
- **Lesion response**: 15-40% change

### Common Adjustments

**Activity too low (< 0.3)?**
```python
I_ext: float = 1.8  # Increase from 1.5
```

**Activity too high (> 0.7)?**
```python
I_ext: float = 1.2  # Decrease from 1.5
```

**No fluctuations (std < 0.05)?**
```python
noise_strength: float = 0.06  # Increase from 0.04
```

**Lesions have minimal effect (< 10% change)?**
```python
global_coupling: float = 1.4  # Increase from 1.0
```

**Simulation too slow?**
```python
dt: float = 0.3  # Increase from 0.2 (trade-off: less temporal precision)
```

### Quick Test After Changes

```bash
python test.py --quick
```

---

## 🌐 VR Server Setup

### Start Server

```bash
python vr_interface.py
```

Server runs at: `http://localhost:5000`

### Test API

```bash
# Health check
curl http://localhost:5000/api/health

# Load brain
curl -X POST http://localhost:5000/api/brain/load \
     -H "Content-Type: application/json" \
     -d '{"num_regions": 68}'

# Run simulation
curl -X POST http://localhost:5000/api/simulation/run \
     -H "Content-Type: application/json" \
     -d '{"duration": 2000, "global_coupling": 1.2}'
```

### Unity Integration

**1. Add package:**
```csharp
using UnityEngine.Networking;
```

**2. Example request:**
```csharp
IEnumerator GetBrainData() {
    string url = "http://localhost:5000/api/brain/info";

    using (UnityWebRequest request = UnityWebRequest.Get(url)) {
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success) {
            string json = request.downloadHandler.text;
            // Parse JSON and use data
            Debug.Log("Brain data: " + json);
        }
    }
}
```

**3. Main endpoints:**
- `/api/brain/load` - Load brain (POST)
- `/api/simulation/run` - Start simulation (POST)
- `/api/simulation/data?region=10` - Get activity (GET)
- `/api/intervention/lesion` - Apply lesion (POST)
- `/api/analysis/metrics` - Get metrics (GET)

---

## 📊 Performance Optimization

### Current Speed

**With defaults:**
- 68 regions
- 2-second simulation
- ~40-60 seconds total

### Speed Up

**1. Increase timestep:**
```python
dt: float = 0.3  # From 0.2 → ~30% faster
```

**2. Reduce regions:**
```python
brain = create_default_brain(num_regions=34)  # Half the regions
```

**3. Shorter simulations:**
```python
duration: float = 1000.0  # From 2000 → 2x faster
```

**4. Reduce save frequency:**
```python
results = sim.run_simulation(save_interval=2)  # Save every 2 steps
```

---

## 🐛 Troubleshooting

### Import Errors

**Problem:**
```
ModuleNotFoundError: No module named 'simulator_fast'
```

**Solution:**
```bash
# Make sure you're in the right directory
cd /path/to/VRBrainLab

# Check file exists
ls simulator_fast.py

# Run from same directory
python test.py
```

### Wrong Results

**Problem:** Activity saturated or too low

**Solution:**
```bash
# Run diagnostic
python test.py --quick

# Follow recommendations in output
# Edit simulator_fast.py parameters
```

### Slow Performance

**Problem:** Simulation takes > 2 minutes

**Solution:**
1. Verify using `simulator_fast.py` (not `simulator.py`)
2. Check imports in demo/test scripts
3. Increase `dt` to 0.3 or 0.4

---

## 📁 File Reference

**Essential files:**
- `data_loader.py` - Brain connectivity
- `simulator_fast.py` - Brain simulation (optimized)
- `intervention.py` - Lesions, stimulation
- `analysis.py` - Metrics extraction
- `vr_interface.py` - REST API server

**Usage files:**
- `demo_brain_lab.py` - Full demonstration
- `test.py` - Test suite

**Config:**
- `requirements.txt` - Dependencies
- `README.md` - Documentation
- `SETUP.md` - This file

---

## ✅ Health Checklist

After setup, verify:

- [ ] `python test.py` passes all tests
- [ ] Mean activity: 0.35-0.60
- [ ] Lesions cause 15-35% change
- [ ] Simulation completes in < 90 seconds
- [ ] VR server starts without errors

**All checked?** → ✅ Ready to use!

---

## 🎯 Next Steps

1. **Run full demo:**
   ```bash
   python demo_brain_lab.py
   ```

2. **Start VR server:**
   ```bash
   python vr_interface.py
   ```

3. **Build VR visualization** (Unity/Unreal)

4. **Load real data** (MRI/DTI connectomes)

5. **Customize** (new interventions, analyses)

---

**For questions, check README.md or run `python test.py` for diagnostics.**
