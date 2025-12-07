# VR Brain Lab 🧠

**A personalized digital-twin brain simulation platform with VR visualization and interactive intervention capabilities.**

Combining computational neuroscience, translational medicine, and immersive visualization for both research and clinical applications.

---

## 🎯 Project Goals

Build a platform that:
- **Simulates** whole-brain dynamics using realistic network models
- **Visualizes** brain activity in immersive VR
- **Predicts** outcomes of interventions (lesions, stimulation, therapy)
- **Supports** both long-term research and clinical decision support

---

## 🏗️ Architecture

```
VRBrainLab/
├── data_loader.py       # Brain connectivity data loading & model setup
├── simulator.py         # Core brain network simulation engine
├── intervention.py      # Lesions, stimulation, perturbations
├── analysis.py          # Metrics extraction & biomarker analysis
├── vr_interface.py      # REST API server for VR communication
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| **data_loader.py** | Load/create brain connectivity (connectome), region definitions, tract lengths |
| **simulator.py** | Neural mass model simulation, network coupling, time-stepping |
| **intervention.py** | Apply lesions, stimulation, parameter changes, plasticity |
| **analysis.py** | Extract network metrics, temporal dynamics, vulnerability maps |
| **vr_interface.py** | Flask API for VR frontend, data streaming, intervention control |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run a Basic Simulation

```python
from data_loader import create_default_brain
from simulator import BrainNetworkSimulator
from analysis import BrainActivityAnalyzer

# Create generic brain model (68 regions)
brain_data = create_default_brain(num_regions=68)

# Run simulation
simulator = BrainNetworkSimulator(brain_data)
results = simulator.run_simulation()

# Analyze results
analyzer = BrainActivityAnalyzer(results)
print(analyzer.generate_report())
```

### 3. Apply an Intervention

```python
from intervention import BrainIntervention

# Create intervention manager
intervention = BrainIntervention(brain_data)

# Apply a lesion (simulate stroke)
intervention.apply_stroke_lesion(
    center_idx=10,  # Central region
    radius=2,       # Affect neighbors within 2 hops
    severity=0.8    # 80% damage
)

# Compare baseline vs intervention
comparison = intervention.run_comparison(duration=2000.0)
```

### 4. Start VR API Server

```bash
python vr_interface.py
```

Server will start at `http://localhost:5000`

API documentation available at: `http://localhost:5000/`

---

## 🧪 Example Use Cases

### Virtual Lesion Simulation
Simulate stroke or injury, analyze network fallout, predict functional loss.

```python
from intervention import quick_lesion_simulation

results = quick_lesion_simulation(
    brain_data,
    lesion_region=15,
    severity=1.0
)
```

### Therapy Optimization
Test which stimulation protocol best restores network activity.

```python
intervention.apply_stimulation(
    region_indices=[10, 11],
    amplitude=1.5,
    frequency=10.0  # Hz
)
```

### Vulnerability Mapping
Identify which brain regions are most at risk.

```python
vulnerability = analyzer.compute_vulnerability_map()
print("Most vulnerable regions:")
for region in vulnerability['top_vulnerable'][:5]:
    print(f"  - {region['region']}: {region['score']:.3f}")
```

---

## 🌐 VR Interface API

### Endpoints

#### Brain Model
- `GET /api/brain/info` - Get brain model information
- `POST /api/brain/load` - Load/create brain model
- `GET /api/brain/connectivity` - Get connectivity matrix

#### Simulation
- `POST /api/simulation/run` - Start simulation
- `GET /api/simulation/status` - Check simulation progress
- `GET /api/simulation/data` - Get activity time series
- `GET /api/simulation/snapshot` - Get activity at specific time

#### Intervention
- `POST /api/intervention/lesion` - Apply lesion
- `POST /api/intervention/stimulate` - Apply stimulation
- `POST /api/intervention/reset` - Reset interventions
- `GET /api/intervention/history` - Get intervention history

#### Analysis
- `GET /api/analysis/metrics` - Get temporal & network metrics
- `GET /api/analysis/vulnerability` - Get vulnerability map
- `GET /api/analysis/report` - Get text report

### Example API Usage (from Unity/VR)

```csharp
// C# example for Unity
IEnumerator RunSimulation() {
    string url = "http://localhost:5000/api/simulation/run";
    string json = "{\"duration\": 3000, \"global_coupling\": 0.6}";

    using (UnityWebRequest request = UnityWebRequest.Post(url, json)) {
        yield return request.SendWebRequest();
        // Handle response...
    }
}
```

---

## 📊 Simulation Parameters

### Neural Mass Model (Wilson-Cowan)
- `tau_e`: Excitatory time constant (default: 10ms)
- `tau_i`: Inhibitory time constant (default: 20ms)
- `I_ext`: External input current (default: 0.5)
- `noise_strength`: Neural noise amplitude (default: 0.01)

### Network Coupling
- `global_coupling`: Overall coupling strength (default: 0.5)
- `conduction_velocity`: Axonal signal speed (default: 3.0 mm/ms)

### Time Parameters
- `dt`: Integration timestep (default: 0.1 ms)
- `duration`: Simulation duration (default: 2000 ms)
- `transient`: Transient period to discard (default: 200 ms)

---

## 🔬 Analysis Metrics

### Network Metrics
- **Density**: Connection density
- **Clustering**: Local clustering coefficient
- **Path Length**: Characteristic path length
- **Hubs**: Highly connected regions

### Temporal Metrics
- **Synchrony**: Global phase synchronization
- **Metastability**: Variability of synchrony over time
- **Dominant Frequency**: Peak oscillation frequency
- **Functional Connectivity**: Time-series correlations

### Vulnerability Metrics
- **Centrality**: Network importance
- **Activity Level**: Mean activity
- **Variability**: Activity fluctuations
- **Combined Score**: Weighted vulnerability index

---

## 🛠️ Advanced Features

### Multi-Scale Simulation (Future)
Integrate spiking neuron models for selected regions (using Arbor-TVB co-simulation framework).

### Personalized Brain Models (Future)
Load real patient data:
- Structural MRI → brain parcellation
- DTI → connectivity matrix & tract lengths
- Functional MRI/EEG → parameter fitting

### Plasticity & Recovery
Simulate long-term adaptation:
```python
intervention.simulate_plasticity(learning_rate=0.1)
intervention.simulate_rewiring(num_new_connections=20)
```

---

## 📖 Scientific Background

This project builds on:
- **Virtual Brain Twins (VBT)**: Personalized brain models for simulation and prediction
- **The Virtual Brain (TVB)**: Open-source whole-brain simulation platform
- **Multi-scale modeling**: Region-level + neuron-level co-simulation
- **Digital twin paradigm**: In-silico testing for translational medicine

### References
- [Virtual brain twins: from basic neuroscience to clinical use](https://academic.oup.com/nsr/article/11/5/nwae079/7616087)
- [The Virtual Brain](https://www.thevirtualbrain.org/)
- [Arbor-TVB co-simulation framework](https://arxiv.org/abs/2505.16861)

---

## 🧩 Extending the Project

### Add New Neural Models
Extend `simulator.py` to support additional neural mass models (e.g., Jansen-Rit, reduced Wong-Wang).

### Add New Interventions
Extend `intervention.py` with:
- Virtual drug effects (parameter perturbations)
- Deep brain stimulation patterns
- Optogenetic-like selective activation

### Add New Analysis
Extend `analysis.py` with:
- Machine learning classifiers
- Seizure detection algorithms
- Outcome prediction models

### VR Visualization
Build Unity/Unreal frontend that:
- Displays 3D brain with connectivity
- Animates activity over time
- Provides interactive intervention controls
- Shows real-time metrics

---

## 🎓 Educational Use

Perfect for:
- Science fair projects
- Computational neuroscience coursework
- Undergraduate/graduate research
- Grant proposals (digital twin medicine)
- Demos of translational neuroscience

---

## 📝 License

This is an educational/research project. Feel free to extend, modify, and build upon it.

---

## 🤝 Contributing

To extend this project:
1. Add new modules (e.g., `plasticity.py`, `ml_models.py`)
2. Improve neural models (more biophysical detail)
3. Add visualization tools (matplotlib dashboards)
4. Build VR frontend (Unity integration)

---

## 📧 Contact & Support

For questions, issues, or collaboration:
- Check the documentation in each module
- Review the example code in `__main__` blocks
- Consult The Virtual Brain documentation for advanced features

---

**Status**: Core simulation engine ✅ | VR API ✅ | Unity/VR Frontend 🚧 | Advanced models 🚧

Built with ❤️ for computational neuroscience and translational medicine.
