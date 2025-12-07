# VRBrainLab — AGENTS CONTEXT FILE

This document describes the architecture, purpose, and internal agents involved in the VRBrainLab project. Codex and other LLM-based assistants should rely on this file to understand the context of the system when generating or modifying code.

---

## 📘 Project Overview

VRBrainLab is a modular computational-neuroscience platform that builds, simulates, analyzes, and visualizes whole-brain network models. It includes:

- A *brain model generator* (structural connectivity, delays, weights)
- A *neural mass simulator* (Wilson–Cowan–style excitatory–inhibitory dynamics per region)
- An *intervention system* (lesions, stroke, stimulation, virtual drugs, plasticity, rewiring)
- A *neuroanalysis engine* (network metrics, temporal dynamics, simulated EEG/BOLD)
- A *VR-ready API layer* that streams simulation results to a Unity/VR frontend

The system is designed as a “digital-twin brain sandbox” supporting both research-style experiments and interactive VR exploration.

---

## 🧠 Core Concepts the Agents Must Understand

### **Brain Model**
- 68-region connectome (generic or personalized)
- Weighted, directed connections with biologically inspired delays
- Structural metrics: degree, strength, clustering, path length

### **Neural Dynamics**
- Each region = E/I populations governed by differential equations
- Simulation timestep ~0.1 ms
- Noise + coupling drive dynamic behavior (oscillations, metastability, synchrony)
- Outputs: region-by-region activity time series

### **Interventions**
Agents must understand these correctly:
- **Lesion:** Remove connections/regions partially or fully
- **Stroke:** Multi-region lesion propagated by hop radius
- **Stimulation:** Increase external drive to 1–N regions
- **Virtual drug:** Adjust system parameters globally (e.g., sedative lowers excitability)
- **Plasticity:** Modify surviving connections (learning rules)
- **Rewiring:** Add new connections after damage

These interventions are run through a baseline vs intervention pipeline.

### **Analysis**
Agents must know how to compute or request:
- Network metrics (clustering, path length, hubness)
- Temporal metrics (global synchrony, metastability)
- Spectral metrics (dominant frequency via FFT)
- Vulnerability scoring
- Simulated EEG (64 channels) and fMRI/BOLD

### **VR Interface**
- Python backend exposes REST/WS endpoints
- Unity frontend requests:
  - connectivity
  - time-series activity
  - analysis metadata
  - intervention-triggered re-simulations

This file ensures agents understand the data structures and interfaces.

---

## 🗂️ Codebase Structure (for reference)

```
data_loader.py     → builds and loads connectomes
simulator.py       → runs fast neural simulations
intervention.py    → defines lesions, drugs, stimulation, strokes, plasticity, rewiring
analysis.py        → computes metrics, EEG/BOLD, vulnerability
vr_interface.py    → REST/WebSocket API for Unity VR frontend
demo_brain_lab.py  → main demonstration runner
```

---

## 🎯 Goals for Agent Behavior

When generating or modifying code, agents should:

1. Maintain consistency with the architecture outlined here.
2. Use correct terminology (regions, coupling, delays, E/I populations).
3. Ensure simulation, intervention, and analysis modules communicate cleanly.
4. Keep VR API responses in expected formats (JSON-serializable).
5. Preserve scientific meaning: dynamics must remain explainable and biologically plausible.
6. Avoid inventing nonexistent files or functions.

---

## 🤖 Future Extensions Agents Should Support

- Support for real patient data (DTI, MRI)
- Improved neural models (e.g., Jansen–Rit, Montbrió mean-field model)
- GPU acceleration
- Real-time VR playback and interactive interventions
- Parameter sweeps & automated regime classification

---

This AGENTS.md file should guide all code completions, refactors, and module interactions in VRBrainLab.
