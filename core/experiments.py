
"""
experiments.py - VR Brain Lab Experiment Controller

This module consolidates the logic for:
1. Physics Validation
2. Dynamical Regime Mapping
3. Parameter Tuning
4. General experimentation

It serves as the logic layer that the GUI (via vr_interface.py) interacts with.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
import threading
from dataclasses import asdict

from core.data_loader import create_default_brain
from core.simulator_fast import BrainNetworkSimulator, SimulationConfig
from core.analysis import BrainActivityAnalyzer

# Try importing optuna for tuner, handle if missing
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class ExperimentController:
    """
    Central controller for running brain experiments.
    """

    def __init__(self):
        self.brain = create_default_brain()
        self.current_simulation = None
        self.lock = threading.Lock()
        
    def run_physics_validation(self) -> Dict[str, Any]:
        """
        Run the standard physics validation suite.
        Returns a dictionary of test results.
        """
        results = {
            "tests": [],
            "overall_pass": True,
            "timestamp": time.time()
        }
        
        # Test 1: Alpha Peak
        try:
            cfg = SimulationConfig(duration=2000, dt=0.1)
            sim = BrainNetworkSimulator(self.brain, cfg, verbose=False)
            sim_res = sim.run_simulation()
            analyzer = BrainActivityAnalyzer(sim_res)
            
            # Use improved nperseg logic from analysis.py
            spectra = analyzer.compute_power_spectra()
            peak = spectra['peak_freq']
            alpha_power = spectra['band_powers']['alpha']
            
            pass_alpha = 8 <= peak <= 13
            results["tests"].append({
                "name": "Alpha Peak Detection",
                "status": "PASS" if pass_alpha else "WARN",
                "message": f"Peak at {peak:.2f} Hz, Alpha Power: {alpha_power:.3f}",
                "data": {"peak": peak, "power": alpha_power}
            })
        except Exception as e:
            results["tests"].append({
                "name": "Alpha Peak Detection",
                "status": "ERROR",
                "message": str(e)
            })
            results["overall_pass"] = False

        # Test 2: Entrainment
        try:
            target_freq = 15.0
            duration = 3000
            cfg = SimulationConfig(duration=duration, dt=0.1)
            t = np.arange(0, duration, cfg.dt)
            stim = 5.0 * np.sin(2 * np.pi * target_freq * (t / 1000.0))
            # Broadcast to all regions
            I_stim = np.tile(stim[:, np.newaxis], (1, self.brain['num_regions']))
            
            sim_driven = BrainNetworkSimulator(self.brain, cfg, verbose=False)
            res_driven = sim_driven.run_simulation(I_stim=I_stim)
            
            analyzer_driven = BrainActivityAnalyzer(res_driven)
            spectra_driven = analyzer_driven.compute_power_spectra()
            peak_driven = spectra_driven['peak_freq']
            
            pass_entrain = abs(peak_driven - target_freq) < 1.0
            if not pass_entrain:
                 results["overall_pass"] = False
                 
            results["tests"].append({
                "name": "15Hz Entrainment",
                "status": "PASS" if pass_entrain else "FAIL",
                "message": f"Driven peak at {peak_driven:.2f} Hz (Target 15.0)",
                "data": {"peak": peak_driven}
            })
        except Exception as e:
            results["tests"].append({
                "name": "15Hz Entrainment",
                "status": "ERROR", 
                "message": str(e)
            })
            results["overall_pass"] = False
            
        return results

    def run_regime_sweep(self, resolution: int = 8) -> Dict[str, Any]:
        """
        Perform a 2D parameter sweep (Coupling vs Input).
        Returns heatmap data.
        """
        coupling_vals = np.linspace(0.0, 1.5, resolution)
        input_vals = np.linspace(0.0, 1.5, resolution)
        
        heatmap = np.zeros((resolution, resolution))
        
        # Mapping: 0=Quiet, 1=Stable, 2=Oscillatory, 3=Metastable, 4=Saturated
        
        for i, I_ext in enumerate(input_vals):
            for j, G in enumerate(coupling_vals):
                cfg = SimulationConfig(
                    duration=1000, 
                    dt=0.5, # Fast
                    global_coupling=G,
                    I_ext=I_ext,
                    noise_strength=0.05
                )
                sim = BrainNetworkSimulator(self.brain, cfg, verbose=False)
                res = sim.run_simulation(suppress_output=True)
                
                # Analysis
                act = res['E'][200:] # remove transient
                mean = np.mean(act)
                std = np.std(act)
                meta = np.std(np.std(act, axis=1))
                
                # Classification logic
                code = 1 # Stable default
                if mean < 0.1: code = 0 # Quiet
                elif mean > 0.95: code = 4 # Saturated
                elif meta > 0.05: code = 3 # Metastable
                elif std > 0.05: code = 2 # Oscillatory
                
                heatmap[i, j] = code
                
        return {
            "heatmap": heatmap.tolist(),
            "x_axis": coupling_vals.tolist(),
            "y_axis": input_vals.tolist(),
            "labels": ["Quiet", "Stable", "Oscillatory", "Metastable", "Saturated"]
        }

    def run_custom_simulation(self, params: Dict) -> Dict:
        """
        Run a single custom simulation with provided parameters.
        """
        # Override default config with provided params
        # Use SimulationConfig default as base
        base_cfg = SimulationConfig()
        
        # Convert params dict to Config object
        # We filter keys to only those in SimulationConfig
        valid_keys = set(base_cfg.__dict__.keys())
        # Also check dataclass fields if __dict__ doesn't work (it should for initialized obj)
        # Actually better to iterate params and set if valid
        
        for k, v in params.items():
            if hasattr(base_cfg, k):
                setattr(base_cfg, k, float(v))
                
        sim = BrainNetworkSimulator(self.brain, base_cfg)
        res = sim.run_simulation()
        
        # Analyze basic metrics
        analyzer = BrainActivityAnalyzer(res)
        metrics = analyzer.compute_temporal_metrics()
        
        # Downsample for web display (max 1000 points)
        step = max(1, len(res['time']) // 1000)
        
        return {
            "time": res['time'][::step].tolist(),
            "activity": np.mean(res['E'], axis=1)[::step].tolist(),
            "metrics": metrics,
            "config": asdict(base_cfg)
        }

# Global instance
experiment_controller = ExperimentController()
