
"""
ai_optimizer.py - AI Neurotherapy Agent

Implements an optimization loop to automatically tune stimulation parameters
to achieve a desired brain state (e.g. maximizing Alpha power).
"""

import numpy as np
from typing import Dict, Any, Callable
from dataclasses import dataclass
from core.simulator_fast import BrainNetworkSimulator, SimulationConfig
from core.analysis import BrainActivityAnalyzer

@dataclass
class OptimizationResult:
    best_params: Dict[str, float]
    best_score: float
    history: list

class NeuroOptimizer:
    """
    Simple Hill-Climbing Agent to optimize stimulation parameters.
    """
    def __init__(self, brain_model):
        self.brain = brain_model
        
    def optimize_stimulation(self, target_metric='alpha', max_steps=10) -> OptimizationResult:
        """
        Tune stimulation amplitude to maximize target metric.
        """
        # Search space: Amplitude 0.0 to 10.0
        current_amp = 1.0
        step_size = 1.0
        best_score = -np.inf
        best_amp = current_amp
        history = []
        
        print(f"Starting Neurotherapy Optimization (Target: {target_metric})...")
        
        for i in range(max_steps):
            # Run simulation with current amp
            score = self._evaluate(current_amp, target_metric)
            history.append({"step": i, "amp": current_amp, "score": score})
            
            print(f"Step {i}: Amp={current_amp:.2f}, Score={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_amp = current_amp
                # Keep going same direction
            else:
                # Worse? Reverse direction and shrink step
                step_size *= -0.5
                
            current_amp += step_size
            current_amp = max(0.0, min(10.0, current_amp)) # Clip
            
            # Simple convergence check
            if abs(step_size) < 0.1:
                break
                
        return OptimizationResult(
            best_params={"amplitude": best_amp},
            best_score=best_score,
            history=history
        )
        
    def _evaluate(self, amplitude: float, metric: str) -> float:
        # Create stimulus
        duration = 1000
        cfg = SimulationConfig(duration=duration, dt=0.5)
        # 10 Hz stimulation target
        t = np.arange(0, duration, cfg.dt)
        freq = 10.0 
        stim = amplitude * np.sin(2 * np.pi * freq * (t / 1000.0))
        I_stim = np.tile(stim[:, np.newaxis], (1, self.brain['num_regions']))
        
        # Run
        sim = BrainNetworkSimulator(self.brain, cfg, verbose=False)
        res = sim.run_simulation(I_stim=I_stim)
        
        # Analyze
        analyzer = BrainActivityAnalyzer(res)
        spectra = analyzer.compute_power_spectra()
        
        if metric == 'alpha':
            return spectra['band_powers']['alpha']
        elif metric == 'synchrony':
             # Return negative synchrony if we want to break seizure? Or positive?
             # Let's assume we want MAX alpha.
             return spectra['band_powers']['alpha']
             
        return 0.0
