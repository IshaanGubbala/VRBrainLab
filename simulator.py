"""
simulator.py - Core Brain Network Simulation Engine

Implements whole-brain network dynamics using coupled neural mass models.
Each brain region is represented as a neural population with internal dynamics,
coupled via structural connectivity with realistic time delays.

Based on principles from The Virtual Brain (TVB), but simplified for clarity.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import time


@dataclass
class SimulationConfig:
    """Configuration parameters for brain simulation."""

    # Time parameters
    dt: float = 0.1  # Integration timestep (ms)
    duration: float = 2000.0  # Simulation duration (ms)
    transient: float = 200.0  # Transient period to discard (ms)

    # Coupling parameters
    global_coupling: float = 1.2  # Global coupling strength (balanced for healthy dynamics)
    conduction_velocity: float = 3.0  # Axonal conduction velocity (mm/ms)

    # Neural mass model parameters (Wilson-Cowan-like)
    tau_e: float = 10.0  # Excitatory time constant (ms)
    tau_i: float = 20.0  # Inhibitory time constant (ms)
    c_ee: float = 16.0  # E→E coupling
    c_ei: float = 12.0  # E→I coupling
    c_ie: float = 15.0  # I→E coupling
    c_ii: float = 3.0   # I→I coupling
    I_ext: float = 2.0  # External input current (balanced for mid-range activity)
    noise_strength: float = 0.03  # Noise amplitude (moderate fluctuations)

    # Activation function parameters (sigmoid)
    a_e: float = 1.3  # Excitatory gain
    theta_e: float = 3.5  # Excitatory threshold (balanced)
    a_i: float = 2.0  # Inhibitory gain
    theta_i: float = 3.0  # Inhibitory threshold (balanced)


class NeuralMassModel:
    """
    Wilson-Cowan-like neural mass model for a single brain region.
    Models excitatory (E) and inhibitory (I) population dynamics.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def sigmoid(self, x: np.ndarray, a: float, theta: float) -> np.ndarray:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-a * (x - theta)))

    def dfun(self, state: np.ndarray, coupling: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """
        Compute state derivatives for the neural mass model.

        Args:
            state: Current state [E, I] for each region, shape (num_regions, 2)
            coupling: Coupling input from other regions, shape (num_regions,)
            noise: Noise input, shape (num_regions, 2)

        Returns:
            Derivatives dE/dt and dI/dt, shape (num_regions, 2)
        """
        E = state[:, 0]  # Excitatory activity
        I = state[:, 1]  # Inhibitory activity

        c = self.config

        # Excitatory input to E population
        input_E = (c.c_ee * self.sigmoid(E, c.a_e, c.theta_e) -
                   c.c_ie * self.sigmoid(I, c.a_i, c.theta_i) +
                   c.I_ext + coupling + noise[:, 0])

        # Excitatory input to I population
        input_I = (c.c_ei * self.sigmoid(E, c.a_e, c.theta_e) -
                   c.c_ii * self.sigmoid(I, c.a_i, c.theta_i) +
                   c.I_ext + noise[:, 1])

        # Rate of change
        dE = (-E + self.sigmoid(input_E, c.a_e, c.theta_e)) / c.tau_e
        dI = (-I + self.sigmoid(input_I, c.a_i, c.theta_i)) / c.tau_i

        return np.stack([dE, dI], axis=1)


class BrainNetworkSimulator:
    """
    Whole-brain network simulator with time-delayed coupling.
    """

    def __init__(self, connectivity_data: Dict, config: Optional[SimulationConfig] = None):
        """
        Initialize the brain simulator.

        Args:
            connectivity_data: Dictionary from data_loader containing:
                - 'weights': Connectivity matrix
                - 'tract_lengths': Fiber tract lengths
                - 'region_labels': Region names
                - 'centres': Region coordinates
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        self.connectivity_data = connectivity_data

        # Extract connectivity data
        self.weights = connectivity_data['weights']
        self.tract_lengths = connectivity_data['tract_lengths']
        self.region_labels = connectivity_data['region_labels']
        self.num_regions = connectivity_data['num_regions']

        # Normalize weights (important for stability)
        self.weights_normalized = self._normalize_weights(self.weights)

        # Compute time delays from tract lengths
        self.delays = self._compute_delays()

        # Initialize neural mass model
        self.model = NeuralMassModel(self.config)

        # Simulation state
        self.state = None
        self.time_points = None
        self.activity_history = None

        print(f"Initialized BrainNetworkSimulator:")
        print(f"  - {self.num_regions} regions")
        print(f"  - Max delay: {np.max(self.delays):.1f} ms")
        print(f"  - dt: {self.config.dt} ms")

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize connectivity weights to prevent runaway dynamics."""
        max_weight = np.max(weights)
        if max_weight > 0:
            return weights / max_weight
        return weights

    def _compute_delays(self) -> np.ndarray:
        """
        Compute time delays from tract lengths and conduction velocity.

        Returns:
            Delay matrix in timesteps
        """
        # Delay (ms) = length (mm) / velocity (mm/ms)
        delay_ms = self.tract_lengths / self.config.conduction_velocity

        # Convert to timesteps
        delay_steps = np.round(delay_ms / self.config.dt).astype(int)

        # Ensure minimum delay of 1 timestep for connected regions
        delay_steps = np.where(self.weights > 0, np.maximum(delay_steps, 1), 0)

        return delay_steps

    def initialize_state(self, initial_state: Optional[np.ndarray] = None) -> None:
        """
        Initialize the simulation state.

        Args:
            initial_state: Optional initial conditions, shape (num_regions, 2)
                          If None, uses small random values
        """
        if initial_state is not None:
            self.state = initial_state.copy()
        else:
            # Random initial conditions near resting state
            self.state = np.random.uniform(0.1, 0.3, (self.num_regions, 2))

        # Initialize history buffer for delays
        max_delay = int(np.max(self.delays))
        self.history_buffer = np.zeros((max_delay + 1, self.num_regions, 2))
        self.history_buffer[0] = self.state
        self.buffer_idx = 0

    def _get_delayed_coupling(self, current_step: int) -> np.ndarray:
        """
        Compute coupling input with time delays.

        Args:
            current_step: Current simulation step

        Returns:
            Delayed coupling input for each region
        """
        coupling = np.zeros(self.num_regions)

        for i in range(self.num_regions):
            for j in range(self.num_regions):
                if self.weights[i, j] > 0:
                    delay = self.delays[i, j]

                    # Get delayed state (use modular indexing for circular buffer)
                    if current_step >= delay:
                        delayed_idx = (self.buffer_idx - delay) % len(self.history_buffer)
                        delayed_E = self.history_buffer[delayed_idx, j, 0]

                        # Add weighted coupling
                        coupling[i] += (self.config.global_coupling *
                                       self.weights_normalized[i, j] *
                                       delayed_E)

        return coupling

    def run_simulation(self,
                      initial_state: Optional[np.ndarray] = None,
                      save_interval: int = 1,
                      progress_callback: Optional[Callable] = None) -> Dict:
        """
        Run the brain network simulation.

        Args:
            initial_state: Optional initial conditions
            save_interval: Save every N timesteps (to reduce memory)
            progress_callback: Optional function to call with progress updates

        Returns:
            Dictionary containing:
                - 'time': Time points (ms)
                - 'activity': Activity traces, shape (num_timepoints, num_regions, 2)
                - 'E': Excitatory activity only, shape (num_timepoints, num_regions)
                - 'I': Inhibitory activity only
                - 'config': Simulation configuration
        """
        print(f"\nRunning simulation:")
        print(f"  Duration: {self.config.duration} ms")
        print(f"  dt: {self.config.dt} ms")

        start_time = time.time()

        # Initialize
        self.initialize_state(initial_state)

        # Time setup
        num_steps = int(self.config.duration / self.config.dt)
        num_save_steps = num_steps // save_interval
        self.time_points = np.arange(0, num_steps) * self.config.dt

        # Preallocate output arrays
        self.activity_history = np.zeros((num_save_steps, self.num_regions, 2))

        # Integration loop (Euler method)
        save_counter = 0

        for step in range(num_steps):
            # Generate noise
            noise = (self.config.noise_strength *
                    np.random.randn(self.num_regions, 2))

            # Get delayed coupling input
            coupling = self._get_delayed_coupling(step)

            # Compute derivatives
            derivatives = self.model.dfun(self.state, coupling, noise)

            # Euler integration step
            self.state += self.config.dt * derivatives

            # Clip to prevent numerical issues
            self.state = np.clip(self.state, -10, 10)

            # Update history buffer
            self.buffer_idx = (self.buffer_idx + 1) % len(self.history_buffer)
            self.history_buffer[self.buffer_idx] = self.state

            # Save data
            if step % save_interval == 0:
                self.activity_history[save_counter] = self.state
                save_counter += 1

            # Progress updates
            if progress_callback and step % 1000 == 0:
                progress = (step / num_steps) * 100
                progress_callback(progress, step, num_steps)

        elapsed = time.time() - start_time
        print(f"✓ Simulation complete in {elapsed:.2f} seconds")

        # Remove transient period
        transient_steps = int(self.config.transient / (self.config.dt * save_interval))
        time_final = self.time_points[::save_interval][transient_steps:]
        activity_final = self.activity_history[transient_steps:]

        # Package results
        results = {
            'time': time_final,
            'activity': activity_final,
            'E': activity_final[:, :, 0],  # Excitatory only
            'I': activity_final[:, :, 1],  # Inhibitory only
            'region_labels': self.region_labels,
            'config': self.config,
            'num_regions': self.num_regions,
            'connectivity': self.weights
        }

        return results

    def get_mean_activity(self) -> float:
        """Get mean activity across all regions and time."""
        if self.activity_history is None:
            return 0.0
        return float(np.mean(self.activity_history[:, :, 0]))  # Use E population

    def get_region_activity(self, region_idx: int) -> np.ndarray:
        """Get activity trace for a specific region."""
        if self.activity_history is None:
            return np.array([])
        return self.activity_history[:, region_idx, 0]


# ========== Convenience Functions ==========

def run_quick_simulation(connectivity_data: Dict,
                         duration: float = 2000.0,
                         **kwargs) -> Dict:
    """
    Quick function to run a simulation with default parameters.

    Args:
        connectivity_data: Brain connectivity data from data_loader
        duration: Simulation duration in ms
        **kwargs: Additional config parameters

    Returns:
        Simulation results dictionary
    """
    config = SimulationConfig(duration=duration, **kwargs)
    simulator = BrainNetworkSimulator(connectivity_data, config)
    return simulator.run_simulation()


def simulate_with_external_input(connectivity_data: Dict,
                                 input_region: int,
                                 input_strength: float,
                                 duration: float = 2000.0) -> Dict:
    """
    Run simulation with external input to a specific region.

    Args:
        connectivity_data: Brain connectivity data
        input_region: Index of region to stimulate
        input_strength: Strength of external input
        duration: Simulation duration

    Returns:
        Simulation results
    """
    simulator = BrainNetworkSimulator(connectivity_data)

    # Create initial state with input
    initial_state = np.random.uniform(0.1, 0.3, (connectivity_data['num_regions'], 2))
    initial_state[input_region, 0] += input_strength

    return simulator.run_simulation(initial_state=initial_state)


if __name__ == "__main__":
    # Demo: Load generic brain and run simulation
    print("=" * 60)
    print("Brain Network Simulator - Demo")
    print("=" * 60)

    # Import data loader
    from data_loader import create_default_brain

    # Create generic brain
    brain_data = create_default_brain(num_regions=68)

    # Configure simulation
    config = SimulationConfig(
        duration=3000.0,  # 3 seconds
        dt=0.1,
        global_coupling=0.6,
        noise_strength=0.02
    )

    # Create simulator
    sim = BrainNetworkSimulator(brain_data, config)

    # Run simulation
    def progress(pct, step, total):
        if int(pct) % 20 == 0:
            print(f"  Progress: {int(pct)}% ({step}/{total} steps)")

    results = sim.run_simulation(progress_callback=progress)

    # Display results
    print(f"\nResults:")
    print(f"  Time points: {len(results['time'])}")
    print(f"  Mean activity: {np.mean(results['E']):.3f}")
    print(f"  Activity std: {np.std(results['E']):.3f}")
    print(f"  Activity range: [{np.min(results['E']):.3f}, {np.max(results['E']):.3f}]")

    # Show activity for first few regions
    print(f"\nSample region activities (final 5 timepoints):")
    for i in range(min(5, brain_data['num_regions'])):
        label = brain_data['region_labels'][i]
        activity = results['E'][-5:, i]
        print(f"  {label}: {activity}")

    print("\n" + "=" * 60)
    print("✓ Demo complete!")
