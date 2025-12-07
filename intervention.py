"""
intervention.py - Brain Intervention and Perturbation Module

Implements various interventions on brain models:
- Lesions (region removal, connection loss)
- Stimulation (external input, neuromodulation)
- Parameter perturbations (simulated drugs, plasticity changes)
- Recovery simulation (rewiring, adaptation)

Designed to work with simulator.py to run modified brain simulations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from copy import deepcopy
from dataclasses import dataclass
# Use fast optimized simulator
try:
    from simulator_fast import BrainNetworkSimulator, SimulationConfig
except ImportError:
    from simulator import BrainNetworkSimulator, SimulationConfig


@dataclass
class InterventionConfig:
    """Configuration for an intervention."""
    name: str
    description: str
    intervention_type: str  # 'lesion', 'stimulation', 'parameter', 'plasticity'


class BrainIntervention:
    """
    Manages interventions on brain connectivity and dynamics.
    """

    def __init__(self, connectivity_data: Dict, sim_config: Optional[SimulationConfig] = None):
        """
        Initialize intervention manager.

        Args:
            connectivity_data: Original brain connectivity data
            sim_config: Simulation configuration
        """
        self.original_data = deepcopy(connectivity_data)
        self.current_data = deepcopy(connectivity_data)
        self.sim_config = sim_config or SimulationConfig()

        self.num_regions = connectivity_data['num_regions']
        self.region_labels = connectivity_data['region_labels']

        # Track applied interventions
        self.intervention_history = []

    def reset(self):
        """Reset to original connectivity (undo all interventions)."""
        self.current_data = deepcopy(self.original_data)
        self.intervention_history = []
        print("✓ Reset to original connectivity")

    # ========== LESION INTERVENTIONS ==========

    def apply_region_lesion(self, region_indices: Union[int, List[int]],
                           severity: float = 1.0) -> Dict:
        """
        Simulate a lesion by removing/damaging brain regions.

        Args:
            region_indices: Single region index or list of indices to lesion
            severity: Lesion severity (0=no damage, 1=complete removal)

        Returns:
            Modified connectivity data
        """
        if isinstance(region_indices, int):
            region_indices = [region_indices]

        region_names = [self.region_labels[i] for i in region_indices]
        print(f"Applying lesion to region(s): {region_names}")
        print(f"  Severity: {severity * 100:.0f}%")

        # Reduce outgoing and incoming connections
        for idx in region_indices:
            # Outgoing connections
            self.current_data['weights'][idx, :] *= (1 - severity)
            # Incoming connections
            self.current_data['weights'][:, idx] *= (1 - severity)

        # Record intervention
        self.intervention_history.append({
            'type': 'lesion',
            'regions': region_names,
            'indices': region_indices,
            'severity': severity
        })

        print(f"✓ Lesion applied")
        return self.current_data

    def apply_connection_lesion(self, source_idx: int, target_idx: int,
                                severity: float = 1.0) -> Dict:
        """
        Lesion specific connection(s) between regions.

        Args:
            source_idx: Source region index
            target_idx: Target region index
            severity: Connection damage severity

        Returns:
            Modified connectivity data
        """
        source_name = self.region_labels[source_idx]
        target_name = self.region_labels[target_idx]

        print(f"Lesioning connection: {source_name} → {target_name}")
        print(f"  Severity: {severity * 100:.0f}%")

        # Reduce connection strength (bidirectional)
        self.current_data['weights'][source_idx, target_idx] *= (1 - severity)
        self.current_data['weights'][target_idx, source_idx] *= (1 - severity)

        self.intervention_history.append({
            'type': 'connection_lesion',
            'source': source_name,
            'target': target_name,
            'severity': severity
        })

        print("✓ Connection lesion applied")
        return self.current_data

    def apply_stroke_lesion(self, center_idx: int, radius: int = 2,
                           severity: float = 1.0) -> Dict:
        """
        Simulate a stroke-like lesion affecting a region and its neighbors.

        Args:
            center_idx: Central region of stroke
            radius: Number of neighbor hops to affect
            severity: Damage severity (decreases with distance)

        Returns:
            Modified connectivity data
        """
        center_name = self.region_labels[center_idx]
        print(f"Simulating stroke centered at: {center_name}")
        print(f"  Radius: {radius} hops")

        # Find regions within radius using connectivity
        affected_regions = self._find_neighbors(center_idx, radius)

        # Apply graded lesion (stronger at center, weaker at periphery)
        for region, distance in affected_regions.items():
            # Distance-dependent severity
            local_severity = severity * (1 - distance / (radius + 1))
            if local_severity > 0:
                self.current_data['weights'][region, :] *= (1 - local_severity)
                self.current_data['weights'][:, region] *= (1 - local_severity)

        affected_names = [self.region_labels[r] for r in affected_regions.keys()]
        print(f"  Affected {len(affected_regions)} regions: {affected_names[:5]}...")

        self.intervention_history.append({
            'type': 'stroke',
            'center': center_name,
            'radius': radius,
            'affected_regions': affected_names,
            'severity': severity
        })

        print("✓ Stroke lesion applied")
        return self.current_data

    # ========== STIMULATION INTERVENTIONS ==========

    def apply_stimulation(self, region_indices: Union[int, List[int]],
                         amplitude: float = 1.0,
                         frequency: Optional[float] = None) -> Dict:
        """
        Apply external stimulation to brain region(s).

        Note: This creates initial conditions; actual stimulation happens in simulation.

        Args:
            region_indices: Region(s) to stimulate
            amplitude: Stimulation strength
            frequency: Optional oscillation frequency (Hz)

        Returns:
            Initial conditions for simulation with stimulation
        """
        if isinstance(region_indices, int):
            region_indices = [region_indices]

        region_names = [self.region_labels[i] for i in region_indices]
        print(f"Configuring stimulation for: {region_names}")
        print(f"  Amplitude: {amplitude}")
        if frequency:
            print(f"  Frequency: {frequency} Hz")

        # Create initial state with boosted activity
        initial_state = np.random.uniform(0.1, 0.3, (self.num_regions, 2))

        for idx in region_indices:
            initial_state[idx, 0] += amplitude  # Boost excitatory population

        self.intervention_history.append({
            'type': 'stimulation',
            'regions': region_names,
            'amplitude': amplitude,
            'frequency': frequency
        })

        print("✓ Stimulation configured")
        return initial_state

    def apply_neuromodulation(self, region_indices: Union[int, List[int]],
                             excitability_change: float = 0.2) -> SimulationConfig:
        """
        Simulate neuromodulation (change region excitability/parameters).

        Args:
            region_indices: Regions to modulate
            excitability_change: Change in excitability (+/- relative change)

        Returns:
            Modified simulation config
        """
        if isinstance(region_indices, int):
            region_indices = [region_indices]

        region_names = [self.region_labels[i] for i in region_indices]
        print(f"Applying neuromodulation to: {region_names}")
        print(f"  Excitability change: {excitability_change:+.1%}")

        # Modify simulation parameters (simplified - affects all regions)
        # In a more advanced version, this would be region-specific
        modified_config = deepcopy(self.sim_config)
        modified_config.I_ext *= (1 + excitability_change)

        self.intervention_history.append({
            'type': 'neuromodulation',
            'regions': region_names,
            'excitability_change': excitability_change
        })

        print("✓ Neuromodulation applied")
        return modified_config

    # ========== PARAMETER PERTURBATIONS ==========

    def apply_virtual_drug(self, drug_effect: str, strength: float = 0.2) -> SimulationConfig:
        """
        Simulate drug effects by changing neural parameters.

        Args:
            drug_effect: Type of drug effect:
                - 'sedative': Decrease excitability
                - 'stimulant': Increase excitability
                - 'stabilizer': Reduce noise, increase inhibition
                - 'disruptor': Increase noise
            strength: Effect strength (relative change)

        Returns:
            Modified simulation configuration
        """
        print(f"Applying virtual drug: {drug_effect}")
        print(f"  Strength: {strength:.1%}")

        modified_config = deepcopy(self.sim_config)

        if drug_effect == 'sedative':
            modified_config.I_ext *= (1 - strength)
            modified_config.c_ei *= (1 + strength)  # Increase inhibition

        elif drug_effect == 'stimulant':
            modified_config.I_ext *= (1 + strength)
            modified_config.c_ee *= (1 + strength)  # Increase excitation

        elif drug_effect == 'stabilizer':
            modified_config.noise_strength *= (1 - strength)
            modified_config.c_ie *= (1 + strength)  # Increase inhibition

        elif drug_effect == 'disruptor':
            modified_config.noise_strength *= (1 + 2 * strength)

        else:
            print(f"Unknown drug effect: {drug_effect}")
            return self.sim_config

        self.intervention_history.append({
            'type': 'virtual_drug',
            'effect': drug_effect,
            'strength': strength
        })

        print(f"✓ Virtual drug applied")
        return modified_config

    # ========== PLASTICITY & RECOVERY ==========

    def simulate_plasticity(self, learning_rate: float = 0.1,
                          target_activity: float = 0.5) -> Dict:
        """
        Simulate activity-dependent plasticity (strengthening active connections).

        Note: This is a simplified homeostatic plasticity model.

        Args:
            learning_rate: Rate of synaptic change
            target_activity: Target activity level for homeostasis

        Returns:
            Modified connectivity with plasticity
        """
        print(f"Simulating plasticity:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Target activity: {target_activity}")

        # This is a placeholder - real plasticity would require:
        # 1. Running a simulation
        # 2. Measuring activity
        # 3. Adjusting connections based on activity

        # Simplified: strengthen existing connections slightly
        mask = self.current_data['weights'] > 0
        self.current_data['weights'][mask] *= (1 + learning_rate * 0.1)

        self.intervention_history.append({
            'type': 'plasticity',
            'learning_rate': learning_rate,
            'target_activity': target_activity
        })

        print("✓ Plasticity applied")
        return self.current_data

    def simulate_rewiring(self, num_new_connections: int = 10,
                         strength: float = 0.5) -> Dict:
        """
        Simulate recovery by adding new connections (rewiring).

        Args:
            num_new_connections: Number of connections to add
            strength: Strength of new connections

        Returns:
            Modified connectivity with new connections
        """
        print(f"Simulating rewiring:")
        print(f"  Adding {num_new_connections} new connections")
        print(f"  Connection strength: {strength}")

        weights = self.current_data['weights']
        added = 0

        # Add connections between nearby regions that aren't connected
        for _ in range(num_new_connections * 10):  # Try multiple times
            i, j = np.random.randint(0, self.num_regions, 2)

            if i != j and weights[i, j] == 0:
                weights[i, j] = strength
                weights[j, i] = strength  # Symmetric
                added += 1

                if added >= num_new_connections:
                    break

        print(f"✓ Added {added} new connections")

        self.intervention_history.append({
            'type': 'rewiring',
            'num_connections': added,
            'strength': strength
        })

        return self.current_data

    # ========== HELPER METHODS ==========

    def _find_neighbors(self, center_idx: int, max_distance: int) -> Dict[int, int]:
        """
        Find regions within a certain graph distance from center.

        Returns:
            Dictionary mapping region_idx -> distance
        """
        weights = self.current_data['weights']
        visited = {center_idx: 0}
        current_layer = [center_idx]

        for distance in range(1, max_distance + 1):
            next_layer = []

            for node in current_layer:
                # Find neighbors
                neighbors = np.where(weights[node, :] > 0)[0]

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited[neighbor] = distance
                        next_layer.append(neighbor)

            current_layer = next_layer

            if not current_layer:
                break

        return visited

    def get_intervention_summary(self) -> str:
        """Get a text summary of all applied interventions."""
        if not self.intervention_history:
            return "No interventions applied"

        summary = f"Applied {len(self.intervention_history)} intervention(s):\n"
        for i, intervention in enumerate(self.intervention_history, 1):
            summary += f"{i}. {intervention['type']}: "

            if 'regions' in intervention:
                summary += f"{intervention['regions']}"
            if 'severity' in intervention:
                summary += f" (severity: {intervention['severity']})"

            summary += "\n"

        return summary

    def run_comparison(self, duration: float = 2000.0) -> Dict:
        """
        Run baseline vs intervention simulation comparison.

        Args:
            duration: Simulation duration

        Returns:
            Dictionary with baseline and intervention results
        """
        print("\n" + "=" * 60)
        print("Running Baseline vs Intervention Comparison")
        print("=" * 60)

        # Run baseline
        print("\n1. Running baseline simulation...")
        baseline_sim = BrainNetworkSimulator(self.original_data, self.sim_config)
        baseline_results = baseline_sim.run_simulation()

        # Run intervention
        print("\n2. Running intervention simulation...")
        print(self.get_intervention_summary())
        intervention_sim = BrainNetworkSimulator(self.current_data, self.sim_config)
        intervention_results = intervention_sim.run_simulation()

        print("\n" + "=" * 60)
        print("Comparison Results:")
        print("=" * 60)
        print(f"Baseline mean activity:     {np.mean(baseline_results['E']):.3f}")
        print(f"Intervention mean activity: {np.mean(intervention_results['E']):.3f}")
        print(f"Activity change:            {(np.mean(intervention_results['E']) - np.mean(baseline_results['E'])):.3f}")
        print("=" * 60)

        return {
            'baseline': baseline_results,
            'intervention': intervention_results,
            'history': self.intervention_history
        }


# ========== Convenience Functions ==========

def quick_lesion_simulation(connectivity_data: Dict,
                           lesion_region: int,
                           severity: float = 1.0) -> Dict:
    """
    Quick function to simulate a lesion and compare to baseline.

    Args:
        connectivity_data: Brain connectivity data
        lesion_region: Region to lesion
        severity: Lesion severity

    Returns:
        Comparison results
    """
    intervention = BrainIntervention(connectivity_data)
    intervention.apply_region_lesion(lesion_region, severity)
    return intervention.run_comparison()


def quick_stimulation_simulation(connectivity_data: Dict,
                                 stim_region: int,
                                 amplitude: float = 1.0) -> Dict:
    """
    Quick function to simulate stimulation and compare to baseline.

    Args:
        connectivity_data: Brain connectivity data
        stim_region: Region to stimulate
        amplitude: Stimulation amplitude

    Returns:
        Comparison results
    """
    intervention = BrainIntervention(connectivity_data)
    initial_state = intervention.apply_stimulation(stim_region, amplitude)

    # Run with stimulation
    sim = BrainNetworkSimulator(connectivity_data)
    results = sim.run_simulation(initial_state=initial_state)

    return results


if __name__ == "__main__":
    # Demo: Lesion and stimulation
    print("=" * 60)
    print("Brain Intervention Module - Demo")
    print("=" * 60)

    from data_loader import create_default_brain

    # Create brain
    brain_data = create_default_brain(num_regions=68)

    # Create intervention manager
    intervention = BrainIntervention(brain_data)

    # Apply a stroke lesion
    print("\n1. Applying stroke lesion...")
    intervention.apply_stroke_lesion(center_idx=10, radius=2, severity=0.8)

    # Run comparison
    results = intervention.run_comparison(duration=2000.0)

    print("\n" + "=" * 60)
    print("✓ Demo complete!")
