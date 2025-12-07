#!/usr/bin/env python3
"""
auto_tuner.py - Automatic Parameter Tuner for Brain Simulation

Automatically searches for parameters that produce healthy brain dynamics.

Target criteria:
- Mean activity: 0.35-0.65
- Activity std: 0.05-0.20
- Lesion response: 15-40% change
- Not saturated (max < 0.95)

Usage:
    python auto_tuner.py

The tuner will:
1. Search parameter space (I_ext, global_coupling, noise, theta)
2. Test each combination
3. Score based on how "healthy" the dynamics are
4. Return best parameters
5. Optionally update simulator.py automatically
"""

import numpy as np
from data_loader import create_default_brain
from simulator_fast import BrainNetworkSimulator, SimulationConfig
from intervention import BrainIntervention
from typing import Dict, List, Tuple
import time


class BrainParameterTuner:
    """
    Automatic parameter tuner for brain simulations.
    """

    def __init__(self, brain_data: Dict, test_duration: float = 1000.0):
        """
        Initialize tuner.

        Args:
            brain_data: Brain connectivity data
            test_duration: Duration for test simulations (ms)
        """
        self.brain_data = brain_data
        self.test_duration = test_duration
        self.best_params = None
        self.best_score = -np.inf
        self.search_history = []

    def score_dynamics(self, results: Dict, lesion_results: Dict = None) -> Tuple[float, Dict]:
        """
        Score how "healthy" the dynamics are.

        Args:
            results: Simulation results
            lesion_results: Optional lesion simulation results

        Returns:
            (score, metrics_dict)
            score: 0-100, higher is better
            metrics: breakdown of score components
        """
        E = results['E']
        mean_activity = E.mean()
        std_activity = E.std()
        max_activity = E.max()
        min_activity = E.min()

        score = 0
        penalties = []
        bonuses = []

        # Component 1: Mean activity in healthy range (0.35-0.65) - worth 30 points
        if 0.35 <= mean_activity <= 0.65:
            activity_score = 30
            bonuses.append(f"Activity in range: +30")
        elif 0.25 <= mean_activity <= 0.75:
            activity_score = 20
            bonuses.append(f"Activity near range: +20")
        elif mean_activity < 0.15:
            activity_score = 0
            penalties.append(f"Activity too low ({mean_activity:.3f}): 0")
        elif mean_activity > 0.85:
            activity_score = 0
            penalties.append(f"Activity too high ({mean_activity:.3f}): 0")
        else:
            activity_score = 10
            penalties.append(f"Activity suboptimal: +10")

        score += activity_score

        # Component 2: Good variance (0.05-0.20) - worth 25 points
        if 0.05 <= std_activity <= 0.20:
            variance_score = 25
            bonuses.append(f"Variance healthy: +25")
        elif 0.02 <= std_activity <= 0.30:
            variance_score = 15
            bonuses.append(f"Variance ok: +15")
        elif std_activity < 0.01:
            variance_score = 0
            penalties.append(f"Variance too low ({std_activity:.3f}): 0")
        else:
            variance_score = 5
            penalties.append(f"Variance suboptimal: +5")

        score += variance_score

        # Component 3: Not saturated (max < 0.95) - worth 20 points
        if max_activity < 0.85:
            saturation_score = 20
            bonuses.append(f"No saturation: +20")
        elif max_activity < 0.95:
            saturation_score = 10
            bonuses.append(f"Slight saturation: +10")
        else:
            saturation_score = 0
            penalties.append(f"Saturated (max={max_activity:.3f}): 0")

        score += saturation_score

        # Component 4: Dynamic range (not stuck at one value) - worth 15 points
        activity_range = max_activity - min_activity
        if activity_range > 0.3:
            range_score = 15
            bonuses.append(f"Good range: +15")
        elif activity_range > 0.1:
            range_score = 10
            bonuses.append(f"Ok range: +10")
        else:
            range_score = 0
            penalties.append(f"Poor range ({activity_range:.3f}): 0")

        score += range_score

        # Component 5: Lesion responsiveness (if tested) - worth 10 points
        lesion_score = 0
        if lesion_results is not None:
            lesion_change_pct = abs(
                (lesion_results['E'].mean() - mean_activity) / mean_activity * 100
            )

            if 15 <= lesion_change_pct <= 40:
                lesion_score = 10
                bonuses.append(f"Lesion response ideal: +10")
            elif 5 <= lesion_change_pct <= 50:
                lesion_score = 5
                bonuses.append(f"Lesion response ok: +5")
            else:
                penalties.append(f"Lesion response poor ({lesion_change_pct:.1f}%): 0")

        score += lesion_score

        metrics = {
            'total_score': score,
            'mean_activity': mean_activity,
            'std_activity': std_activity,
            'max_activity': max_activity,
            'activity_range': activity_range,
            'bonuses': bonuses,
            'penalties': penalties
        }

        return score, metrics

    def test_parameters(self, I_ext: float, global_coupling: float,
                       noise_strength: float, theta_e: float,
                       test_lesion: bool = False) -> Tuple[float, Dict]:
        """
        Test a specific parameter set.

        Returns:
            (score, metrics)
        """
        try:
            # Create config
            config = SimulationConfig(
                duration=self.test_duration,
                transient=100.0,
                I_ext=I_ext,
                global_coupling=global_coupling,
                noise_strength=noise_strength,
                theta_e=theta_e
            )

            # Run simulation
            sim = BrainNetworkSimulator(self.brain_data, config)
            results = sim.run_simulation()

            # Optionally test lesion
            lesion_results = None
            if test_lesion:
                intervention = BrainIntervention(self.brain_data, config)
                intervention.apply_region_lesion(10, severity=0.9)
                sim_lesion = BrainNetworkSimulator(intervention.current_data, config)
                lesion_results = sim_lesion.run_simulation()

            # Score
            score, metrics = self.score_dynamics(results, lesion_results)

            return score, metrics

        except Exception as e:
            print(f"  ⚠️  Error testing params: {e}")
            return 0, {'error': str(e)}

    def grid_search(self, test_lesion: bool = True) -> Dict:
        """
        Search parameter space using grid search.

        Args:
            test_lesion: Whether to test lesion response (slower but more accurate)

        Returns:
            Best parameters and score
        """
        print("=" * 70)
        print("AUTOMATIC PARAMETER TUNER - Grid Search")
        print("=" * 70)

        # Define search space
        I_ext_values = np.arange(1.0, 3.5, 0.5)
        coupling_values = np.arange(0.8, 2.0, 0.3)
        noise_values = [0.02, 0.03, 0.05, 0.08]
        theta_values = [3.0, 3.5, 4.0]

        total_combinations = (len(I_ext_values) * len(coupling_values) *
                            len(noise_values) * len(theta_values))

        print(f"\nSearch space:")
        print(f"  I_ext: {I_ext_values}")
        print(f"  global_coupling: {coupling_values}")
        print(f"  noise_strength: {noise_values}")
        print(f"  theta_e: {theta_values}")
        print(f"\n  Total combinations: {total_combinations}")
        print(f"  Test lesion: {test_lesion}")

        print("\nSearching...\n")

        best_score = -np.inf
        best_params = None
        best_metrics = None

        tested = 0
        start_time = time.time()

        for I_ext in I_ext_values:
            for coupling in coupling_values:
                for noise in noise_values:
                    for theta in theta_values:
                        tested += 1

                        # Test this combination
                        score, metrics = self.test_parameters(
                            I_ext, coupling, noise, theta, test_lesion
                        )

                        # Update best
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'I_ext': I_ext,
                                'global_coupling': coupling,
                                'noise_strength': noise,
                                'theta_e': theta,
                                'theta_i': theta - 0.5  # Keep offset
                            }
                            best_metrics = metrics

                            print(f"✓ New best! Score: {score:.1f}/100")
                            print(f"  I_ext={I_ext}, coupling={coupling}, "
                                  f"noise={noise}, theta={theta}")
                            print(f"  Mean activity: {metrics['mean_activity']:.3f}, "
                                  f"Std: {metrics['std_activity']:.3f}")

                        # Progress
                        if tested % 10 == 0:
                            elapsed = time.time() - start_time
                            eta = (elapsed / tested) * (total_combinations - tested)
                            print(f"  Progress: {tested}/{total_combinations} "
                                  f"({100*tested/total_combinations:.0f}%) "
                                  f"ETA: {eta:.0f}s")

        elapsed_total = time.time() - start_time

        print("\n" + "=" * 70)
        print("TUNING COMPLETE")
        print("=" * 70)

        print(f"\nTime elapsed: {elapsed_total:.1f} seconds")
        print(f"Combinations tested: {total_combinations}")

        if best_params:
            print(f"\n✅ BEST PARAMETERS FOUND (Score: {best_score:.1f}/100)")
            print("-" * 70)
            for key, value in best_params.items():
                print(f"  {key}: {value}")

            print(f"\nDynamics Metrics:")
            print(f"  Mean activity: {best_metrics['mean_activity']:.3f}")
            print(f"  Activity std: {best_metrics['std_activity']:.3f}")
            print(f"  Max activity: {best_metrics['max_activity']:.3f}")
            print(f"  Activity range: {best_metrics['activity_range']:.3f}")

            print(f"\nScore breakdown:")
            for bonus in best_metrics['bonuses']:
                print(f"  ✓ {bonus}")
            for penalty in best_metrics['penalties']:
                print(f"  ⚠️  {penalty}")

            # Store
            self.best_params = best_params
            self.best_score = best_score

            return best_params
        else:
            print("\n❌ No good parameters found!")
            return None

    def apply_to_simulator(self, simulator_path: str = "simulator_fast.py"):
        """
        Automatically update the simulator file with best parameters.

        Args:
            simulator_path: Path to simulator file
        """
        if self.best_params is None:
            print("❌ No best parameters to apply! Run grid_search first.")
            return

        print(f"\n{'=' * 70}")
        print(f"Updating {simulator_path} with best parameters...")
        print(f"{'=' * 70}")

        # Read file
        with open(simulator_path, 'r') as f:
            lines = f.readlines()

        # Find and replace parameter lines
        updates = 0
        for i, line in enumerate(lines):
            if 'global_coupling: float =' in line and 'Global coupling' in line:
                old = line
                lines[i] = f"    global_coupling: float = {self.best_params['global_coupling']}  # Auto-tuned\n"
                print(f"✓ Updated global_coupling")
                updates += 1

            elif 'I_ext: float =' in line and 'External input' in line:
                old = line
                lines[i] = f"    I_ext: float = {self.best_params['I_ext']}  # Auto-tuned\n"
                print(f"✓ Updated I_ext")
                updates += 1

            elif 'noise_strength: float =' in line and 'Noise' in line:
                old = line
                lines[i] = f"    noise_strength: float = {self.best_params['noise_strength']}  # Auto-tuned\n"
                print(f"✓ Updated noise_strength")
                updates += 1

            elif 'theta_e: float =' in line and 'Excitatory threshold' in line:
                old = line
                lines[i] = f"    theta_e: float = {self.best_params['theta_e']}  # Auto-tuned\n"
                print(f"✓ Updated theta_e")
                updates += 1

            elif 'theta_i: float =' in line and 'Inhibitory threshold' in line:
                old = line
                lines[i] = f"    theta_i: float = {self.best_params['theta_i']}  # Auto-tuned\n"
                print(f"✓ Updated theta_i")
                updates += 1

        # Write back
        with open(simulator_path, 'w') as f:
            f.writelines(lines)

        print(f"\n✅ Updated {updates} parameters in {simulator_path}")
        print("=" * 70)


def quick_tune(num_regions: int = 68, apply_automatically: bool = False) -> Dict:
    """
    Quick function to tune parameters and optionally apply them.

    Args:
        num_regions: Number of brain regions
        apply_automatically: Whether to auto-update simulator.py

    Returns:
        Best parameters dictionary
    """
    print("Creating brain model...")
    brain = create_default_brain(num_regions)

    tuner = BrainParameterTuner(brain, test_duration=800.0)

    # Search
    best_params = tuner.grid_search(test_lesion=True)

    # Apply
    if apply_automatically and best_params:
        tuner.apply_to_simulator("simulator_fast.py")
        tuner.apply_to_simulator("simulator.py")

    return best_params


if __name__ == "__main__":
    import sys

    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "BRAIN PARAMETER AUTO-TUNER" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Check if user wants to auto-apply
    auto_apply = "--apply" in sys.argv

    if auto_apply:
        print("⚙️  Auto-apply mode: Will update simulator files automatically\n")
    else:
        print("💡 Run with --apply flag to automatically update simulator files\n")

    # Run tuner
    best_params = quick_tune(num_regions=68, apply_automatically=auto_apply)

    if best_params:
        print("\n" + "=" * 70)
        print("✅ TUNING SUCCESSFUL!")
        print("=" * 70)
        print("\nTo use these parameters:")
        if not auto_apply:
            print("  1. Run again with: python auto_tuner.py --apply")
            print("  OR")
            print("  2. Manually edit simulator.py SimulationConfig with values above")
        else:
            print("  ✓ Parameters already applied to simulator files!")
            print("  → Just run: python demo_brain_lab.py")

        print("\n" + "=" * 70)
    else:
        print("\n❌ Tuning failed - no good parameters found")
        print("Try expanding the search space or adjusting target criteria")
