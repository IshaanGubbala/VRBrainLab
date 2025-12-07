#!/usr/bin/env python3
"""
demo_brain_lab.py - Comprehensive Demo of VR Brain Lab

This script demonstrates the full pipeline:
1. Load/create brain connectivity
2. Run baseline simulation
3. Apply interventions (lesion, stimulation)
4. Analyze and compare results
5. Generate reports

Run this to verify your installation and see the system in action.
"""

import numpy as np
from data_loader import BrainDataLoader, create_default_brain
# Use fast optimized simulator (10-20x speedup)
from simulator_fast import BrainNetworkSimulator, SimulationConfig
from intervention import BrainIntervention
from analysis import BrainActivityAnalyzer


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_1_create_brain():
    """Demo 1: Create and explore brain connectivity."""
    print_section("DEMO 1: Create Brain Connectivity")

    # Create brain data loader
    loader = BrainDataLoader()

    # Load generic connectome
    brain_data = loader.load_generic_connectome(num_regions=68)

    # Show info
    info = loader.get_connectivity_info()
    print("Brain Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Save for later use
    loader.save_connectivity("brain_data", prefix="demo_brain")

    return brain_data


def demo_2_baseline_simulation(brain_data):
    """Demo 2: Run baseline brain simulation."""
    print_section("DEMO 2: Baseline Brain Simulation")

    # Configure simulation (use defaults from simulator.py for balanced parameters)
    config = SimulationConfig(
        duration=3000.0,  # 3 seconds
        transient=500.0
        # All other parameters use defaults from SimulationConfig
    )

    print(f"Simulation parameters:")
    print(f"  Duration: {config.duration} ms")
    print(f"  Timestep: {config.dt} ms")
    print(f"  Global coupling: {config.global_coupling}")
    print(f"  Noise: {config.noise_strength}")

    # Create simulator
    simulator = BrainNetworkSimulator(brain_data, config)

    # Progress callback
    def progress(pct, step, total):
        if int(pct) % 25 == 0 and int(pct) > 0:
            print(f"  Progress: {int(pct)}%")

    # Run simulation
    results = simulator.run_simulation(progress_callback=progress)

    print(f"\nResults summary:")
    print(f"  Time points: {len(results['time'])}")
    print(f"  Mean activity: {np.mean(results['E']):.3f}")
    print(f"  Activity std: {np.std(results['E']):.3f}")
    print(f"  Activity range: [{np.min(results['E']):.3f}, {np.max(results['E']):.3f}]")

    return results


def demo_3_analysis(results):
    """Demo 3: Analyze simulation results."""
    print_section("DEMO 3: Brain Activity Analysis")

    # Create analyzer
    analyzer = BrainActivityAnalyzer(results)

    # Compute all metrics
    print("Computing network metrics...")
    network_metrics = analyzer.compute_network_metrics()

    print("\nNetwork Structure:")
    print(f"  Density: {network_metrics['density']:.3f}")
    print(f"  Clustering: {network_metrics['clustering_coefficient']:.3f}")
    print(f"  Path length: {network_metrics['path_length']:.2f}")

    print("\n  Top 5 Hub Regions:")
    for hub in network_metrics['hub_regions'][:5]:
        print(f"    - {hub['region']}: strength={hub['strength']:.2f}, degree={hub['degree']}")

    print("\nComputing temporal metrics...")
    temporal_metrics = analyzer.compute_temporal_metrics()

    print("\nTemporal Dynamics:")
    print(f"  Mean activity: {temporal_metrics['mean_activity']:.3f}")
    print(f"  Global synchrony: {temporal_metrics['global_synchrony']:.3f}")
    print(f"  Metastability: {temporal_metrics['metastability']:.3f}")
    print(f"  Dominant frequency: {temporal_metrics['dominant_frequency']['frequency_hz']:.1f} Hz ({temporal_metrics['dominant_frequency']['band']} band)")

    print("\nComputing vulnerability map...")
    vulnerability = analyzer.compute_vulnerability_map()

    print("\n  Top 5 Most Vulnerable Regions:")
    for v in vulnerability['top_vulnerable'][:5]:
        print(f"    - {v['region']}: score={v['score']:.3f}")

    # Generate simulated readouts
    print("\nGenerating simulated neuroimaging readouts...")
    eeg = analyzer.generate_simulated_eeg()
    print(f"  Simulated EEG: {eeg['signals'].shape[0]} timepoints × {eeg['num_channels']} channels")

    fmri = analyzer.generate_simulated_fmri(tr=2000.0)
    print(f"  Simulated fMRI BOLD: {fmri['bold'].shape[0]} timepoints × {fmri['num_regions']} regions")

    # Full report
    print("\n" + "-" * 70)
    print("FULL ANALYSIS REPORT:")
    print("-" * 70)
    print(analyzer.generate_report())

    return analyzer


def demo_4_intervention(brain_data):
    """Demo 4: Apply interventions and compare."""
    print_section("DEMO 4: Brain Interventions")

    # Create intervention manager
    intervention = BrainIntervention(brain_data)

    # Scenario 1: Region lesion
    print("Scenario 1: Single Region Lesion")
    print("-" * 70)
    intervention.reset()
    intervention.apply_region_lesion(region_indices=10, severity=0.9)

    # Run comparison
    print("\nRunning baseline vs lesion comparison...")
    comparison_1 = intervention.run_comparison(duration=2000.0)

    baseline_1 = comparison_1['baseline']
    lesion_1 = comparison_1['intervention']

    print("\nComparison Results:")
    print(f"  Baseline mean activity: {np.mean(baseline_1['E']):.3f}")
    print(f"  Lesion mean activity: {np.mean(lesion_1['E']):.3f}")
    print(f"  Change: {(np.mean(lesion_1['E']) - np.mean(baseline_1['E'])):.3f}")

    # Scenario 2: Stroke (spatially extended lesion)
    print("\n\nScenario 2: Stroke Lesion (Extended)")
    print("-" * 70)
    intervention.reset()
    intervention.apply_stroke_lesion(center_idx=15, radius=3, severity=0.8)

    print("\nRunning baseline vs stroke comparison...")
    comparison_2 = intervention.run_comparison(duration=2000.0)

    stroke_results = comparison_2['intervention']
    print(f"\n  Stroke mean activity: {np.mean(stroke_results['E']):.3f}")

    # Scenario 3: Stimulation
    print("\n\nScenario 3: Brain Stimulation")
    print("-" * 70)
    intervention.reset()

    # Configure stimulation
    stim_regions = [20, 21]
    initial_state = intervention.apply_stimulation(
        region_indices=stim_regions,
        amplitude=1.5
    )

    # Run stimulation simulation
    sim = BrainNetworkSimulator(brain_data)
    stim_results = sim.run_simulation(initial_state=initial_state)

    print(f"\n  Stimulation mean activity: {np.mean(stim_results['E']):.3f}")
    print(f"  Change from baseline: {(np.mean(stim_results['E']) - np.mean(baseline_1['E'])):.3f}")

    # Scenario 4: Virtual drug
    print("\n\nScenario 4: Virtual Drug Effect")
    print("-" * 70)
    intervention.reset()

    # Apply sedative drug
    modified_config = intervention.apply_virtual_drug(
        drug_effect='sedative',
        strength=0.3
    )

    # Run with drug
    sim_drug = BrainNetworkSimulator(brain_data, modified_config)
    drug_results = sim_drug.run_simulation()

    print(f"\n  Drug mean activity: {np.mean(drug_results['E']):.3f}")
    print(f"  Change from baseline: {(np.mean(drug_results['E']) - np.mean(baseline_1['E'])):.3f}")

    print("\n" + "=" * 70)
    print("Summary of All Interventions:")
    print("=" * 70)
    print(f"  Baseline: {np.mean(baseline_1['E']):.3f}")
    print(f"  Region lesion: {np.mean(lesion_1['E']):.3f} ({(np.mean(lesion_1['E']) - np.mean(baseline_1['E'])) / np.mean(baseline_1['E']) * 100:+.1f}%)")
    print(f"  Stroke: {np.mean(stroke_results['E']):.3f} ({(np.mean(stroke_results['E']) - np.mean(baseline_1['E'])) / np.mean(baseline_1['E']) * 100:+.1f}%)")
    print(f"  Stimulation: {np.mean(stim_results['E']):.3f} ({(np.mean(stim_results['E']) - np.mean(baseline_1['E'])) / np.mean(baseline_1['E']) * 100:+.1f}%)")
    print(f"  Sedative drug: {np.mean(drug_results['E']):.3f} ({(np.mean(drug_results['E']) - np.mean(baseline_1['E'])) / np.mean(baseline_1['E']) * 100:+.1f}%)")


def demo_5_recovery(brain_data):
    """Demo 5: Recovery and plasticity simulation."""
    print_section("DEMO 5: Recovery & Plasticity")

    intervention = BrainIntervention(brain_data)

    # Apply lesion
    print("Step 1: Apply lesion")
    intervention.apply_region_lesion(region_indices=[10, 11], severity=0.9)

    # Simulate baseline lesion
    sim1 = BrainNetworkSimulator(intervention.current_data)
    lesion_results = sim1.run_simulation()
    print(f"  Lesion activity: {np.mean(lesion_results['E']):.3f}")

    # Apply plasticity
    print("\nStep 2: Apply plasticity (strengthen remaining connections)")
    intervention.simulate_plasticity(learning_rate=0.15)

    # Simulate with plasticity
    sim2 = BrainNetworkSimulator(intervention.current_data)
    plasticity_results = sim2.run_simulation()
    print(f"  Post-plasticity activity: {np.mean(plasticity_results['E']):.3f}")

    # Apply rewiring
    print("\nStep 3: Apply rewiring (add new connections)")
    intervention.simulate_rewiring(num_new_connections=15, strength=0.6)

    # Simulate with rewiring
    sim3 = BrainNetworkSimulator(intervention.current_data)
    rewiring_results = sim3.run_simulation()
    print(f"  Post-rewiring activity: {np.mean(rewiring_results['E']):.3f}")

    print("\nRecovery trajectory:")
    print(f"  1. After lesion: {np.mean(lesion_results['E']):.3f}")
    print(f"  2. After plasticity: {np.mean(plasticity_results['E']):.3f} ({(np.mean(plasticity_results['E']) - np.mean(lesion_results['E'])):.3f} improvement)")
    print(f"  3. After rewiring: {np.mean(rewiring_results['E']):.3f} ({(np.mean(rewiring_results['E']) - np.mean(lesion_results['E'])):.3f} improvement)")


def main():
    """Run full demo pipeline."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "VR BRAIN LAB - FULL SYSTEM DEMO" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")

    try:
        # Demo 1: Create brain
        brain_data = demo_1_create_brain()

        # Demo 2: Baseline simulation
        baseline_results = demo_2_baseline_simulation(brain_data)

        # Demo 3: Analysis
        analyzer = demo_3_analysis(baseline_results)

        # Demo 4: Interventions
        demo_4_intervention(brain_data)

        # Demo 5: Recovery
        demo_5_recovery(brain_data)

        # Success
        print_section("DEMO COMPLETE!")
        print("✓ All modules working correctly")
        print("✓ Brain simulation engine: OK")
        print("✓ Intervention system: OK")
        print("✓ Analysis pipeline: OK")
        print("\nNext steps:")
        print("  1. Start VR API server: python vr_interface.py")
        print("  2. Build Unity/VR frontend to visualize and interact")
        print("  3. Load real patient data (structural MRI, DTI)")
        print("  4. Extend with more advanced neural models")
        print("\nFor more info, see README.md")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
