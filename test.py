#!/usr/bin/env python3
"""
test.py - VR Brain Lab Test Suite + Parameter Sweep

Single consolidated script that:
1. Checks if simulator is working
2. Verifies brain dynamics are healthy
3. Tests interventions
4. Shows quick demo
5. Runs parameter sweeps over coupling/I_ext/c_ee:c_ie with heterogeneity and delays (parallelized)

Usage:
    python test.py                        # Run all tests
    python test.py --quick                # Quick health check only
    python test.py --demo                 # Run mini demo
    python test.py --sweep                # Run full parameter sweep
    python test.py --sweep --quick-sweep  # Smaller sweep grid
    python test.py --sweep --jobs 8       # Parallel sweep with 8 workers
"""

import argparse
import contextlib
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy import signal

# Configure Matplotlib to avoid cache permission issues in sandboxed runs
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache")))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache")))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).joinpath("fontconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import create_default_brain
from simulator_fast import BrainNetworkSimulator, SimulationConfig
from intervention import BrainIntervention

METRIC_NAMES = ["mean", "std", "fc", "metastability", "spectral_entropy"]


# ========= Parameter Sweep Utilities =========

def compute_fc_mean(activity: np.ndarray) -> float:
    """Mean functional connectivity (upper triangle correlation)."""
    fc_matrix = np.corrcoef(activity, rowvar=False)
    upper = fc_matrix[np.triu_indices_from(fc_matrix, k=1)]
    return float(np.nanmean(upper))


def compute_metastability(activity: np.ndarray) -> float:
    """Metastability = std of synchrony over time."""
    synchrony_ts = np.std(activity, axis=1)
    return float(np.std(synchrony_ts))


def spectral_entropy(signal_1d: np.ndarray, dt_ms: float,
                     fmin: float = 1.0, fmax: float = 80.0) -> float:
    """Spectral entropy of a 1D signal using Welch PSD within [fmin, fmax]."""
    fs_hz = 1000.0 / dt_ms
    freqs, psd = signal.welch(signal_1d, fs=fs_hz, nperseg=min(4096, len(signal_1d)))
    band = (freqs >= fmin) & (freqs <= fmax)
    psd_band = psd[band]
    if psd_band.size == 0 or np.sum(psd_band) <= 0:
        return float("nan")

    psd_norm = psd_band / np.sum(psd_band)
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
    return float(entropy / np.log(len(psd_norm)))


def summarize_dynamics(activity: np.ndarray, dt_ms: float) -> dict:
    """Compute sweep metrics from excitatory activity (time x regions)."""
    global_mean = float(np.mean(activity))
    global_std = float(np.std(activity))
    fc_mean = compute_fc_mean(activity)
    metastability_val = compute_metastability(activity)
    global_signal = np.mean(activity, axis=1)
    spec_ent = spectral_entropy(global_signal, dt_ms)

    return {
        "mean": global_mean,
        "std": global_std,
        "fc": fc_mean,
        "metastability": metastability_val,
        "spectral_entropy": spec_ent,
    }


def build_grid(start: float, stop: float, steps: int) -> np.ndarray:
    """Helper to build inclusive linear grids."""
    if steps <= 1:
        return np.array([start])
    return np.linspace(start, stop, steps)


def plot_heatmaps(metrics_slice: dict,
                  couplings: np.ndarray,
                  i_exts: np.ndarray,
                  ratio_value: float,
                  output_dir: Path) -> None:
    """Plot heatmaps for a single ratio slice."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharex=True, sharey=True)
    titles = [
        "Mean activity",
        "Activity std",
        "Mean FC",
        "Metastability",
        "Spectral entropy",
    ]

    for ax, metric_name, title in zip(axes, METRIC_NAMES, titles):
        data = metrics_slice[metric_name]
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=[i_exts.min(), i_exts.max(), couplings.min(), couplings.max()],
        )
        ax.set_title(title)
        ax.set_xlabel("I_ext")
        ax.set_ylabel("Global coupling")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"c_ee/c_ie ratio = {ratio_value:.2f}", y=1.03)
    plt.tight_layout()
    output_path = output_dir / f"heatmaps_ratio_{ratio_value:.2f}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def score_candidate(metrics: dict) -> float:
    """Heuristic score favoring realistic regimes."""
    score = 0.0
    mean_val = metrics["mean"]
    score -= abs(mean_val - 0.45) / 0.2
    if 0.35 <= mean_val <= 0.55:
        score += 1.0

    std_val = metrics["std"]
    score -= abs(std_val - 0.12) / 0.2
    if 0.05 <= std_val <= 0.20:
        score += 0.5

    fc_val = metrics["fc"]
    score -= abs(fc_val - 0.4) / 0.4
    if 0.20 <= fc_val <= 0.60:
        score += 1.0

    meta_val = metrics["metastability"]
    score -= abs(meta_val - 0.04) / 0.08
    if 0.01 <= meta_val <= 0.08:
        score += 0.75

    spec_val = metrics["spectral_entropy"]
    if np.isfinite(spec_val):
        score += (spec_val - 0.6) * 0.5

    return score


def _sweep_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """Worker to run a single parameter combination (multiprocessing-safe)."""
    r_idx = job["r_idx"]
    c_idx = job["c_idx"]
    i_idx = job["i_idx"]
    ratio = job["ratio"]
    coupling = job["coupling"]
    i_ext = job["i_ext"]
    save_interval = job["save_interval"]
    cfg = job["cfg"]
    brain = job["brain"]
    seed = job["seed"]
    job_idx = job["job_idx"]

    if seed is not None:
        np.random.seed(seed)

    config = SimulationConfig(
        dt=cfg["dt"],
        duration=cfg["duration"],
        transient=cfg["transient"],
        global_coupling=coupling,
        I_ext=i_ext,
        c_ie=cfg["c_ie_base"],
        c_ee=ratio * cfg["c_ie_base"],
        noise_strength=cfg["noise_strength"],
        conduction_velocity=cfg["conduction_velocity"],
        i_ext_heterogeneity=cfg["i_ext_heterogeneity"],
        theta_e_heterogeneity=cfg["theta_e_heterogeneity"],
        delay_jitter_pct=cfg["delay_jitter_pct"],
        heterogeneity_seed=seed,
        use_ou_noise=cfg["use_ou_noise"],
        ou_tau=cfg["ou_tau"],
        ou_sigma=cfg["ou_sigma"],
        slow_drive_sigma=cfg["slow_drive_sigma"],
        slow_drive_tau=cfg["slow_drive_tau"],
    )

    try:
        simulator = BrainNetworkSimulator(brain, config)
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            results = simulator.run_simulation(save_interval=save_interval)
        metrics = summarize_dynamics(results["E"], config.dt)
        return {
            "job_idx": job_idx,
            "r_idx": r_idx,
            "c_idx": c_idx,
            "i_idx": i_idx,
            "ratio": ratio,
            "coupling": coupling,
            "i_ext": i_ext,
            "metrics": metrics,
        }
    except Exception as exc:  # pragma: no cover - robustness for large sweeps
        return {
            "job_idx": job_idx,
            "r_idx": r_idx,
            "c_idx": c_idx,
            "i_idx": i_idx,
            "ratio": ratio,
            "coupling": coupling,
            "i_ext": i_ext,
            "error": str(exc),
        }


def run_parameter_sweep(args: argparse.Namespace) -> None:
    """Sweep coupling, I_ext, and c_ee/c_ie ratio with parallel execution."""
    if args.quick_sweep:
        args.coupling_steps = min(args.coupling_steps, 3)
        args.i_ext_steps = min(args.i_ext_steps, 3)
        args.ratio_steps = min(args.ratio_steps, 3)
        args.duration = min(args.duration, 1200.0)
        args.transient = min(args.transient, 200.0)

    if args.seed is not None:
        np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    couplings = build_grid(args.coupling_start, args.coupling_stop, args.coupling_steps)
    i_exts = build_grid(args.i_ext_start, args.i_ext_stop, args.i_ext_steps)
    ratios = build_grid(args.ratio_start, args.ratio_stop, args.ratio_steps)
    grid_shape = (len(ratios), len(couplings), len(i_exts))

    metrics_data = {name: np.full(grid_shape, np.nan) for name in METRIC_NAMES}
    scored_candidates = []

    total_runs = len(ratios) * len(couplings) * len(i_exts)
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP (parallel)")
    print("=" * 70)
    print(f"Total simulations: {total_runs}")
    print(f"Coupling grid: {couplings}")
    print(f"I_ext grid: {i_exts}")
    print(f"c_ee/c_ie ratios: {ratios}")
    print(f"Output directory: {output_dir.resolve()}")

    brain = create_default_brain(args.num_regions)

    cfg = {
        "dt": args.dt,
        "duration": args.duration,
        "transient": args.transient,
        "noise_strength": args.noise_strength,
        "conduction_velocity": args.conduction_velocity,
        "c_ie_base": args.c_ie_base,
        "i_ext_heterogeneity": args.i_ext_hetero,
        "theta_e_heterogeneity": args.theta_hetero,
        "delay_jitter_pct": args.delay_jitter_pct,
        "use_ou_noise": args.use_ou_noise,
        "ou_tau": args.ou_tau,
        "ou_sigma": args.ou_sigma,
        "slow_drive_sigma": args.slow_drive_sigma,
        "slow_drive_tau": args.slow_drive_tau,
    }

    base_seed = args.seed if args.seed is not None else np.random.randint(1_000_000)

    jobs = []
    job_idx = 0
    for r_idx, ratio in enumerate(ratios):
        for c_idx, coupling in enumerate(couplings):
            for i_idx, i_ext in enumerate(i_exts):
                seed = base_seed + job_idx if base_seed is not None else None
                jobs.append({
                    "job_idx": job_idx,
                    "r_idx": r_idx,
                    "c_idx": c_idx,
                    "i_idx": i_idx,
                    "ratio": ratio,
                    "coupling": coupling,
                    "i_ext": i_ext,
                    "save_interval": args.save_interval,
                    "cfg": cfg,
                    "brain": brain,
                    "seed": seed,
                })
                job_idx += 1

    n_jobs = args.jobs
    if n_jobs is None or n_jobs <= 0:
        cpu_cnt = os.cpu_count() or 1
        n_jobs = max(1, cpu_cnt - 1)

    print(f"\nUsing {n_jobs} worker(s)")

    completed = 0

    def process_result(result: Dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        if completed % max(1, total_runs // 10) == 0:
            print(f"  Progress: {completed}/{total_runs}")

        if "error" in result:
            print(f"   ! Failed combo ratio={result['ratio']:.2f}, "
                  f"coupling={result['coupling']:.2f}, I_ext={result['i_ext']:.2f} "
                  f"-> {result['error']}")
            return

        metrics = result["metrics"]
        r_idx = result["r_idx"]
        c_idx = result["c_idx"]
        i_idx = result["i_idx"]
        for name in METRIC_NAMES:
            metrics_data[name][r_idx, c_idx, i_idx] = metrics[name]

        scored_candidates.append({
            "score": score_candidate(metrics),
            "ratio": result["ratio"],
            "coupling": result["coupling"],
            "i_ext": result["i_ext"],
            **metrics,
        })

    if n_jobs == 1:
        for job in jobs:
            process_result(_sweep_worker(job))
    else:
        # Chunk submission to reduce overhead for large sweeps
        chunk_size = max(8, len(jobs) // (n_jobs * 4))
        Executor = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
        with Executor(max_workers=n_jobs) as executor:
            for start in range(0, len(jobs), chunk_size):
                chunk = jobs[start:start + chunk_size]
                futures = [executor.submit(_sweep_worker, job) for job in chunk]
                for fut in as_completed(futures):
                    process_result(fut.result())

    np.savez(
        output_dir / "parameter_sweep_results.npz",
        couplings=couplings,
        i_exts=i_exts,
        ratios=ratios,
        **metrics_data,
    )

    if not args.no_plots:
        for idx, ratio in enumerate(ratios):
            ratio_slice = {name: metrics_data[name][idx] for name in METRIC_NAMES}
            plot_heatmaps(ratio_slice, couplings, i_exts, ratio, output_dir)

    scored_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
    top_k = min(5, len(scored_candidates))
    print("\nTop candidates (higher score = closer to realistic targets):")
    for cand in scored_candidates[:top_k]:
        print(f"  ratio={cand['ratio']:.2f}, coupling={cand['coupling']:.2f}, "
              f"I_ext={cand['i_ext']:.2f} | "
              f"mean={cand['mean']:.3f}, std={cand['std']:.3f}, "
              f"FC={cand['fc']:.3f}, meta={cand['metastability']:.3f}, "
              f"specH={cand['spectral_entropy']:.3f}, score={cand['score']:.2f}")

    print("\nSaved:")
    print(f"  Numeric grid -> {output_dir / 'parameter_sweep_results.npz'}")
    if not args.no_plots:
        print(f"  Heatmaps -> {output_dir / 'heatmaps_ratio_*.png'}")
    print("\nDone.")


def test_simulator():
    """Test basic simulator functionality."""
    print("\n" + "="*70)
    print("TEST 1: Simulator Functionality")
    print("="*70)

    brain = create_default_brain(68)
    config = SimulationConfig(duration=500.0, transient=50.0)

    sim = BrainNetworkSimulator(brain, config)
    results = sim.run_simulation()

    print(f"✓ Simulation completed")
    print(f"  Timepoints: {len(results['time'])}")
    print(f"  Regions: {results['num_regions']}")

    return results, brain


def test_dynamics(results):
    """Test if dynamics are healthy."""
    print("\n" + "="*70)
    print("TEST 2: Brain Dynamics Health Check")
    print("="*70)

    E = results['E']
    mean_act = E.mean()
    std_act = E.std()
    max_act = E.max()

    print(f"\nActivity Metrics:")
    print(f"  Mean:  {mean_act:.3f}")
    print(f"  Std:   {std_act:.3f}")
    print(f"  Range: [{E.min():.3f}, {max_act:.3f}]")

    # Check health
    issues = []
    if mean_act < 0.25:
        issues.append("⚠️  Mean activity too low (increase I_ext)")
    elif mean_act > 0.75:
        issues.append("⚠️  Mean activity too high (decrease I_ext)")
    else:
        print("\n✓ Mean activity: HEALTHY")

    if std_act < 0.03:
        issues.append("⚠️  Variance too low (increase noise_strength)")
    else:
        print("✓ Activity variance: HEALTHY")

    if max_act > 0.95:
        issues.append("⚠️  Saturation detected (decrease I_ext or increase theta)")
    else:
        print("✓ No saturation: HEALTHY")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ All dynamics checks PASSED")
        return True


def test_interventions(brain):
    """Test if interventions work."""
    print("\n" + "="*70)
    print("TEST 3: Intervention Functionality")
    print("="*70)

    config = SimulationConfig(duration=500.0, transient=50.0)

    # Baseline
    sim_baseline = BrainNetworkSimulator(brain, config)
    baseline = sim_baseline.run_simulation()
    baseline_mean = baseline['E'].mean()

    # Lesion
    print("\nTesting lesion intervention...")
    intervention = BrainIntervention(brain, config)
    intervention.apply_region_lesion(10, severity=0.9)

    sim_lesion = BrainNetworkSimulator(intervention.current_data, config)
    lesion = sim_lesion.run_simulation()
    lesion_mean = lesion['E'].mean()

    change = ((lesion_mean - baseline_mean) / baseline_mean) * 100

    print(f"  Baseline: {baseline_mean:.3f}")
    print(f"  Lesion:   {lesion_mean:.3f}")
    print(f"  Change:   {change:+.1f}%")

    if abs(change) < 5:
        print("  ⚠️  Lesion has minimal effect (increase global_coupling)")
        return False
    elif 10 <= abs(change) <= 50:
        print("  ✓ Lesion response: HEALTHY")
        return True
    else:
        print(f"  ⚠️  Lesion response outside expected range")
        return False


def run_mini_demo():
    """Run a quick demonstration."""
    print("\n" + "="*70)
    print("MINI DEMO: Brain Simulation + Intervention")
    print("="*70)

    # Create brain
    print("\n1. Creating 68-region brain model...")
    brain = create_default_brain(68)

    # Run baseline
    print("\n2. Running baseline simulation...")
    config = SimulationConfig(duration=1000.0, transient=100.0)
    sim = BrainNetworkSimulator(brain, config)
    baseline = sim.run_simulation()

    print(f"  Mean activity: {baseline['E'].mean():.3f}")
    print(f"  Activity std:  {baseline['E'].std():.3f}")

    # Apply stroke
    print("\n3. Applying stroke lesion...")
    intervention = BrainIntervention(brain, config)
    intervention.apply_stroke_lesion(center_idx=15, radius=2, severity=0.8)

    # Run with stroke
    print("\n4. Running post-stroke simulation...")
    sim_stroke = BrainNetworkSimulator(intervention.current_data, config)
    stroke = sim_stroke.run_simulation()

    change = ((stroke['E'].mean() - baseline['E'].mean()) / baseline['E'].mean()) * 100

    print(f"  Baseline activity: {baseline['E'].mean():.3f}")
    print(f"  Stroke activity:   {stroke['E'].mean():.3f}")
    print(f"  Change:            {change:+.1f}%")

    print("\n" + "="*70)
    print("✓ Demo complete!")
    print("="*70)


def main():
    """Run tests, demo, or parameter sweep based on args."""

    parser = argparse.ArgumentParser(
        description="VRBrainLab test suite and parameter sweep runner."
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick health check (skip interventions).")
    parser.add_argument("--demo", action="store_true",
                        help="Run mini demo instead of tests.")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sweep and exit.")
    parser.add_argument("--quick-sweep", action="store_true",
                        help="Smaller sweep grid (3x3x3, shorter duration).")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip heatmap plotting for sweeps.")
    parser.add_argument("--jobs", type=int, default=0,
                        help="Worker processes/threads for sweeps (0=auto).")
    parser.add_argument("--executor", choices=["process", "thread"], default="thread",
                        help="Executor type for sweeps (threads avoid pickle overhead).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sweeps.")
    parser.add_argument("--output-dir", type=str, default="sweep_outputs",
                        help="Directory for sweep outputs.")
    parser.add_argument("--num-regions", type=int, default=68,
                        help="Number of regions for default brain.")
    parser.add_argument("--save-interval", type=int, default=2,
                        help="Save every N steps during sweeps.")
    parser.add_argument("--duration", type=float, default=2500.0,
                        help="Sweep simulation duration (ms).")
    parser.add_argument("--transient", type=float, default=300.0,
                        help="Sweep transient to discard (ms).")
    parser.add_argument("--dt", type=float, default=0.2,
                        help="Sweep integration step (ms).")
    parser.add_argument("--noise-strength", type=float, default=0.22,
                        help="Noise strength for sweeps (higher to break synchrony).")
    parser.add_argument("--use-ou-noise", action="store_true",
                        help="Use OU (colored) noise instead of pure white.")
    parser.add_argument("--ou-tau", type=float, default=50.0,
                        help="OU noise time constant (ms).")
    parser.add_argument("--ou-sigma", type=float, default=0.5,
                        help="OU noise amplitude.")
    parser.add_argument("--slow-drive-sigma", type=float, default=0.3,
                        help="Slow common-drive OU amplitude (0 disables).")
    parser.add_argument("--slow-drive-tau", type=float, default=600.0,
                        help="Slow drive time constant (ms).")
    parser.add_argument("--conduction-velocity", type=float, default=3.0,
                        help="Conduction velocity for sweeps.")
    parser.add_argument("--coupling-start", type=float, default=0.05)
    parser.add_argument("--coupling-stop", type=float, default=0.4)
    parser.add_argument("--coupling-steps", type=int, default=7)
    parser.add_argument("--i-ext-start", type=float, default=0.9,
                        dest="i_ext_start")
    parser.add_argument("--i-ext-stop", type=float, default=1.9,
                        dest="i_ext_stop")
    parser.add_argument("--i-ext-steps", type=int, default=7,
                        dest="i_ext_steps")
    parser.add_argument("--ratio-start", type=float, default=0.6,
                        help="Starting c_ee/c_ie ratio.")
    parser.add_argument("--ratio-stop", type=float, default=1.2,
                        help="Ending c_ee/c_ie ratio.")
    parser.add_argument("--ratio-steps", type=int, default=5,
                        help="Number of ratio samples.")
    parser.add_argument("--c-ie-base", type=float, default=24.0,
                        help="Baseline c_ie; c_ee = ratio * c_ie.")
    parser.add_argument("--i-ext-hetero", type=float, default=0.35,
                        help="Fractional std for per-region I_ext jitter (e.g., 0.35 = ±35%).")
    parser.add_argument("--theta-hetero", type=float, default=0.45,
                        help="Absolute std for per-region theta_e jitter.")
    parser.add_argument("--delay-jitter-pct", type=float, default=0.40,
                        help="Fractional jitter on delays (0.4 = ±40%).")

    args = parser.parse_args()

    if args.sweep or args.quick_sweep:
        run_parameter_sweep(args)
        return

    print("\n╔" + "═"*68 + "╗")
    print("║" + " "*20 + "VR BRAIN LAB - TEST SUITE" + " "*23 + "║")
    print("╚" + "═"*68 + "╝")

    if args.demo:
        run_mini_demo()
        return

    if args.quick:
        results, brain = test_simulator()
        healthy = test_dynamics(results)

        if healthy:
            print("\n✅ SYSTEM HEALTHY - Ready to use!")
        else:
            print("\n⚠️  SYSTEM NEEDS TUNING - Check messages above")
        return

    results, brain = test_simulator()
    dynamics_ok = test_dynamics(results)
    interventions_ok = test_interventions(brain)

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    print(f"  Simulator:     ✓ PASS")
    print(f"  Dynamics:      {'✓ PASS' if dynamics_ok else '⚠️  NEEDS TUNING'}")
    print(f"  Interventions: {'✓ PASS' if interventions_ok else '⚠️  NEEDS TUNING'}")

    if dynamics_ok and interventions_ok:
        print("\n✅ ALL TESTS PASSED - System ready!")
        print("\nNext steps:")
        print("  1. Run full demo: python demo_brain_lab.py")
        print("  2. Start VR server: python vr_interface.py")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nTo fix: Edit simulator_fast.py SimulationConfig (line ~30)")
        print("  - If activity too low: increase I_ext")
        print("  - If saturated: decrease I_ext")
        print("  - If no variance: increase noise_strength")
        print("  - If lesions weak: increase global_coupling")

    print("="*70)


if __name__ == "__main__":
    main()
