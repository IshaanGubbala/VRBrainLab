#!/usr/bin/env python3
"""
auto_tuner.py - Adaptive Parameter Tuner for VRBrainLab

Continuously searches parameter space (parallelized) until target dynamical
metrics are reached or max rounds are exhausted. Targets:
  - Mean:           0.35–0.55
  - Std:            0.10–0.25
  - FC (corr):      0.30–0.60
  - Metastability:  0.01–0.08
  - Spectral entropy: >= 0.50

Defaults focus on the desynchronized-yet-coordinated regime (moderate coupling,
heterogeneity, delay jitter, OU/slow-drive noise).

Usage:
  python auto_tuner.py                 # run until target_score met
  python auto_tuner.py --quick         # smaller grids, shorter sims
  python auto_tuner.py --apply         # apply best params to simulator_fast.py
"""

import argparse
import contextlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from scipy import signal

from data_loader import create_default_brain
from simulator_fast import BrainNetworkSimulator, SimulationConfig


# ---------- Metrics ----------

def compute_fc_mean(activity: np.ndarray) -> float:
    fc_matrix = np.corrcoef(activity, rowvar=False)
    upper = fc_matrix[np.triu_indices_from(fc_matrix, k=1)]
    return float(np.nanmean(upper))


def compute_metastability(activity: np.ndarray) -> float:
    synchrony_ts = np.std(activity, axis=1)
    return float(np.std(synchrony_ts))


def spectral_entropy(signal_1d: np.ndarray, dt_ms: float,
                     fmin: float = 1.0, fmax: float = 80.0) -> float:
    fs_hz = 1000.0 / dt_ms
    freqs, psd = signal.welch(signal_1d, fs=fs_hz, nperseg=min(4096, len(signal_1d)))
    band = (freqs >= fmin) & (freqs <= fmax)
    psd_band = psd[band]
    if psd_band.size == 0 or np.sum(psd_band) <= 0:
        return float("nan")
    psd_norm = psd_band / np.sum(psd_band)
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
    return float(entropy / np.log(len(psd_norm)))


def summarize_metrics(activity: np.ndarray, dt_ms: float) -> Dict[str, float]:
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


def score_metrics(metrics: Dict[str, float]) -> float:
    """Higher is better. Centered on target bands."""
    score = 0.0
    mean_val = metrics["mean"]
    score -= abs(mean_val - 0.45) / 0.15
    if 0.35 <= mean_val <= 0.55:
        score += 1.0

    std_val = metrics["std"]
    score -= abs(std_val - 0.16) / 0.15
    if 0.10 <= std_val <= 0.25:
        score += 1.0

    fc_val = metrics["fc"]
    score -= abs(fc_val - 0.45) / 0.4
    if 0.30 <= fc_val <= 0.60:
        score += 1.0

    meta_val = metrics["metastability"]
    score -= abs(meta_val - 0.04) / 0.08
    if 0.01 <= meta_val <= 0.08:
        score += 1.0

    spec_val = metrics["spectral_entropy"]
    if np.isfinite(spec_val):
        score += (spec_val - 0.5) * 1.0  # reward higher entropy above 0.5

    return score


# ---------- Simulation ----------

def run_combo(brain: Dict, params: Dict, base_cfg: Dict) -> Tuple[float, Dict[str, float], Dict]:
    """Run one parameter combo and return (score, metrics, params)."""
    cfg = SimulationConfig(
        dt=base_cfg["dt"],
        duration=base_cfg["duration"],
        transient=base_cfg["transient"],
        global_coupling=params["coupling"],
        I_ext=params["i_ext"],
        c_ie=params["c_ie_base"],
        c_ee=params["ratio"] * params["c_ie_base"],
        noise_strength=params["noise_strength"],
        conduction_velocity=base_cfg["conduction_velocity"],
        i_ext_heterogeneity=params["i_ext_hetero"],
        theta_e_heterogeneity=params["theta_hetero"],
        delay_jitter_pct=params["delay_jitter"],
        heterogeneity_seed=params["seed"],
        use_ou_noise=params["use_ou"],
        ou_tau=params["ou_tau"],
        ou_sigma=params["ou_sigma"],
        slow_drive_sigma=params["slow_drive_sigma"],
        slow_drive_tau=params["slow_drive_tau"],
    )

    simulator = BrainNetworkSimulator(brain, cfg, verbose=False)
    results = simulator.run_simulation(save_interval=base_cfg["save_interval"],
                                       suppress_output=True)
    metrics = summarize_metrics(results["E"], cfg.dt)
    score = score_metrics(metrics)
    return score, metrics, params


# ---------- Search logic ----------

def build_grid(values: List[float], label: str) -> List[float]:
    if not values:
        raise ValueError(f"No values provided for {label}")
    return list(values)


def generate_combos(args: argparse.Namespace, round_idx: int) -> List[Dict]:
    """Generate parameter combinations for a round."""
    couplings = np.linspace(args.coupling_start, args.coupling_stop, args.coupling_steps)
    i_exts = np.linspace(args.i_ext_start, args.i_ext_stop, args.i_ext_steps)
    ratios = np.linspace(args.ratio_start, args.ratio_stop, args.ratio_steps)
    noise_values = np.linspace(args.noise_min, args.noise_max, args.noise_steps)
    hetero_i_values = np.linspace(args.i_ext_hetero_min, args.i_ext_hetero_max, args.hetero_steps)
    hetero_theta_values = np.linspace(args.theta_hetero_min, args.theta_hetero_max, args.hetero_steps)
    delay_values = np.linspace(args.delay_jitter_min, args.delay_jitter_max, args.delay_steps)

    combos = []
    seed_base = args.seed if args.seed is not None else np.random.randint(1_000_000)
    idx = 0
    for ratio in ratios:
        for coupling in couplings:
            for i_ext in i_exts:
                for noise in noise_values:
                    for i_het in hetero_i_values:
                        for th_het in hetero_theta_values:
                            for delay in delay_values:
                                combos.append({
                                    "ratio": float(ratio),
                                    "coupling": float(coupling),
                                    "i_ext": float(i_ext),
                                    "noise_strength": float(noise),
                                    "i_ext_hetero": float(i_het),
                                    "theta_hetero": float(th_het),
                                    "delay_jitter": float(delay),
                                    "c_ie_base": args.c_ie_base,
                                    "use_ou": args.use_ou_noise,
                                    "ou_tau": args.ou_tau,
                                    "ou_sigma": args.ou_sigma,
                                    "slow_drive_sigma": args.slow_drive_sigma,
                                    "slow_drive_tau": args.slow_drive_tau,
                                    "seed": seed_base + idx,
                                })
                                idx += 1
    # Shuffle to diversify early results
    np.random.shuffle(combos)
    return combos


def search_round(brain: Dict, combos: List[Dict], base_cfg: Dict, args: argparse.Namespace):
    """Evaluate one round of combos in parallel and return best."""
    best_score = -1e9
    best_metrics = None
    best_params = None

    n_jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 2) - 1)
    chunk_size = max(8, len(combos) // (n_jobs * 4))

    print(f"\nRound combos: {len(combos)} | workers: {n_jobs} | chunk: {chunk_size}")

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        for start in range(0, len(combos), chunk_size):
            chunk = combos[start:start + chunk_size]
            for params in chunk:
                futures.append(executor.submit(run_combo, brain, params, base_cfg))

        for idx, fut in enumerate(as_completed(futures), 1):
            try:
                score, metrics, params = fut.result()
            except Exception as exc:  # pragma: no cover (robustness)
                print(f"   ! combo failed: {exc}")
                continue
            if score > best_score:
                best_score = score
                best_metrics = metrics
                best_params = params
                print(f"✓ New best (score {score:.2f}) "
                      f"ratio={params['ratio']:.2f}, coupling={params['coupling']:.3f}, "
                      f"I_ext={params['i_ext']:.2f}, noise={params['noise_strength']:.2f}, "
                      f"het=({params['i_ext_hetero']:.2f},{params['theta_hetero']:.2f}), "
                      f"delay={params['delay_jitter']:.2f} | "
                      f"mean={metrics['mean']:.3f}, std={metrics['std']:.3f}, "
                      f"FC={metrics['fc']:.3f}, meta={metrics['metastability']:.3f}, "
                      f"specH={metrics['spectral_entropy']:.3f}")

    return best_score, best_metrics, best_params


def apply_params(params: Dict, filepath: str = "simulator_fast.py"):
    """Apply key parameters to simulator defaults."""
    print(f"\nApplying best parameters to {filepath} ...")
    with open(filepath, "r") as f:
        lines = f.readlines()

    updates = 0
    replacements = {
        "global_coupling: float =": params["coupling"],
        "I_ext: float =": params["i_ext"],
        "noise_strength: float =": params["noise_strength"],
        "theta_e: float =": params.get("theta_e", 3.0),
        "theta_i: float =": params.get("theta_e", 3.0) - 0.5,
        "c_ee: float =": params["ratio"] * params["c_ie_base"],
        "c_ie: float =": params["c_ie_base"],
        "i_ext_heterogeneity: float =": params["i_ext_hetero"],
        "theta_e_heterogeneity: float =": params["theta_hetero"],
        "delay_jitter_pct: float =": params["delay_jitter"],
        "use_ou_noise: bool =": params["use_ou"],
        "ou_tau: float =": params["ou_tau"],
        "ou_sigma: float =": params["ou_sigma"],
        "slow_drive_sigma: float =": params["slow_drive_sigma"],
        "slow_drive_tau: float =": params["slow_drive_tau"],
    }

    for i, line in enumerate(lines):
        for key, val in replacements.items():
            if key in line:
                comment = "  # Auto-tuned\n" if "#" in line else "\n"
                prefix = line.split(key)[0]
                lines[i] = f"{prefix}{key} {val}{comment}"
                updates += 1
                break

    with open(filepath, "w") as f:
        f.writelines(lines)

    print(f"✅ Applied {updates} parameter defaults to {filepath}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive auto-tuner for VRBrainLab.")
    parser.add_argument("--apply", action="store_true", help="Apply best params to simulator_fast.py")
    parser.add_argument("--quick", action="store_true", help="Smaller grids and shorter simulations")
    parser.add_argument("--target-score", type=float, default=2.5,
                        help="Stop when best score reaches this.")
    parser.add_argument("--max-rounds", type=int, default=5,
                        help="Maximum tuning rounds (ignored if --until-target).")
    parser.add_argument("--until-target", action="store_true",
                        help="Ignore max rounds; keep searching until target_score is hit.")
    parser.add_argument("--jobs", type=int, default=0,
                        help="Parallel workers (0=auto).")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed.")

    # Simulation timing
    parser.add_argument("--duration", type=float, default=2000.0, help="Simulation duration (ms).")
    parser.add_argument("--transient", type=float, default=250.0, help="Transient trim (ms).")
    parser.add_argument("--dt", type=float, default=0.2, help="Timestep (ms).")
    parser.add_argument("--save-interval", type=int, default=4, help="Save every N steps.")
    parser.add_argument("--conduction-velocity", type=float, default=3.0,
                        help="Axonal conduction velocity (mm/ms).")

    # Parameter ranges
    parser.add_argument("--coupling-start", type=float, default=0.20)
    parser.add_argument("--coupling-stop", type=float, default=0.8)
    parser.add_argument("--coupling-steps", type=int, default=5)
    parser.add_argument("--i-ext-start", type=float, default=0.9)
    parser.add_argument("--i-ext-stop", type=float, default=1.9)
    parser.add_argument("--i-ext-steps", type=int, default=5)
    parser.add_argument("--ratio-start", type=float, default=0.8)
    parser.add_argument("--ratio-stop", type=float, default=1.2)
    parser.add_argument("--ratio-steps", type=int, default=3)
    parser.add_argument("--c-ie-base", type=float, default=23.0)
    parser.add_argument("--noise-min", type=float, default=0.16)
    parser.add_argument("--noise-max", type=float, default=0.26)
    parser.add_argument("--noise-steps", type=int, default=3)
    parser.add_argument("--i-ext-hetero-min", type=float, default=0.20)
    parser.add_argument("--i-ext-hetero-max", type=float, default=0.35)
    parser.add_argument("--theta-hetero-min", type=float, default=0.35)
    parser.add_argument("--theta-hetero-max", type=float, default=0.55)
    parser.add_argument("--hetero-steps", type=int, default=2)
    parser.add_argument("--delay-jitter-min", type=float, default=0.25)
    parser.add_argument("--delay-jitter-max", type=float, default=0.40)
    parser.add_argument("--delay-steps", type=int, default=2)

    # OU / slow drive
    parser.add_argument("--use-ou-noise", action="store_true", default=True,
                        help="Enable OU colored noise.")
    parser.add_argument("--ou-tau", type=float, default=80.0)
    parser.add_argument("--ou-sigma", type=float, default=0.45)
    parser.add_argument("--slow-drive-sigma", type=float, default=0.30)
    parser.add_argument("--slow-drive-tau", type=float, default=800.0)

    args = parser.parse_args()

    if args.quick:
        args.coupling_steps = min(args.coupling_steps, 3)
        args.i_ext_steps = min(args.i_ext_steps, 3)
        args.ratio_steps = min(args.ratio_steps, 2)
        args.noise_steps = min(args.noise_steps, 2)
        args.hetero_steps = 1
        args.delay_steps = 1
        args.duration = min(args.duration, 1400.0)
        args.transient = min(args.transient, 200.0)
        args.save_interval = max(args.save_interval, 4)

    return args


def adapt_search_space(args: argparse.Namespace, best_metrics: Dict[str, float]):
    """Adjust search bounds based on observed best metrics to steer toward targets."""
    if not best_metrics:
        return

    fc = best_metrics.get("fc", 0)
    meta = best_metrics.get("metastability", 0)

    # If FC too low, increase coupling range and reduce heterogeneity/jitter
    if fc < 0.25:
        args.coupling_stop = min(args.coupling_stop * 1.3, 1.2)
        args.coupling_start = min(args.coupling_start + 0.05, args.coupling_stop * 0.8)
        args.i_ext_hetero_max = max(0.15, args.i_ext_hetero_max * 0.85)
        args.theta_hetero_max = max(0.25, args.theta_hetero_max * 0.9)
        args.delay_jitter_max = max(args.delay_jitter_min, args.delay_jitter_max * 0.9)

    # If FC too high, back off coupling and noise/hetero tweaks
    if fc > 0.65:
        args.coupling_stop = max(0.2, args.coupling_stop * 0.8)
        args.noise_min = min(0.3, args.noise_min + 0.02)

    # If metastability too low, increase slow drive and noise slightly
    if meta < 0.01:
        args.slow_drive_sigma = min(0.6, args.slow_drive_sigma * 1.2)
        args.noise_max = min(0.35, args.noise_max + 0.02)
        args.delay_jitter_max = min(0.6, args.delay_jitter_max * 1.05)

    # Keep ranges sane
    args.coupling_start = max(0.02, min(args.coupling_start, args.coupling_stop - 0.05))
    args.coupling_stop = min(1.2, max(args.coupling_stop, args.coupling_start + 0.05))
    args.noise_min = max(0.05, min(args.noise_min, args.noise_max))
    args.noise_max = min(0.6, max(args.noise_max, args.noise_min + 0.01))
    args.i_ext_hetero_min = max(0.0, min(args.i_ext_hetero_min, args.i_ext_hetero_max))
    args.theta_hetero_min = max(0.0, min(args.theta_hetero_min, args.theta_hetero_max))


def main():
    args = parse_args()

    print("\n╔" + "═"*68 + "╗")
    print("║" + " "*18 + "VR BRAIN AUTO-TUNER" + " "*24 + "║")
    print("╚" + "═"*68 + "╝")

    print("\nBuilding brain ...")
    brain = create_default_brain(68)

    base_cfg = {
        "dt": args.dt,
        "duration": args.duration,
        "transient": args.transient,
        "save_interval": args.save_interval,
        "conduction_velocity": args.conduction_velocity,
    }

    best_overall = (-1e9, None, None)

    round_idx = 0
    max_rounds = args.max_rounds if not args.until_target else 10_000_000

    while True:
        round_idx += 1
        if round_idx > max_rounds:
            break

        print(f"\n=== ROUND {round_idx}/{args.max_rounds} ===")
        combos = generate_combos(args, round_idx)
        start = time.time()
        best_score, best_metrics, best_params = search_round(brain, combos, base_cfg, args)
        elapsed = time.time() - start
        print(f"Round {round_idx} complete in {elapsed:.1f}s | best score {best_score:.2f}")

        if best_score > best_overall[0]:
            best_overall = (best_score, best_metrics, best_params)

        adapt_search_space(args, best_metrics)

        if best_score >= args.target_score:
            print("\n🎯 Target score reached. Stopping.")
            break

    final_score, final_metrics, final_params = best_overall
    if final_params:
        print("\n" + "="*70)
        print(f"BEST FOUND (score {final_score:.2f}):")
        print(f"  ratio={final_params['ratio']:.2f}, coupling={final_params['coupling']:.3f}, "
              f"I_ext={final_params['i_ext']:.2f}, noise={final_params['noise_strength']:.2f}")
        print(f"  hetero I_ext={final_params['i_ext_hetero']:.2f}, "
              f"theta={final_params['theta_hetero']:.2f}, delay={final_params['delay_jitter']:.2f}")
        print(f"  OU={final_params['use_ou']}, ou_tau={final_params['ou_tau']:.1f}, "
              f"ou_sigma={final_params['ou_sigma']:.2f}, slow_sigma={final_params['slow_drive_sigma']:.2f}")
        print(f"Metrics: mean={final_metrics['mean']:.3f}, std={final_metrics['std']:.3f}, "
              f"FC={final_metrics['fc']:.3f}, meta={final_metrics['metastability']:.3f}, "
              f"specH={final_metrics['spectral_entropy']:.3f}")
        print("="*70)

        if args.apply:
            apply_params(final_params)
            print("\n✅ Parameters applied to simulator_fast.py")
        else:
            print("\n💡 To apply these defaults: python auto_tuner.py --apply")
    else:
        print("\n❌ No valid parameters found.")


if __name__ == "__main__":
    main()
