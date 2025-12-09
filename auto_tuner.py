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
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from scipy import signal

# Avoid matplotlib cache permission issues on macOS sandbox
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache")))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
    """
    Higher is better. Centered on target bands.
    Scaled to keep rewards mostly positive when near targets.
    """
    score = 5.0  # base offset to keep values positive

    mean_val = metrics["mean"]
    if 0.35 <= mean_val <= 0.55:
        score += 1.0
    else:
        score -= 2.0 * abs(mean_val - 0.45) / 0.15
    if mean_val > 0.70 or mean_val < 0.25:
        score -= 3.0

    std_val = metrics["std"]
    if 0.10 <= std_val <= 0.25:
        score += 1.25
    else:
        score -= 1.5 * abs(std_val - 0.16) / 0.15

    fc_val = metrics["fc"]
    if 0.30 <= fc_val <= 0.60:
        score += 2.5
    else:
        score -= 2.0 * abs(fc_val - 0.45) / 0.4
    if fc_val < 0.10:
        score -= 2.0

    meta_val = metrics["metastability"]
    if 0.01 <= meta_val <= 0.08:
        score += 2.0
    else:
        score -= 2.0 * abs(meta_val - 0.04) / 0.08
    if meta_val < 0.005:
        score -= 1.5

    spec_val = metrics["spectral_entropy"]
    if np.isfinite(spec_val):
        score += 2.0 * max(0.0, spec_val - 0.5)  # reward higher entropy above 0.5

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
        global_noise_frac=params["global_noise_frac"],
        sin_drive_amp=params["sin_amp"],
        sin_drive_freq_hz=params["sin_freq"],
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
    """Generate parameter combinations using a CMA-ES-like sampler."""
    bounds = get_bounds(args)

    keys = list(bounds.keys())
    dim = len(keys)

    # Initialize center/sigma if not present
    if not hasattr(args, "_cma_center"):
        args._cma_center = np.array([(bounds[k][0] + bounds[k][1]) / 2 for k in keys])
        args._cma_sigma = np.array([(bounds[k][1] - bounds[k][0]) * 0.15 for k in keys])

    center = args._cma_center
    sigma = args._cma_sigma

    lambda_samples = max(8, args.jobs * 2) if args.jobs > 0 else 12
    combos = []
    seed_base = args.seed if args.seed is not None else np.random.randint(1_000_000)

    for i in range(lambda_samples):
        z = np.random.randn(dim)
        x = center + sigma * z
        params_vec = np.clip(x, [bounds[k][0] for k in keys], [bounds[k][1] for k in keys])
        params = dict(zip(keys, params_vec))
        params.update({
            "c_ie_base": args.c_ie_base,
            "use_ou": args.use_ou_noise,
            "ou_tau": args.ou_tau,
            "ou_sigma": args.ou_sigma,
            "slow_drive_tau": args.slow_drive_tau,
            "global_noise_frac": args.global_noise_frac,
            "sin_amp": args.sin_amp,
            "sin_freq": args.sin_freq,
            "seed": seed_base + i + round_idx * 1000,
        })
        combos.append(params)

    # Store keys for later update
    args._cma_keys = keys
    args._cma_bounds = bounds
    return combos


def update_cma(args: argparse.Namespace, scored: List[Tuple[float, Dict]]):
    """Update CMA-like center and sigma based on top samples."""
    keys = args._cma_keys
    bounds = args._cma_bounds
    center = args._cma_center
    sigma = args._cma_sigma

    # Sort by score descending and take top mu
    scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
    mu = max(4, len(scored_sorted) // 3)
    top = scored_sorted[:mu]
    if not top:
        return
    weight = np.linspace(1, 0.1, mu)
    weight /= weight.sum()

    top_vecs = []
    for _, params in top:
        vec = np.array([params[k] for k in keys])
        top_vecs.append(vec)
    top_vecs = np.vstack(top_vecs)

    new_center = np.average(top_vecs, axis=0, weights=weight)
    # Adapt sigma based on spread of top samples
    spread = np.std(top_vecs, axis=0)
    new_sigma = 0.7 * sigma + 0.3 * spread

    # Clip
    low = np.array([bounds[k][0] for k in keys])
    high = np.array([bounds[k][1] for k in keys])
    args._cma_center = np.clip(new_center, low, high)
    args._cma_sigma = np.clip(new_sigma, (high - low) * 0.02, (high - low) * 0.25)


def search_round(brain: Dict, combos: List[Dict], base_cfg: Dict, args: argparse.Namespace):
    """Evaluate one round of combos in parallel and return best."""
    best_score = -1e9
    best_metrics = None
    best_params = None
    scored_samples: List[Tuple[float, Dict]] = []
    round_records: List[Dict] = []

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
            scored_samples.append((score, params))
            round_records.append({"score": score, "metrics": metrics, "params": params})

    update_cma(args, scored_samples)
    return best_score, best_metrics, best_params, round_records


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
    parser.add_argument("--log-dir", type=str, default="tuner_logs",
                        help="Directory to save round logs/plots.")
    parser.add_argument("--no-plots", action="store_true", help="Disable per-round plots.")
    parser.add_argument("--num-regions", type=int, default=68, help="Number of brain regions.")
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
    parser.add_argument("--global-noise-frac", type=float, default=0.05,
                        help="Fraction of noise that is global/shared.")
    parser.add_argument("--sin-amp", type=float, default=0.05,
                        help="Amplitude of sinusoidal drive.")
    parser.add_argument("--sin-freq", type=float, default=10.0,
                        help="Frequency of sinusoidal drive (Hz).")

    # PBT options
    parser.add_argument("--pbt", action="store_true", help="Use PBT-style search instead of CMA.")
    parser.add_argument("--pbt-pop", type=int, default=12, help="Population size for PBT.")
    parser.add_argument("--pbt-gens", type=int, default=8, help="Number of PBT generations.")
    parser.add_argument("--pbt-elite-frac", type=float, default=0.25, help="Elite fraction for PBT.")
    parser.add_argument("--pbt-mutate-scale", type=float, default=0.08,
                        help="Mutation scale as fraction of param range.")

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


def get_bounds(args: argparse.Namespace) -> Dict[str, Tuple[float, float]]:
    """Bounds for parameters."""
    return {
        "ratio": (args.ratio_start, args.ratio_stop),
        "coupling": (args.coupling_start, args.coupling_stop),
        "i_ext": (args.i_ext_start, args.i_ext_stop),
        "noise_strength": (args.noise_min, args.noise_max),
        "i_ext_hetero": (args.i_ext_hetero_min, args.i_ext_hetero_max),
        "theta_hetero": (args.theta_hetero_min, args.theta_hetero_max),
        "delay_jitter": (args.delay_jitter_min, args.delay_jitter_max),
        "slow_drive_sigma": (args.slow_drive_sigma * 0.7, args.slow_drive_sigma * 1.3),
        "global_noise_frac": (0.0, 0.2),
        "sin_amp": (0.0, max(0.2, args.sin_amp)),
        "sin_freq": (6.0, 14.0),
    }


def adapt_search_space(args: argparse.Namespace, best_metrics: Dict[str, float]):
    """Adjust search bounds based on observed best metrics to steer toward targets."""
    if not best_metrics:
        return

    fc = best_metrics.get("fc", 0)
    meta = best_metrics.get("metastability", 0)
    mean_val = best_metrics.get("mean", 0)

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

    # Mean too high -> reduce I_ext and coupling upper
    if mean_val > 0.65:
        args.i_ext_stop = max(args.i_ext_start + 0.05, args.i_ext_stop * 0.9)
        args.coupling_stop = max(args.coupling_start + 0.05, args.coupling_stop * 0.9)
        args.noise_min = min(args.noise_min + 0.01, args.noise_max)

    # Mean too low -> increase I_ext and coupling lower bound
    if mean_val < 0.30:
        args.i_ext_start = min(args.i_ext_start + 0.05, args.i_ext_stop - 0.05)
        args.coupling_start = min(args.coupling_start + 0.05, args.coupling_stop - 0.05)

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

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("\n╔" + "═"*68 + "╗")
    print("║" + " "*18 + "VR BRAIN AUTO-TUNER" + " "*24 + "║")
    print("╚" + "═"*68 + "╝")

    print("\nBuilding brain ...")
    brain = create_default_brain(args.num_regions)

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

    if args.pbt:
        best_overall = run_pbt(brain, args, base_cfg, log_dir, max_rounds)
    else:
        while True:
            round_idx += 1
            if round_idx > max_rounds:
                break

            print(f"\n=== ROUND {round_idx}/{args.max_rounds if not args.until_target else '∞'} ===")
            combos = generate_combos(args, round_idx)
            start = time.time()
            best_score, best_metrics, best_params, round_records = search_round(brain, combos, base_cfg, args)
            elapsed = time.time() - start
            print(f"Round {round_idx} complete in {elapsed:.1f}s | best score {best_score:.2f}")

            # Log round data
            round_path = log_dir / f"round_{round_idx}.json"
            with open(round_path, "w") as f:
                json.dump(round_records, f, indent=2)

            if not args.no_plots:
                save_round_plot(round_records, round_idx, log_dir)

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


def save_round_plot(records: List[Dict], round_idx: int, log_dir: Path):
    """Save scatter of FC vs metastability for a round."""
    if not records:
        return
    fc = [r["metrics"]["fc"] for r in records]
    meta = [r["metrics"]["metastability"] for r in records]
    scores = [r["score"] for r in records]

    plt.figure(figsize=(5, 4))
    sc = plt.scatter(fc, meta, c=scores, cmap="viridis", s=60, edgecolors="k")
    plt.colorbar(sc, label="Score")
    plt.xlabel("FC mean")
    plt.ylabel("Metastability")
    plt.title(f"Round {round_idx}")
    plt.axvspan(0.30, 0.60, color="green", alpha=0.1, label="Target FC")
    plt.axhspan(0.01, 0.08, color="orange", alpha=0.1, label="Target meta")
    plt.legend(loc="upper right", fontsize=8)
    out_path = log_dir / f"round_{round_idx}_fc_meta.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def run_pbt(brain: Dict, args: argparse.Namespace, base_cfg: Dict,
            log_dir: Path, max_rounds: int):
    """Simple PBT-style loop with exploit/explore."""
    bounds = get_bounds(args)
    keys = list(bounds.keys())

    def sample_params():
        p = {}
        for k, (lo, hi) in bounds.items():
            p[k] = float(np.random.uniform(lo, hi))
        p.update({
            "c_ie_base": args.c_ie_base,
            "use_ou": args.use_ou_noise,
            "ou_tau": args.ou_tau,
            "ou_sigma": args.ou_sigma,
            "slow_drive_tau": args.slow_drive_tau,
            "global_noise_frac": args.global_noise_frac,
            "sin_amp": args.sin_amp,
            "sin_freq": args.sin_freq,
            "seed": np.random.randint(1_000_000),
        })
        return p

    def mutate(base: Dict) -> Dict:
        new = base.copy()
        for k in keys:
            lo, hi = bounds[k]
            span = hi - lo
            new[k] = float(np.clip(new[k] + np.random.randn() * args.pbt_mutate_scale * span, lo, hi))
        return new

    pop = [sample_params() for _ in range(args.pbt_pop)]
    best_overall = (-1e9, None, None)

    for gen in range(1, args.pbt_gens + 1):
        print(f"\n=== PBT GEN {gen}/{args.pbt_gens} ===")
        records = []

        with ThreadPoolExecutor(max_workers=args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 2) - 1)) as ex:
            futures = [ex.submit(run_combo, brain, p, base_cfg) for p in pop]
            for fut in as_completed(futures):
                try:
                    score, metrics, params = fut.result()
                except Exception as exc:  # pragma: no cover
                    print(f"   ! combo failed: {exc}")
                    continue
                records.append({"score": score, "metrics": metrics, "params": params})
                if score > best_overall[0]:
                    best_overall = (score, metrics, params)
                    print(f"✓ New best (score {score:.2f}) "
                          f"coupling={params['coupling']:.3f}, I_ext={params['i_ext']:.2f}, "
                          f"FC={metrics['fc']:.3f}, meta={metrics['metastability']:.3f}, "
                          f"mean={metrics['mean']:.3f}")

        # Logging/plot
        round_path = log_dir / f"pbt_gen_{gen}.json"
        with open(round_path, "w") as f:
            json.dump(records, f, indent=2)
        if not args.no_plots:
            save_round_plot(records, gen, log_dir)

        if best_overall[0] >= args.target_score:
            print("\n🎯 Target score reached. Stopping PBT.")
            break

        # Exploit/explore
        records_sorted = sorted(records, key=lambda x: x["score"], reverse=True)
        elite_n = max(1, int(args.pbt_pop * args.pbt_elite_frac))
        elites = records_sorted[:elite_n] if records_sorted else []
        if not elites:
            pop = [sample_params() for _ in range(args.pbt_pop)]
            continue
        new_pop = [rec["params"] for rec in elites]
        while len(new_pop) < args.pbt_pop:
            parent = elites[np.random.randint(0, elite_n)]["params"]
            child = mutate(parent)
            new_pop.append(child)
        pop = new_pop

    return best_overall


if __name__ == "__main__":
    main()
