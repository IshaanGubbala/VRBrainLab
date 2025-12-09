#!/usr/bin/env python3
"""
auto_tuner.py - Two-phase tuner (LHS explore → local refine) with live heatmap

Phase 1: Latin Hypercube samples the space evenly to find good regions.
Phase 2: Local Nelder-Mead refinement around the best point.

Noise handling: tightened bounds, deterministic seed, optional torch backend
(single-process to avoid Mac pickling issues). A live heatmap (global_coupling
vs I_ext colored by loss) is saved to the log directory after each evaluation.

Usage:
    python tools/auto_tuner.py --lhs-samples 40 --refine-iters 60 --log-dir tuner_logs
    python tools/auto_tuner.py --quick                   # 20 LHS + 30 refine, shorter sims
    python tools/auto_tuner.py --apply                   # write best params into core/simulator_fast.py
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Ensure project root on path for local imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import create_default_brain
from core.simulator_fast import BrainNetworkSimulator, SimulationConfig


# ============================================================================
# TARGET DYNAMICS (What we want to achieve)
# ============================================================================
TARGETS = {
    "mean": 0.45,
    "std": 0.18,
    "fc_mean": 0.45,
    "metastability": 0.04,
    "max_activity": 0.80,
}

WEIGHTS = {
    "mean": 10.0,
    "std": 5.0,
    "fc_mean": 8.0,
    "metastability": 6.0,
    "saturation": 15.0,
}


# ============================================================================
# PARAMETER BOUNDS (tight, biologically plausible)
# ============================================================================
PARAM_BOUNDS = {
    "global_coupling": (0.65, 0.85),
    "I_ext": (0.65, 0.85),
    "c_ee": (7.5, 10.5),
    "c_ie": (18.0, 22.0),
    "noise_strength": (0.12, 0.18),
    "theta_e": (3.2, 3.8),
    "slow_drive_sigma": (0.50, 0.70),
    "delay_jitter_pct": (0.10, 0.15),
}


# ---------------------------------------------------------------------------
# Metric and loss utilities
# ---------------------------------------------------------------------------
def compute_metrics(activity: np.ndarray) -> Dict[str, float]:
    mean = float(np.mean(activity))
    std = float(np.std(activity))
    max_act = float(np.max(activity))

    fc_matrix = np.corrcoef(activity.T)  # Regions x regions
    upper_tri = fc_matrix[np.triu_indices_from(fc_matrix, k=1)]
    fc_mean = float(np.nanmean(upper_tri))

    sync_timeseries = np.std(activity, axis=1)  # Synchrony at each timepoint
    metastability = float(np.std(sync_timeseries))

    return {
        "mean": mean,
        "std": std,
        "max_activity": max_act,
        "fc_mean": fc_mean,
        "metastability": metastability,
    }


def compute_loss(metrics: Dict[str, float]) -> float:
    loss = 0.0
    loss += WEIGHTS["mean"] * (metrics["mean"] - TARGETS["mean"]) ** 2
    loss += WEIGHTS["std"] * (metrics["std"] - TARGETS["std"]) ** 2
    loss += WEIGHTS["fc_mean"] * (metrics["fc_mean"] - TARGETS["fc_mean"]) ** 2
    loss += WEIGHTS["metastability"] * (metrics["metastability"] - TARGETS["metastability"]) ** 2

    if metrics["max_activity"] > TARGETS["max_activity"]:
        loss += WEIGHTS["saturation"] * (metrics["max_activity"] - TARGETS["max_activity"]) ** 2

    return float(loss)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_params(params: np.ndarray, brain: Dict, duration: float = 1500.0, backend: str = "numpy") -> Tuple[float, Dict]:
    global_coupling, I_ext, c_ee, c_ie, noise, theta_e, slow_drive_sigma, delay_jitter = params

    config = SimulationConfig(
        duration=duration,
        transient=200.0,
        dt=0.1,
        global_coupling=float(global_coupling),
        I_ext=float(I_ext),
        c_ee=float(c_ee),
        c_ie=float(c_ie),
        noise_strength=float(noise),
        theta_e=float(theta_e),
        slow_drive_sigma=float(slow_drive_sigma),
        delay_jitter_pct=float(delay_jitter),
        backend=backend,
    )

    try:
        sim = BrainNetworkSimulator(brain, config, verbose=False)
        results = sim.run_simulation(suppress_output=True)
        activity = results["E"]
        metrics = compute_metrics(activity)
        loss = compute_loss(metrics)
        return loss, metrics
    except Exception as e:
        print(f"  ⚠️  Simulation failed: {e}")
        return 1e6, {}


# ---------------------------------------------------------------------------
# Sampling and logging
# ---------------------------------------------------------------------------
def lhs_samples(n_samples: int, bounds: List[Tuple[float, float]], rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sampling over bounds."""
    dim = len(bounds)
    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.random((n_samples, dim))
    a = cut[:n_samples]
    b = cut[1 : n_samples + 1]
    rdpoints = u * (b - a)[:, None] + a[:, None]  # shape (n_samples, dim)
    for j in range(dim):
        rng.shuffle(rdpoints[:, j])
    for j, (lo, hi) in enumerate(bounds):
        rdpoints[:, j] = lo + rdpoints[:, j] * (hi - lo)
    return rdpoints


class LogBook:
    """Tracks evaluations and saves live heatmaps."""

    def __init__(self, log_dir: Path):
        self.records: List[Dict] = []
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def add(self, stage: str, params: np.ndarray, loss: float, metrics: Dict[str, float]):
        rec = {
            "stage": stage,
            "params": [float(x) for x in params],
            "loss": float(loss),
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        self.records.append(rec)
        self._save_heatmap()

    def _save_heatmap(self):
        if not self.records:
            return
        xs = [r["params"][0] for r in self.records]
        ys = [r["params"][1] for r in self.records]
        zs = [r["loss"] for r in self.records]
        stages = [r["stage"] for r in self.records]

        plt.figure(figsize=(6, 5))
        vmin, vmax = min(zs), max(zs)
        for stage, marker in [("lhs", "o"), ("refine", "s")]:
            idx = [i for i, s in enumerate(stages) if s == stage]
            if idx:
                plt.scatter(
                    np.array(xs)[idx],
                    np.array(ys)[idx],
                    c=np.array(zs)[idx],
                    cmap="viridis",
                    marker=marker,
                    edgecolor="k",
                    s=60,
                    vmin=vmin,
                    vmax=vmax,
                    label=stage,
                )
        plt.xlabel("global_coupling")
        plt.ylabel("I_ext")
        plt.title("Loss heatmap (lower is better)")
        cb = plt.colorbar()
        cb.set_label("Loss")
        plt.legend()
        plt.tight_layout()
        out = self.log_dir / "heatmap_latest.png"
        plt.savefig(out)
        plt.close()


# ---------------------------------------------------------------------------
# Optimization driver
# ---------------------------------------------------------------------------
def run_optimization(
    brain: Dict,
    max_iters_lhs: int = 40,
    max_iters_refine: int = 60,
    duration: float = 1500.0,
    backend: str = "numpy",
    log_dir: Path = Path("tuner_logs"),
    seed: int = 42,
) -> Dict:
    print("\n🔧 Running two-phase optimization...")
    print(f"   LHS samples: {max_iters_lhs}")
    print(f"   Refine iterations: {max_iters_refine}")
    print(f"   Simulation duration: {duration} ms")

    bounds = [
        PARAM_BOUNDS[k]
        for k in [
            "global_coupling",
            "I_ext",
            "c_ee",
            "c_ie",
            "noise_strength",
            "theta_e",
            "slow_drive_sigma",
            "delay_jitter_pct",
        ]
    ]

    rng = np.random.default_rng(seed)
    logger = LogBook(log_dir)
    start_time = time.time()

    # Phase 1: LHS exploration
    lhs = lhs_samples(max_iters_lhs, bounds, rng)
    best_loss = float("inf")
    best_metrics: Dict[str, float] = {}
    best_params: Optional[np.ndarray] = None

    for i, p in enumerate(lhs, 1):
        loss, metrics = evaluate_params(p, brain, duration, backend=backend)
        logger.add("lhs", p, loss, metrics)
        if loss < best_loss:
            best_loss = float(loss)
            best_metrics = {k: float(v) for k, v in metrics.items()}
            best_params = np.array(p, dtype=float)
        print(f"LHS {i:03d}/{max_iters_lhs} | loss={loss:.3f} best={best_loss:.3f}", flush=True)

    if best_params is None:
        raise RuntimeError("No successful evaluations in LHS phase.")

    # Phase 2: Nelder-Mead refinement around best LHS point
    refine_count = 0
    refine_best = best_loss
    refine_best_params = best_params

    def nm_objective(x):
        nonlocal refine_count, refine_best, refine_best_params
        loss, metrics = evaluate_params(x, brain, duration, backend=backend)
        logger.add("refine", x, loss, metrics)
        refine_count += 1

        if loss < refine_best:
            refine_best = float(loss)
            refine_best_params = np.array(x, dtype=float)
            print(
                f"Refine {refine_count:03d}/{max_iters_refine} | loss={loss:.3f} best={refine_best:.3f} "
                f"(mean={metrics.get('mean', 0):.3f}, std={metrics.get('std', 0):.3f}, "
                f"fc={metrics.get('fc_mean', 0):.3f}, meta={metrics.get('metastability', 0):.3f})",
                flush=True,
            )
        elif refine_count % 5 == 0:
            print(f"Refine {refine_count:03d}/{max_iters_refine} | loss={loss:.3f} best={refine_best:.3f}", flush=True)
        return loss

    minimize(
        nm_objective,
        x0=best_params,
        method="Nelder-Mead",
        options={"maxiter": max_iters_refine, "xatol": 0.005, "fatol": 0.005, "adaptive": True},
    )

    elapsed = time.time() - start_time
    best_record = min(logger.records, key=lambda r: r["loss"])

    print(f"\n✅ Optimization complete in {elapsed:.1f}s")
    print(f"   Total evaluations: {len(logger.records)}")
    print(f"   Final loss: {best_record['loss']:.3f}")
    print(f"   Live heatmap: {log_dir / 'heatmap_latest.png'}")

    return {
        "params": {
            "global_coupling": best_record["params"][0],
            "I_ext": best_record["params"][1],
            "c_ee": best_record["params"][2],
            "c_ie": best_record["params"][3],
            "noise_strength": best_record["params"][4],
            "theta_e": best_record["params"][5],
            "slow_drive_sigma": best_record["params"][6],
            "delay_jitter_pct": best_record["params"][7],
        },
        "loss": best_record["loss"],
        "metrics": best_record["metrics"],
        "iterations": len(logger.records),
        "time": elapsed,
        "log_dir": str(log_dir),
    }


# ---------------------------------------------------------------------------
# Apply parameters
# ---------------------------------------------------------------------------
def apply_parameters(params: Dict):
    config_file = Path(__file__).resolve().parents[1] / "core" / "simulator_fast.py"

    updates = {
        "global_coupling": params["global_coupling"],
        "I_ext": params["I_ext"],
        "c_ee": params["c_ee"],
        "c_ie": params["c_ie"],
        "noise_strength": params["noise_strength"],
        "theta_e": params["theta_e"],
        "slow_drive_sigma": params.get("slow_drive_sigma", None),
        "delay_jitter_pct": params.get("delay_jitter_pct", None),
    }

    print(f"\n📝 Updating {config_file}...")
    with open(config_file, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        for param_name, value in updates.items():
            if value is None:
                continue
            if f"{param_name}: float" in line and "=" in line:
                comment = "#" + line.split("#")[1] if "#" in line else "\n"
                indent = len(line) - len(line.lstrip())
                lines[i] = f"{' ' * indent}{param_name}: float = {value:.3f}  {comment}"

    with open(config_file, "w") as f:
        f.writelines(lines)

    print("✅ Parameters updated!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Two-phase Brain Parameter Optimizer")
    parser.add_argument("--lhs-samples", type=int, default=40, help="Latin Hypercube samples (exploration)")
    parser.add_argument("--refine-iters", type=int, default=60, help="Nelder-Mead iterations (refinement)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 20 LHS, 30 refine, 800ms sims")
    parser.add_argument("--apply", action="store_true", help="Apply best parameters to core/simulator_fast.py")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy", help="Simulation backend")
    parser.add_argument("--log-dir", type=str, default="tuner_logs", help="Directory to save logs/plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.quick:
        lhs_samples = 20
        refine_iters = 30
        duration = 800.0
    else:
        lhs_samples = args.lhs_samples
        refine_iters = args.refine_iters
        duration = 1500.0

    print("╔" + "═" * 70 + "╗")
    print("║" + " " * 20 + "FAST PARAMETER OPTIMIZER" + " " * 25 + "║")
    print("╚" + "═" * 70 + "╝")

    print("\n🧠 Creating brain model...")
    brain = create_default_brain(68)

    print("\n🎯 Target Dynamics:")
    for key, val in TARGETS.items():
        print(f"   {key}: {val}")

    result = run_optimization(
        brain,
        max_iters_lhs=lhs_samples,
        max_iters_refine=refine_iters,
        duration=duration,
        backend=args.backend,
        log_dir=Path(args.log_dir),
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    print("\n📊 Best Parameters Found:")
    for key, val in result["params"].items():
        bounds = PARAM_BOUNDS[key]
        print(f"   {key:20s}: {val:6.3f}  (range: {bounds})")

    print("\n📈 Metrics Achieved:")
    for key, val in result["metrics"].items():
        target = TARGETS.get(key, "N/A")
        print(f"   {key:20s}: {val:6.3f}  (target: {target})")

    print(f"\n🎯 Final Loss: {result['loss']:.3f}")
    print(f"🖼️  Heatmap saved to: {result['log_dir']}/heatmap_latest.png")

    output_file = Path("tuner_results.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Results saved to {output_file}")

    if args.apply:
        apply_parameters(result["params"])
        print("\n✅ Run 'python tools/test.py' to verify the new parameters!")
    else:
        print("\n💡 Run with --apply to automatically update simulator_fast.py")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
