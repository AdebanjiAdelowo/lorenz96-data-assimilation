"""Performance benchmark (brief section 19): cost per forecast-analysis cycle and scaling with
ensemble size. Lorenz-96 is inexpensive (this is exactly what makes the many-seed/many-configuration
studies elsewhere in this project affordable), so this script is a secondary sanity check, not the
project's main focus -- consistent with the brief's own instruction to prioritise statistically
meaningful repeated experiments over micro-optimising code.

Usage: python scripts/performance_benchmark.py [--config smoke|local|full]
"""
from __future__ import annotations

import argparse
import pathlib
import platform
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.l96 import L96Params
from src.observations import make_observation_operator
from src.experiment import run_assimilation_experiment

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


def get_hardware_info() -> str:
    import subprocess
    cpu = "unknown"
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        pass
    return (f"Platform: {platform.platform()}\nCPU: {cpu}\nPython: {platform.python_version()}\n"
            f"NumPy: {np.__version__}\nPrecision: float64, CPU only, single-threaded\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    base, s = cfg["base"], cfg["performance"]

    params = L96Params(K=base["K"], F=base["F"], dt=base["dt"])
    obs_op = make_observation_operator(base["K"], stride=1, obs_std=1.0)

    lines = [get_hardware_info(),
             f"K={base['K']}, {s['n_cycles']} cycles/run, {s['n_warmup']} warm-up run(s) discarded, "
             f"{s['n_repeats']} timed repeats", "",
             f"{'Ne':>6} {'median s/cycle':>16} {'min':>10} {'max':>10} {'total s (median run)':>22}"]

    Ne_values, med_per_cycle = [], []
    for Ne in s["Ne_values"]:
        def call():
            run_assimilation_experiment(params, obs_op, Ne=Ne, obs_interval_steps=10,
                                         n_cycles=s["n_cycles"], ens_init_std=1.0, seed=0,
                                         t_spinup=base["t_spinup"])
        for _ in range(s["n_warmup"]):
            call()
        times = []
        for _ in range(s["n_repeats"]):
            t0 = time.perf_counter()
            call()
            times.append(time.perf_counter() - t0)
        med, mn, mx = np.median(times), np.min(times), np.max(times)
        Ne_values.append(Ne)
        med_per_cycle.append(med / s["n_cycles"])
        lines.append(f"{Ne:6d} {med/s['n_cycles']:16.5f} {mn/s['n_cycles']:10.5f} "
                     f"{mx/s['n_cycles']:10.5f} {med:22.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"performance_benchmark_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(Ne_values, med_per_cycle, "o-", label="measured")
    ref = med_per_cycle[0] * (np.array(Ne_values) / Ne_values[0])
    ax.loglog(Ne_values, ref, "k--", alpha=0.5, label=r"$O(N_e)$ reference")
    ax.set_xlabel("ensemble size $N_e$")
    ax.set_ylabel("median seconds per forecast-analysis cycle")
    ax.set_title(f"Runtime scaling with ensemble size ({args.config})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"performance_scaling_{args.config}.png", dpi=150)

    print(f"\nWrote figures/performance_scaling_{args.config}.png and "
          f"results/performance_benchmark_{args.config}.txt")


if __name__ == "__main__":
    main()
