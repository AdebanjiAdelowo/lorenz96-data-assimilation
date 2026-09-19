"""Ensemble-size study (brief section 9): analysis/forecast RMSE, spread, and runtime vs. ensemble
size, over multiple independent seeds (mean +/- std). No localization/inflation, matching the
baseline, so this isolates the finite-ensemble sampling effect the baseline experiment demonstrated.

Usage: python scripts/ensemble_size_study.py [--config smoke|local|full]
"""
from __future__ import annotations

import argparse
import pathlib
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
from src.experiment import repeated_experiment

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    base, s = cfg["base"], cfg["ensemble_size_study"]

    params = L96Params(K=base["K"], F=base["F"], dt=base["dt"])
    obs_op = make_observation_operator(base["K"], stride=s["stride"], obs_std=s["obs_std"])
    seeds = list(range(s["n_seeds"]))

    lines = [f"Ensemble-size study ({args.config} config): {s['n_seeds']} seeds, "
             f"stride={s['stride']}, obs_std={s['obs_std']}, {s['n_cycles']} cycles", "",
             f"{'Ne':>5} {'analysis RMSE (mean+-std)':>28} {'forecast RMSE (mean+-std)':>28} "
             f"{'spread (mean+-std)':>22} {'runtime(s)':>11} {'diverged frac':>14}"]

    Ne_values = s["Ne_values"]
    a_means, a_stds, spread_means = [], [], []
    for Ne in Ne_values:
        t0 = time.perf_counter()
        out = repeated_experiment(seeds, params=params, obs_op=obs_op, Ne=Ne,
                                   obs_interval_steps=s["obs_interval_steps"], n_cycles=s["n_cycles"],
                                   ens_init_std=s["ens_init_std"], t_spinup=base["t_spinup"])
        runtime = (time.perf_counter() - t0) / len(seeds)
        a, f, sp = out["analysis_rmse"], out["forecast_rmse"], out["analysis_spread"]
        a_means.append(a.mean); a_stds.append(a.std); spread_means.append(sp.mean)
        lines.append(f"{Ne:5d} {a.mean:12.4f}+-{a.std:<12.4f} {f.mean:12.4f}+-{f.std:<12.4f} "
                     f"{sp.mean:9.4f}+-{sp.std:<9.4f} {runtime:11.3f} {out['diverged_fraction']:14.2f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"ensemble_size_study_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(Ne_values, a_means, yerr=a_stds, marker="o", label="analysis RMSE")
    ax.plot(Ne_values, spread_means, marker="s", label="analysis spread")
    ax.set_xlabel("ensemble size $N_e$")
    ax.set_ylabel("value")
    ax.set_yscale("log")
    ax.set_title(f"RMSE and spread vs. ensemble size ({args.config})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"ensemble_size_study_{args.config}.png", dpi=150)

    print(f"\nWrote figures/ensemble_size_study_{args.config}.png and "
          f"results/ensemble_size_study_{args.config}.txt")


if __name__ == "__main__":
    main()
