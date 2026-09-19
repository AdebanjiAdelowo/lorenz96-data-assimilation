"""Observation-frequency study (brief section 11): vary the interval between observations, showing
performance deteriorate as the (chaotic) system evolves freely for longer between analyses -- directly
related to the Lyapunov-exponent-driven error growth quantified in scripts/estimate_lyapunov.py.

Usage: python scripts/observation_frequency_study.py [--config smoke|local|full]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    base, s = cfg["base"], cfg["obs_frequency_study"]

    params = L96Params(K=base["K"], F=base["F"], dt=base["dt"])
    obs_op = make_observation_operator(base["K"], stride=s["stride"], obs_std=s["obs_std"])
    seeds = list(range(s["n_seeds"]))

    lines = [f"Observation-frequency study ({args.config} config): Ne={s['Ne']}, stride={s['stride']}, "
             f"obs_std={s['obs_std']}, {s['n_seeds']} seeds", "",
             f"{'interval steps':>15} {'interval (time)':>16} {'analysis RMSE (mean+-std)':>28} "
             f"{'spread (mean+-std)':>22}"]

    intervals, times, a_means, a_stds = [], [], [], []
    for interval_steps in s["intervals"]:
        n_cycles = max(20, int(round(s["n_cycles"] * s["intervals"][0] / interval_steps)))
        out = repeated_experiment(seeds, params=params, obs_op=obs_op, Ne=s["Ne"],
                                   obs_interval_steps=interval_steps, n_cycles=n_cycles,
                                   ens_init_std=s["ens_init_std"], t_spinup=base["t_spinup"])
        a, sp = out["analysis_rmse"], out["analysis_spread"]
        t_units = interval_steps * params.dt
        intervals.append(interval_steps); times.append(t_units)
        a_means.append(a.mean); a_stds.append(a.std)
        lines.append(f"{interval_steps:15d} {t_units:16.3f} {a.mean:12.4f}+-{a.std:<12.4f} "
                     f"{sp.mean:9.4f}+-{sp.std:<9.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"obs_frequency_study_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(times, a_means, yerr=a_stds, marker="o")
    ax.set_xlabel("observation interval (model time units)")
    ax.set_ylabel("time-averaged analysis RMSE")
    ax.set_title(f"RMSE vs. observation interval ({args.config})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"obs_frequency_study_{args.config}.png", dpi=150)

    print(f"\nWrote figures/obs_frequency_study_{args.config}.png and "
          f"results/obs_frequency_study_{args.config}.txt")


if __name__ == "__main__":
    main()
