"""Observation-density study (brief section 10): vary the fraction of the state observed (stride),
holding Ne, noise, and interval fixed at a "moderately working" configuration (see configs/*.yaml
comment) so the density effect is visible on its own, not swamped by ensemble collapse. Shows how
unobserved components are inferred through the ensemble's dynamical (flow-dependent) covariance.

Usage: python scripts/observation_density_study.py [--config smoke|local|full]
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
    base, s = cfg["base"], cfg["obs_density_study"]

    params = L96Params(K=base["K"], F=base["F"], dt=base["dt"])
    seeds = list(range(s["n_seeds"]))

    lines = [f"Observation-density study ({args.config} config): Ne={s['Ne']}, "
             f"obs_std={s['obs_std']}, {s['n_seeds']} seeds, {s['n_cycles']} cycles", "",
             f"{'stride':>7} {'m/K observed':>14} {'analysis RMSE (mean+-std)':>28} "
             f"{'spread (mean+-std)':>22}"]

    strides, a_means, a_stds = [], [], []
    for stride in s["strides"]:
        obs_op = make_observation_operator(base["K"], stride=stride, obs_std=s["obs_std"])
        out = repeated_experiment(seeds, params=params, obs_op=obs_op, Ne=s["Ne"],
                                   obs_interval_steps=s["obs_interval_steps"], n_cycles=s["n_cycles"],
                                   ens_init_std=s["ens_init_std"], t_spinup=base["t_spinup"])
        a, sp = out["analysis_rmse"], out["analysis_spread"]
        strides.append(stride); a_means.append(a.mean); a_stds.append(a.std)
        m_of_K = f"{obs_op.m}/{base['K']}"
        lines.append(f"{stride:7d} {m_of_K:>14} {a.mean:12.4f}+-{a.std:<12.4f} "
                     f"{sp.mean:9.4f}+-{sp.std:<9.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"obs_density_study_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(strides, a_means, yerr=a_stds, marker="o")
    ax.set_xlabel("observation stride (1 = every state observed)")
    ax.set_ylabel("time-averaged analysis RMSE")
    ax.set_title(f"RMSE vs. observation density ({args.config})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"obs_density_study_{args.config}.png", dpi=150)

    print(f"\nWrote figures/obs_density_study_{args.config}.png and "
          f"results/obs_density_study_{args.config}.txt")


if __name__ == "__main__":
    main()
