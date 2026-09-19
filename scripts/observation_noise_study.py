"""Observation-noise study (brief section 12): vary observation standard deviation R^0.5, measuring
analysis RMSE and ensemble spread. Lower observation noise does NOT automatically fix finite-ensemble
sampling problems (spurious correlations, rank deficiency) -- this is checked directly by also running
this sweep at the baseline's small ensemble size and comparing against the density study's larger,
"working" ensemble size.

Usage: python scripts/observation_noise_study.py [--config smoke|local|full]
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
    base, s = cfg["base"], cfg["obs_noise_study"]

    params = L96Params(K=base["K"], F=base["F"], dt=base["dt"])
    seeds = list(range(s["n_seeds"]))

    lines = [f"Observation-noise study ({args.config} config): Ne={s['Ne']}, stride={s['stride']}, "
             f"{s['n_seeds']} seeds, {s['n_cycles']} cycles", "",
             f"{'obs_std':>8} {'analysis RMSE (mean+-std)':>28} {'spread (mean+-std)':>22}"]

    stds, a_means, a_stds = [], [], []
    for obs_std in s["stds"]:
        obs_op = make_observation_operator(base["K"], stride=s["stride"], obs_std=obs_std)
        out = repeated_experiment(seeds, params=params, obs_op=obs_op, Ne=s["Ne"],
                                   obs_interval_steps=s["obs_interval_steps"], n_cycles=s["n_cycles"],
                                   ens_init_std=s["ens_init_std"], t_spinup=base["t_spinup"])
        a, sp = out["analysis_rmse"], out["analysis_spread"]
        stds.append(obs_std); a_means.append(a.mean); a_stds.append(a.std)
        lines.append(f"{obs_std:8.2f} {a.mean:12.4f}+-{a.std:<12.4f} {sp.mean:9.4f}+-{sp.std:<9.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"obs_noise_study_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(stds, a_means, yerr=a_stds, marker="o", label="analysis RMSE")
    ax.plot(stds, stds, "k:", alpha=0.5, label="RMSE = obs_std (reference)")
    ax.set_xlabel("observation noise std")
    ax.set_ylabel("time-averaged analysis RMSE")
    ax.set_title(f"RMSE vs. observation noise ({args.config})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"obs_noise_study_{args.config}.png", dpi=150)

    print(f"\nWrote figures/obs_noise_study_{args.config}.png and "
          f"results/obs_noise_study_{args.config}.txt")


if __name__ == "__main__":
    main()
