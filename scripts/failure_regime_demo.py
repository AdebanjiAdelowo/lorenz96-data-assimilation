"""Failure-regime demonstration (brief section 16): an ensemble small enough that even Gaspari-Cohn
localization (radius chosen from the localization study) cannot fully rescue it, tracked over many
cycles to show PERSISTENT filter collapse -- ensemble spread crashes to a small, overconfident value
while analysis RMSE stabilises at a large, uncorrected level (the filter has stopped trusting new
observations enough to correct itself), a well-documented EnKF pathology (distinct from, and more
representative of typical EnKF failure than, literal numerical blow-up).

Usage: python scripts/failure_regime_demo.py [--config smoke|local|full]
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
from src.localization import localization_matrix
from src.experiment import run_assimilation_experiment, time_averaged_rmse

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    base, s = cfg["base"], cfg["failure_regime"]

    params = L96Params(K=base["K"], F=base["F"], dt=base["dt"])
    obs_op = make_observation_operator(base["K"], stride=s["stride"], obs_std=s["obs_std"])
    loc_xy, loc_yy = localization_matrix(base["K"], obs_op.obs_indices, radius=s["loc_radius"])

    lines = [f"Failure-regime demonstration ({args.config} config): very small ensembles, "
             f"WITH localization (radius={s['loc_radius']}) -- even localization cannot fully "
             f"compensate for too few members", "",
             f"{'Ne':>5} {'diverged':>9} {'time-avg analysis RMSE':>24} {'final spread':>13}"]

    fig, axes = plt.subplots(len(s["Ne_values"]), 1, figsize=(8, 2.3 * len(s["Ne_values"])),
                              sharex=True)
    if len(s["Ne_values"]) == 1:
        axes = [axes]

    for ax, Ne in zip(axes, s["Ne_values"]):
        res = run_assimilation_experiment(params, obs_op, Ne=Ne,
                                           obs_interval_steps=s["obs_interval_steps"],
                                           n_cycles=s["n_cycles"], ens_init_std=s["ens_init_std"],
                                           seed=0, loc_xy=loc_xy, loc_yy=loc_yy,
                                           t_spinup=base["t_spinup"])
        a_rmse, _ = time_averaged_rmse(res)
        lines.append(f"{Ne:5d} {str(res.diverged):>9} {a_rmse:24.4f} {res.analysis_spread[-1]:13.4f}")

        ax.plot(res.t, res.analysis_rmse, label="analysis RMSE")
        ax.plot(res.t, res.analysis_spread, label="analysis spread")
        ax.set_ylabel(f"Ne={Ne}")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("t")
    fig.suptitle(f"Failure regime: RMSE/spread mismatch persists at very small Ne "
                 f"even with localization ({args.config})")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"failure_regime_{args.config}.png", dpi=150)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"failure_regime_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    print(f"\nWrote figures/failure_regime_{args.config}.png and "
          f"results/failure_regime_{args.config}.txt")


if __name__ == "__main__":
    main()
