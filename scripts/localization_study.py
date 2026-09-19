"""Localization study (brief section 14): applied because the baseline experiment
(scripts/run_baseline_assimilation.py) demonstrated the need -- Ne=20 for a K=40 system, no
localization, shows severe spread collapse (spread << RMSE) and growing analysis RMSE. Compares
several Gaspari-Cohn radii against the unlocalized baseline, quantitatively.

Usage: python scripts/localization_study.py [--config smoke|local|full]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.l96 import L96Params
from src.observations import make_observation_operator
from src.localization import localization_matrix
from src.experiment import repeated_experiment

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    base, s = cfg["base"], cfg["localization_study"]

    params = L96Params(K=base["K"], F=base["F"], dt=base["dt"])
    obs_op = make_observation_operator(base["K"], stride=s["stride"], obs_std=s["obs_std"])
    seeds = list(range(s["n_seeds"]))
    common = dict(params=params, obs_op=obs_op, Ne=s["Ne"], obs_interval_steps=s["obs_interval_steps"],
                  n_cycles=s["n_cycles"], ens_init_std=s["ens_init_std"], t_spinup=base["t_spinup"])

    lines = [f"Localization study ({args.config} config): Ne={s['Ne']} (deliberately undersampled for "
             f"K={base['K']}), stride={s['stride']}, obs_std={s['obs_std']}, {s['n_seeds']} seeds", "",
             f"{'radius':>10} {'analysis RMSE (mean+-std)':>28} {'spread (mean+-std)':>22} "
             f"{'diverged frac':>14}"]

    out_none = repeated_experiment(seeds, **common)
    a0 = out_none["analysis_rmse"]
    lines.append(f"{'none':>10} {a0.mean:12.4f}+-{a0.std:<12.4f} "
                 f"{out_none['analysis_spread'].mean:9.4f}+-{out_none['analysis_spread'].std:<9.4f} "
                 f"{out_none['diverged_fraction']:14.2f}")

    radii, a_means, a_stds = [np.nan], [a0.mean], [a0.std]
    for radius in s["radii"]:
        loc_xy, loc_yy = localization_matrix(base["K"], obs_op.obs_indices, radius=radius)
        out = repeated_experiment(seeds, loc_xy=loc_xy, loc_yy=loc_yy, **common)
        a = out["analysis_rmse"]
        radii.append(radius); a_means.append(a.mean); a_stds.append(a.std)
        lines.append(f"{radius:10.1f} {a.mean:12.4f}+-{a.std:<12.4f} "
                     f"{out['analysis_spread'].mean:9.4f}+-{out['analysis_spread'].std:<9.4f} "
                     f"{out['diverged_fraction']:14.2f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"localization_study_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = ["none"] + [str(r) for r in s["radii"]]
    ax.bar(labels, a_means, yerr=a_stds, capsize=4)
    ax.set_xlabel("localization radius")
    ax.set_ylabel("time-averaged analysis RMSE")
    ax.set_title(f"Localized vs. unlocalized filter ({args.config}, Ne={s['Ne']})")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"localization_study_{args.config}.png", dpi=150)

    print(f"\nWrote figures/localization_study_{args.config}.png and "
          f"results/localization_study_{args.config}.txt")


if __name__ == "__main__":
    main()
