"""Inflation study (brief section 15): applied because the baseline experiment demonstrated ensemble
under-dispersion (spread collapse). Studies a small number of multiplicative inflation factors,
chosen a priori from the standard range used in the EnKF literature (1.0-1.5), NOT tuned against this
specific trajectory for the prettiest plot -- the same fixed grid of factors is evaluated over
multiple seeds and the post-transient time-averaged RMSE is reported, exactly as for every other
study in this project.

Usage: python scripts/inflation_study.py [--config smoke|local|full]
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
    base, s = cfg["base"], cfg["inflation_study"]
    loc_cfg = cfg["localization_study"]

    params = L96Params(K=base["K"], F=base["F"], dt=base["dt"])
    obs_op = make_observation_operator(base["K"], stride=s["stride"], obs_std=s["obs_std"])
    seeds = list(range(s["n_seeds"]))
    common = dict(params=params, obs_op=obs_op, Ne=s["Ne"], obs_interval_steps=s["obs_interval_steps"],
                  n_cycles=s["n_cycles"], ens_init_std=s["ens_init_std"], t_spinup=base["t_spinup"])

    lines = [f"Inflation study ({args.config} config): Ne={s['Ne']} (deliberately undersampled for "
             f"K={base['K']}), stride={s['stride']}, obs_std={s['obs_std']}, {s['n_seeds']} seeds", "",
             "-- inflation alone (no localization) --",
             f"{'lambda':>8} {'analysis RMSE (mean+-std)':>28} {'spread (mean+-std)':>22}"]

    factors, a_means_no_loc, a_stds_no_loc = [], [], []
    for lam in s["factors"]:
        out = repeated_experiment(seeds, inflation=lam, **common)
        a = out["analysis_rmse"]
        factors.append(lam); a_means_no_loc.append(a.mean); a_stds_no_loc.append(a.std)
        lines.append(f"{lam:8.2f} {a.mean:12.4f}+-{a.std:<12.4f} "
                     f"{out['analysis_spread'].mean:9.4f}+-{out['analysis_spread'].std:<9.4f}")

    # combined with localization: pick the radius with the lowest mean RMSE on THIS study's own
    # grid (a fixed a priori set of candidates, not tuned against any single seed) rather than an
    # arbitrary "middle of the list" index, which does not track the empirically best radius.
    radius_rmses = []
    for radius in loc_cfg["radii"]:
        loc_xy_r, loc_yy_r = localization_matrix(base["K"], obs_op.obs_indices, radius=radius)
        out_r = repeated_experiment(seeds, loc_xy=loc_xy_r, loc_yy=loc_yy_r, **common)
        radius_rmses.append(out_r["analysis_rmse"].mean)
    best_radius = loc_cfg["radii"][int(np.argmin(radius_rmses))]
    loc_xy, loc_yy = localization_matrix(base["K"], obs_op.obs_indices, radius=best_radius)
    lines += ["", f"-- radius selection: {list(zip(loc_cfg['radii'], [round(r, 4) for r in radius_rmses]))} "
              f"-> best_radius={best_radius}",
              f"-- inflation combined with localization (radius={best_radius}) --",
              f"{'lambda':>8} {'analysis RMSE (mean+-std)':>28} {'spread (mean+-std)':>22}"]
    a_means_loc = []
    for lam in s["factors"]:
        out = repeated_experiment(seeds, inflation=lam, loc_xy=loc_xy, loc_yy=loc_yy, **common)
        a = out["analysis_rmse"]
        a_means_loc.append(a.mean)
        lines.append(f"{lam:8.2f} {a.mean:12.4f}+-{a.std:<12.4f} "
                     f"{out['analysis_spread'].mean:9.4f}+-{out['analysis_spread'].std:<9.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"inflation_study_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(factors, a_means_no_loc, yerr=a_stds_no_loc, marker="o", label="inflation only")
    ax.plot(factors, a_means_loc, marker="s", label=f"inflation + localization (r={best_radius})")
    ax.set_xlabel(r"inflation factor $\lambda$")
    ax.set_ylabel("time-averaged analysis RMSE")
    ax.set_title(f"Inflation study ({args.config}, Ne={s['Ne']})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"inflation_study_{args.config}.png", dpi=150)

    print(f"\nWrote figures/inflation_study_{args.config}.png and "
          f"results/inflation_study_{args.config}.txt")


if __name__ == "__main__":
    main()
