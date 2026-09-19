"""Repeated-seed statistics (brief section 17) and rank-histogram / innovation diagnostics (section
18). EnKF performance depends on the random initial ensemble and (for the stochastic formulation)
perturbed observations, so key comparisons -- here, unlocalized vs. localized baseline -- are run
over many independent seeds and reported as mean +/- std, never a single-seed number.

Usage: python scripts/repeated_seeds_study.py [--config smoke|local|full]
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
from src.experiment import repeated_experiment, run_assimilation_experiment, rank_histogram

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    base, s = cfg["base"], cfg["repeated_seeds"]

    params = L96Params(K=base["K"], F=base["F"], dt=base["dt"])
    obs_op = make_observation_operator(base["K"], stride=s["stride"], obs_std=s["obs_std"])
    seeds = list(range(s["n_seeds"]))
    common = dict(params=params, obs_op=obs_op, Ne=s["Ne"], obs_interval_steps=s["obs_interval_steps"],
                  n_cycles=s["n_cycles"], ens_init_std=s["ens_init_std"], t_spinup=base["t_spinup"])

    out_unloc = repeated_experiment(seeds, **common)
    loc_xy, loc_yy = localization_matrix(base["K"], obs_op.obs_indices, radius=s["loc_radius"])
    out_loc = repeated_experiment(seeds, loc_xy=loc_xy, loc_yy=loc_yy, **common)

    lines = [f"Repeated-seed statistics ({args.config} config): {s['n_seeds']} independent seeds, "
             f"Ne={s['Ne']}, stride={s['stride']}, obs_std={s['obs_std']}", "",
             f"unlocalized: analysis RMSE = {out_unloc['analysis_rmse'].mean:.4f} +/- "
             f"{out_unloc['analysis_rmse'].std:.4f} (per-seed: "
             f"{np.round(out_unloc['analysis_rmse'].values, 3).tolist()})",
             f"localized (radius={s['loc_radius']}): analysis RMSE = {out_loc['analysis_rmse'].mean:.4f} "
             f"+/- {out_loc['analysis_rmse'].std:.4f} (per-seed: "
             f"{np.round(out_loc['analysis_rmse'].values, 3).tolist()})",
             "",
             f"unlocalized: spread = {out_unloc['analysis_spread'].mean:.4f} +/- "
             f"{out_unloc['analysis_spread'].std:.4f}",
             f"localized: spread = {out_loc['analysis_spread'].mean:.4f} +/- "
             f"{out_loc['analysis_spread'].std:.4f}",
    ]

    # rank histogram on a couple of localized-run seeds (needs saved forecast ensembles)
    all_ranks = []
    for seed in seeds[:min(3, len(seeds))]:
        res = run_assimilation_experiment(seed=seed, loc_xy=loc_xy, loc_yy=loc_yy, save_ensembles=True,
                                           **common)
        all_ranks.append(rank_histogram(res))
    ranks = np.concatenate(all_ranks)
    Ne = s["Ne"]
    counts, _ = np.histogram(ranks, bins=np.arange(Ne + 2) - 0.5)
    lines += ["", f"Rank histogram (localized filter, {min(3, len(seeds))} seeds, {len(ranks)} "
              f"truth/forecast-ensemble comparisons, {Ne+1} possible ranks):",
              f"  counts: {counts.tolist()}",
              f"  expected count per bin if flat: {len(ranks)/(Ne+1):.1f}",
              "  (a diagnostic, not proof of calibration -- see README 'Ensemble spread')"]

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"repeated_seeds_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.bar(["unlocalized", "localized"],
            [out_unloc["analysis_rmse"].mean, out_loc["analysis_rmse"].mean],
            yerr=[out_unloc["analysis_rmse"].std, out_loc["analysis_rmse"].std], capsize=5)
    ax1.set_ylabel("analysis RMSE")
    ax1.set_title(f"Mean +/- std over {s['n_seeds']} seeds")
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(np.arange(Ne + 1), counts)
    ax2.axhline(len(ranks) / (Ne + 1), color="k", linestyle="--", alpha=0.6, label="flat reference")
    ax2.set_xlabel("rank of truth among sorted forecast-ensemble members")
    ax2.set_ylabel("count")
    ax2.set_title("Rank histogram (localized filter)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"repeated_seeds_{args.config}.png", dpi=150)

    print(f"\nWrote figures/repeated_seeds_{args.config}.png and results/repeated_seeds_{args.config}.txt")


if __name__ == "__main__":
    main()
