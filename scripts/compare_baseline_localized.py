"""Side-by-side trajectory view: the unlocalized baseline EnKF vs. the same filter with Gaspari-Cohn
localization, on the SAME truth, observations and seed.

Both runs use the `baseline` block of the chosen config (Ne, obs stride/std/interval, n_cycles, seed);
the only change in the second run is localization at the given radius (default 2.0, the best radius
in the `full`-config localization study). No inflation. This is a single-seed illustration; the
multi-seed statistics are in scripts/localization_study.py.

Usage: python scripts/compare_baseline_localized.py [--config smoke|local|full] [--radius 2.0]
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
from src.experiment import run_assimilation_experiment, time_averaged_rmse

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    parser.add_argument("--radius", type=float, default=2.0)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    base, b = cfg["base"], cfg["baseline"]

    params = L96Params(K=base["K"], F=base["F"], dt=base["dt"])
    obs_op = make_observation_operator(base["K"], stride=b["stride"], obs_std=b["obs_std"])
    common = dict(Ne=b["Ne"], obs_interval_steps=b["obs_interval_steps"], n_cycles=b["n_cycles"],
                  ens_init_std=b["ens_init_std"], seed=b["seed"], t_spinup=base["t_spinup"])

    loc_xy, loc_yy = localization_matrix(base["K"], obs_op.obs_indices, radius=args.radius)
    runs = {
        "no localization": run_assimilation_experiment(params, obs_op, **common),
        f"localization, radius {args.radius:g}": run_assimilation_experiment(
            params, obs_op, loc_xy=loc_xy, loc_yy=loc_yy, **common),
    }

    comp = 0
    j = int(np.where(obs_op.obs_indices == comp)[0][0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 5.6), sharex=True,
                             gridspec_kw={"width_ratios": [1.6, 1]})
    for row, (label, res) in enumerate(runs.items()):
        a_rmse, _ = time_averaged_rmse(res)
        spread = np.mean(res.analysis_spread[len(res.analysis_spread) // 5:])
        print(f"{label}: time-averaged analysis RMSE {a_rmse:.4f}, spread {spread:.4f}")

        ax = axes[row, 0]
        ax.plot(res.t[1:], [o[j] for o in res.obs], "C1.", ms=3, alpha=0.5, label="observations")
        ax.plot(res.t, res.truth[:, comp], "k-", lw=1.3, label="truth")
        ax.plot(res.t, res.analysis_mean[:, comp], "C0--", lw=1.0, label="EnKF analysis mean")
        ax.set_ylabel(f"$x_{{{comp}}}$")
        ax.set_title(f"{label}: time-avg. analysis RMSE {a_rmse:.2f}", fontsize=10)

        ax = axes[row, 1]
        ax.semilogy(res.t, res.analysis_rmse, label="analysis RMSE")
        ax.semilogy(res.t, res.analysis_spread, label="analysis spread")
        ax.set_ylim(5e-2, 2e1)
        ax.grid(alpha=0.3, which="both")
        ax.set_title(f"RMSE vs. spread ({label})", fontsize=10)

    axes[0, 0].legend(fontsize=7, loc="upper right", ncol=3)
    axes[0, 1].legend(fontsize=7, loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel("t")
    fig.suptitle(f"Lorenz-96 EnKF, K={base['K']}, Ne={b['Ne']}, fully observed, obs std {b['obs_std']}, "
                 f"seed {b['seed']} ({args.config} config)", fontsize=10)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"baseline_vs_localized_{args.config}.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
