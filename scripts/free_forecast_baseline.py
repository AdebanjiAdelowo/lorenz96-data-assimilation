"""Free-forecast baseline (brief section 5): initialize a perturbed state near the truth and
integrate it FREELY (no observations, no assimilation), tracking RMSE(t) growth against the truth.
This is the baseline against which every assimilation result in this project must be judged -- if
assimilation cannot beat this, it has added nothing.

Usage: python scripts/free_forecast_baseline.py [--config smoke|local|full]
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
from src.l96 import L96Params, spin_up, integrate
from src.experiment import rmse

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    base, ff_cfg = cfg["base"], cfg["free_forecast"]

    params = L96Params(K=base["K"], F=base["F"], dt=base["dt"])
    x0 = np.full(params.K, params.F)
    x0[0] += 0.01
    truth0 = spin_up(x0, params, base["t_spinup"])

    rng = np.random.default_rng(ff_cfg["seed"])
    forecast0 = truth0 + ff_cfg["ens_init_std"] * rng.standard_normal(params.K)

    n_steps = int(round(ff_cfg["t_end"] / params.dt))
    save_every = max(1, n_steps // 300)
    t, truth_traj = integrate(truth0, params, n_steps, save_every)
    _, forecast_traj = integrate(forecast0, params, n_steps, save_every)

    rmse_t = np.array([rmse(forecast_traj[i], truth_traj[i]) for i in range(len(t))])

    lines = [
        f"Free-forecast baseline ({args.config} config): K={params.K}, F={params.F}, "
        f"initial perturbation std={ff_cfg['ens_init_std']}",
        f"RMSE(0) = {rmse_t[0]:.4f}",
        f"RMSE at t={t[len(t)//4]:.2f} (25%): {rmse_t[len(t)//4]:.4f}",
        f"RMSE at t={t[len(t)//2]:.2f} (50%): {rmse_t[len(t)//2]:.4f}",
        f"RMSE at t={t[-1]:.2f} (end): {rmse_t[-1]:.4f}",
        "(saturates near the climatological RMS separation of two independent attractor states, "
        "not a numerical artefact -- errors cannot exceed the attractor's own scale)",
    ]
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"free_forecast_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(t, truth_traj[:, 0], label="truth, $x_0$")
    axes[0].plot(t, forecast_traj[:, 0], "--", label="free forecast, $x_0$")
    axes[0].set_ylabel("$x_0$")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Truth vs. free forecast (one component)")
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(t, rmse_t)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("RMSE(t)")
    axes[1].set_title("Free-forecast RMSE growth (chaotic error amplification)")
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"free_forecast_{args.config}.png", dpi=150)

    print(f"\nWrote figures/free_forecast_{args.config}.png and results/free_forecast_{args.config}.txt")


if __name__ == "__main__":
    main()
