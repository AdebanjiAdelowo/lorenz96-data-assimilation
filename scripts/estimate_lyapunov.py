"""Quantify chaotic behaviour of the K=40, F=8 Lorenz-96 configuration: raw perturbation growth
(unbounded, showing exponential divergence then nonlinear saturation) and the largest Lyapunov
exponent via the Benettin et al. (1980) renormalization algorithm, with its running estimate plotted
to show convergence rather than trusting a single point value.

Usage: python scripts/estimate_lyapunov.py [--config smoke|local|full]
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
from src.l96 import L96Params, spin_up
from src.chaos import perturbation_growth, lyapunov_exponent_benettin

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    base, lyap_cfg = cfg["base"], cfg["lyapunov"]

    params = L96Params(K=base["K"], F=base["F"], dt=base["dt"])
    x0 = np.full(params.K, params.F)
    x0[0] += 0.01
    x_attr = spin_up(x0, params, base["t_spinup"])

    rng = np.random.default_rng(lyap_cfg["seed"])
    delta0 = lyap_cfg["delta0_norm"] * rng.standard_normal(params.K)
    t_grow, sep = perturbation_growth(x_attr, params, delta0, t_end=min(10.0, lyap_cfg["t_total"]),
                                       save_every=5)

    lam, running, times = lyapunov_exponent_benettin(
        x_attr, params, t_total=lyap_cfg["t_total"], tau=lyap_cfg["tau"],
        delta0_norm=lyap_cfg["delta0_norm"], seed=lyap_cfg["seed"],
    )

    lines = [
        f"Chaos diagnostics ({args.config} config): K={params.K}, F={params.F}",
        f"Largest Lyapunov exponent estimate (Benettin algorithm): lambda_1 = {lam:.4f}",
        f"  tau (renormalization interval) = {lyap_cfg['tau']}, t_total = {lyap_cfg['t_total']}, "
        f"delta0_norm = {lyap_cfg['delta0_norm']}",
        f"  running estimate at t={times[len(times)//2]:.2f} (halfway): {running[len(running)//2]:.4f}",
        f"  running estimate at t={times[-1]:.2f} (final): {running[-1]:.4f}",
        f"  e-folding (Lyapunov) time 1/lambda_1 = {1.0/lam:.4f} model time units",
    ]
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"lyapunov_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.semilogy(t_grow, sep, "o-", ms=3)
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$\|x_{\mathrm{pert}}(t) - x_{\mathrm{true}}(t)\|$")
    ax1.set_title("Perturbation growth (raw, unbounded)")
    ax1.grid(alpha=0.3, which="both")

    ax2.plot(times, running)
    ax2.axhline(lam, color="k", linestyle="--", alpha=0.5, label=f"final estimate = {lam:.3f}")
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"running estimate of $\lambda_1$")
    ax2.set_title("Lyapunov-exponent convergence")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"lyapunov_{args.config}.png", dpi=150)

    print(f"\nWrote figures/lyapunov_{args.config}.png and results/lyapunov_{args.config}.txt")


if __name__ == "__main__":
    main()
