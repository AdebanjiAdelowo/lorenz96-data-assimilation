"""The Lorenz-96 model (Lorenz, 1996, "Predictability: a problem partly solved", ECMWF Seminar on
Predictability; Lorenz & Emanuel, 1998, "Optimal sites for supplementary weather observations:
simulation with a small model", J. Atmos. Sci. 55) and a classical RK4 integrator.

## The model

A chain of K scalar variables $x_0,\\dots,x_{K-1}$ on a periodic (cyclic) index, intended as a
minimal caricature of a single latitude circle of the atmosphere: each variable advects nonlinearly
with its neighbours, is linearly damped, and is driven by a constant external forcing F,

    dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F,     i = 0, ..., K-1 (indices mod K).

The quadratic term conserves the total "energy" sum(x_i^2) exactly BY ITSELF (it is a discrete
analogue of nonlinear advection, and satisfies a discrete energy identity -- see
`tests/test_l96.py::test_quadratic_term_alone_conserves_energy`); the full system is dissipative
(linear damping -x_i) and forced (+F), so the FULL system's energy is NOT conserved -- there is no
invented conserved quantity for the forced/dissipative system, only this one exact structural
property of the quadratic term in isolation, which the tests check directly.

## Standard chaotic configuration

K=40, F=8 is the standard configuration used throughout the data-assimilation literature (Lorenz &
Emanuel, 1998; Evensen, 2009), chosen here for the same reason: F=8 places the system deep in a
chaotic regime for K=40 (see "Establishing chaotic behavior" in the README), giving a genuinely hard,
well-studied sequential-estimation testbed.

## Fixed point

x_i = F for all i is an EXACT fixed point of the ODE for any F, K (substituting: (F-F)*F - F + F = 0
identically) -- a genuine, non-invented structural property used as a verification check
(`tests/test_l96.py::test_uniform_forcing_state_is_a_fixed_point`), distinct from any claim about
energy conservation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def l96_rhs(x: np.ndarray, F: float) -> np.ndarray:
    """dx/dt for the Lorenz-96 system, cyclic indices via np.roll (roll by -1 shifts index i+1 into
    position i, i.e. np.roll(x, -1)[i] = x[(i+1) % K]; np.roll(x, 2)[i] = x[(i-2) % K])."""
    x_p1 = np.roll(x, -1)
    x_m2 = np.roll(x, 2)
    x_m1 = np.roll(x, 1)
    return (x_p1 - x_m2) * x_m1 - x + F


def rk4_step(x: np.ndarray, F: float, dt: float) -> np.ndarray:
    k1 = l96_rhs(x, F)
    k2 = l96_rhs(x + 0.5 * dt * k1, F)
    k3 = l96_rhs(x + 0.5 * dt * k2, F)
    k4 = l96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@dataclass
class L96Params:
    K: int = 40
    F: float = 8.0
    dt: float = 0.01  # standard L96 time unit; ~0.05 time units ~ 6 hours of "atmospheric" time
    #                   (Lorenz & Emanuel 1998's own approximate dimensional calibration, cited for
    #                   context only -- not used quantitatively anywhere in this project).


def integrate(x0: np.ndarray, params: L96Params, n_steps: int,
              save_every: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Integrate from x0 for n_steps of size params.dt. Returns (t (n_saved,), x (n_saved, K))."""
    x = x0.copy()
    t = 0.0
    t_list = [0.0]
    x_list = [x0.copy()]
    for step in range(1, n_steps + 1):
        x = rk4_step(x, params.F, params.dt)
        t = step * params.dt
        if step % save_every == 0 or step == n_steps:
            t_list.append(t)
            x_list.append(x.copy())
    return np.array(t_list), np.array(x_list)


def spin_up(x0: np.ndarray, params: L96Params, t_spinup: float) -> np.ndarray:
    """Integrate from an arbitrary/generic initial condition long enough to reach the system's
    (chaotic) attractor, discarding the transient -- standard practice before using a trajectory as
    "truth" for a data-assimilation experiment (starting ON the attractor, not near an arbitrary
    initial condition, so early-time results are not an artefact of transient relaxation)."""
    n_steps = int(round(t_spinup / params.dt))
    _, x = integrate(x0, params, n_steps, save_every=n_steps)
    return x[-1]
