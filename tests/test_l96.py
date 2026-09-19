import numpy as np

from src.l96 import (
    l96_rhs, rk4_step, L96Params, integrate, spin_up, l96_rhs_ensemble, integrate_ensemble,
)


def test_cyclic_indexing_matches_hand_expansion_for_small_K():
    """K=6 hand-checked expansion of the cyclic indices, comparing l96_rhs against an explicit
    Python-loop implementation (deliberately not vectorised, so it cannot share a bug with the
    np.roll-based implementation)."""
    K = 6
    F = 3.0
    rng = np.random.default_rng(0)
    x = rng.standard_normal(K)

    expected = np.zeros(K)
    for i in range(K):
        ip1 = (i + 1) % K
        im1 = (i - 1) % K
        im2 = (i - 2) % K
        expected[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + F

    assert np.allclose(l96_rhs(x, F), expected)


def test_uniform_forcing_state_is_a_fixed_point():
    """x_i = F for all i is an exact fixed point of the ODE for any F, K (a genuine structural
    property, not an invented conserved quantity)."""
    for K, F in [(6, 3.0), (40, 8.0), (40, 0.0), (20, -2.0)]:
        x = np.full(K, F)
        assert np.allclose(l96_rhs(x, F), 0.0, atol=1e-12)


def test_quadratic_term_alone_conserves_energy():
    """The quadratic (advection-like) term (x_{i+1}-x_{i-2})*x_{i-1}, IN ISOLATION (no damping, no
    forcing), satisfies sum_i x_i * quad_i(x) = 0 exactly -- the continuous-time energy-derivative
    identity for this term alone. This does NOT imply the full forced/dissipative system conserves
    energy (it does not); only this one term's structural property is checked."""
    rng = np.random.default_rng(1)
    for K in [6, 12, 40]:
        x = rng.standard_normal(K)
        quad = (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1)
        assert abs(np.dot(x, quad)) < 1e-10


def test_full_system_does_not_conserve_energy():
    """The full forced/dissipative system's energy sum(x_i^2) changes over time (damping and forcing
    break the quadratic term's own conservation property) -- checked directly to guard against ever
    mistakenly treating this system as energy-conserving."""
    K = 40
    F = 8.0
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal(K) + F
    params = L96Params(K=K, F=F, dt=0.01)
    _, x = integrate(x0, params, n_steps=200, save_every=50)
    energy = np.sum(x ** 2, axis=1)
    assert not np.allclose(energy, energy[0], rtol=1e-3)


def test_deterministic_integration():
    K = 40
    rng = np.random.default_rng(3)
    x0 = rng.standard_normal(K) + 8.0
    params = L96Params(K=K, F=8.0, dt=0.01)
    _, x1 = integrate(x0, params, n_steps=100, save_every=10)
    _, x2 = integrate(x0, params, n_steps=100, save_every=10)
    assert np.array_equal(x1, x2)


def test_rk4_step_matches_explicit_formula():
    K = 6
    F = 2.0
    x = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 0.2])
    dt = 0.01
    k1 = l96_rhs(x, F)
    k2 = l96_rhs(x + 0.5 * dt * k1, F)
    k3 = l96_rhs(x + 0.5 * dt * k2, F)
    k4 = l96_rhs(x + dt * k3, F)
    expected = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    assert np.allclose(rk4_step(x, F, dt), expected)


def test_integrator_converges_at_fourth_order_against_a_fine_reference():
    K = 40
    F = 8.0
    rng = np.random.default_rng(4)
    x0 = rng.standard_normal(K) * 2 + F
    t_end = 0.5

    dt_ref = 1e-5
    p_ref = L96Params(K=K, F=F, dt=dt_ref)
    n_ref = int(round(t_end / dt_ref))
    _, x_ref = integrate(x0, p_ref, n_ref, save_every=n_ref)
    x_ref_final = x_ref[-1]

    dts = [0.02, 0.01, 0.005]
    errs = []
    for dt in dts:
        p = L96Params(K=K, F=F, dt=dt)
        n = int(round(t_end / dt))
        _, x = integrate(x0, p, n, save_every=n)
        errs.append(np.linalg.norm(x[-1] - x_ref_final))

    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    assert all(3.5 < o < 4.5 for o in orders)


def test_spin_up_reaches_bounded_attractor_state():
    """Spin-up from a generic (off-attractor) initial condition should produce a bounded state with
    magnitude consistent with the known L96 attractor scale (|x_i| typically O(1)-O(10) for F=8),
    not diverge."""
    K = 40
    F = 8.0
    x0 = np.full(K, F)
    x0[0] += 0.01  # generic small perturbation
    params = L96Params(K=K, F=F, dt=0.01)
    x_spun = spin_up(x0, params, t_spinup=20.0)
    assert np.all(np.isfinite(x_spun))
    assert np.max(np.abs(x_spun)) < 50.0


def test_vectorised_ensemble_integration_matches_per_member_integration():
    K = 40
    F = 8.0
    Ne = 7
    rng = np.random.default_rng(5)
    X0 = F + rng.standard_normal((K, Ne))
    params = L96Params(K=K, F=F, dt=0.01)

    X_final_vectorised = integrate_ensemble(X0, params, n_steps=50)

    X_final_loop = np.empty_like(X0)
    for i in range(Ne):
        _, xi = integrate(X0[:, i], params, n_steps=50, save_every=50)
        X_final_loop[:, i] = xi[-1]

    assert np.allclose(X_final_vectorised, X_final_loop, atol=1e-10)


def test_output_shapes_and_save_every():
    K = 10
    params = L96Params(K=K, F=8.0, dt=0.01)
    x0 = np.full(K, 8.0)
    t, x = integrate(x0, params, n_steps=100, save_every=10)
    assert x.shape == (11, K)
    assert t.shape == (11,)
    assert np.isclose(t[-1], 1.0)
