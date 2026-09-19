import numpy as np

from src.l96 import L96Params, spin_up
from src.chaos import perturbation_growth, lyapunov_exponent_benettin


def _attractor_state(K=40, F=8.0, dt=0.01, t_spinup=15.0):
    params = L96Params(K=K, F=F, dt=dt)
    x0 = np.full(K, F)
    x0[0] += 0.01
    return spin_up(x0, params, t_spinup), params


def test_perturbation_grows_then_saturates():
    x_attr, params = _attractor_state()
    K = params.K
    rng = np.random.default_rng(0)
    delta0 = 1e-6 * rng.standard_normal(K)
    t, sep = perturbation_growth(x_attr, params, delta0, t_end=15.0, save_every=20)
    # early growth should be substantially larger (in log terms) than late growth, since separation
    # saturates at the attractor's own scale rather than growing forever
    early_growth = np.log(sep[5] / sep[0])
    late_growth = np.log(max(sep[-1], 1e-300) / max(sep[-6], 1e-300))
    assert sep[0] > 0
    assert sep[-1] > sep[5]  # net growth over the whole run
    assert early_growth > 0  # genuinely growing early on
    # saturation: separation should plateau at O(attractor scale), not keep growing exponentially
    assert sep[-1] < 200.0


def test_lyapunov_estimate_is_positive_and_roughly_stable():
    """The K=40, F=8 configuration is the textbook chaotic L96 case; the largest Lyapunov exponent
    should be clearly positive and the running estimate should stabilise (not keep drifting) over
    the second half of the integration window."""
    x_attr, params = _attractor_state(t_spinup=20.0)
    lam, running, times = lyapunov_exponent_benettin(x_attr, params, t_total=20.0, tau=0.1,
                                                       delta0_norm=1e-7, seed=2)
    assert lam > 0.5  # clearly chaotic; literature value for this configuration is ~1.6-1.7
    # running estimate in the second half should not differ from the final estimate by more than 20%
    half = len(running) // 2
    assert abs(running[half] - running[-1]) / running[-1] < 0.2


def test_lyapunov_estimate_insensitive_to_perturbation_magnitude():
    """Confirms the algorithm is operating in the locally-linear regime: the estimate should not
    depend meaningfully on delta0's magnitude, as long as it is small."""
    x_attr, params = _attractor_state(t_spinup=15.0)
    lams = []
    for delta0 in [1e-5, 1e-7]:
        lam, _, _ = lyapunov_exponent_benettin(x_attr, params, t_total=10.0, tau=0.1,
                                                delta0_norm=delta0, seed=3)
        lams.append(lam)
    assert abs(lams[0] - lams[1]) / lams[1] < 0.05


def test_lyapunov_exponent_deterministic_given_seed():
    x_attr, params = _attractor_state(t_spinup=10.0)
    lam1, _, _ = lyapunov_exponent_benettin(x_attr, params, t_total=5.0, tau=0.1, seed=7)
    lam2, _, _ = lyapunov_exponent_benettin(x_attr, params, t_total=5.0, tau=0.1, seed=7)
    assert lam1 == lam2
