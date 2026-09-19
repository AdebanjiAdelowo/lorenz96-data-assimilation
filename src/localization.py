"""Gaspari-Cohn (1999) compactly-supported localization function ("Construction of correlation
functions in two and three dimensions and convolution of anisotropic fields", Q. J. R. Meteorol. Soc.
125), applied on the cyclic Lorenz-96 ring.

    rho(z) =  -1/4 z^5 + 1/2 z^4 + 5/8 z^3 - 5/3 z^2 + 1,                          0 <= z <= 1
              1/12 z^5 - 1/2 z^4 + 5/8 z^3 + 5/3 z^2 - 5 z + 4 - 2/(3z),           1 <  z <= 2
              0,                                                                   z >  2

with z = d/c, d the (cyclic-ring) distance between two state locations and c the localization radius
(rho vanishes exactly beyond distance 2c -- a genuinely COMPACT support, unlike a Gaussian taper,
which is the main practical reason this function is preferred for covariance localization: exact
zeros let a large fraction of long-range spurious sample correlations be removed entirely rather than
merely damped).
"""
from __future__ import annotations

import numpy as np


def gaspari_cohn(z: np.ndarray) -> np.ndarray:
    z = np.abs(np.asarray(z, dtype=float))
    rho = np.zeros_like(z)

    m1 = z <= 1.0
    zz = z[m1]
    rho[m1] = (-0.25 * zz ** 5 + 0.5 * zz ** 4 + 0.625 * zz ** 3 - (5.0 / 3.0) * zz ** 2 + 1.0)

    m2 = (z > 1.0) & (z <= 2.0)
    zz = z[m2]
    with np.errstate(divide="ignore", invalid="ignore"):
        rho[m2] = ((1.0 / 12.0) * zz ** 5 - 0.5 * zz ** 4 + 0.625 * zz ** 3 + (5.0 / 3.0) * zz ** 2
                   - 5.0 * zz + 4.0 - (2.0 / 3.0) / zz)

    return rho


def cyclic_distance(i: np.ndarray, j: np.ndarray, K: int) -> np.ndarray:
    """Shortest distance between indices i, j on a cyclic ring of length K."""
    d = np.abs(i - j)
    return np.minimum(d, K - d)


def localization_matrix(K: int, obs_indices: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (loc_xy (K, m), loc_yy (m, m)): Gaspari-Cohn localization weights between every state
    location and every observation location, and between every pair of observation locations, on the
    cyclic K-ring, with radius `radius` (rho vanishes beyond cyclic distance 2*radius)."""
    state_idx = np.arange(K)
    d_xy = cyclic_distance(state_idx[:, None], obs_indices[None, :], K)
    loc_xy = gaspari_cohn(d_xy / radius)

    d_yy = cyclic_distance(obs_indices[:, None], obs_indices[None, :], K)
    loc_yy = gaspari_cohn(d_yy / radius)

    return loc_xy, loc_yy
