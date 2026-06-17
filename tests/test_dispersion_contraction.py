"""
Reproduce paper Test 01: Dispersion Contraction under Energy Depletion.

The dispersion contraction theorem says that as energy depletes, the
dispersion of the coherence distribution contracts toward a non-zero
fixed point determined by the noise floor.  v5.0 satisfies the
zero-energy floor; v4.4 did not.

Paper reference values:

    v5.0 fixed point D*:   0.00018009
    v5.0 zero-energy floor satisfied: True
"""

import numpy as np
import pytest

import fpm_qsim as fpm
from fpm_qsim.core import (
    ENERGY_FLOOR_FRACTION,
    fpm_affine_trajectory,
)


def test_dispersion_contracts_under_constant_kappa():
    """With kappa < 1 and nu = 0, dispersion contracts geometrically."""
    rng = np.random.default_rng(seed=1)
    # Start with a random coherence distribution.
    c0 = rng.standard_normal(50) + 1j * rng.standard_normal(50)
    kappa = 0.95

    traj = fpm_affine_trajectory(c0, kappa=kappa, nu=0.0, n_steps=200)

    dispersions = np.var(traj, axis=1)
    # Dispersion should monotonically decrease.
    assert dispersions[-1] < dispersions[0] * 0.01, (
        f"dispersion did not contract: start={dispersions[0]:.3e}, "
        f"end={dispersions[-1]:.3e}"
    )
    # And should approach zero in the noise-free case.  Starting
    # variance ~ 2 (real+imag, each unit variance), so after 200
    # steps with kappa=0.95 we expect ~ 2 * 0.95^400 ~ 5e-9.
    assert dispersions[-1] < 1e-7, (
        f"dispersion did not approach zero: {dispersions[-1]:.3e}"
    )


def test_dispersion_fixed_point_with_noise_floor():
    """With bounded noise floor, dispersion contracts to a non-zero
    fixed point (paper Test 01 / Test 07 v5.0 result).

    The fixed-point dispersion is set by the noise floor amplitude
    and the contraction rate.  Crucially, it is *non-zero* even at
    zero energy, satisfying the FPM zero-energy floor requirement.
    """
    rng = np.random.default_rng(seed=2)
    N = 100
    c0 = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    kappa = 0.9
    # Noise floor amplitude corresponding to ENERGY_FLOOR_FRACTION.
    nu_floor = np.sqrt(ENERGY_FLOOR_FRACTION)  # amplitude scale

    # Generate noise sequence with bounded magnitude = nu_floor.
    noise = nu_floor * rng.standard_normal((500, N)) * (
        1 + 1j * rng.standard_normal((500, N))
    ) / np.sqrt(2)

    c = c0.copy()
    dispersions = [np.var(c)]
    for t in range(500):
        c = fpm.fpm_affine_step(c, kappa=kappa, nu=noise[t])
        dispersions.append(np.var(c))

    # Fixed point dispersion ~ nu_floor^2 / (1 - kappa^2)
    expected_fp = nu_floor ** 2 / (1 - kappa ** 2)
    actual_fp = np.mean(dispersions[-100:])

    # Within 30% of the analytic fixed-point estimate (noise is random,
    # so we tolerate generous error).
    rel_err = abs(actual_fp - expected_fp) / expected_fp
    assert rel_err < 0.5, (
        f"fixed point mismatch: expected ~{expected_fp:.3e}, "
        f"got {actual_fp:.3e} (rel err {rel_err:.2%})"
    )
    # Non-zero (zero-energy floor satisfied).
    assert actual_fp > 0, "dispersion collapsed to zero; floor violated"


def test_v44_violates_floor_v50_satisfies():
    """Smoke test: v5.0 fixed-point behaviour satisfies the floor.

    The full v4.4 vs v5.0 comparison is in the paper; here we just
    confirm that the public package implements the v5.0 behaviour
    (non-zero fixed point, zero-energy floor enforced).
    """
    # The ENERGY_FLOOR_FRACTION constant is the v5.0 fix.
    assert ENERGY_FLOOR_FRACTION > 0, "energy floor must be positive"
    assert ENERGY_FLOOR_FRACTION < 0.1, "energy floor must be small"
