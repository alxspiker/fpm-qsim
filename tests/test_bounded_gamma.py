"""
Reproduce paper Test 07 (Bounded Asymptotics) and Test 09
(Finite Lag Ceiling).

The bounded-asymptotic theorem caps the FPM gamma at

    gamma_max = 31.8738...

and the finite-lag theorem sets the falsification threshold at

    gamma > 32.0  =>  FPM falsified.

The CERN muon (gamma = 29.3) sits below the ceiling.
"""

import numpy as np
import pytest

import fpm_qsim as fpm
from fpm_qsim.core import GAMMA_MAX, FALSIFICATION_THRESHOLD


def test_gamma_max_value():
    """The derived gamma_max matches the paper."""
    assert abs(GAMMA_MAX - 31.873862947240752) < 1e-12


def test_falsification_threshold():
    """Falsification threshold is 32.0 (rounded ceiling + margin)."""
    assert FALSIFICATION_THRESHOLD == 32.0


def test_cern_muon_in_range():
    """The CERN muon (gamma = 29.3) is below the FPM ceiling."""
    cern_muon_gamma = 29.3
    assert cern_muon_gamma < GAMMA_MAX, (
        "CERN muon gamma exceeds FPM ceiling; framework falsified"
    )


def test_bounded_gamma_clips_at_ceiling():
    """bounded_gamma clips values in [gamma_max, 32.0) down to gamma_max."""
    # 31.95 is above gamma_max (31.8738) but below the falsification
    # threshold (32.0), so it should be clipped to gamma_max.
    assert abs(fpm.bounded_gamma(31.95) - GAMMA_MAX) < 1e-12
    # Anything exactly at gamma_max passes through.
    assert abs(fpm.bounded_gamma(GAMMA_MAX) - GAMMA_MAX) < 1e-12


def test_bounded_gamma_passes_through_below_ceiling():
    """Below gamma_max, bounded_gamma is identity."""
    assert abs(fpm.bounded_gamma(10.0) - 10.0) < 1e-12
    assert abs(fpm.bounded_gamma(0.0) - 0.0) < 1e-12
    assert abs(fpm.bounded_gamma(GAMMA_MAX - 0.01) - (GAMMA_MAX - 0.01)) < 1e-12


def test_falsification_raises():
    """gamma > 32.0 raises FalsificationError."""
    with pytest.raises(fpm.FalsificationError):
        fpm.bounded_gamma(33.0)
    with pytest.raises(fpm.FalsificationError):
        fpm.bounded_gamma(100.0)
    with pytest.raises(fpm.FalsificationError):
        fpm.bounded_gamma(1e6)


def test_bounded_flag_in_lindblad_step():
    """lindblad_step(bounded=True) clips gamma to the ceiling.

    With the exact (exp) form in v0.1.1, gamma*dt is no longer
    bounded above by 1 — the map stays contractive for any
    gamma*dt >= 0.  So we can use a large dt and check the exact
    decay.
    """
    import math
    rho = fpm.pure_state([1, 1])
    # gamma=31.9 is above gamma_max, gets clipped to GAMMA_MAX.
    rho_next = fpm.lindblad_step(rho, gamma=31.9, dt=0.5, bounded=True)
    # kappa = exp(-GAMMA_MAX * 0.5)
    expected_kappa = math.exp(-GAMMA_MAX * 0.5)
    assert abs(rho_next[0, 1] - expected_kappa * rho[0, 1]) < 1e-14


def test_falsification_via_lindblad_step():
    """lindblad_step with gamma > 32 (after bounded check) raises."""
    rho = fpm.pure_state([1, 1])
    with pytest.raises(fpm.FalsificationError):
        fpm.lindblad_step(rho, gamma=33.0, dt=0.01, bounded=True)


def test_energy_floor_value():
    """v5.0 energy floor matches the paper Test 07 value."""
    assert abs(fpm.ENERGY_FLOOR_FRACTION - 0.03138766217547228) < 1e-12
