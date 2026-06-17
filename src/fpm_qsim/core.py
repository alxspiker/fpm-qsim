"""
fpm_qsim.core
=============

Core FPM (Finite Possibility Mechanics) primitives.

This module implements the affine coherence map

    c_{t+1} = kappa_t * c_t + nu_t

and its identification with the Euler-discretized Lindblad dephasing
master equation under

    gamma_t = (1 - kappa_t) / dt   <==>   kappa_t = 1 - gamma_t * dt

Reference
---------
Theorem 3 (Lindblad Correspondence) in
"Finite Possibility Mechanics: A Unified Information-Theoretic
Framework" (Spiker, 2026) establishes that the Euler discretization
of the Lindblad master equation for a dephasing channel with H = 0
is algebraically equivalent to the FPM affine map.  Numerical
verification in the paper achieves RMSE ~ 6e-17 between the two
representations on off-diagonal density-matrix elements over
600 ticks and 10 paths.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants baked into the FPM framework (derived, not fitted)
# ---------------------------------------------------------------------------

#: Falsifiable Lorentz-factor ceiling derived from the finite-lag theorem.
#: Paper Test 09: L_max = 3.285  ->  gamma_max = 31.8738...
#: The CERN muon gamma = 29.3 sits below this ceiling.
#: Any observation with gamma > 32.0 falsifies the framework.
GAMMA_MAX: float = 31.873862947240752

#: Companion falsification threshold (rounded ceiling + 1% margin).
FALSIFICATION_THRESHOLD: float = 32.0

#: Energy floor fraction derived from the bounded-asymptotic theorem
#: (Test 07).  Prevents the energy variable from collapsing to exactly
#: zero, which would create a thermodynamic contradiction.
ENERGY_FLOOR_FRACTION: float = 0.03138766217547228

#: Spectral-gap isotropic limit weight (Test 04).  In the isotropic
#: (zero-shear) regime the derived entropy-balance weight is exactly
#: 1/3, recovering the symmetric mean.
ISOTROPIC_WEIGHT_LIMIT: float = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Affine coherence map
# ---------------------------------------------------------------------------

def kappa_from_gamma(gamma: float, dt: float = 1.0) -> float:
    """Convert a dephasing rate to the FPM contraction coefficient
    using the **Euler** identification (Theorem 3 form).

    Identifies the Lindblad rate ``gamma`` with the FPM affine
    contraction via

        kappa = 1 - gamma * dt

    This is the form identified by Theorem 3 of the FPM paper as
    algebraically equivalent to the Euler-discretized Lindblad
    dephasing equation.  The public :func:`fpm_qsim.lindblad_step`
    uses :func:`kappa_exact` instead, which has no Euler ``O(dt)``
    error.

    Parameters
    ----------
    gamma : float
        Lindblad dephasing rate (per unit time).
    dt : float, optional
        Time step used in the Euler discretization.  Default 1.0.

    Returns
    -------
    float
        The FPM contraction coefficient kappa in [0, 1].

    Raises
    ------
    ValueError
        If ``gamma * dt`` falls outside [0, 1], which would yield a
        non-contractive or sign-flipping affine map (unphysical for
        dephasing).
    """
    product = float(gamma) * float(dt)
    if product < 0.0 or product > 1.0:
        raise ValueError(
            f"gamma * dt = {product} is outside [0, 1]; the affine map "
            "would be non-contractive or sign-flipping. Reduce dt or "
            "clip gamma. (Use kappa_exact for the unbounded continuous form.)"
        )
    return 1.0 - product


def kappa_exact(gamma: float, dt: float = 1.0) -> float:
    """Convert a dephasing rate to the FPM contraction coefficient
    using the **exact continuous** form.

    Returns

        kappa = exp(-gamma * dt)

    which is a valid FPM affine-map coefficient (in ``[0, 1]`` for
    all ``gamma >= 0``, ``dt >= 0``) and which makes the FPM affine
    map machine-precise for continuous Lindblad dephasing.  This is
    the form used by :func:`fpm_qsim.lindblad_step`.

    Parameters
    ----------
    gamma : float
        Lindblad dephasing rate (per unit time).  Must be >= 0.
    dt : float, optional
        Time step.  Default 1.0.  Must be >= 0.

    Returns
    -------
    float
        The FPM contraction coefficient kappa in [0, 1].
    """
    if float(gamma) < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}.")
    if float(dt) < 0.0:
        raise ValueError(f"dt must be non-negative, got {dt}.")
    return float(np.exp(-float(gamma) * float(dt)))


def gamma_from_kappa(kappa: float, dt: float = 1.0) -> float:
    """Inverse of :func:`kappa_from_gamma`."""
    if not 0.0 <= float(kappa) <= 1.0:
        raise ValueError(
            f"kappa = {kappa} is outside [0, 1]; cannot invert to a "
            "physical dephasing rate."
        )
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    return (1.0 - float(kappa)) / float(dt)


def gamma_from_energy(
    daemon,
    gate_power: float,
    *,
    load: Optional[float] = None,
    dt: float = 1.0,
    C_N: float = 1.0,
    bounded: bool = True,
) -> float:
    """Derive an endogenous dephasing rate from daemon energy and load.

    Operationalizes the FPM contraction ansatz

        kappa_t = C_N * (1 + B_t)^(-3/4)
        gamma_t = (1 - kappa_t) / dt

    with a minimal gate-noise load model

        B_t = load + gate_power / energy_fraction

    so larger gate power, larger baryonic/load terms, or lower daemon
    energy all increase dephasing.  This makes ``gamma`` endogenous to
    the simulated daemon rather than a free external parameter.

    Parameters
    ----------
    daemon : object
        Daemon-like object with an ``energy_fraction`` property, or
        ``E`` and ``E_max`` attributes.  ``DaemonState`` satisfies this
        contract.
    gate_power : float
        Non-negative gate load applied during this step.
    load : float, optional
        Additional non-negative baryonic/load term.  If omitted, reads
        ``daemon.load`` or ``daemon.baryonic_load`` when present,
        otherwise uses ``0.0``.
    dt : float, optional
        Positive timestep for converting ``kappa`` to ``gamma``.
    C_N : float, optional
        Normalization coefficient in ``[0, 1]``.  Default ``1.0``.
    bounded : bool, optional
        If True, apply the FPM falsification ceiling via
        :func:`bounded_gamma`.

    Returns
    -------
    float
        Endogenous dephasing rate suitable for ``lindblad_step``.
    """
    gate_power = float(gate_power)
    if gate_power < 0.0:
        raise ValueError(f"gate_power must be non-negative, got {gate_power}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    C_N = float(C_N)
    if not 0.0 <= C_N <= 1.0:
        raise ValueError(f"C_N must be in [0, 1], got {C_N}.")

    if load is None:
        load = getattr(daemon, "load", getattr(daemon, "baryonic_load", 0.0))
    load = float(load)
    if load < 0.0:
        raise ValueError(f"load must be non-negative, got {load}.")

    energy_fraction = getattr(daemon, "energy_fraction", None)
    if energy_fraction is None:
        E = getattr(daemon, "E", None)
        E_max = getattr(daemon, "E_max", None)
        if E is None or E_max is None:
            raise TypeError(
                "daemon must provide energy_fraction or E and E_max attributes."
            )
        energy_fraction = float(E) / float(E_max) if float(E_max) > 0 else 0.0
    else:
        energy_fraction = float(energy_fraction)
    if energy_fraction < 0.0:
        raise ValueError(
            f"daemon energy_fraction must be non-negative, got {energy_fraction}."
        )

    effective_energy = max(
        energy_fraction, ENERGY_FLOOR_FRACTION, np.finfo(float).eps,
    )
    effective_load = load + gate_power / effective_energy
    kappa = C_N * (1.0 + effective_load) ** (-0.75)
    kappa = min(1.0, max(0.0, float(kappa)))
    gamma = gamma_from_kappa(kappa, dt=dt)
    return float(bounded_gamma(gamma)) if bounded else gamma


def fpm_affine_step(c, kappa, nu=0.0):
    """One tick of the FPM affine coherence map.

        c_{t+1} = kappa * c_t + nu

    Parameters
    ----------
    c : array_like or scalar
        Current coherence value(s).  May be a scalar complex number,
        a 1-D array of coherence amplitudes, or any array-like.
    kappa : float or array_like
        Contraction coefficient in [0, 1].  May be a scalar or
        broadcast-compatible with ``c``.
    nu : float or array_like, optional
        Bounded innovation noise.  Default 0.0 (pure dephasing limit).

    Returns
    -------
    ndarray or scalar
        The next coherence value(s).  Same shape as ``c``.
    """
    c_arr = np.asarray(c, dtype=np.complex128)
    out = kappa * c_arr + np.asarray(nu, dtype=np.complex128)
    # Preserve scalar-ness for ergonomics.
    if np.isscalar(c) or out.shape == ():
        return out.item()
    return out


def fpm_affine_trajectory(c0, kappa, nu=0.0, n_steps=1):
    """Roll out the FPM affine map for ``n_steps`` ticks.

    For a constant ``kappa`` and ``nu`` the closed-form solution is

        c_t = kappa**t * c_0 + nu * (1 - kappa**t) / (1 - kappa)

    This function uses the closed form when possible (constant
    kappa != 1) and falls back to per-tick iteration otherwise.

    Parameters
    ----------
    c0 : array_like or scalar
        Initial coherence value(s).
    kappa : float
        Constant contraction coefficient.  Scalar only.
    nu : float or array_like, optional
        Constant innovation noise.  Default 0.0.
    n_steps : int, optional
        Number of ticks to roll out.  Default 1.

    Returns
    -------
    ndarray
        Trajectory of shape ``(n_steps + 1, *c0.shape)``.  Entry
        ``[0]`` is the initial state.
    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}.")

    c0_arr = np.asarray(c0, dtype=np.complex128)
    out = np.empty((n_steps + 1, *c0_arr.shape), dtype=np.complex128)
    out[0] = c0_arr

    nu_arr = np.asarray(nu, dtype=np.complex128)
    if np.isclose(float(kappa), 1.0):
        # Pure accumulation, no contraction.
        for t in range(1, n_steps + 1):
            out[t] = out[t - 1] + nu_arr
        return out

    # Closed form for constant kappa, nu.
    powers = np.power(kappa, np.arange(1, n_steps + 1, dtype=np.float64))
    # c_t = kappa**t * c_0 + nu * (1 - kappa**t) / (1 - kappa)
    geom_sum = (1.0 - powers) / (1.0 - float(kappa))
    for t in range(1, n_steps + 1):
        out[t] = powers[t - 1] * c0_arr + geom_sum[t - 1] * nu_arr
    return out


# ---------------------------------------------------------------------------
# Bounded gamma form (falsifiable ceiling)
# ---------------------------------------------------------------------------

def bounded_gamma(gamma_raw, gamma_max=GAMMA_MAX):
    """Clip a raw dephasing rate at the FPM-derived ceiling.

    The finite-lag ceiling theorem (paper Test 07 and Test 09) caps
    the physically admissible Lorentz factor at

        gamma_max = 31.8738...

    Any rate exceeding this ceiling is clipped, and observations
    exceeding the rounded threshold of 32.0 falsify the framework.
    This function does NOT silently clip values above the
    falsification threshold; instead it raises so the caller can
    decide whether to log the observation or halt.

    Parameters
    ----------
    gamma_raw : float or array_like
        Proposed dephasing rate(s).
    gamma_max : float, optional
        Soft ceiling.  Default :data:`GAMMA_MAX`.

    Returns
    -------
    ndarray or float
        Clipped rate(s).

    Raises
    ------
    FalsificationError
        If any rate exceeds :data:`FALSIFICATION_THRESHOLD` (32.0),
        indicating an observation that would falsify FPM.
    """
    arr = np.asarray(gamma_raw, dtype=np.float64)
    scalar = arr.shape == ()
    if np.any(arr > FALSIFICATION_THRESHOLD):
        bad = arr[arr > FALSIFICATION_THRESHOLD]
        raise FalsificationError(
            f"gamma = {bad.tolist()} exceeds the FPM falsification "
            f"threshold {FALSIFICATION_THRESHOLD}. This observation "
            "would falsify the framework; refusing to clip silently."
        )
    clipped = np.minimum(arr, gamma_max)
    return float(clipped) if scalar else clipped


class FalsificationError(RuntimeError):
    """Raised when an observation would falsify the FPM framework."""


__all__ = [
    "GAMMA_MAX",
    "FALSIFICATION_THRESHOLD",
    "ENERGY_FLOOR_FRACTION",
    "ISOTROPIC_WEIGHT_LIMIT",
    "FalsificationError",
    "kappa_from_gamma",
    "kappa_exact",
    "gamma_from_kappa",
    "gamma_from_energy",
    "fpm_affine_step",
    "fpm_affine_trajectory",
    "bounded_gamma",
]
