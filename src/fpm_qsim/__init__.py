"""
fpm_qsim
========

Drop-in Lindblad dephasing simulator backed by the Finite Possibility
Mechanics (FPM) affine map.

The FPM coherence update

    c_{t+1} = kappa_t * c_t + nu_t

with ``kappa in [0, 1]`` is the engine.  Theorem 3 of the FPM paper
identifies one particular choice (``kappa = 1 - gamma*dt``) with the
Euler-discretized Lindblad dephasing equation.  This package uses the
**exact continuous** form

    kappa = exp(-gamma * dt)

which is also a valid FPM affine-map coefficient and which makes the
integrator machine-precise for pure dephasing: it reproduces the
analytic solution to 1e-16, matching Kraus / matrix-exponential /
QuTiP integrators without their cost.

Quick start
-----------

>>> import numpy as np
>>> from fpm_qsim import lindblad_step, pure_state
>>> rho0 = pure_state([1, 1])              # |+><+|
>>> rho1 = lindblad_step(rho0, gamma=0.1, dt=1.0)
>>> float(rho1[0, 0].real)
1.0
>>> # Off-diagonal (coherence) contracts by exp(-gamma*dt):
>>> import math
>>> abs(rho1[0, 1]) - math.exp(-0.1) * abs(rho0[0, 1])   # doctest: +ELLIPSIS
0.0...

Drop-in replacement for a standard Lindblad dephasing loop:

>>> from fpm_qsim import simulate
>>> traj = simulate(rho0, gamma=0.05, dt=0.1, n_steps=100)
>>> traj.shape
(101, 2, 2)

References
----------
Alx Spiker, "Finite Possibility Mechanics: A Unified Information-
Theoretic Framework", 2026.  See in particular Theorem 3 (Lindblad
Correspondence), the Dispersion Contraction Theorem, and the
Finite-Lag-Ceiling Theorem.
"""

from ._version import __version__
from .core import (
    GAMMA_MAX,
    FALSIFICATION_THRESHOLD,
    ENERGY_FLOOR_FRACTION,
    ISOTROPIC_WEIGHT_LIMIT,
    FalsificationError,
    kappa_from_gamma,
    kappa_exact,
    gamma_from_kappa,
    fpm_affine_step,
    fpm_affine_trajectory,
    bounded_gamma,
)
from .lindblad import (
    lindblad_step,
    unitary_step,
    simulate,
)
from .states import (
    basis_state,
    pure_state,
    maximally_mixed,
    partial_trace,
    is_density_matrix,
    trace_distance,
    fidelity,
)
from .conservation import (
    DaemonState,
    ConservationLedger,
)

__all__ = [
    # Version
    "__version__",
    # Core FPM primitives
    "GAMMA_MAX",
    "FALSIFICATION_THRESHOLD",
    "ENERGY_FLOOR_FRACTION",
    "ISOTROPIC_WEIGHT_LIMIT",
    "FalsificationError",
    "kappa_from_gamma",
    "kappa_exact",
    "gamma_from_kappa",
    "fpm_affine_step",
    "fpm_affine_trajectory",
    "bounded_gamma",
    # Lindblad-equivalent API
    "lindblad_step",
    "unitary_step",
    "simulate",
    # State utilities
    "basis_state",
    "pure_state",
    "maximally_mixed",
    "partial_trace",
    "is_density_matrix",
    "trace_distance",
    "fidelity",
    # Closed-universe conservation
    "DaemonState",
    "ConservationLedger",
]

