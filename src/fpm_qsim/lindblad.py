"""
fpm_qsim.lindblad
=================

Drop-in Lindblad dephasing simulator backed by the FPM affine map.

This module provides the user-facing API that mirrors common quantum
open-systems libraries (QuTiP, qiskit_aer, etc.) so that existing
code can be ported with a single import change:

Before::

    from qutip import mesolve  # or your favorite Lindblad integrator
    rho_t = mesolve(H, rho0, tlist, c_ops=[np.sqrt(gamma)*sigma_z()])

After::

    from fpm_qsim import lindblad_step
    rho_t = rho0
    for _ in range(n_steps):
        rho_t = lindblad_step(rho_t, gamma=gamma, dt=dt)

The FPM affine map is

    c_{t+1} = kappa * c_t + nu_t

with ``kappa in [0, 1]``.  Theorem 3 of the FPM paper identifies one
particular choice of kappa, ``kappa = 1 - gamma*dt``, with the
Euler-discretized Lindblad dephasing equation.  This package uses the
exact continuous-dephasing form

    kappa = exp(-gamma * dt)

which is also a valid FPM affine-map coefficient (still in [0, 1])
and which makes the integrator **machine-precise** for pure
dephasing: it reproduces the analytic solution

    rho(t) = exp(-gamma*t) * (rho_0 - diag(rho_0)) + diag(rho_0)

to 1e-16, matching Kraus / matrix-exponential / QuTiP integrators
without the matrix-exponential cost.

The Euler form (``kappa = 1 - gamma*dt``) remains available as a
private reference implementation for Theorem 3 verification; see
:mod:`fpm_qsim._reference`.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .core import (
    GAMMA_MAX,
    FalsificationError,
    bounded_gamma,
)


ArrayLike = Union[np.ndarray, "list"]


# ---------------------------------------------------------------------------
# Density-matrix helpers
# ---------------------------------------------------------------------------

def _as_density_matrix(rho: ArrayLike) -> np.ndarray:
    """Coerce ``rho`` to a 2-D complex128 ndarray with shape (N, N)."""
    arr = np.asarray(rho, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"rho must be a square 2-D array; got shape {arr.shape}."
        )
    return arr


# ---------------------------------------------------------------------------
# The drop-in dephasing step (exact continuous form)
# ---------------------------------------------------------------------------

def lindblad_step(
    rho: ArrayLike,
    gamma: float,
    dt: float = 1.0,
    *,
    H: Optional[ArrayLike] = None,
    bounded: bool = False,
) -> np.ndarray:
    """Advance a density matrix by one exact FPM-affine dephasing step.

    Implements

        rho_{t+1} = exp(-gamma*dt) * rho_t
                    + (1 - exp(-gamma*dt)) * diag(rho_t)

    which is the **exact** solution of the continuous Lindblad
    dephasing master equation

        d(rho)/dt = -gamma * (rho - diag(rho))

    over one time step ``dt``.  This is the FPM affine map

        c_{t+1} = kappa * c_t + nu_t

    with ``kappa = exp(-gamma*dt)`` (a valid FPM contraction
    coefficient in ``[0, 1]``) and ``nu`` given by the diagonal
    restoration.  Theorem 3 of the FPM paper identifies the
    Euler-discretized form (``kappa = 1 - gamma*dt``) with the
    FPM map; the exact form here is the same affine map with a
    strictly better kappa choice.

    Parameters
    ----------
    rho : array_like, shape (N, N)
        Current density matrix.  Must be Hermitian.  Trace
        preservation is enforced by construction.
    gamma : float
        Dephasing rate (1/time units).  Must be non-negative.
    dt : float, optional
        Time step.  Default 1.0.  Must be non-negative.
    H : array_like, optional
        Hamiltonian.  If provided, a single Euler step of
        ``-i*[H, rho]/hbar`` is applied BEFORE the dephasing
        contraction.  This extends the integrator beyond pure
        dephasing; the exact FPM correspondence is proven only for
        H = 0, so callers using H != 0 should validate against a
        reference integrator.
    bounded : bool, optional
        If True, clip ``gamma`` at :data:`GAMMA_MAX` before use and
        raise :class:`FalsificationError` if it exceeds the
        falsification threshold.  Default False (caller is
        responsible).

    Returns
    -------
    ndarray, shape (N, N)
        Next density matrix.  A fresh array; ``rho`` is not modified.

    Notes
    -----
    The operation is trace-preserving by construction (the diagonal
    is a fixed point of the affine map).  Positive semidefiniteness
    is preserved for all ``gamma >= 0`` and ``dt >= 0`` because
    ``exp(-gamma*dt) in [0, 1]`` always.

    Accuracy: this step reproduces the analytic continuous-dephasing
    solution to machine precision (max abs error ~ 1e-16).  No
    Euler-style ``O(dt)`` per-step error.
    """
    rho_arr = _as_density_matrix(rho).copy()

    if bounded:
        gamma = float(bounded_gamma(gamma))
    else:
        gamma = float(gamma)

    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}.")
    if dt < 0.0:
        raise ValueError(f"dt must be non-negative, got {dt}.")

    # Optional unitary kick (extension; not part of the exact theorem).
    if H is not None:
        H_arr = np.asarray(H, dtype=np.complex128)
        if H_arr.shape != rho_arr.shape:
            raise ValueError(
                f"H shape {H_arr.shape} does not match rho shape "
                f"{rho_arr.shape}."
            )
        # Euler step of -i [H, rho] dt   (hbar = 1)
        rho_arr += -1j * (H_arr @ rho_arr - rho_arr @ H_arr) * dt

    # Exact FPM affine-map coefficient for continuous dephasing.
    kappa = float(np.exp(-gamma * dt))
    diag = np.diagonal(rho_arr).copy()
    out = kappa * rho_arr
    # Restore the diagonal to its fixed-point value.
    np.fill_diagonal(out, diag)
    return out


def simulate(
    rho0: ArrayLike,
    gamma: float,
    dt: float = 1.0,
    n_steps: int = 1,
    *,
    H: Optional[ArrayLike] = None,
    bounded: bool = False,
    record: bool = True,
):
    """Roll out a Lindblad dephasing trajectory.

    Convenience wrapper around :func:`lindblad_step` that records
    the full trajectory.

    Parameters
    ----------
    rho0 : array_like, shape (N, N)
        Initial density matrix.
    gamma : float
        Dephasing rate.
    dt : float, optional
        Time step.  Default 1.0.
    n_steps : int, optional
        Number of steps to integrate.  Default 1.
    H : array_like, optional
        Optional Hamiltonian.  See :func:`lindblad_step`.
    bounded : bool, optional
        If True, apply the FPM gamma ceiling.  Default False.
    record : bool, optional
        If True (default), return the full trajectory of shape
        ``(n_steps + 1, N, N)``.  If False, return only the final
        state.

    Returns
    -------
    ndarray
        Trajectory of shape ``(n_steps + 1, N, N)`` if ``record`` is
        True, else shape ``(N, N)``.
    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}.")

    rho = _as_density_matrix(rho0).copy()
    N = rho.shape[0]

    if not record:
        for _ in range(n_steps):
            rho = lindblad_step(rho, gamma=gamma, dt=dt, H=H, bounded=bounded)
        return rho

    traj = np.empty((n_steps + 1, N, N), dtype=np.complex128)
    traj[0] = rho
    for t in range(1, n_steps + 1):
        rho = lindblad_step(rho, gamma=gamma, dt=dt, H=H, bounded=bounded)
        traj[t] = rho
    return traj


__all__ = [
    "lindblad_step",
    "simulate",
]
