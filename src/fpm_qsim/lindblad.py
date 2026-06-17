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

with ``kappa in [0, 1]``.  Two choices of ``kappa`` are supported,
with **strict ontological boundaries** between them:

**``method="euler"`` (FPM-native lattice map).**

    kappa = 1 - gamma * dt

This is the literal Theorem 3 identification: the FPM affine map with
``kappa = 1 - gamma*dt`` is algebraically equivalent to the
Euler-discretized Lindblad dephasing equation.  Under FPM's discrete
action principle, this is the physically realizable lattice
operation.  Per state variable per step, the simulated route cost is
exactly one multiplication and one addition &mdash; billable to the
:class:`~fpm_qsim.conservation.ConservationLedger` as
``bill_compute_cost(daemon, n_multiplies=1, n_adds=1)``.  This
preserves closed-universe structural coherence.

**``method="exact"`` (legacy continuous-math oracle).**

    kappa = exp(-gamma * dt)

This is the exact continuous-dephasing solution &mdash; it reproduces
the analytic result ``rho(t) = exp(-gamma*t) * (rho_0 - diag(rho_0))
+ diag(rho_0)`` to machine precision (~1e-16), matching Kraus /
matrix-exponential / QuTiP integrators.

**Ontological warning.** Under FPM, ``exp`` is a continuous-math
idealization.  The simulated system cannot natively evaluate
``exp(-gamma*dt)``; doing so requires a discrete computational
construction (Taylor series, CORDIC, Pade approximant) whose
finite-integer route cost must be paid by the simulated daemons.
When ``method="exact"`` is used, the host evaluates ``np.exp`` and
hands the result to the simulated system as a **zero-cost oracle**,
deliberately breaking the FPM closed-universe ledger.  This is a
non-physical override used to benchmark FPM against legacy
continuous quantum mechanics.  To preserve closed-universe
accounting while using the exact map, callers must explicitly bill
the simulated construction cost via
:func:`~fpm_qsim.conservation.bill_exp_route_cost`.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .core import (
    GAMMA_MAX,
    FalsificationError,
    bounded_gamma,
    gamma_from_energy,
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
# The drop-in dephasing step
# ---------------------------------------------------------------------------

def lindblad_step(
    rho: ArrayLike,
    gamma: Optional[float] = None,
    dt: float = 1.0,
    *,
    daemon=None,
    gate_power: Optional[float] = None,
    load: Optional[float] = None,
    method: str = "exact",
    bounded: bool = False,
) -> np.ndarray:
    """Advance a density matrix by one FPM-affine dephasing step.

    Implements the FPM affine map

        c_{t+1} = kappa * c_t + nu_t

    where ``nu`` restores the diagonal (a fixed point of the
    dephasing map).  The choice of ``kappa`` is governed by
    ``method`` and carries strict ontological consequences (see
    module docstring).

    Parameters
    ----------
    rho : array_like, shape (N, N)
        Current density matrix.  Must be Hermitian.  Trace
        preservation is enforced by construction.
    gamma : float, optional
        Explicit dephasing rate (1/time units).  Must be non-negative.
        If omitted, provide ``daemon`` and ``gate_power`` to derive
        gamma endogenously from daemon energy.
    dt : float, optional
        Time step.  Default 1.0.  Must be non-negative.
    daemon : object, optional
        Daemon-like object used by :func:`fpm_qsim.gamma_from_energy`
        when ``gamma`` is omitted.
    gate_power : float, optional
        Gate load used by :func:`fpm_qsim.gamma_from_energy` when
        ``gamma`` is omitted.
    load : float, optional
        Additional baryonic/load term for energy-derived gamma.
    method : {"exact", "euler"}, optional
        Selects the FPM affine-map coefficient:

        * ``"exact"`` (default): ``kappa = exp(-gamma*dt)``.
          Machine-precise continuous-dephasing solution.  **Oracle
          break** under FPM &mdash; the host's ``np.exp`` is handed
          to the simulated system without billing the simulated
          route cost.  Use for legacy-continuous-QM comparison only.
          To preserve closed-universe accounting with this method,
          the caller must explicitly bill the construction cost via
          :func:`~fpm_qsim.conservation.bill_exp_route_cost`.
        * ``"euler"``: ``kappa = 1 - gamma*dt``.  The FPM-native
          lattice map identified by Theorem 3.  Per state variable
          per step, the simulated route cost is exactly 1 multiply
          + 1 add &mdash; billable to the
          :class:`~fpm_qsim.conservation.ConservationLedger`.
          Requires ``0 <= gamma*dt <= 1`` for the map to remain
          contractive.

    bounded : bool, optional
        If True, clip ``gamma`` at :data:`GAMMA_MAX` before use and
        raise :class:`FalsificationError` if it exceeds the
        falsification threshold.  Default False.

    Returns
    -------
    ndarray, shape (N, N)
        Next density matrix.  A fresh array; ``rho`` is not modified.

    Raises
    ------
    ValueError
        If ``method`` is not "exact" or "euler", or if ``gamma`` /
        ``dt`` are negative, or if ``method="euler"`` is used with
        ``gamma*dt`` outside ``[0, 1]``.

    Notes
    -----
    The operation is trace-preserving by construction (the diagonal
    is a fixed point of the affine map).  Positive semidefiniteness
    is preserved for all ``gamma >= 0`` and ``dt >= 0`` when
    ``method="exact"`` (because ``exp(-gamma*dt) in [0, 1]`` always);
    for ``method="euler"``, positivity is preserved only when
    ``0 <= gamma*dt <= 1``.

    **H != 0 removed in v0.1.3.** Earlier versions accepted an ``H``
    parameter that applied an Euler unitary kick before the dephasing
    contraction.  This reintroduced ``O(dt^2)`` per-step error from
    the Lie-Trotter splitting (because ``[H, L_dephasing] != 0`` in
    general), silently breaking the machine-precision guarantee.
    Use :func:`unitary_step` to compose Hamiltonian evolution
    manually with your chosen splitting strategy.
    """
    rho_arr = _as_density_matrix(rho).copy()

    if gamma is None:
        if daemon is None or gate_power is None:
            raise ValueError(
                "Provide either explicit gamma or both daemon and gate_power."
            )
        gamma = gamma_from_energy(
            daemon, gate_power=gate_power, load=load, dt=dt, bounded=bounded,
        )
        apply_bounded = False
    else:
        if daemon is not None or gate_power is not None or load is not None:
            raise ValueError(
                "Provide either explicit gamma or energy-derived inputs "
                "(daemon, gate_power, load), not both."
            )
        gamma = float(gamma)
        apply_bounded = bounded

    if apply_bounded:
        gamma = float(bounded_gamma(gamma))

    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}.")
    if dt < 0.0:
        raise ValueError(f"dt must be non-negative, got {dt}.")

    if method == "exact":
        # Legacy continuous-math oracle.  Host's np.exp is handed to
        # the simulated system as a zero-cost oracle; the FPM
        # closed-universe ledger is deliberately broken.  Callers who
        # need closed-universe accounting must bill the construction
        # cost separately via bill_exp_route_cost().
        kappa = float(np.exp(-gamma * dt))
    elif method == "euler":
        # FPM-native lattice map.  Per state variable: 1 multiply +
        # 1 add.  Billable to ConservationLedger as
        # bill_compute_cost(daemon, n_multiplies=1, n_adds=1).
        product = gamma * dt
        if product < 0.0 or product > 1.0:
            raise ValueError(
                f"method='euler' requires 0 <= gamma*dt <= 1 for the "
                f"affine map to remain contractive; got gamma*dt = "
                f"{product:.6g}. Use method='exact' for unbounded "
                f"gamma*dt, or reduce dt."
            )
        kappa = 1.0 - product
    else:
        raise ValueError(
            f"method must be 'exact' or 'euler'; got {method!r}."
        )

    diag = np.diagonal(rho_arr).copy()
    out = kappa * rho_arr
    # Restore the diagonal to its fixed-point value.
    np.fill_diagonal(out, diag)
    return out


def unitary_step(
    rho: ArrayLike,
    H: ArrayLike,
    dt: float = 1.0,
) -> np.ndarray:
    """Apply a single exact unitary (Hamiltonian) step.

    Implements

        rho_{t+1} = U @ rho_t @ U^dagger,    U = exp(-i H dt)

    which is the exact solution of the von Neumann equation

        d(rho)/dt = -i [H, rho]

    over one time step ``dt``.  Uses a matrix exponential, so it is
    machine-precise for any Hermitian ``H`` and any ``dt``.

    This function exists so callers can compose Hamiltonian evolution
    with dephasing (:func:`lindblad_step`) using their chosen
    splitting strategy.  For pure dephasing, just use
    :func:`lindblad_step` directly.  For combined unitary + dephasing
    dynamics, use a Strang splitting:

        rho = unitary_step(rho, H, dt/2)
        rho = lindblad_step(rho, gamma, dt)
        rho = unitary_step(rho, H, dt/2)

    This gives ``O(dt^3)`` per-step error, much better than the
    naive Euler splitting (``O(dt^2)``).

    Parameters
    ----------
    rho : array_like, shape (N, N)
        Current density matrix.
    H : array_like, shape (N, N)
        Hermitian Hamiltonian.
    dt : float, optional
        Time step.  Default 1.0.

    Returns
    -------
    ndarray, shape (N, N)
        Next density matrix.  A fresh array; ``rho`` is not modified.

    Notes
    -----
    Requires SciPy for the matrix exponential.  If SciPy is not
    installed, raises ``ImportError``.
    """
    from scipy.linalg import expm

    rho_arr = _as_density_matrix(rho)
    H_arr = np.asarray(H, dtype=np.complex128)
    if H_arr.shape != rho_arr.shape:
        raise ValueError(
            f"H shape {H_arr.shape} does not match rho shape "
            f"{rho_arr.shape}."
        )
    U = expm(-1j * H_arr * float(dt))
    return U @ rho_arr @ U.conj().T


def simulate(
    rho0: ArrayLike,
    gamma: Optional[float] = None,
    dt: float = 1.0,
    n_steps: int = 1,
    *,
    daemon=None,
    gate_power: Optional[float] = None,
    load: Optional[float] = None,
    method: str = "exact",
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
    gamma : float, optional
        Explicit dephasing rate.  If omitted, provide ``daemon`` and
        ``gate_power`` to derive gamma from daemon energy.
    dt : float, optional
        Time step.  Default 1.0.
    n_steps : int, optional
        Number of steps to integrate.  Default 1.
    daemon : object, optional
        Daemon-like object for energy-derived gamma.
    gate_power : float, optional
        Gate load for energy-derived gamma.
    load : float, optional
        Additional baryonic/load term for energy-derived gamma.
    method : {"exact", "euler"}, optional
        See :func:`lindblad_step`.  Default ``"exact"``.
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
            rho = lindblad_step(
                rho,
                gamma=gamma,
                dt=dt,
                daemon=daemon,
                gate_power=gate_power,
                load=load,
                method=method,
                bounded=bounded,
            )
        return rho

    traj = np.empty((n_steps + 1, N, N), dtype=np.complex128)
    traj[0] = rho
    for t in range(1, n_steps + 1):
        rho = lindblad_step(
            rho,
            gamma=gamma,
            dt=dt,
            daemon=daemon,
            gate_power=gate_power,
            load=load,
            method=method,
            bounded=bounded,
        )
        traj[t] = rho
    return traj


__all__ = [
    "lindblad_step",
    "unitary_step",
    "simulate",
]
