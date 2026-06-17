"""
fpm_qsim._reference
===================

Private reference implementations for Theorem 3 verification.

This module contains the literal Euler-discretized Lindblad dephasing
step

    rho_{t+1} = (1 - gamma*dt) * rho_t + gamma*dt * diag(rho_t)

which is the form identified by Theorem 3 of the FPM paper as
algebraically equivalent to the FPM affine map under
``kappa = 1 - gamma*dt``.

It is **not** used by the public API (which uses the exact continuous
form ``kappa = exp(-gamma*dt)`` for machine precision).  It exists
only so the Theorem 3 test in ``tests/test_lindblad_correspondence.py``
can verify the algebraic identity directly.
"""

from __future__ import annotations

from typing import Union

import numpy as np


ArrayLike = Union[np.ndarray, "list"]


def euler_lindblad_step(
    rho: ArrayLike,
    gamma: float,
    dt: float = 1.0,
) -> np.ndarray:
    """Euler-discretized Lindblad dephasing step (Theorem 3 form).

    Implements

        rho_{t+1} = (1 - gamma*dt) * rho_t + gamma*dt * diag(rho_t)

    This is the Euler discretization of

        d(rho)/dt = -gamma * (rho - diag(rho))

    and is the form identified by Theorem 3 of the FPM paper as
    algebraically equivalent to the FPM affine map under
    ``kappa = 1 - gamma*dt``.

    Notes
    -----
    This function is **not** part of the public API.  It exists for
    Theorem 3 verification only.  Use :func:`fpm_qsim.lindblad_step`
    for actual simulations; it uses the exact continuous form and
    has no Euler-style ``O(dt)`` error.
    """
    arr = np.asarray(rho, dtype=np.complex128).copy()
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"rho must be a square 2-D array; got shape {arr.shape}."
        )
    product = float(gamma) * float(dt)
    if product < 0.0 or product > 1.0:
        raise ValueError(
            f"gamma * dt = {product:.6g} is outside [0, 1]; the Euler "
            "affine map would be non-contractive or sign-flipping."
        )
    diag = np.diagonal(arr).copy()
    kappa = 1.0 - product
    out = kappa * arr
    np.fill_diagonal(out, diag)
    return out


__all__ = ["euler_lindblad_step"]
