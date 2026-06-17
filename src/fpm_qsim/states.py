"""
fpm_qsim.states
===============

Density-matrix construction and validation helpers.

These utilities are deliberately minimal and dependency-free (NumPy
only) so that the package can be dropped into any simulation
pipeline without dragging in a full quantum-information toolkit.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def basis_state(index: int, dim: int) -> np.ndarray:
    """Return the computational basis state ``|index>`` as a column vector.

    Parameters
    ----------
    index : int
        Index of the basis state in ``[0, dim)``.
    dim : int
        Hilbert-space dimension.

    Returns
    -------
    ndarray, shape (dim, 1)
        Complex unit column vector.
    """
    if not 0 <= index < dim:
        raise ValueError(f"index {index} out of range for dim {dim}.")
    v = np.zeros((dim, 1), dtype=np.complex128)
    v[index, 0] = 1.0
    return v


def pure_state(amplitudes: Sequence[complex]) -> np.ndarray:
    """Build a normalised pure state ``|psi><psi|`` from amplitudes.

    Parameters
    ----------
    amplitudes : sequence of complex
        Unnormalised amplitudes.

    Returns
    -------
    ndarray, shape (N, N)
        Density matrix of the pure state.
    """
    v = np.asarray(amplitudes, dtype=np.complex128).reshape(-1, 1)
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("amplitudes must be non-zero.")
    v = v / norm
    return v @ v.conj().T


def maximally_mixed(dim: int) -> np.ndarray:
    """Return I_dim / dim, the maximally mixed state."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}.")
    return np.eye(dim, dtype=np.complex128) / dim


def partial_trace(rho: np.ndarray, keep: Sequence[int], dims: Sequence[int]) -> np.ndarray:
    """Partial trace over a tensor-product Hilbert space.

    Parameters
    ----------
    rho : ndarray, shape (prod(dims), prod(dims))
        Density matrix of the composite system.
    keep : sequence of int
        Indices of subsystems to KEEP (0-based).
    dims : sequence of int
        Dimensions of each subsystem.

    Returns
    -------
    ndarray
        Reduced density matrix of the kept subsystems.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    dims = list(dims)
    n_sys = len(dims)
    keep_set = set(keep)
    trace_out = [i for i in range(n_sys) if i not in keep_set]

    # Reshape into tensor with 2*n_sys axes.
    shape = dims + dims
    rho_t = rho.reshape(shape)
    # Move traced axes to the end in pairs.
    for axis in sorted(trace_out, reverse=True):
        # Contract axis `axis` with axis `axis + n_sys`.
        rho_t = np.trace(rho_t, axis1=axis, axis2=axis + n_sys - 1
                         if False else axis + n_sys - len(trace_out)
                         + trace_out.index(axis) if False else axis + n_sys)
        # np.trace removes two axes, so the indices shift; easier: use einsum.
        # Fall through to einsum approach below.
        break

    # Use einsum for correctness.
    rho_t = rho.reshape(shape)
    keep_dims = [dims[i] for i in range(n_sys) if i in keep_set]
    n_keep = len(keep_dims)
    # Build subscripts: a b c ... a' b' c' -> (kept) (kept')
    # where indices in trace_out are summed with their primed counterparts.
    letters = "abcdefghijklmnopqrstuvwxyz"
    if 2 * n_sys > len(letters):
        raise ValueError("Too many subsystems for einsum subscripting.")
    left = letters[:n_sys]
    right = letters[n_sys:2 * n_sys]
    # For traced-out subsystems, prime index must equal left index.
    right_list = list(right)
    for i in trace_out:
        right_list[i] = left[i]
    out = "".join(left[i] for i in range(n_sys) if i in keep_set)
    out += "".join(right_list[i] for i in range(n_sys) if i in keep_set)
    subscript = f"{left}{right}->{out}"
    reduced = np.einsum(subscript, rho_t)
    out_dim = int(np.prod(keep_dims)) if keep_dims else 1
    return reduced.reshape(out_dim, out_dim)


def is_density_matrix(rho: np.ndarray, tol: float = 1e-9) -> bool:
    """Check whether ``rho`` is a valid density matrix.

    Tests:
        1. Square 2-D array.
        2. Hermitian.
        3. Trace equals 1 (within ``tol``).
        4. Positive semidefinite (smallest eigenvalue >= -``tol``).
    """
    arr = np.asarray(rho)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return False
    if not np.allclose(arr, arr.conj().T, atol=tol):
        return False
    tr = np.trace(arr)
    if abs(tr - 1.0) > tol:
        return False
    w = np.linalg.eigvalsh(arr)
    return bool(w.min() >= -tol)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance ``0.5 * ||rho - sigma||_1`` between two states."""
    diff = np.asarray(rho, dtype=np.complex128) - np.asarray(sigma, dtype=np.complex128)
    # ||A||_1 = sum of singular values = sum of |eigenvalues| for Hermitian A.
    w = np.linalg.eigvalsh(0.5 * (diff + diff.conj().T))
    return float(0.5 * np.sum(np.abs(w)))


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity ``Tr(sqrt(sqrt(rho) sigma sqrt(rho)))``.

    For pure states this reduces to ``|<psi|phi>|^2``.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    sigma = np.asarray(sigma, dtype=np.complex128)
    # eigendecomposition of rho (Hermitian, PSD).
    w, v = np.linalg.eigh(rho)
    w = np.clip(w, 0.0, None)
    sqrt_rho = v @ np.diag(np.sqrt(w)) @ v.conj().T
    m = sqrt_rho @ sigma @ sqrt_rho
    w2, v2 = np.linalg.eigh(m)
    w2 = np.clip(w2, 0.0, None)
    return float(np.sum(np.sqrt(w2)))


__all__ = [
    "basis_state",
    "pure_state",
    "maximally_mixed",
    "partial_trace",
    "is_density_matrix",
    "trace_distance",
    "fidelity",
]
