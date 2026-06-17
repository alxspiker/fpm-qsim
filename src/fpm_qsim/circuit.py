"""
fpm_qsim.circuit
================

Quantum circuit layer that composes unitary gates with FPM dephasing
under a single closed-universe ledger.

This is the v0.1.6 layer that turns ``fpm_qsim`` from a density-matrix
stepper into a small circuit simulator.  It exists to remove the
boilerplate of manually composing :func:`fpm_qsim.unitary_step` and
:func:`fpm_qsim.lindblad_step` (and the associated ledger billing)
for the common case of "apply these gates, then dephase, repeat N
times".

The circuit is a **queue of unitary gates and dephasing layers**.
Each call to :meth:`Circuit.step` applies the full queue once.
:meth:`Circuit.run` repeats the queue ``n_steps`` times and returns
the trajectory.

Honest scope
------------
Only the gate set with a clean FPM correspondence is provided:

* Single-qubit Clifford+T: ``h``, ``x``, ``y``, ``z``, ``s``, ``t``.
* Two-qubit: ``cx`` (CNOT), ``cz``, ``swap``.
* Generic single-qubit rotation ``u(theta, phi, lam, i)``.
* Arbitrary user-supplied k-qubit unitary via :meth:`apply_unitary`.

General dissipative channels (amplitude damping, depolarizing, etc.)
remain out of scope: they have no FPM correspondence theorem.  Only
the pure-dephasing affine map of Theorem 3 is wired in via
:meth:`dephase`.

Ontological billing
-------------------
When a ``daemon`` and ``ledger`` are attached to the circuit, every
queued operation bills the closed-universe ledger for its simulated
construction cost under FPM's discrete action principle:

* **Unitary gate** (matrix exponential ``expm(-1j H dt)``): billed
  as ``N**2`` scalar ``exp`` constructions (one per matrix element),
  each via a K-term Taylor series.  ``N = 2**n_qubits``.  This is
  the same oracle-construction cost model used by
  :func:`fpm_qsim.bill_exp_route_cost` for the scalar case.
* **Dephasing layer, ``method="euler"``**: 1 multiply + 1 addition
  per off-diagonal state variable.  This is the literal Theorem 3
  lattice operation, billable as
  ``bill_compute_cost(daemon, n_multiplies=N*(N-1), n_adds=N*(N-1))``.
* **Dephasing layer, ``method="exact"``**: 1 scalar ``exp`` per
  off-diagonal state variable, each billed via a K-term Taylor
  series.  Oracle break, billed explicitly.

Billing is **opt-in**: if ``ledger`` is ``None`` or ``daemon`` is
``None``, no billing happens and the circuit behaves as a pure
state-stepper.  This keeps the SciPy-free / ledger-free path simple
for engineering users.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .core import FalsificationError, gamma_from_energy
from .lindblad import lindblad_step, unitary_step
from .conservation import (
    ConservationLedger,
    DaemonState,
    exp_route_cost,
)


# ---------------------------------------------------------------------------
# Standard gate matrices
# ---------------------------------------------------------------------------

# Single-qubit gates (2x2, complex).
_H = (1.0 / np.sqrt(2.0)) * np.array(
    [[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128
)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_S = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
_T = np.array(
    [[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]], dtype=np.complex128
)

# Two-qubit gates (4x4, complex), in the standard |00>,|01>,|10>,|11> basis.
# CNOT with qubit 0 as control and qubit 1 as target.
_CX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.complex128,
)
# CZ: diagonal [1, 1, 1, -1].
_CZ = np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)
# SWAP.
_SWAP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.complex128,
)


def _u_gate(theta: float, phi: float, lam: float) -> np.ndarray:
    """Qiskit-style single-qubit unitary.

        U(theta, phi, lam) =
            [[cos(theta/2),                       -exp(i lam) sin(theta/2)],
             [exp(i phi) sin(theta/2),   exp(i (phi + lam)) cos(theta/2)]]

    Parameters
    ----------
    theta, phi, lam : float
        Euler angles in radians.
    """
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array(
        [
            [c, -np.exp(1.0j * lam) * s],
            [np.exp(1.0j * phi) * s, np.exp(1.0j * (phi + lam)) * c],
        ],
        dtype=np.complex128,
    )


# ---------------------------------------------------------------------------
# Gate embedding
# ---------------------------------------------------------------------------

def _embed_gate(
    U: np.ndarray, targets: Sequence[int], n_qubits: int
) -> np.ndarray:
    """Embed a k-qubit unitary into the full ``n_qubits``-qubit Hilbert space.

    The resulting operator is the tensor product of ``U`` acting on the
    qubits listed in ``targets`` (in the order given) with the identity
    on all other qubits, with appropriate qubit-index permutation so
    that ``U`` acts on the *named* targets.

    Parameters
    ----------
    U : ndarray, shape (2**k, 2**k)
        Unitary acting on ``k = len(targets)`` qubits.
    targets : sequence of int
        Distinct qubit indices in ``[0, n_qubits)``.  Order matters: the
        first axis of ``U`` corresponds to ``targets[0]``, etc.
    n_qubits : int
        Total qubit count.

    Returns
    -------
    ndarray, shape (2**n_qubits, 2**n_qubits)
        The full-Hilbert-space unitary.
    """
    targets = list(targets)
    k = len(targets)
    if k == 0:
        raise ValueError("targets must be non-empty.")
    if len(set(targets)) != k:
        raise ValueError(f"targets must be distinct; got {targets}.")
    for t in targets:
        if not 0 <= t < n_qubits:
            raise ValueError(
                f"target {t} out of range for {n_qubits} qubits."
            )
    if U.shape != (2 ** k, 2 ** k):
        raise ValueError(
            f"U shape {U.shape} does not match {k} targets "
            f"(expected {(2 ** k, 2 ** k)})."
        )
    if n_qubits > 10:
        raise ValueError(
            "n_qubits > 10 not supported by _embed_gate (einsum letter "
            "budget). Use apply_unitary_full for larger systems."
        )

    U_arr = np.asarray(U, dtype=np.complex128)
    U_t = U_arr.reshape([2] * (2 * k))

    # Letters: lowercase for input (column) axes, uppercase for output
    # (row) axes.  Per qubit we use one lowercase letter and one
    # uppercase letter.
    lower = "abcdefghij"
    upper = "ABCDEFGHIJ"

    # U_t axes: (out_0, ..., out_{k-1}, in_0, ..., in_{k-1})
    U_subscript = (
        "".join(upper[t] for t in targets)
        + "".join(lower[t] for t in targets)
    )

    # Output subscript: (out_0, ..., out_{n-1}, in_0, ..., in_{n-1}).
    # For target qubits, the output uses the corresponding uppercase
    # letter (bound by U_t).  For non-target qubits, the output uses
    # the lowercase letter of the input axis (so output == input, i.e.
    # identity on those qubits).  We must supply identity tensors for
    # the non-target qubits so einsum can resolve those letters.
    inputs = [U_t]
    input_subs = [U_subscript]
    for i in range(n_qubits):
        if i not in targets:
            inputs.append(np.eye(2, dtype=np.complex128))
            input_subs.append(f"{upper[i]}{lower[i]}")

    out_part = "".join(upper[i] for i in range(n_qubits))
    in_part = "".join(lower[i] for i in range(n_qubits))
    output_subscript = out_part + in_part

    subscript = ",".join(input_subs) + "->" + output_subscript
    full_U = np.einsum(subscript, *inputs, optimize=True)
    full_dim = 2 ** n_qubits
    return full_U.reshape(full_dim, full_dim)


# ---------------------------------------------------------------------------
# Operation record
# ---------------------------------------------------------------------------

class _Op:
    """Internal record of a queued operation."""

    __slots__ = ("kind", "U", "targets", "dt", "gamma", "gate_power", "load")

    def __init__(
        self,
        kind: str,
        *,
        U: Optional[np.ndarray] = None,
        targets: Optional[Tuple[int, ...]] = None,
        dt: float = 1.0,
        gamma: Optional[float] = None,
        gate_power: Optional[float] = None,
        load: Optional[float] = None,
    ):
        self.kind = kind  # "U" or "D"
        self.U = U
        self.targets = targets
        self.dt = dt
        self.gamma = gamma
        self.gate_power = gate_power
        self.load = load

    def describe(self) -> str:
        if self.kind == "U":
            return f"U(targets={list(self.targets)})"
        parts = [f"dt={self.dt}"]
        if self.gamma is not None:
            parts.append(f"gamma={self.gamma:.4g}")
        if self.gate_power is not None:
            parts.append(f"gate_power={self.gate_power:.4g}")
        if self.load is not None:
            parts.append(f"load={self.load:.4g}")
        return "D(" + ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# The Circuit
# ---------------------------------------------------------------------------

class Circuit:
    """A queue of unitary gates and dephasing layers.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.  Must be >= 1.  Supported up to 10 qubits
        (Hilbert-space dimension 1024) due to the einsum letter budget
        in :func:`_embed_gate`.  For larger systems, build the full
        unitary externally and pass it to :meth:`apply_unitary_full`.
    daemon : DaemonState, optional
        Daemon paying for simulated operations.  When provided together
        with ``ledger``, every gate and dephasing layer bills the
        ledger automatically.
    ledger : ConservationLedger, optional
        Closed-universe ledger.  When provided together with
        ``daemon``, every gate and dephasing layer bills the ledger
        automatically.
    method : {"exact", "euler"}, optional
        Default FPM affine-map form for dephasing layers.  See
        :func:`fpm_qsim.lindblad_step`.  Default ``"exact"``.
    bounded : bool, optional
        If True, dephasing layers clip gamma at
        :data:`fpm_qsim.GAMMA_MAX` and raise
        :class:`fpm_qsim.FalsificationError` if a gamma would exceed
        :data:`fpm_qsim.FALSIFICATION_THRESHOLD`.  Default False.
    default_load : float, optional
        Default baryonic load for endogenous-gamma dephasing layers
        when ``load`` is not specified on the layer.  Falls back to
        ``daemon.load`` if available, else 0.0.
    default_gate_power : float, optional
        Default gate-power input for endogenous-gamma dephasing layers
        when ``gate_power`` is not specified on the layer.  Default
        0.0.
    cost_per_op : float, optional
        Energy cost per simulated op (as a fraction of ``E_max``) used
        when billing the ledger.  See
        :meth:`ConservationLedger.bill_compute_cost` for guidance.
        Default ``1e-5`` (reversible-compute regime).
    taylor_order : int, optional
        Number of Taylor terms assumed when billing the construction
        of ``exp`` evaluations.  Default 8.

    Notes
    -----
    The circuit is **stateless**: it does not store the density matrix
    between calls.  Callers pass ``rho0`` to :meth:`run` and receive
    the trajectory (or final state).

    Examples
    --------
    Minimal usage (no daemon, no ledger)::

        >>> import fpm_qsim as fpm
        >>> circ = fpm.Circuit(2)
        >>> circ.h(0).cx(0, 1).dephase(gamma=0.1)
        >>> rho0 = fpm.pure_state([1, 0])
        >>> traj = circ.run(rho0, n_steps=5)
        >>> traj.shape
        (6, 4, 4)

    FPM-aligned usage (closed-universe billing)::

        >>> ledger = fpm.ConservationLedger(E_max_total=100.0)
        >>> daemon = ledger.add_daemon(80.0)
        >>> circ = fpm.Circuit(
        ...     2, daemon=daemon, ledger=ledger, method="euler",
        ... )
        >>> circ.h(0).cx(0, 1).dephase(gate_power=0.1)
        >>> rho0 = fpm.pure_state([1, 0])
        >>> traj = circ.run(rho0, n_steps=5)
        >>> # daemon.E has been debited for every simulated op.
    """

    def __init__(
        self,
        n_qubits: int,
        *,
        daemon: Optional[DaemonState] = None,
        ledger: Optional[ConservationLedger] = None,
        method: str = "exact",
        bounded: bool = False,
        default_load: Optional[float] = None,
        default_gate_power: Optional[float] = None,
        cost_per_op: float = 1e-5,
        taylor_order: int = 8,
    ):
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}.")
        if n_qubits > 10:
            raise ValueError(
                f"n_qubits > 10 not supported (einsum letter budget); "
                f"got {n_qubits}. Use apply_unitary_full for larger "
                f"systems."
            )
        if method not in ("exact", "euler"):
            raise ValueError(
                f"method must be 'exact' or 'euler'; got {method!r}."
            )
        if (daemon is None) != (ledger is None):
            raise ValueError(
                "Provide both daemon and ledger, or neither. "
                f"Got daemon={'set' if daemon else 'None'}, "
                f"ledger={'set' if ledger else 'None'}."
            )
        if cost_per_op < 0.0:
            raise ValueError(
                f"cost_per_op must be >= 0, got {cost_per_op}."
            )
        if taylor_order < 0:
            raise ValueError(
                f"taylor_order must be >= 0, got {taylor_order}."
            )

        self.n_qubits = int(n_qubits)
        self.dim = 2 ** self.n_qubits
        self.daemon = daemon
        self.ledger = ledger
        self.method = method
        self.bounded = bool(bounded)
        self.default_load = default_load
        self.default_gate_power = (
            0.0 if default_gate_power is None else float(default_gate_power)
        )
        self.cost_per_op = float(cost_per_op)
        self.taylor_order = int(taylor_order)

        self._ops: List[_Op] = []
        self.gates_applied = 0
        self.dephase_layers_applied = 0

    # ------------------------------------------------------------------
    # Queue builders (fluent)
    # ------------------------------------------------------------------

    def apply_unitary(
        self,
        U: np.ndarray,
        targets: Sequence[int],
    ) -> "Circuit":
        """Append a k-qubit unitary acting on ``targets``.

        The unitary is applied directly via ``U @ rho @ U^dagger`` —
        **not** as a Hamiltonian time evolution.  This matches the
        standard quantum-circuit convention.  For Hamiltonian time
        evolution, use :meth:`strang_step`.

        Parameters
        ----------
        U : ndarray, shape (2**k, 2**k)
            Unitary acting on ``k = len(targets)`` qubits.
        targets : sequence of int
            Distinct qubit indices in ``[0, n_qubits)``.
        """
        targets_t = tuple(int(t) for t in targets)
        U_arr = np.asarray(U, dtype=np.complex128)
        # Validate by embedding now (raises if dimensions mismatch).
        U_full = _embed_gate(U_arr, targets_t, self.n_qubits)
        self._ops.append(_Op("U", U=U_full, targets=targets_t))
        return self

    def apply_unitary_full(
        self, U_full: np.ndarray
    ) -> "Circuit":
        """Append a pre-expanded full-Hilbert-space unitary.

        Use this for systems with more than 10 qubits where
        :meth:`apply_unitary` cannot be used (einsum letter budget).
        The caller is responsible for ensuring ``U_full`` has shape
        ``(2**n_qubits, 2**n_qubits)`` and is unitary.  Applied via
        direct conjugation ``U @ rho @ U^dagger``.
        """
        U_arr = np.asarray(U_full, dtype=np.complex128)
        if U_arr.shape != (self.dim, self.dim):
            raise ValueError(
                f"U_full shape {U_arr.shape} does not match "
                f"({self.dim}, {self.dim})."
            )
        self._ops.append(
            _Op("U", U=U_arr, targets=tuple(range(self.n_qubits)))
        )
        return self

    def h(self, i: int) -> "Circuit":
        """Hadamard on qubit ``i``."""
        return self.apply_unitary(_H, [i])

    def x(self, i: int) -> "Circuit":
        """Pauli-X on qubit ``i``."""
        return self.apply_unitary(_X, [i])

    def y(self, i: int) -> "Circuit":
        """Pauli-Y on qubit ``i``."""
        return self.apply_unitary(_Y, [i])

    def z(self, i: int) -> "Circuit":
        """Pauli-Z on qubit ``i``."""
        return self.apply_unitary(_Z, [i])

    def s(self, i: int) -> "Circuit":
        """S gate (phase) on qubit ``i``."""
        return self.apply_unitary(_S, [i])

    def t(self, i: int) -> "Circuit":
        """T gate (pi/8) on qubit ``i``."""
        return self.apply_unitary(_T, [i])

    def u(
        self,
        theta: float,
        phi: float,
        lam: float,
        i: int,
    ) -> "Circuit":
        """General single-qubit unitary U(theta, phi, lam) on qubit ``i``."""
        return self.apply_unitary(_u_gate(theta, phi, lam), [i])

    def cx(
        self, control: int, target: int
    ) -> "Circuit":
        """CNOT with ``control`` qubit and ``target`` qubit."""
        if control == target:
            raise ValueError(
                f"control and target must differ; got {control}."
            )
        return self.apply_unitary(_CX, [control, target])

    def cz(self, i: int, j: int) -> "Circuit":
        """Controlled-Z on qubits ``i`` and ``j``."""
        if i == j:
            raise ValueError(f"qubits must differ; got {i}.")
        return self.apply_unitary(_CZ, [i, j])

    def swap(self, i: int, j: int) -> "Circuit":
        """SWAP qubits ``i`` and ``j``."""
        if i == j:
            raise ValueError(f"qubits must differ; got {i}.")
        return self.apply_unitary(_SWAP, [i, j])

    def dephase(
        self,
        gamma: Optional[float] = None,
        *,
        dt: float = 1.0,
        gate_power: Optional[float] = None,
        load: Optional[float] = None,
    ) -> "Circuit":
        """Append a dephasing layer.

        Parameters
        ----------
        gamma : float, optional
            Explicit dephasing rate.  If omitted, derives gamma from
            the attached ``daemon`` (requires the circuit to have been
            constructed with a ``daemon``).  See
            :func:`fpm_qsim.gamma_from_energy`.
        dt : float, optional
            Time step for the dephasing layer.  Default 1.0.
        gate_power : float, optional
            Gate-power input for endogenous-gamma derivation.  Falls
            back to ``default_gate_power`` if not specified.
        load : float, optional
            Baryonic load for endogenous-gamma derivation.  Falls back
            to ``default_load`` -> ``daemon.load`` -> 0.0.
        """
        if dt < 0.0:
            raise ValueError(f"dt must be >= 0, got {dt}.")
        if gamma is None and self.daemon is None:
            raise ValueError(
                "dephase() requires either explicit gamma or a daemon "
                "attached to the circuit."
            )
        if gamma is not None and (gate_power is not None or load is not None):
            raise ValueError(
                "Provide either explicit gamma or endogenous-gamma "
                "inputs (gate_power/load), not both."
            )
        self._ops.append(
            _Op(
                "D",
                dt=dt,
                gamma=gamma,
                gate_power=gate_power,
                load=load,
            )
        )
        return self

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def operations(self) -> List[str]:
        """Human-readable description of each queued operation."""
        return [op.describe() for op in self._ops]

    @property
    def n_operations(self) -> int:
        return len(self._ops)

    def reset(self) -> "Circuit":
        """Clear the queue.  Billing stats are not reset."""
        self._ops = []
        return self

    def reset_stats(self) -> "Circuit":
        """Reset the billing counters (does not clear the queue)."""
        self.gates_applied = 0
        self.dephase_layers_applied = 0
        return self

    # ------------------------------------------------------------------
    # Billing helpers
    # ------------------------------------------------------------------

    def _bill_unitary(self, dim: int) -> None:
        """Bill the simulated Taylor construction of a dim×dim matrix exp.

        Under FPM, ``expm(-1j H dt)`` is a continuous-math oracle.  We
        bill it as ``dim**2`` scalar ``exp`` constructions, each via a
        K-term Taylor series.  This is a conservative approximation of
        the actual scaling-and-squaring cost; it is honest about the
        oracle break without modeling the full Padé algorithm.
        """
        if self.ledger is None or self.daemon is None:
            return
        n_elements = dim * dim
        n_mul, n_add = exp_route_cost(self.taylor_order)
        self.ledger.bill_compute_cost(
            self.daemon,
            n_multiplies=n_mul * n_elements,
            n_adds=n_add * n_elements,
            cost_per_op=self.cost_per_op,
        )

    def _bill_dephase(self, dim: int, method: str) -> None:
        """Bill one dephasing layer's simulated construction cost.

        For ``method="euler"``: 1 mul + 1 add per off-diagonal state
        variable (``dim * (dim - 1)`` total).  This is the literal
        Theorem 3 lattice operation, billable as
        ``bill_compute_cost(daemon, n_multiplies=dim*(dim-1), n_adds=dim*(dim-1))``.

        For ``method="exact"``: 1 scalar ``exp`` per off-diagonal
        state variable, each billed via a K-term Taylor series.
        """
        if self.ledger is None or self.daemon is None:
            return
        n_state_vars = dim * (dim - 1)
        if method == "euler":
            self.ledger.bill_compute_cost(
                self.daemon,
                n_multiplies=n_state_vars,
                n_adds=n_state_vars,
                cost_per_op=self.cost_per_op,
            )
        else:  # exact
            n_mul, n_add = exp_route_cost(self.taylor_order)
            self.ledger.bill_compute_cost(
                self.daemon,
                n_multiplies=n_mul * n_state_vars,
                n_adds=n_add * n_state_vars,
                cost_per_op=self.cost_per_op,
            )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def step(self, rho: np.ndarray) -> np.ndarray:
        """Apply the full queued sequence once.

        Parameters
        ----------
        rho : ndarray, shape (dim, dim)
            Current density matrix.

        Returns
        -------
        ndarray, shape (dim, dim)
            Density matrix after applying all queued operations.
            A fresh array; ``rho`` is not modified.
        """
        rho_arr = np.asarray(rho, dtype=np.complex128).copy()
        if rho_arr.shape != (self.dim, self.dim):
            raise ValueError(
                f"rho shape {rho_arr.shape} does not match "
                f"({self.dim}, {self.dim})."
            )

        for op in self._ops:
            if op.kind == "U":
                # Direct conjugation U @ rho @ U^dagger.  This is the
                # standard quantum-circuit convention; the gate matrix
                # IS the unitary, not a Hamiltonian.  No matrix exp
                # is needed for the application (only for the
                # ontological billing of the simulated construction
                # cost via _bill_unitary).
                rho_arr = op.U @ rho_arr @ op.U.conj().T
                self._bill_unitary(self.dim)
                self.gates_applied += 1
            elif op.kind == "D":
                # Resolve endogenous-gamma inputs.
                gate_power = op.gate_power
                if gate_power is None:
                    gate_power = self.default_gate_power
                load = op.load
                if load is None:
                    load = self.default_load
                rho_arr = lindblad_step(
                    rho_arr,
                    gamma=op.gamma,
                    dt=op.dt,
                    daemon=self.daemon if op.gamma is None else None,
                    gate_power=gate_power if op.gamma is None else None,
                    load=load if op.gamma is None else None,
                    method=self.method,
                    bounded=self.bounded,
                )
                self._bill_dephase(self.dim, self.method)
                self.dephase_layers_applied += 1
            else:  # pragma: no cover - defensive
                raise RuntimeError(f"unknown op kind: {op.kind!r}")

        return rho_arr

    def run(
        self,
        rho0: np.ndarray,
        n_steps: int = 1,
        *,
        record: bool = True,
    ):
        """Apply :meth:`step` ``n_steps`` times.

        Parameters
        ----------
        rho0 : ndarray, shape (dim, dim)
            Initial density matrix.
        n_steps : int, optional
            Number of times to apply the queued sequence.  Default 1.
        record : bool, optional
            If True (default), return the full trajectory of shape
            ``(n_steps + 1, dim, dim)``.  If False, return only the
            final state.

        Returns
        -------
        ndarray
            Trajectory if ``record=True``, else final state.
        """
        if n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {n_steps}.")
        rho = np.asarray(rho0, dtype=np.complex128).copy()
        if rho.shape != (self.dim, self.dim):
            raise ValueError(
                f"rho0 shape {rho.shape} does not match "
                f"({self.dim}, {self.dim})."
            )

        if not record:
            for _ in range(n_steps):
                rho = self.step(rho)
            return rho

        traj = np.empty(
            (n_steps + 1, self.dim, self.dim), dtype=np.complex128
        )
        traj[0] = rho
        for t in range(1, n_steps + 1):
            rho = self.step(rho)
            traj[t] = rho
        return traj

    def run_with_replenishment(
        self,
        rho0: np.ndarray,
        n_steps: int = 1,
        *,
        record: bool = True,
    ):
        """Apply :meth:`step` ``n_steps`` times, replenishing the
        daemon's energy each tick to keep the closed-universe ledger
        balanced.

        After each step, replenishes the daemon by exactly the energy
        debited that tick (spend + landauer), keeping the closed-
        universe identity

            total_replenish == total_spend + total_landauer

        satisfied to within energy-floor / energy-ceiling clipping.
        This is the FPM closed-universe conservation theorem (paper
        Test 03) made operational at the circuit level.

        Requires ``daemon`` and ``ledger`` to be attached to the
        circuit.  Use plain :meth:`run` if you want to manage
        replenishment manually (or skip it for open-system
        simulations).

        Parameters
        ----------
        rho0 : ndarray, shape (dim, dim)
            Initial density matrix.
        n_steps : int, optional
            Number of times to apply the queued sequence.  Default 1.
        record : bool, optional
            If True (default), return the full trajectory of shape
            ``(n_steps + 1, dim, dim)``.  If False, return only the
            final state.

        Returns
        -------
        ndarray
            Trajectory if ``record=True``, else final state.

        Raises
        ------
        ValueError
            If ``daemon`` or ``ledger`` is None, if ``n_steps < 0``,
            or if ``rho0`` has the wrong shape.

        Notes
        -----
        The replenishment each tick is::

            spend_delta   = daemon.cumulative_spend   - prev_spend
            landauer_delta = daemon.cumulative_landauer - prev_landauer
            ledger.record_replenish(daemon, spend_delta + landauer_delta)

        If the daemon is near ``E_max``, the actual replenishment is
        capped (``record_replenish`` clips at ``E_max - E``).  If the
        daemon is near the energy floor, the spend is also capped.
        Both clippings can produce non-zero drift; this is honest
        behavior &mdash; the framework is reporting that the configured
        ``E_max_total`` is too small to absorb the requested
        computation.

        Landauer debits charged externally between ``step()`` calls
        (via ``ledger.record_landauer``) are NOT replenished by this
        method.  The caller is responsible for those.

        Examples
        --------
        ::

            ledger = fpm.ConservationLedger(E_max_total=100.0)
            daemon = ledger.add_daemon(80.0)
            circ = fpm.Circuit(
                2, daemon=daemon, ledger=ledger, method="euler",
            )
            circ.h(0).cx(0, 1).dephase(gate_power=0.05)

            rho0 = fpm.pure_state([1, 0, 0, 0])
            traj = circ.run_with_replenishment(rho0, n_steps=20)
            # ledger.drift() should be ~0 (no external landauer).
        """
        if self.daemon is None or self.ledger is None:
            raise ValueError(
                "run_with_replenishment requires a daemon and ledger "
                "attached to the circuit. Construct with "
                "Circuit(..., daemon=..., ledger=...), or use run() "
                "for open-system simulations without replenishment."
            )
        if n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {n_steps}.")

        rho = np.asarray(rho0, dtype=np.complex128).copy()
        if rho.shape != (self.dim, self.dim):
            raise ValueError(
                f"rho0 shape {rho.shape} does not match "
                f"({self.dim}, {self.dim})."
            )

        if not record:
            for _ in range(n_steps):
                rho = self._step_with_replenishment(rho)
            return rho

        traj = np.empty(
            (n_steps + 1, self.dim, self.dim), dtype=np.complex128
        )
        traj[0] = rho
        for t in range(1, n_steps + 1):
            rho = self._step_with_replenishment(rho)
            traj[t] = rho
        return traj

    def _step_with_replenishment(self, rho: np.ndarray) -> np.ndarray:
        """Apply one ``step()`` and replenish the daemon by the
        debited amount (spend + landauer).

        This is the closed-universe conservation primitive: every
        unit of energy debited from the daemon during the step is
        returned to it immediately afterward, keeping the ledger
        identity ``replenish == spend + landauer`` satisfied to
        within floor/ceiling clipping.
        """
        prev_spend = self.daemon.cumulative_spend
        prev_landauer = self.daemon.cumulative_landauer
        rho = self.step(rho)
        spend_delta = self.daemon.cumulative_spend - prev_spend
        landauer_delta = self.daemon.cumulative_landauer - prev_landauer
        self.ledger.record_replenish(
            self.daemon, spend_delta + landauer_delta
        )
        return rho

    def strang_step(
        self,
        rho: np.ndarray,
        H: np.ndarray,
        gamma: Optional[float],
        dt: float,
        *,
        gate_power: Optional[float] = None,
        load: Optional[float] = None,
    ) -> np.ndarray:
        """Apply one Strang-splitting round of Hamiltonian + dephasing.

        Implements the standard second-order splitting::

            rho -> U(dt/2) rho U^dag(dt/2)         # half unitary
                 -> lindblad_step(rho, gamma, dt)  # full dephasing
                 -> U(dt/2) rho U^dag(dt/2)         # half unitary

        which has ``O(dt**3)`` per-step error.  Bills the ledger for
        both unitary halves (as matrix-exp constructions) and the
        dephasing layer (per the circuit's ``method``).

        Use this when you want the standard splitting without building
        the queue explicitly.  It does **not** consume the queue.

        Parameters
        ----------
        rho : ndarray, shape (dim, dim)
            Current density matrix.
        H : ndarray, shape (dim, dim)
            Hermitian Hamiltonian.
        gamma : float, optional
            Explicit dephasing rate.  If None, derives from the
            attached daemon (requires construction with a daemon).
        dt : float
            Time step.  The full round advances time by ``dt``.
        gate_power, load : float, optional
            Endogenous-gamma inputs (see :meth:`dephase`).
        """
        rho_arr = np.asarray(rho, dtype=np.complex128).copy()
        H_arr = np.asarray(H, dtype=np.complex128)
        if rho_arr.shape != (self.dim, self.dim):
            raise ValueError(
                f"rho shape {rho_arr.shape} does not match "
                f"({self.dim}, {self.dim})."
            )
        if H_arr.shape != (self.dim, self.dim):
            raise ValueError(
                f"H shape {H_arr.shape} does not match "
                f"({self.dim}, {self.dim})."
            )
        if dt < 0.0:
            raise ValueError(f"dt must be >= 0, got {dt}.")
        if gamma is None and self.daemon is None:
            raise ValueError(
                "strang_step requires either explicit gamma or a "
                "daemon attached to the circuit."
            )

        half_dt = 0.5 * dt

        # First half unitary.
        rho_arr = unitary_step(rho_arr, H_arr, dt=half_dt)
        self._bill_unitary(self.dim)

        # Full dephasing.
        rho_arr = lindblad_step(
            rho_arr,
            gamma=gamma,
            dt=dt,
            daemon=self.daemon if gamma is None else None,
            gate_power=gate_power if gamma is None else None,
            load=load if gamma is None else None,
            method=self.method,
            bounded=self.bounded,
        )
        self._bill_dephase(self.dim, self.method)

        # Second half unitary.
        rho_arr = unitary_step(rho_arr, H_arr, dt=half_dt)
        self._bill_unitary(self.dim)

        self.gates_applied += 2
        self.dephase_layers_applied += 1
        return rho_arr


__all__ = [
    "Circuit",
]
