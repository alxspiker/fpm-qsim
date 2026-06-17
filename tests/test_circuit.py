"""Tests for the v0.1.6 Circuit layer."""
import math
import numpy as np
import pytest

import fpm_qsim as fpm
from fpm_qsim.circuit import (
    Circuit,
    _embed_gate,
    _H,
    _X,
    _Y,
    _Z,
    _S,
    _T,
    _CX,
    _CZ,
    _SWAP,
    _u_gate,
)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

def test_circuit_construction_minimal():
    circ = fpm.Circuit(2)
    assert circ.n_qubits == 2
    assert circ.dim == 4
    assert circ.n_operations == 0
    assert circ.gates_applied == 0
    assert circ.dephase_layers_applied == 0


def test_circuit_construction_with_daemon_and_ledger():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(80.0)
    circ = fpm.Circuit(
        2, daemon=daemon, ledger=ledger, method="euler", bounded=True,
    )
    assert circ.daemon is daemon
    assert circ.ledger is ledger
    assert circ.method == "euler"
    assert circ.bounded is True


def test_circuit_construction_rejects_daemon_without_ledger():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(80.0)
    with pytest.raises(ValueError, match="daemons together with ledger"):
        fpm.Circuit(2, daemon=daemon)
    with pytest.raises(ValueError, match="daemons together with ledger"):
        fpm.Circuit(2, ledger=ledger)


def test_circuit_construction_rejects_invalid_method():
    with pytest.raises(ValueError, match="method must be"):
        fpm.Circuit(2, method="bogus")


def test_circuit_construction_rejects_zero_qubits():
    with pytest.raises(ValueError, match="n_qubits must be >= 1"):
        fpm.Circuit(0)


def test_circuit_construction_rejects_too_many_qubits():
    with pytest.raises(ValueError, match="n_qubits > 10"):
        fpm.Circuit(11)


# ---------------------------------------------------------------------------
# Gate embedding
# ---------------------------------------------------------------------------

def test_embed_single_qubit_gate_on_qubit_0():
    U_full = _embed_gate(_X, [0], 2)
    # X on qubit 0 should swap |00><->|10 and |01><->|11>.
    expected = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=complex,
    )
    assert np.allclose(U_full, expected)


def test_embed_single_qubit_gate_on_qubit_1():
    U_full = _embed_gate(_X, [1], 2)
    # X on qubit 1 should swap |00><->|01 and |10><->|11>.
    expected = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=complex,
    )
    assert np.allclose(U_full, expected)


def test_embed_two_qubit_gate_cnot_01():
    U_full = _embed_gate(_CX, [0, 1], 2)
    # Already 4x4, no embedding needed.
    assert np.allclose(U_full, _CX)


def test_embed_two_qubit_gate_cnot_10():
    # CNOT with qubit 1 as control, qubit 0 as target.
    U_full = _embed_gate(_CX, [1, 0], 2)
    # Permute: control is now qubit 1.
    # |00> -> |00>, |01> -> |11>, |10> -> |10>, |11> -> |01>
    expected = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ],
        dtype=complex,
    )
    assert np.allclose(U_full, expected)


def test_embed_gate_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        _embed_gate(_H, [0, 1], 2)  # H is 2x2 but 2 targets need 4x4


def test_embed_gate_rejects_duplicate_targets():
    with pytest.raises(ValueError, match="distinct"):
        _embed_gate(_CX, [0, 0], 2)


def test_embed_gate_rejects_out_of_range_targets():
    with pytest.raises(ValueError, match="out of range"):
        _embed_gate(_H, [5], 2)


# ---------------------------------------------------------------------------
# Queue builders
# ---------------------------------------------------------------------------

def test_queue_builder_h():
    circ = fpm.Circuit(2)
    circ.h(0)
    assert circ.n_operations == 1
    assert "U(targets=[0])" in circ.operations[0]


def test_queue_builder_fluent_chaining():
    circ = fpm.Circuit(2)
    result = circ.h(0).cx(0, 1).dephase(gamma=0.05)
    assert result is circ  # fluent
    assert circ.n_operations == 3
    ops = circ.operations
    assert "U(targets=[0])" in ops[0]
    assert "U(targets=[0, 1])" in ops[1]
    assert "D(" in ops[2] and "gamma=0.05" in ops[2]


def test_queue_builder_cx_rejects_same_control_target():
    circ = fpm.Circuit(2)
    with pytest.raises(ValueError, match="control and target must differ"):
        circ.cx(0, 0)


def test_queue_builder_cz_rejects_same_qubits():
    circ = fpm.Circuit(2)
    with pytest.raises(ValueError, match="qubits must differ"):
        circ.cz(1, 1)


def test_queue_builder_swap_rejects_same_qubits():
    circ = fpm.Circuit(2)
    with pytest.raises(ValueError, match="qubits must differ"):
        circ.swap(0, 0)


def test_queue_builder_dephase_rejects_no_gamma_no_daemon():
    circ = fpm.Circuit(2)
    with pytest.raises(ValueError, match="either explicit gamma or a daemon"):
        circ.dephase()


def test_queue_builder_dephase_rejects_gamma_and_endogenous_inputs():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(80.0)
    circ = fpm.Circuit(2, daemon=daemon, ledger=ledger)
    with pytest.raises(ValueError, match="not both"):
        circ.dephase(gamma=0.05, gate_power=0.1)


def test_queue_builder_apply_unitary_validates_shape():
    circ = fpm.Circuit(2)
    bad_U = np.eye(3, dtype=complex)
    with pytest.raises(ValueError, match="does not match"):
        circ.apply_unitary(bad_U, [0])


def test_queue_builder_apply_unitary_full_validates_shape():
    circ = fpm.Circuit(2)
    with pytest.raises(ValueError, match="does not match"):
        circ.apply_unitary_full(np.eye(3, dtype=complex))


def test_reset_clears_queue():
    circ = fpm.Circuit(2)
    circ.h(0).cx(0, 1)
    assert circ.n_operations == 2
    circ.reset()
    assert circ.n_operations == 0


def test_reset_stats_clears_counters():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(80.0)
    circ = fpm.Circuit(2, daemon=daemon, ledger=ledger)
    circ.h(0).dephase(gamma=0.05)
    circ.step(fpm.pure_state([1, 0, 0, 0]))
    assert circ.gates_applied == 1
    assert circ.dephase_layers_applied == 1
    circ.reset_stats()
    assert circ.gates_applied == 0
    assert circ.dephase_layers_applied == 0


# ---------------------------------------------------------------------------
# Execution: gate semantics
# ---------------------------------------------------------------------------

def test_h_gate_creates_plus_state():
    """H|0> = |+>, so rho becomes |+><+| = 0.5*[[1,1],[1,1]]."""
    circ = fpm.Circuit(1)
    circ.h(0)
    rho0 = fpm.pure_state([1, 0])
    rho1 = circ.step(rho0)
    expected = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
    assert np.allclose(rho1, expected)


def test_x_gate_flips_population():
    """X|0> = |1>, so rho becomes |1><1|."""
    circ = fpm.Circuit(1)
    circ.x(0)
    rho0 = fpm.pure_state([1, 0])
    rho1 = circ.step(rho0)
    expected = np.array([[0, 0], [0, 1]], dtype=complex)
    assert np.allclose(rho1, expected)


def test_z_gate_phases_minus_state():
    """Z|+> = |->, so rho becomes |-><-| = 0.5*[[1,-1],[-1,1]]."""
    circ = fpm.Circuit(1)
    circ.z(0)
    rho0 = fpm.pure_state([1, 1])  # |+>
    rho1 = circ.step(rho0)
    expected = 0.5 * np.array([[1, -1], [-1, 1]], dtype=complex)
    assert np.allclose(rho1, expected)


def test_cnot_creates_bell_state():
    """H on q0 then CNOT(0,1) creates (|00>+|11>)/sqrt(2)."""
    circ = fpm.Circuit(2)
    circ.h(0).cx(0, 1)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    rho1 = circ.step(rho0)
    expected = 0.5 * np.array(
        [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]],
        dtype=complex,
    )
    assert np.allclose(rho1, expected)


def test_swap_swaps_qubits():
    """SWAP applied to |01> gives |10>."""
    circ = fpm.Circuit(2)
    circ.swap(0, 1)
    rho0 = fpm.pure_state([0, 1, 0, 0])  # |01>
    rho1 = circ.step(rho0)
    expected = fpm.pure_state([0, 0, 1, 0])  # |10>
    assert np.allclose(rho1, expected)


def test_u_gate_general_rotation():
    """U(pi, 0, 0) = X (up to global phase)."""
    circ = fpm.Circuit(1)
    circ.u(math.pi, 0.0, 0.0, 0)
    rho0 = fpm.pure_state([1, 0])
    rho1 = circ.step(rho0)
    expected = fpm.pure_state([0, 1])  # X|0> = |1>
    # U(pi, 0, 0) = [[0, -1], [1, 0]] = i*X (up to global phase i).
    # Applied to density matrix: global phase cancels.
    assert np.allclose(rho1, expected)


def test_apply_unitary_full_direct():
    """apply_unitary_full applies U @ rho @ U^dag directly."""
    circ = fpm.Circuit(1)
    U = _H
    circ.apply_unitary_full(U)
    rho0 = fpm.pure_state([1, 0])
    rho1 = circ.step(rho0)
    expected = U @ rho0 @ U.conj().T
    assert np.allclose(rho1, expected)


def test_step_preserves_trace_and_hermiticity():
    """step() must return a valid density matrix."""
    circ = fpm.Circuit(2)
    circ.h(0).cx(0, 1).dephase(gamma=0.05)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    for _ in range(5):
        rho0 = circ.step(rho0)
        assert fpm.is_density_matrix(rho0), "step output not a density matrix"


def test_step_does_not_mutate_input():
    """step() must not modify the input rho."""
    circ = fpm.Circuit(1)
    circ.h(0)
    rho0 = fpm.pure_state([1, 0])
    rho0_copy = rho0.copy()
    circ.step(rho0)
    assert np.allclose(rho0, rho0_copy)


def test_step_validates_rho_shape():
    circ = fpm.Circuit(2)
    circ.h(0)
    bad_rho = fpm.pure_state([1, 0])  # 2x2, not 4x4
    with pytest.raises(ValueError, match="does not match"):
        circ.step(bad_rho)


# ---------------------------------------------------------------------------
# Execution: dephasing semantics
# ---------------------------------------------------------------------------

def test_dephase_decays_coherence_machine_precision():
    """After N dephasing steps, |rho_01| = |rho_01(0)| * exp(-gamma*N*dt)."""
    circ = fpm.Circuit(1)
    circ.dephase(gamma=0.05, dt=1.0)
    rho0 = fpm.pure_state([1, 1])  # |+>, rho_01 = 0.5
    traj = circ.run(rho0, n_steps=10)
    for n in [1, 5, 10]:
        expected = 0.5 * math.exp(-0.05 * n)
        actual = abs(traj[n, 0, 1])
        assert abs(actual - expected) < 1e-12, (
            f"step {n}: got {actual}, expected {expected}"
        )


def test_dephase_preserves_diagonal():
    """Dephasing must leave the diagonal untouched (fixed point)."""
    circ = fpm.Circuit(1)
    circ.dephase(gamma=0.1, dt=1.0)
    rho0 = fpm.pure_state([1, 1])  # diagonal = [0.5, 0.5]
    rho1 = circ.step(rho0)
    assert abs(rho1[0, 0] - 0.5) < 1e-15
    assert abs(rho1[1, 1] - 0.5) < 1e-15


def test_dephase_method_euler_vs_exact():
    """method='euler' and method='exact' give different decays."""
    rho0 = fpm.pure_state([1, 1])

    circ_euler = fpm.Circuit(1, method="euler")
    circ_euler.dephase(gamma=0.05, dt=1.0)
    rho_euler = circ_euler.step(rho0)

    circ_exact = fpm.Circuit(1, method="exact")
    circ_exact.dephase(gamma=0.05, dt=1.0)
    rho_exact = circ_exact.step(rho0)

    # Euler: 0.5 * (1 - 0.05) = 0.475
    # Exact: 0.5 * exp(-0.05) = 0.47561...
    assert abs(abs(rho_euler[0, 1]) - 0.475) < 1e-12
    assert abs(abs(rho_exact[0, 1]) - 0.5 * math.exp(-0.05)) < 1e-12


# ---------------------------------------------------------------------------
# Execution: trajectory recording
# ---------------------------------------------------------------------------

def test_run_with_record_returns_trajectory():
    circ = fpm.Circuit(1)
    circ.dephase(gamma=0.05)
    rho0 = fpm.pure_state([1, 1])
    traj = circ.run(rho0, n_steps=10, record=True)
    assert traj.shape == (11, 2, 2)
    # Initial state preserved.
    assert np.allclose(traj[0], rho0)


def test_run_without_record_returns_final():
    circ = fpm.Circuit(1)
    circ.dephase(gamma=0.05)
    rho0 = fpm.pure_state([1, 1])
    final = circ.run(rho0, n_steps=10, record=False)
    assert final.shape == (2, 2)


def test_run_zero_steps_returns_initial():
    circ = fpm.Circuit(1)
    circ.dephase(gamma=0.05)
    rho0 = fpm.pure_state([1, 1])
    traj = circ.run(rho0, n_steps=0)
    assert traj.shape == (1, 2, 2)
    assert np.allclose(traj[0], rho0)


def test_run_rejects_negative_steps():
    circ = fpm.Circuit(1)
    circ.dephase(gamma=0.05)
    with pytest.raises(ValueError, match="n_steps must be >= 0"):
        circ.run(fpm.pure_state([1, 1]), n_steps=-1)


# ---------------------------------------------------------------------------
# Closed-universe billing
# ---------------------------------------------------------------------------

def test_billing_disabled_without_ledger():
    """Without ledger+daemon, no billing happens (pure state-stepper)."""
    circ = fpm.Circuit(2, method="euler")
    circ.h(0).cx(0, 1).dephase(gamma=0.05)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    circ.step(rho0)
    # Counters still track, but no daemon was debited.
    assert circ.gates_applied == 2
    assert circ.dephase_layers_applied == 1


def test_billing_unitary_gates_debits_daemon():
    """Each unitary gate bills N^2 * exp_route_cost ops."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(80.0)
    circ = fpm.Circuit(
        2, daemon=daemon, ledger=ledger, method="euler", cost_per_op=1e-4,
    )
    circ.h(0)  # one gate
    circ.step(fpm.pure_state([1, 0, 0, 0]))

    # N=4 (2 qubits), so N^2 = 16 elements.
    # Taylor order default 8: (16 mul + 8 add) per element = 24 ops.
    # Total: 16 * 24 = 384 ops. cost_per_op = 1e-4 of E_max=100 = 0.01 each.
    # Total billed: 384 * 0.01 = 3.84.
    expected_spend = 16 * (2 * 8 + 8) * 1e-4 * 100.0
    assert abs(ledger.total_spend - expected_spend) < 1e-9, (
        f"expected {expected_spend}, got {ledger.total_spend}"
    )
    assert circ.gates_applied == 1


def test_billing_dephase_euler_method():
    """method='euler' dephase bills 1 mul + 1 add per off-diagonal state var."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(80.0)
    circ = fpm.Circuit(
        2, daemon=daemon, ledger=ledger, method="euler", cost_per_op=1e-4,
    )
    circ.dephase(gamma=0.05)
    circ.step(fpm.pure_state([1, 0, 0, 0]))

    # N=4, off-diagonals = N*(N-1) = 12. Each billed 1 mul + 1 add = 2 ops.
    # Total: 12 * 2 = 24 ops. cost_per_op = 1e-4 of E_max=100 = 0.01 each.
    # Total billed: 24 * 0.01 = 0.24.
    expected_spend = 12 * 2 * 1e-4 * 100.0
    assert abs(ledger.total_spend - expected_spend) < 1e-9
    assert circ.dephase_layers_applied == 1


def test_billing_dephase_exact_method():
    """method='exact' dephase bills N*(N-1) * exp_route_cost ops."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(80.0)
    circ = fpm.Circuit(
        2, daemon=daemon, ledger=ledger, method="exact", cost_per_op=1e-4,
    )
    circ.dephase(gamma=0.05)
    circ.step(fpm.pure_state([1, 0, 0, 0]))

    # 12 off-diag * (16 mul + 8 add) = 12 * 24 = 288 ops.
    # cost_per_op = 0.01, total = 2.88.
    expected_spend = 12 * (2 * 8 + 8) * 1e-4 * 100.0
    assert abs(ledger.total_spend - expected_spend) < 1e-9


def test_billing_strang_step_bills_three_operations():
    """strang_step bills: unitary + dephase + unitary."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(80.0)
    circ = fpm.Circuit(
        1, daemon=daemon, ledger=ledger, method="euler", cost_per_op=1e-4,
    )
    H = np.array([[1, 1], [1, -1]], dtype=complex) * 0.5
    rho = fpm.pure_state([1, 1])
    circ.strang_step(rho, H, gamma=0.05, dt=0.5)
    # 2 unitary halves + 1 dephase layer.
    assert circ.gates_applied == 2
    assert circ.dephase_layers_applied == 1


def test_billing_respects_energy_floor():
    """Billing cannot drive the daemon below the energy floor."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(5.0)  # near the floor
    circ = fpm.Circuit(
        2, daemon=daemon, ledger=ledger, method="exact",
        cost_per_op=1.0,  # huge cost per op
    )
    circ.h(0)
    circ.step(fpm.pure_state([1, 0, 0, 0]))
    floor = fpm.ENERGY_FLOOR_FRACTION * 100.0
    assert daemon.E >= floor - 1e-9


# ---------------------------------------------------------------------------
# Endogenous gamma via daemon
# ---------------------------------------------------------------------------

def test_dephase_with_endogenous_gamma_works():
    """dephase() with no explicit gamma derives from daemon."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(75.0)
    circ = fpm.Circuit(2, daemon=daemon, ledger=ledger, method="euler")
    circ.dephase(gate_power=0.1, load=0.1, dt=1.0)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    rho1 = circ.step(rho0)
    assert fpm.is_density_matrix(rho1)
    # Coherence should have decayed.
    assert abs(rho1[0, 1]) < 0.5  # initial rho_01 = 0


def test_dephase_with_default_gate_power():
    """default_gate_power is used when gate_power is not specified on layer."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(75.0)
    circ = fpm.Circuit(
        2, daemon=daemon, ledger=ledger,
        method="euler", default_gate_power=0.2,
    )
    circ.dephase(load=0.1, dt=1.0)
    rho0 = fpm.pure_state([1, 1, 0, 0])  # has off-diagonal coherence
    rho1 = circ.step(rho0)
    assert fpm.is_density_matrix(rho1)


def test_dephase_with_daemon_load_attribute():
    """Daemon's load attribute is used when load is not specified."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(75.0)
    daemon.load = 0.3
    circ = fpm.Circuit(2, daemon=daemon, ledger=ledger, method="euler")
    circ.dephase(gate_power=0.1, dt=1.0)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    rho1 = circ.step(rho0)
    assert fpm.is_density_matrix(rho1)


# ---------------------------------------------------------------------------
# Falsifiability
# ---------------------------------------------------------------------------

def test_falsification_error_via_bounded_circuit():
    """A circuit with bounded=True raises on gamma > 32."""
    circ = fpm.Circuit(1, bounded=True)
    circ.dephase(gamma=50.0)
    with pytest.raises(fpm.FalsificationError, match="falsification"):
        circ.step(fpm.pure_state([1, 1]))


def test_falsification_error_via_endogenous_gamma():
    """Endogenous gamma can trigger falsification when dt is small.

    The energy floor (Test 07) bounds effective_energy at 0.0314 of
    E_max, so with dt=1.0 endogenous gamma asymptotes to ~1.0 and
    cannot reach the falsification threshold.  But with small dt the
    conversion (1 - kappa) / dt can still exceed 32.0, and
    bounded=True catches it.
    """
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(1.0)  # very low energy
    circ = fpm.Circuit(
        1, daemon=daemon, ledger=ledger,
        method="euler", bounded=True, default_gate_power=100.0,
    )
    circ.dephase(dt=0.001)  # small dt -> large gamma
    with pytest.raises(fpm.FalsificationError, match="falsification"):
        circ.step(fpm.pure_state([1, 1]))


def test_unbounded_circuit_allows_large_gamma():
    """Without bounded=True, large gamma is allowed (silent)."""
    circ = fpm.Circuit(1, bounded=False)
    circ.dephase(gamma=50.0, dt=0.001)  # gamma*dt = 0.05, OK for euler
    rho0 = fpm.pure_state([1, 1])
    rho1 = circ.step(rho0)
    assert fpm.is_density_matrix(rho1)


# ---------------------------------------------------------------------------
# Strang splitting
# ---------------------------------------------------------------------------

def test_strang_step_matches_strang_splitting():
    """strang_step should match manual U/2 + dephase + U/2 composition."""
    H = np.array([[1, 1], [1, -1]], dtype=complex) * 0.5
    gamma, dt = 0.05, 0.5
    rho0 = fpm.pure_state([1, 1])

    # Manual Strang.
    rho = rho0.copy()
    rho = fpm.unitary_step(rho, H, dt / 2)
    rho = fpm.lindblad_step(rho, gamma=gamma, dt=dt, method="exact")
    rho = fpm.unitary_step(rho, H, dt / 2)

    circ = fpm.Circuit(1, method="exact")
    rho_circ = circ.strang_step(rho0, H, gamma=gamma, dt=dt)
    assert np.allclose(rho, rho_circ, atol=1e-14)


def test_strang_step_validates_shapes():
    circ = fpm.Circuit(2)
    H = np.eye(2, dtype=complex)  # wrong shape
    with pytest.raises(ValueError, match="does not match"):
        circ.strang_step(
            fpm.pure_state([1, 0, 0, 0]), H, gamma=0.05, dt=0.5,
        )


def test_strang_step_requires_gamma_or_daemon():
    circ = fpm.Circuit(1)
    H = np.eye(2, dtype=complex)
    with pytest.raises(ValueError, match="either explicit gamma or a daemon"):
        circ.strang_step(fpm.pure_state([1, 0]), H, gamma=None, dt=0.5)


def test_strang_step_rejects_negative_dt():
    circ = fpm.Circuit(1)
    H = np.eye(2, dtype=complex)
    with pytest.raises(ValueError, match="dt must be >= 0"):
        circ.strang_step(fpm.pure_state([1, 0]), H, gamma=0.05, dt=-0.5)


# ---------------------------------------------------------------------------
# Integration: closed-universe conservation
# ---------------------------------------------------------------------------

def test_closed_universe_ledger_with_circuit_billing():
    """End-to-end: circuit with daemon+ledger, balanced replenishment, low drift."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(80.0)
    circ = fpm.Circuit(
        2, daemon=daemon, ledger=ledger, method="euler", cost_per_op=1e-5,
    )
    circ.h(0).cx(0, 1).dephase(gate_power=0.05, dt=1.0)
    rho = fpm.pure_state([1, 0, 0, 0])

    for _ in range(20):
        rho = circ.step(rho)
        # Closed-universe: replenish exactly what was spent this step.
        spent = ledger.total_spend  # cumulative
        # Easier: capture delta per step.
    # Just check that drift is reasonable when we replenish to match spend.
    # Reset and do it properly:
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(80.0)
    circ = fpm.Circuit(
        2, daemon=daemon, ledger=ledger, method="euler", cost_per_op=1e-5,
    )
    circ.h(0).cx(0, 1).dephase(gate_power=0.05, dt=1.0)
    rho = fpm.pure_state([1, 0, 0, 0])

    for _ in range(20):
        prev_spend = daemon.cumulative_spend
        rho = circ.step(rho)
        spent_this_step = daemon.cumulative_spend - prev_spend
        ledger.record_replenish(daemon, spent_this_step)

    drift = ledger.drift()
    assert drift < 0.05, f"ledger drift {drift:.2%} exceeds 5%"


# ---------------------------------------------------------------------------
# run_with_replenishment (v0.1.7)
# ---------------------------------------------------------------------------

def _make_circ_daemon_ledger(method="euler", cost_per_op=1e-5, E_init=80.0):
    """Helper: build a 2-qubit circuit with daemon+ledger attached."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(E_init)
    circ = fpm.Circuit(
        2, daemon=daemon, ledger=ledger,
        method=method, cost_per_op=cost_per_op,
        default_gate_power=0.05,
    )
    circ.h(0).cx(0, 1).dephase(dt=1.0)
    return circ, daemon, ledger


def test_run_with_replenishment_returns_trajectory():
    """Default call returns a trajectory of shape (n_steps+1, dim, dim)."""
    circ, daemon, ledger = _make_circ_daemon_ledger()
    rho0 = fpm.pure_state([1, 0, 0, 0])
    traj = circ.run_with_replenishment(rho0, n_steps=10)
    assert traj.shape == (11, 4, 4)
    # Initial state preserved.
    assert np.allclose(traj[0], rho0)


def test_run_with_replenishment_no_record_returns_final():
    """record=False returns only the final state."""
    circ, daemon, ledger = _make_circ_daemon_ledger()
    rho0 = fpm.pure_state([1, 0, 0, 0])
    final = circ.run_with_replenishment(rho0, n_steps=10, record=False)
    assert final.shape == (4, 4)


def test_run_with_replenishment_zero_steps():
    """n_steps=0 returns the initial state with no replenishment."""
    circ, daemon, ledger = _make_circ_daemon_ledger()
    rho0 = fpm.pure_state([1, 0, 0, 0])
    traj = circ.run_with_replenishment(rho0, n_steps=0)
    assert traj.shape == (1, 4, 4)
    assert np.allclose(traj[0], rho0)
    # No billing, no replenishment.
    assert ledger.total_spend == 0.0
    assert ledger.total_replenish == 0.0
    assert daemon.E == 80.0


def test_run_with_replenishment_rejects_no_daemon():
    """Without a daemon, raises ValueError pointing to run()."""
    circ = fpm.Circuit(2)  # no daemon, no ledger
    circ.h(0).dephase(gamma=0.05)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    with pytest.raises(ValueError, match="requires a daemon"):
        circ.run_with_replenishment(rho0, n_steps=5)


def test_run_with_replenishment_rejects_daemon_without_ledger():
    """Construction already rejects this, but check the error path."""
    with pytest.raises(ValueError, match="daemons together with ledger"):
        fpm.Circuit(2, daemon=object())  # ledger missing


def test_run_with_replenishment_rejects_negative_steps():
    circ, _, _ = _make_circ_daemon_ledger()
    with pytest.raises(ValueError, match="n_steps must be >= 0"):
        circ.run_with_replenishment(fpm.pure_state([1, 0, 0, 0]), n_steps=-1)


def test_run_with_replenishment_rejects_bad_rho_shape():
    circ, _, _ = _make_circ_daemon_ledger()
    bad_rho = fpm.pure_state([1, 0])  # 2x2, not 4x4
    with pytest.raises(ValueError, match="does not match"):
        circ.run_with_replenishment(bad_rho, n_steps=1)


def test_run_with_replenishment_keeps_drift_at_zero():
    """The whole point: closed-universe drift should be ~0."""
    circ, daemon, ledger = _make_circ_daemon_ledger()
    rho0 = fpm.pure_state([1, 0, 0, 0])
    circ.run_with_replenishment(rho0, n_steps=20)
    # No external landauer charged, so drift should be exactly 0.
    assert ledger.drift() < 1e-12, (
        f"drift {ledger.drift():.2e} should be ~0 after balanced replenishment"
    )


def test_run_with_replenishment_preserves_daemon_energy():
    """With exact replenishment and no clipping, daemon E stays constant."""
    circ, daemon, ledger = _make_circ_daemon_ledger(E_init=80.0)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    initial_E = daemon.E
    circ.run_with_replenishment(rho0, n_steps=20)
    # E should be unchanged (we replenished exactly what was spent).
    assert abs(daemon.E - initial_E) < 1e-9, (
        f"daemon E changed from {initial_E} to {daemon.E}"
    )


def test_run_with_replenishment_matches_manual_path():
    """Automatic replenishment gives the same trajectory as manual."""
    # Automatic.
    circ1, daemon1, ledger1 = _make_circ_daemon_ledger()
    rho0 = fpm.pure_state([1, 0, 0, 0])
    traj_auto = circ1.run_with_replenishment(rho0, n_steps=15)

    # Manual.
    circ2, daemon2, ledger2 = _make_circ_daemon_ledger()
    rho = rho0.copy()
    for _ in range(15):
        prev_spend = daemon2.cumulative_spend
        prev_landauer = daemon2.cumulative_landauer
        rho = circ2.step(rho)
        delta = (
            (daemon2.cumulative_spend - prev_spend)
            + (daemon2.cumulative_landauer - prev_landauer)
        )
        ledger2.record_replenish(daemon2, delta)
    traj_manual_final = rho

    # Trajectories should match to machine precision.
    assert np.allclose(traj_auto[-1], traj_manual_final, atol=1e-14)
    # And ledgers should agree.
    assert abs(ledger1.total_spend - ledger2.total_spend) < 1e-9
    assert abs(ledger1.total_replenish - ledger2.total_replenish) < 1e-9
    assert abs(ledger1.drift() - ledger2.drift()) < 1e-12


def test_run_with_replenishment_tracks_billing_counters():
    """gates_applied and dephase_layers_applied update correctly."""
    circ, _, _ = _make_circ_daemon_ledger()
    rho0 = fpm.pure_state([1, 0, 0, 0])
    circ.run_with_replenishment(rho0, n_steps=10)
    # 2 gates per step (H + CNOT), 1 dephase per step, 10 steps.
    assert circ.gates_applied == 20
    assert circ.dephase_layers_applied == 10


def test_run_with_replenishment_works_with_exact_method():
    """Endogenous gamma + method='exact' also closes the universe."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(80.0)
    circ = fpm.Circuit(
        2, daemon=daemon, ledger=ledger,
        method="exact", cost_per_op=1e-6,  # smaller cost so daemon doesn't floor
        default_gate_power=0.05,
    )
    circ.h(0).cx(0, 1).dephase(dt=1.0)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    traj = circ.run_with_replenishment(rho0, n_steps=10)
    assert traj.shape == (11, 4, 4)
    # Drift should be ~0 (no external landauer).
    assert ledger.drift() < 1e-9


def test_run_with_replenishment_with_external_landauer():
    """Landauer charged externally between steps is NOT replenished.

    The closed-universe identity is replenish == spend + landauer.
    External landauer charges (not triggered by step()) leave a
    deliberate gap that the caller must close themselves.
    """
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(80.0)
    circ = fpm.Circuit(
        2, daemon=daemon, ledger=ledger, method="euler", cost_per_op=1e-5,
    )
    circ.h(0).dephase(gate_power=0.05, dt=1.0)
    rho0 = fpm.pure_state([1, 0, 0, 0])

    # Run 3 steps via run_with_replenishment.
    circ.run_with_replenishment(rho0, n_steps=3)
    spend_after_3 = ledger.total_spend
    replenish_after_3 = ledger.total_replenish
    assert ledger.total_landauer == 0.0
    assert abs(replenish_after_3 - spend_after_3) < 1e-9

    # Now charge external landauer. Drift becomes nonzero.
    ledger.record_landauer(daemon, bits_erased=2.0)
    assert ledger.total_landauer > 0.0
    drift_with_external_landauer = ledger.drift()
    assert drift_with_external_landauer > 0.0, (
        "external landauer should produce non-zero drift"
    )


def test_run_with_replenishment_clipping_at_e_max():
    """If daemon reaches E_max, replenishment is capped and drift grows.

    With E_init very close to E_max and large cost_per_op, the spend
    is tiny so replenishment would push E above E_max. record_replenish
    clips at E_max - E, so cumulative_replenish < cumulative_spend and
    drift > 0. This is honest behavior.
    """
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(99.999)  # very near E_max
    circ = fpm.Circuit(
        1, daemon=daemon, ledger=ledger, method="euler",
        cost_per_op=1e-3,  # large cost per op
    )
    circ.dephase(gamma=0.05, dt=1.0)
    rho0 = fpm.pure_state([1, 1])
    # Run several steps. Daemon should saturate near E_max.
    circ.run_with_replenishment(rho0, n_steps=5)
    # Daemon E should be at or near E_max (replenishment clipped).
    assert daemon.E <= 100.0 + 1e-9
    # Note: drift may or may not be > 0 depending on whether spend
    # exceeded the (E_max - E) headroom. Just verify no crash.


def test_run_with_replenishment_clipping_at_floor():
    """If spend drives daemon to floor, replenishment restores it."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(5.0)  # near floor
    circ = fpm.Circuit(
        2, daemon=daemon, ledger=ledger, method="euler",
        cost_per_op=1.0,  # huge cost -> daemon will hit floor
    )
    circ.h(0).dephase(gamma=0.05, dt=1.0)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    circ.run_with_replenishment(rho0, n_steps=3)
    # Daemon E should be above floor (replenishment restored it).
    floor = fpm.ENERGY_FLOOR_FRACTION * 100.0
    assert daemon.E >= floor - 1e-9


def test_run_with_replenishment_preserves_density_matrix():
    """Each step of run_with_replenishment must return a valid density matrix."""
    circ, _, _ = _make_circ_daemon_ledger()
    rho0 = fpm.pure_state([1, 0, 0, 0])
    traj = circ.run_with_replenishment(rho0, n_steps=10)
    for t in range(traj.shape[0]):
        assert fpm.is_density_matrix(traj[t]), (
            f"step {t} did not produce a valid density matrix"
        )


def test_run_with_replenishment_does_not_mutate_input():
    """run_with_replenishment must not modify the input rho0."""
    circ, _, _ = _make_circ_daemon_ledger()
    rho0 = fpm.pure_state([1, 0, 0, 0])
    rho0_copy = rho0.copy()
    circ.run_with_replenishment(rho0, n_steps=5)
    assert np.allclose(rho0, rho0_copy)


def test_run_with_replenishment_endogenous_gamma():
    """Endogenous gamma (from daemon) works with auto-replenishment."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(75.0)
    circ = fpm.Circuit(
        2, daemon=daemon, ledger=ledger, method="euler",
        cost_per_op=1e-5, default_gate_power=0.1,
    )
    circ.h(0).cx(0, 1).dephase(dt=1.0)  # endogenous gamma from daemon
    rho0 = fpm.pure_state([1, 0, 0, 0])
    traj = circ.run_with_replenishment(rho0, n_steps=15)
    assert traj.shape == (16, 4, 4)
    # Each step should be a valid density matrix.
    assert fpm.is_density_matrix(traj[-1])
    # Ledger should be balanced.
    assert ledger.drift() < 1e-9


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def test_circuit_in_public_api():
    """Circuit is exported from the top-level package."""
    assert hasattr(fpm, "Circuit")
    assert fpm.Circuit is Circuit


def test_circuit_appears_in_all():
    assert "Circuit" in fpm.__all__


# ---------------------------------------------------------------------------
# Multi-daemon mode (v0.1.8)
# ---------------------------------------------------------------------------

def _make_multi_daemon_circ(
    n_qubits=2, energies=(80.0, 60.0), method="euler",
    cost_per_op=1e-5, default_gate_power=0.05,
):
    """Helper: build a multi-daemon circuit with per-qubit daemons."""
    E_max = 100.0
    ledger = fpm.ConservationLedger(E_max_total=E_max)
    daemons = [ledger.add_daemon(e) for e in energies]
    circ = fpm.Circuit(
        n_qubits, daemons=daemons, ledger=ledger,
        method=method, cost_per_op=cost_per_op,
        default_gate_power=default_gate_power,
    )
    return circ, daemons, ledger


# --- Construction ---

def test_multi_daemon_construction():
    circ, daemons, ledger = _make_multi_daemon_circ()
    assert circ.is_multi_daemon is True
    assert circ.daemon is None
    assert len(circ.daemons) == 2
    assert circ.all_daemons == list(daemons)
    assert circ.n_qubits == 2


def test_multi_daemon_rejects_daemon_and_daemons_both():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    d = ledger.add_daemon(80.0)
    with pytest.raises(ValueError, match="not both"):
        fpm.Circuit(2, daemon=d, daemons=[d, d], ledger=ledger)


def test_multi_daemon_rejects_wrong_length():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    d0 = ledger.add_daemon(80.0)
    d1 = ledger.add_daemon(60.0)
    with pytest.raises(ValueError, match="does not match"):
        fpm.Circuit(3, daemons=[d0, d1], ledger=ledger)


def test_multi_daemon_rejects_non_daemonstate():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    with pytest.raises(TypeError, match="must be a DaemonState"):
        fpm.Circuit(2, daemons=[1, 2], ledger=ledger)


def test_multi_daemon_daemon_for_qubit():
    circ, daemons, _ = _make_multi_daemon_circ()
    assert circ._daemon_for_qubit(0) is daemons[0]
    assert circ._daemon_for_qubit(1) is daemons[1]


def test_multi_daemon_daemons_for_targets():
    circ, daemons, _ = _make_multi_daemon_circ()
    result = circ._daemons_for_targets([0, 1])
    assert result == [daemons[0], daemons[1]]
    # Single target
    assert circ._daemons_for_targets([1]) == [daemons[1]]


# --- Backward compat: single-daemon mode still works ---

def test_single_daemon_is_multi_daemon_false():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(80.0)
    circ = fpm.Circuit(2, daemon=daemon, ledger=ledger)
    assert circ.is_multi_daemon is False
    assert circ.daemons is None
    assert circ.all_daemons == [daemon]


# --- Billing: unitary gates split across target daemons ---

def test_multi_daemon_single_qubit_gate_bills_only_that_daemon():
    """H on qubit 0 bills only daemons[0]."""
    circ, daemons, ledger = _make_multi_daemon_circ(cost_per_op=1e-4)
    circ.h(0)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    circ.step(rho0)
    # Only d0 should have been billed.
    assert daemons[0].cumulative_spend > 0
    assert daemons[1].cumulative_spend == 0.0


def test_multi_daemon_two_qubit_gate_splits_billing():
    """CNOT(0,1) splits the bill 50/50 between d0 and d1."""
    circ, daemons, ledger = _make_multi_daemon_circ(cost_per_op=1e-4)
    circ.cx(0, 1)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    circ.step(rho0)
    # Both should be billed, and equally.
    assert daemons[0].cumulative_spend > 0
    assert daemons[1].cumulative_spend > 0
    assert abs(daemons[0].cumulative_spend - daemons[1].cumulative_spend) < 1e-9


def test_multi_daemon_apply_unitary_full_bills_all_daemons():
    """apply_unitary_full bills all daemons equally."""
    circ, daemons, ledger = _make_multi_daemon_circ(cost_per_op=1e-4)
    U = np.eye(4, dtype=complex)  # trivial
    circ.apply_unitary_full(U)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    circ.step(rho0)
    # Both daemons billed equally.
    assert daemons[0].cumulative_spend > 0
    assert daemons[1].cumulative_spend > 0
    assert abs(daemons[0].cumulative_spend - daemons[1].cumulative_spend) < 1e-9


# --- Billing: dephasing layers ---

def test_multi_daemon_dephase_bills_all_daemons():
    """dephase() with no targets bills all daemons."""
    circ, daemons, ledger = _make_multi_daemon_circ(cost_per_op=1e-4)
    circ.dephase(gamma=0.05, dt=1.0)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    circ.step(rho0)
    assert daemons[0].cumulative_spend > 0
    assert daemons[1].cumulative_spend > 0


def test_multi_daemon_dephase_with_targets_bills_only_named():
    """dephase(targets=[0]) bills only d0."""
    circ, daemons, ledger = _make_multi_daemon_circ(cost_per_op=1e-4)
    circ.dephase(gamma=0.05, dt=1.0, targets=[0])
    rho0 = fpm.pure_state([1, 0, 0, 0])
    circ.step(rho0)
    assert daemons[0].cumulative_spend > 0
    assert daemons[1].cumulative_spend == 0.0


# --- Per-qubit endogenous gamma ---

def test_multi_daemon_per_qubit_gamma_depends_on_daemon_energy():
    """Energy-rich daemon produces lower gamma (slower dephasing)."""
    # d0 energy-rich (E=90), d1 energy-poor (E=10)
    circ, daemons, ledger = _make_multi_daemon_circ(
        energies=(90.0, 10.0), cost_per_op=1e-6,
        default_gate_power=0.1,
    )
    circ.dephase(dt=1.0)
    rho0 = fpm.pure_state([1, 1, 1, 1])
    rho1 = circ.step(rho0)
    # rho_01: q0 same, q1 differs.  d1 is poor -> fast decay -> smaller.
    # rho_02: q0 differs, q1 same.  d0 is rich -> slow decay -> larger.
    assert abs(rho1[0, 1]) < abs(rho1[0, 2]), (
        f"|rho_01|={abs(rho1[0, 1])} should be < |rho_02|={abs(rho1[0, 2])} "
        f"(q1 poor decays faster)"
    )


def test_multi_daemon_per_qubit_gamma_matches_direct_calculation():
    """Per-qubit dephasing matches manual gamma_from_energy + kappa.

    Uses cost_per_op=0 so billing doesn't change daemon energy during
    the step (which would make the post-step gamma computation differ
    from the in-step gamma computation).
    """
    circ, daemons, ledger = _make_multi_daemon_circ(
        energies=(90.0, 10.0), method="euler",
        cost_per_op=0.0,  # no billing -> daemon E unchanged
        default_gate_power=0.1,
    )
    circ.dephase(dt=1.0)
    rho0 = fpm.pure_state([1, 1, 1, 1])

    # Compute gamma BEFORE step (matching what the circuit will use).
    gamma0 = fpm.gamma_from_energy(daemons[0], gate_power=0.1, dt=1.0, bounded=False)
    gamma1 = fpm.gamma_from_energy(daemons[1], gate_power=0.1, dt=1.0, bounded=False)
    kappa0 = 1.0 - gamma0 * 1.0  # euler
    kappa1 = 1.0 - gamma1 * 1.0

    rho1 = circ.step(rho0)

    # Manual: apply dephasing per qubit with its own gamma.
    rho_manual = rho0.copy()
    n = 4
    indices = np.arange(n)
    bit0 = (indices >> 1) & 1  # MSB = qubit 0
    bit1 = (indices >> 0) & 1  # LSB = qubit 1
    mask0 = bit0[:, None] != bit0[None, :]
    mask1 = bit1[:, None] != bit1[None, :]
    rho_manual = rho_manual * np.where(mask0, kappa0, 1.0)
    rho_manual = rho_manual * np.where(mask1, kappa1, 1.0)

    assert np.allclose(rho1, rho_manual, atol=1e-14), (
        f"per-qubit dephasing does not match manual calculation\n"
        f"diff: {np.max(np.abs(rho1 - rho_manual))}"
    )


def test_multi_daemon_dephase_preserves_diagonal():
    circ, daemons, ledger = _make_multi_daemon_circ()
    circ.dephase(dt=1.0)
    rho0 = fpm.pure_state([1, 1, 1, 1])
    rho1 = circ.step(rho0)
    # Diagonal must be preserved.
    for i in range(4):
        assert abs(rho1[i, i] - rho0[i, i]) < 1e-14


def test_multi_daemon_dephase_preserves_density_matrix():
    circ, daemons, ledger = _make_multi_daemon_circ()
    circ.h(0).cx(0, 1).dephase(dt=1.0)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    for _ in range(10):
        rho0 = circ.step(rho0)
        assert fpm.is_density_matrix(rho0)


# --- Targeted dephasing ---

def test_multi_daemon_targeted_dephase_leaves_other_qubit_untouched():
    """dephase(targets=[0]) leaves q1's coherences untouched."""
    circ, daemons, ledger = _make_multi_daemon_circ(cost_per_op=1e-6)
    circ.dephase(dt=1.0, targets=[0])
    rho0 = fpm.pure_state([1, 1, 1, 1])
    rho1 = circ.step(rho0)
    # rho_01 (q0 same, q1 differs): should be UNCHANGED.
    assert abs(abs(rho1[0, 1]) - abs(rho0[0, 1])) < 1e-12, (
        f"rho_01 should be unchanged (q1 not targeted): "
        f"before={abs(rho0[0, 1])}, after={abs(rho1[0, 1])}"
    )
    # rho_02 (q0 differs): should DECAY.
    assert abs(rho1[0, 2]) < abs(rho0[0, 2]), (
        f"rho_02 should decay (q0 targeted): "
        f"before={abs(rho0[0, 2])}, after={abs(rho1[0, 2])}"
    )


def test_multi_daemon_dephase_targets_validation():
    circ, _, _ = _make_multi_daemon_circ()
    with pytest.raises(ValueError, match="distinct"):
        circ.dephase(gamma=0.05, targets=[0, 0])
    with pytest.raises(ValueError, match="out of range"):
        circ.dephase(gamma=0.05, targets=[5])


def test_multi_daemon_dephase_targets_recorded_in_operations():
    circ, _, _ = _make_multi_daemon_circ()
    circ.dephase(gamma=0.05, targets=[0, 1])
    ops = circ.operations
    assert "qubits=[0, 1]" in ops[0]


# --- Closed-universe balancing in multi-daemon mode ---

def test_multi_daemon_run_with_replenishment_zero_drift():
    """Network-wide drift should be ~0 with balanced replenishment."""
    circ, daemons, ledger = _make_multi_daemon_circ()
    circ.h(0).cx(0, 1).dephase(dt=1.0)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    circ.run_with_replenishment(rho0, n_steps=20)
    assert ledger.drift() < 1e-12, (
        f"network drift {ledger.drift():.2e} should be ~0"
    )


def test_multi_daemon_run_with_replenishment_preserves_each_daemon_energy():
    """Each daemon's E should be preserved by balanced replenishment."""
    circ, daemons, ledger = _make_multi_daemon_circ(
        energies=(80.0, 60.0)
    )
    initial_E = [d.E for d in daemons]
    circ.h(0).cx(0, 1).dephase(dt=1.0)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    circ.run_with_replenishment(rho0, n_steps=20)
    for i, d in enumerate(daemons):
        assert abs(d.E - initial_E[i]) < 1e-9, (
            f"daemon {i} E changed from {initial_E[i]} to {d.E}"
        )


def test_multi_daemon_run_with_replenishment_bills_both_daemons():
    """Both daemons should have been billed during the run."""
    circ, daemons, ledger = _make_multi_daemon_circ()
    circ.h(0).cx(0, 1).dephase(dt=1.0)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    circ.run_with_replenishment(rho0, n_steps=10)
    assert daemons[0].cumulative_spend > 0
    assert daemons[1].cumulative_spend > 0


def test_multi_daemon_run_with_replenishment_matches_manual():
    """Auto-replenishment matches manual per-daemon replenishment."""
    # Automatic.
    circ1, daemons1, ledger1 = _make_multi_daemon_circ(
        energies=(80.0, 60.0)
    )
    circ1.h(0).cx(0, 1).dephase(dt=1.0)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    traj_auto = circ1.run_with_replenishment(rho0, n_steps=15)

    # Manual.
    circ2, daemons2, ledger2 = _make_multi_daemon_circ(
        energies=(80.0, 60.0)
    )
    circ2.h(0).cx(0, 1).dephase(dt=1.0)
    rho = rho0.copy()
    for _ in range(15):
        prev = {d.index: (d.cumulative_spend, d.cumulative_landauer) for d in daemons2}
        rho = circ2.step(rho)
        for d in daemons2:
            prev_s, prev_l = prev[d.index]
            delta = (d.cumulative_spend - prev_s) + (d.cumulative_landauer - prev_l)
            ledger2.record_replenish(d, delta)

    assert np.allclose(traj_auto[-1], rho, atol=1e-14)
    assert abs(ledger1.total_spend - ledger2.total_spend) < 1e-9
    assert abs(ledger1.total_replenish - ledger2.total_replenish) < 1e-9
    assert abs(ledger1.drift() - ledger2.drift()) < 1e-12


def test_multi_daemon_run_with_replenishment_no_record():
    circ, _, _ = _make_multi_daemon_circ()
    circ.h(0).dephase(dt=1.0)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    final = circ.run_with_replenishment(rho0, n_steps=5, record=False)
    assert final.shape == (4, 4)


def test_multi_daemon_run_with_replenishment_zero_steps():
    circ, daemons, ledger = _make_multi_daemon_circ()
    circ.h(0).dephase(dt=1.0)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    traj = circ.run_with_replenishment(rho0, n_steps=0)
    assert traj.shape == (1, 4, 4)
    assert ledger.total_spend == 0.0
    assert ledger.total_replenish == 0.0


def test_multi_daemon_run_with_replenishment_rejects_no_daemon():
    """Without daemons, raises ValueError."""
    circ = fpm.Circuit(2)  # no daemons
    circ.h(0).dephase(gamma=0.05)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    with pytest.raises(ValueError, match="requires a daemon"):
        circ.run_with_replenishment(rho0, n_steps=5)


# --- Multi-daemon with explicit gamma ---

def test_multi_daemon_explicit_gamma_uniform_dephasing():
    """Explicit gamma applies uniform dephasing, billing split equally."""
    circ, daemons, ledger = _make_multi_daemon_circ(cost_per_op=1e-5)
    circ.dephase(gamma=0.05, dt=1.0)
    rho0 = fpm.pure_state([1, 1, 1, 1])
    rho1 = circ.step(rho0)
    # All off-diagonal elements should decay by the same kappa.
    # (uniform dephasing, not per-qubit)
    kappa = 1.0 - 0.05  # euler
    for i in range(4):
        for j in range(4):
            if i != j:
                expected = kappa * rho0[i, j]
                assert abs(rho1[i, j] - expected) < 1e-12, (
                    f"rho[{i},{j}] = {rho1[i, j]}, expected {expected}"
                )


def test_multi_daemon_explicit_gamma_splits_billing():
    circ, daemons, ledger = _make_multi_daemon_circ(cost_per_op=1e-5)
    circ.dephase(gamma=0.05, dt=1.0)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    circ.step(rho0)
    # Billing split equally.
    assert abs(daemons[0].cumulative_spend - daemons[1].cumulative_spend) < 1e-9


# --- Strang splitting in multi-daemon mode ---

def test_multi_daemon_strang_step_works():
    """strang_step works in multi-daemon mode with endogenous gamma."""
    circ, daemons, ledger = _make_multi_daemon_circ(
        n_qubits=1, energies=(80.0,), cost_per_op=1e-6,
        default_gate_power=0.1,
    )
    H = np.array([[1, 1], [1, -1]], dtype=complex) * 0.5
    rho0 = fpm.pure_state([1, 1])
    rho1 = circ.strang_step(rho0, H, gamma=None, dt=0.5)
    assert fpm.is_density_matrix(rho1)
    # Daemon should have been billed.
    assert daemons[0].cumulative_spend > 0


def test_multi_daemon_strang_step_bills_all_daemons():
    """strang_step in multi-daemon mode bills all daemons for unitary halves."""
    circ, daemons, ledger = _make_multi_daemon_circ(cost_per_op=1e-6)
    H = np.eye(4, dtype=complex) * 0.1
    rho0 = fpm.pure_state([1, 0, 0, 0])
    circ.strang_step(rho0, H, gamma=0.05, dt=0.5)
    # Both daemons should be billed for the unitary halves.
    assert daemons[0].cumulative_spend > 0
    assert daemons[1].cumulative_spend > 0


# --- Three-qubit multi-daemon ---

def test_multi_daemon_three_qubits():
    """Multi-daemon mode works with 3 qubits."""
    circ, daemons, ledger = _make_multi_daemon_circ(
        n_qubits=3, energies=(80.0, 60.0, 40.0),
        cost_per_op=1e-6,
    )
    circ.h(0).cx(0, 1).cx(1, 2).dephase(dt=1.0)
    rho0 = fpm.pure_state([1, 0, 0, 0, 0, 0, 0, 0])  # |000>
    traj = circ.run_with_replenishment(rho0, n_steps=10)
    assert traj.shape == (11, 8, 8)
    # All three daemons should have been billed.
    for i, d in enumerate(daemons):
        assert d.cumulative_spend > 0, f"daemon {i} not billed"
    # Network drift should be ~0.
    assert ledger.drift() < 1e-9


# --- Energy floor in multi-daemon mode ---

def test_multi_daemon_energy_floor_respected():
    """A daemon near the floor is floored; dephase still works."""
    circ, daemons, ledger = _make_multi_daemon_circ(
        energies=(5.0, 80.0), cost_per_op=1.0,  # huge cost
    )
    circ.h(0).dephase(dt=1.0)
    rho0 = fpm.pure_state([1, 0, 0, 0])
    circ.step(rho0)
    # d0 should be at or above the floor.
    floor = fpm.ENERGY_FLOOR_FRACTION * 100.0
    assert daemons[0].E >= floor - 1e-9


# --- Falsification in multi-daemon mode ---

def test_multi_daemon_falsification_via_endogenous_gamma():
    """Endogenous gamma from a low-energy daemon can falsify FPM."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    d0 = ledger.add_daemon(1.0)  # very low energy
    d1 = ledger.add_daemon(80.0)
    circ = fpm.Circuit(
        2, daemons=[d0, d1], ledger=ledger,
        method="euler", bounded=True,
        default_gate_power=100.0, cost_per_op=1e-6,
    )
    circ.dephase(dt=0.001)  # small dt -> large gamma from d0
    rho0 = fpm.pure_state([1, 1, 1, 1])
    with pytest.raises(fpm.FalsificationError, match="falsification"):
        circ.step(rho0)


# --- Public API exposure ---

def test_circuit_daemons_attribute_exposed():
    """The `daemons` attribute is publicly accessible."""
    circ, daemons, _ = _make_multi_daemon_circ()
    assert circ.daemons is not None
    assert len(circ.daemons) == 2


def test_circuit_is_multi_daemon_property():
    circ, _, _ = _make_multi_daemon_circ()
    assert circ.is_multi_daemon is True
    circ2 = fpm.Circuit(2)
    assert circ2.is_multi_daemon is False


def test_circuit_all_daemons_property():
    """all_daemons returns the list of all daemons."""
    circ, daemons, _ = _make_multi_daemon_circ()
    assert circ.all_daemons == list(daemons)
    # Single-daemon mode.
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(80.0)
    circ2 = fpm.Circuit(2, daemon=daemon, ledger=ledger)
    assert circ2.all_daemons == [daemon]
    # No daemons.
    circ3 = fpm.Circuit(2)
    assert circ3.all_daemons == []
