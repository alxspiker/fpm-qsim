"""Tests for v0.1.4 method='euler' / 'exact' ontological distinction."""
import math
import numpy as np
import pytest

import fpm_qsim as fpm
from fpm_qsim._reference import euler_lindblad_step


def test_method_euler_matches_theorem3():
    """method='euler' is the literal Theorem 3 affine map (kappa = 1 - gamma*dt).
    It must match the private reference Euler implementation exactly.
    """
    rho0 = fpm.pure_state([1, 1])
    gamma, dt = 0.1, 1.0
    rho_euler_public = fpm.lindblad_step(rho0, gamma=gamma, dt=dt, method="euler")
    rho_euler_ref = euler_lindblad_step(rho0, gamma=gamma, dt=dt)
    assert np.allclose(rho_euler_public, rho_euler_ref, atol=1e-15)


def test_method_exact_uses_exp():
    """method='exact' uses kappa = exp(-gamma*dt)."""
    rho0 = fpm.pure_state([1, 1])
    gamma, dt = 0.1, 1.0
    rho_exact = fpm.lindblad_step(rho0, gamma=gamma, dt=dt, method="exact")
    # kappa = exp(-0.1) = 0.9048...
    expected_kappa = math.exp(-0.1)
    expected = expected_kappa * rho0
    np.fill_diagonal(expected, np.diagonal(rho0))
    assert np.allclose(rho_exact, expected, atol=1e-15)


def test_method_exact_is_default():
    """Default method is 'exact' (backward compat for engineering users)."""
    rho0 = fpm.pure_state([1, 1])
    rho_default = fpm.lindblad_step(rho0, gamma=0.1, dt=1.0)
    rho_explicit = fpm.lindblad_step(rho0, gamma=0.1, dt=1.0, method="exact")
    assert np.allclose(rho_default, rho_explicit, atol=1e-15)


def test_method_euler_invalid_for_large_gamma_dt():
    """method='euler' raises ValueError when gamma*dt > 1 (non-contractive)."""
    rho = fpm.pure_state([1, 1])
    with pytest.raises(ValueError, match="contractive"):
        fpm.lindblad_step(rho, gamma=2.0, dt=1.0, method="euler")


def test_method_invalid_raises():
    """Invalid method names raise ValueError."""
    rho = fpm.pure_state([1, 1])
    with pytest.raises(ValueError, match="method must be"):
        fpm.lindblad_step(rho, gamma=0.1, dt=1.0, method="bogus")


def test_method_euler_is_oracle_free():
    """method='euler' uses no transcendental functions in the kappa construction.

    This is the FPM-native lattice map.  The kappa value 1 - gamma*dt
    is constructed via one multiply and one add — both finite-integer
    operations on the discrete lattice.
    """
    # We can't directly test "no transcendental functions were called"
    # but we can verify that the kappa value matches 1 - gamma*dt
    # exactly (to machine precision), which it must by construction.
    rho0 = fpm.pure_state([1, 1])
    gamma, dt = 0.05, 1.0
    rho_euler = fpm.lindblad_step(rho0, gamma=gamma, dt=dt, method="euler")
    expected_kappa = 1.0 - gamma * dt  # 0.95
    expected = expected_kappa * rho0
    np.fill_diagonal(expected, np.diagonal(rho0))
    assert np.allclose(rho_euler, expected, atol=1e-15)


def test_exp_route_cost_returns_reasonable_counts():
    """exp_route_cost returns (2K multiplies, K adds) for K-term Taylor."""
    n_mul, n_add = fpm.exp_route_cost(taylor_order=8)
    assert n_mul == 16
    assert n_add == 8


def test_exp_route_cost_zero_order():
    """Zero-order Taylor is the constant 1 — costs nothing."""
    n_mul, n_add = fpm.exp_route_cost(taylor_order=0)
    assert n_mul == 0
    assert n_add == 0


def test_bill_compute_cost_debits_daemon():
    """bill_compute_cost deducts energy from the daemon."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    d = ledger.add_daemon(50.0)
    initial_E = d.E
    debited = ledger.bill_compute_cost(
        d, n_multiplies=10, n_adds=5, cost_per_op=1e-4,
    )
    assert debited > 0
    assert d.E < initial_E
    assert d.cumulative_spend == debited
    # Total spend should reflect the debit.
    assert ledger.total_spend == debited


def test_bill_compute_cost_respects_floor():
    """bill_compute_cost cannot drive the daemon below the energy floor."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    d = ledger.add_daemon(50.0)
    # Huge bill — should hit floor.
    debited = ledger.bill_compute_cost(
        d, n_multiplies=10_000, n_adds=10_000, cost_per_op=1e-2,
    )
    floor = fpm.ENERGY_FLOOR_FRACTION * 100.0
    assert d.E >= floor - 1e-9
    # And the actual debit is less than the (huge) requested amount.
    assert debited < 50.0


def test_bill_exp_route_cost_combined_helper():
    """bill_exp_route_cost bills the simulated Taylor construction."""
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    d = ledger.add_daemon(50.0)
    initial_E = d.E
    debited = fpm.bill_exp_route_cost(
        ledger, d, taylor_order=8, cost_per_op=1e-4,
    )
    assert debited > 0
    assert d.E < initial_E
    # Should bill (16 multiplies + 8 adds) = 24 ops.
    assert d.cumulative_spend == debited


def test_closed_universe_ledger_with_euler_billing():
    """End-to-end: use method='euler' with explicit billing, check ledger closes.

    For the closed-universe identity to hold (replenish == spend +
    landauer), the replenishment must exactly cover what was spent.
    """
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    d = ledger.add_daemon(50.0)
    rho0 = fpm.pure_state([1, 1])
    rho = rho0.copy()

    total_billed = 0.0
    # Run 10 euler steps, billing 1 mul + 1 add per step.
    for _ in range(10):
        rho = fpm.lindblad_step(rho, gamma=0.05, dt=1.0, method="euler")
        # Bill the simulated compute cost.
        billed = ledger.bill_compute_cost(
            d, n_multiplies=1, n_adds=1, cost_per_op=1e-4,
        )
        total_billed += billed
        # Closed-universe replenishment: cover exactly what was spent.
        ledger.record_replenish(d, billed)

    # With balanced replenishment, drift should be near zero.
    drift = ledger.drift()
    assert drift < 0.05, f"ledger drift {drift:.2%} exceeds 5%"


def test_closed_universe_ledger_with_exact_billing():
    """End-to-end: use method='exact' with bill_exp_route_cost, check ledger closes.

    This addresses the audit's core point: when using the oracle
    method='exact', the caller can keep the ledger closed by
    explicitly billing the simulated Taylor construction cost.
    """
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    d = ledger.add_daemon(50.0)
    rho0 = fpm.pure_state([1, 1])
    rho = rho0.copy()

    total_billed = 0.0
    # Run 10 exact steps, billing the Taylor construction of exp each step.
    for _ in range(10):
        rho = fpm.lindblad_step(rho, gamma=0.05, dt=1.0, method="exact")
        # Bill the simulated Taylor construction of exp(-gamma*dt).
        billed = fpm.bill_exp_route_cost(
            ledger, d, taylor_order=8, cost_per_op=1e-4,
        )
        total_billed += billed
        # Closed-universe replenishment: cover exactly what was spent.
        ledger.record_replenish(d, billed)

    drift = ledger.drift()
    assert drift < 0.05, f"ledger drift {drift:.2%} exceeds 5%"


def test_simulate_passes_method_through():
    """simulate() forwards the method parameter to lindblad_step."""
    rho0 = fpm.pure_state([1, 1])
    traj_euler = fpm.simulate(rho0, gamma=0.05, dt=1.0, n_steps=10, method="euler")
    traj_exact = fpm.simulate(rho0, gamma=0.05, dt=1.0, n_steps=10, method="exact")
    # The two trajectories should differ (Euler has O(dt) error per step).
    assert not np.allclose(traj_euler[-1], traj_exact[-1], atol=1e-3)
