"""Tests for endogenous energy-derived gamma."""

import numpy as np
import pytest

import fpm_qsim as fpm


def test_gamma_from_energy_matches_formula():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(50.0)

    gamma = fpm.gamma_from_energy(
        daemon, gate_power=0.25, load=0.5, dt=2.0, C_N=0.9,
        bounded=False,
    )

    energy_fraction = daemon.energy_fraction
    effective_load = 0.5 + 0.25 / energy_fraction
    expected_kappa = 0.9 * (1.0 + effective_load) ** (-0.75)
    expected_gamma = (1.0 - expected_kappa) / 2.0
    assert abs(gamma - expected_gamma) < 1e-15


def test_gamma_from_energy_increases_with_gate_power():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(80.0)

    low = fpm.gamma_from_energy(daemon, gate_power=0.1, load=0.0)
    high = fpm.gamma_from_energy(daemon, gate_power=0.5, load=0.0)

    assert high > low


def test_gamma_from_energy_increases_with_load():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(80.0)

    low = fpm.gamma_from_energy(daemon, gate_power=0.1, load=0.1)
    high = fpm.gamma_from_energy(daemon, gate_power=0.1, load=1.0)

    assert high > low


def test_gamma_from_energy_increases_when_daemon_energy_falls():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    high_energy = ledger.add_daemon(90.0)
    low_energy = ledger.add_daemon(20.0)

    gamma_high_energy = fpm.gamma_from_energy(
        high_energy, gate_power=0.2, load=0.1,
    )
    gamma_low_energy = fpm.gamma_from_energy(
        low_energy, gate_power=0.2, load=0.1,
    )

    assert gamma_low_energy > gamma_high_energy


def test_gamma_from_energy_reads_daemon_load_attribute():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(50.0)
    daemon.load = 0.25

    implicit = fpm.gamma_from_energy(daemon, gate_power=0.1)
    explicit = fpm.gamma_from_energy(daemon, gate_power=0.1, load=0.25)

    assert implicit == explicit


def test_energy_derived_gamma_works_with_euler_step():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(75.0)
    rho0 = fpm.pure_state([1, 1])

    gamma = fpm.gamma_from_energy(daemon, gate_power=0.1, load=0.1, dt=1.0)
    rho1 = fpm.lindblad_step(rho0, gamma=gamma, dt=1.0, method="euler")

    assert fpm.is_density_matrix(rho1)
    assert np.isclose(rho1[0, 0], rho0[0, 0])
    assert abs(rho1[0, 1]) < abs(rho0[0, 1])


def test_lindblad_step_accepts_daemon_and_gate_power():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(75.0)
    rho0 = fpm.pure_state([1, 1])

    gamma = fpm.gamma_from_energy(daemon, gate_power=0.1, load=0.1, dt=1.0)
    explicit = fpm.lindblad_step(
        rho0, gamma=gamma, dt=1.0, method="euler",
    )
    derived = fpm.lindblad_step(
        rho0,
        dt=1.0,
        daemon=daemon,
        gate_power=0.1,
        load=0.1,
        method="euler",
    )

    assert np.allclose(derived, explicit, atol=1e-15)


def test_simulate_accepts_daemon_and_gate_power():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(75.0)
    rho0 = fpm.pure_state([1, 1])

    traj = fpm.simulate(
        rho0,
        dt=1.0,
        n_steps=5,
        daemon=daemon,
        gate_power=0.05,
        load=0.1,
        method="euler",
    )

    assert traj.shape == (6, 2, 2)
    assert abs(traj[-1, 0, 1]) < abs(traj[0, 0, 1])


def test_lindblad_step_rejects_ambiguous_gamma_sources():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(75.0)
    rho0 = fpm.pure_state([1, 1])

    with pytest.raises(ValueError, match="either explicit gamma"):
        fpm.lindblad_step(rho0, dt=1.0, method="euler")
    with pytest.raises(ValueError, match="not both"):
        fpm.lindblad_step(
            rho0,
            gamma=0.1,
            dt=1.0,
            daemon=daemon,
            gate_power=0.1,
            method="euler",
        )


def test_gamma_from_energy_rejects_invalid_inputs():
    ledger = fpm.ConservationLedger(E_max_total=100.0)
    daemon = ledger.add_daemon(50.0)

    with pytest.raises(ValueError, match="gate_power"):
        fpm.gamma_from_energy(daemon, gate_power=-0.1)
    with pytest.raises(ValueError, match="load"):
        fpm.gamma_from_energy(daemon, gate_power=0.1, load=-0.1)
    with pytest.raises(ValueError, match="dt"):
        fpm.gamma_from_energy(daemon, gate_power=0.1, dt=0.0)
    with pytest.raises(ValueError, match="C_N"):
        fpm.gamma_from_energy(daemon, gate_power=0.1, C_N=1.1)
