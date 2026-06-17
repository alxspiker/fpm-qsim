"""Tests for the public API surface and state utilities."""

import numpy as np
import pytest

import fpm_qsim as fpm


def test_pure_state_is_density_matrix():
    rho = fpm.pure_state([1, 2, 3])
    assert fpm.is_density_matrix(rho)
    assert rho.shape == (3, 3)
    assert abs(np.trace(rho) - 1.0) < 1e-14


def test_maximally_mixed():
    rho = fpm.maximally_mixed(4)
    assert fpm.is_density_matrix(rho)
    assert np.allclose(rho, np.eye(4) / 4)


def test_basis_state():
    v = fpm.basis_state(2, 4)
    assert v.shape == (4, 1)
    assert v[2, 0] == 1.0
    assert np.sum(np.abs(v) ** 2) == 1.0


def test_basis_state_out_of_range():
    with pytest.raises(ValueError):
        fpm.basis_state(5, 4)
    with pytest.raises(ValueError):
        fpm.basis_state(-1, 4)


def test_pure_state_zero_raises():
    with pytest.raises(ValueError):
        fpm.pure_state([0, 0, 0])


def test_fidelity_pure_states():
    """Fidelity of identical pure states is 1."""
    rho = fpm.pure_state([1, 2, 3])
    fid = fpm.fidelity(rho, rho)
    assert abs(fid - 1.0) < 1e-12


def test_fidelity_orthogonal_states():
    """Fidelity of orthogonal pure states is 0."""
    rho1 = fpm.pure_state([1, 0, 0])
    rho2 = fpm.pure_state([0, 1, 0])
    fid = fpm.fidelity(rho1, rho2)
    assert fid < 1e-12


def test_trace_distance_identical():
    rho = fpm.pure_state([1, 1, 1])
    assert fpm.trace_distance(rho, rho) < 1e-14


def test_trace_distance_orthogonal():
    rho1 = fpm.pure_state([1, 0])
    rho2 = fpm.pure_state([0, 1])
    # Trace distance of orthogonal pure states is 1.
    assert abs(fpm.trace_distance(rho1, rho2) - 1.0) < 1e-12


def test_simulate_shapes():
    rho0 = fpm.pure_state([1, 1, 1])
    traj = fpm.simulate(rho0, gamma=0.05, dt=1.0, n_steps=50)
    assert traj.shape == (51, 3, 3)


def test_simulate_no_record():
    rho0 = fpm.pure_state([1, 1])
    final = fpm.simulate(rho0, gamma=0.1, dt=1.0, n_steps=10, record=False)
    assert final.shape == (2, 2)


def test_unitary_step():
    """unitary_step applies an exact Hamiltonian step."""
    rho0 = fpm.pure_state([1, 0])  # |0><0|
    H = np.array([[0, 1], [1, 0]], dtype=complex)  # sigma_x

    # With H = sigma_x and small dt, the state should rotate.
    # |0> -> cos(dt)|0> + i*sin(dt)|1>, so |rho_01| = sin(dt)*cos(dt).
    import math
    dt = 0.1
    rho_next = fpm.unitary_step(rho0, H, dt=dt)
    expected_01 = math.sin(dt) * math.cos(dt)
    assert abs(rho_next[0, 1].imag - expected_01) < 1e-14, (
        f"unitary_step gave rho_01 = {rho_next[0, 1]}, "
        f"expected imaginary part {expected_01}"
    )


def test_lindblad_step_no_H_param():
    """v0.1.3: lindblad_step no longer accepts H parameter."""
    rho = fpm.pure_state([1, 0])
    H = np.array([[0, 1], [1, 0]], dtype=complex)
    with pytest.raises(TypeError):
        # H is no longer a valid keyword argument.
        fpm.lindblad_step(rho, gamma=0.1, dt=1.0, H=H)


def test_version_string():
    assert isinstance(fpm.__version__, str)
    assert len(fpm.__version__) > 0


def test_public_api_exports():
    """Spot-check that the public names exist."""
    expected = [
        "lindblad_step",
        "unitary_step",
        "simulate",
        "fpm_affine_step",
        "fpm_affine_trajectory",
        "kappa_from_gamma",
        "kappa_exact",
        "gamma_from_kappa",
        "gamma_from_energy",
        "bounded_gamma",
        "FalsificationError",
        "GAMMA_MAX",
        "FALSIFICATION_THRESHOLD",
        "pure_state",
        "maximally_mixed",
        "basis_state",
        "is_density_matrix",
        "fidelity",
        "trace_distance",
        "DaemonState",
        "ConservationLedger",
        # v0.1.4 oracle-cost billing helpers
        "exp_route_cost",
        "bill_exp_route_cost",
    ]
    for name in expected:
        assert hasattr(fpm, name), f"missing public name: {name}"


def test_reference_lindblad_step_not_public():
    """The Euler reference is private in v0.1.1; only the exact form
    is public."""
    assert not hasattr(fpm, "reference_lindblad_step"), (
        "reference_lindblad_step should be private in v0.1.1"
    )
    # It should still be importable from the private module:
    from fpm_qsim._reference import euler_lindblad_step
    assert callable(euler_lindblad_step)
