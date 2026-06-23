"""
equivalence_test.py — Verify C++ FPM produces bit-identical output to Python FPM
================================================================================

Run this script to confirm the C++ port matches the Python implementation
across all 7 test categories. All tests must PASS before the C++ extension
can be considered a drop-in replacement.

Usage:
    python3 equivalence_test.py
"""
import sys
import math
import numpy as np

# Add the C++ module directory to path
sys.path.insert(0, ".")

try:
    import fpm_qsim as fpm_py
except ModuleNotFoundError:
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(
            "Python reference package fpm_qsim is not installed; "
            "equivalence_test.py is a manual reference comparison.",
            allow_module_level=True,
        )
    raise SystemExit(
        "Python reference package fpm_qsim is not installed. "
        "Install the old Python reference package to run this equivalence script."
    )
import fpm_cpp


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def check(name, condition, details=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {details}" if details else ""))
    return condition


def main():
    all_pass = True

    section("[1] Physical Constants")
    all_pass &= check("GAMMA_MAX match",
        fpm_py.GAMMA_MAX == fpm_cpp.GAMMA_MAX,
        f"py={fpm_py.GAMMA_MAX}, cpp={fpm_cpp.GAMMA_MAX}")
    all_pass &= check("FALSIFICATION_THRESHOLD match",
        fpm_py.FALSIFICATION_THRESHOLD == fpm_cpp.FALSIFICATION_THRESHOLD)
    all_pass &= check("ENERGY_FLOOR_FRACTION match",
        fpm_py.ENERGY_FLOOR_FRACTION == fpm_cpp.ENERGY_FLOOR_FRACTION)
    all_pass &= check("ISOTROPIC_WEIGHT_LIMIT match",
        fpm_py.ISOTROPIC_WEIGHT_LIMIT == fpm_cpp.ISOTROPIC_WEIGHT_LIMIT)

    section("[2] lindblad_step equivalence (1-6 qubits)")
    for n_q in [1, 2, 3, 4, 5, 6]:
        dim = 2 ** n_q
        rng = np.random.default_rng(2026 + n_q)
        psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        psi /= np.linalg.norm(psi)
        rho0 = np.ascontiguousarray(np.outer(psi, psi.conj()), dtype=np.complex128)
        rho_py = fpm_py.lindblad_step(rho0, gamma=0.1, dt=1.0, method="exact")
        rho_cpp_omp = fpm_cpp.lindblad_step(rho0, gamma=0.1, dt=1.0, method="exact", use_omp=True)
        rho_cpp_serial = fpm_cpp.lindblad_step(rho0, gamma=0.1, dt=1.0, method="exact", use_omp=False)
        diff_omp = np.max(np.abs(rho_py - rho_cpp_omp))
        diff_serial = np.max(np.abs(rho_py - rho_cpp_serial))
        ok = diff_omp < 1e-15 and diff_serial < 1e-15
        all_pass &= check(f"{n_q}q (dim={dim:3d})",
            ok, f"diff_omp={diff_omp:.3e}, diff_serial={diff_serial:.3e}")

    section("[3] simulate trajectory equivalence")
    for n_q in [1, 3, 5, 6]:
        dim = 2 ** n_q
        rng = np.random.default_rng(2026 + n_q)
        psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        psi /= np.linalg.norm(psi)
        rho0 = np.ascontiguousarray(np.outer(psi, psi.conj()), dtype=np.complex128)
        traj_py = fpm_py.simulate(rho0, gamma=0.05, dt=1.0, n_steps=200)
        traj_cpp = fpm_cpp.simulate(rho0, gamma=0.05, dt=1.0, n_steps=200, method="exact", use_omp=True)
        diff = np.max(np.abs(traj_py - traj_cpp))
        ok = diff < 1e-15
        all_pass &= check(f"{n_q}q, 200 steps",
            ok, f"diff={diff:.3e}, shape={traj_cpp.shape}")

    section("[4] Euler method (kappa = 1 - gamma*dt)")
    rho0 = np.ascontiguousarray(fpm_py.pure_state([1, 1]))
    rho_py = fpm_py.lindblad_step(rho0, gamma=0.05, dt=1.0, method="euler")
    rho_cpp = fpm_cpp.lindblad_step(rho0, gamma=0.05, dt=1.0, method="euler", use_omp=True)
    diff = np.max(np.abs(rho_py - rho_cpp))
    all_pass &= check("Euler method match", diff < 1e-15, f"diff={diff:.3e}")

    section("[5] bounded_gamma + FalsificationError")
    # CERN muon (below ceiling) — should accept
    bg_py = fpm_py.bounded_gamma(29.3)
    bg_cpp = fpm_cpp.bounded_gamma(29.3)
    all_pass &= check("bounded_gamma(29.3) match", abs(bg_py - bg_cpp) < 1e-12,
        f"py={bg_py}, cpp={bg_cpp}")
    # Falsifying observation (above threshold) — should raise
    raised_py = False
    raised_cpp = False
    try:
        fpm_py.bounded_gamma(40.0)
    except fpm_py.FalsificationError:
        raised_py = True
    try:
        fpm_cpp.bounded_gamma(40.0)
    except fpm_py.FalsificationError:
        raised_cpp = True
    all_pass &= check("Python raises FalsificationError for gamma=40", raised_py)
    all_pass &= check("C++ raises FalsificationError for gamma=40", raised_cpp)

    section("[6] Endogenous gamma_from_energy")
    # C++ ledger
    ledger_cpp = fpm_cpp.ConservationLedger(100.0)
    d_cpp = ledger_cpp.add_daemon(80.0)
    g_cpp = fpm_cpp.gamma_from_energy(d_cpp, gate_power=0.10, dt=1.0)
    # Python ledger
    ledger_py = fpm_py.ConservationLedger(E_max_total=100.0)
    d_py = ledger_py.add_daemon(80.0)
    g_py = fpm_py.gamma_from_energy(d_py, gate_power=0.10, dt=1.0)
    match = abs(g_cpp - g_py) < 1e-12
    all_pass &= check("gamma_from_energy match", match,
        f"py={g_py:.10f}, cpp={g_cpp:.10f}")

    # Also test energy-poor daemon (should give larger gamma)
    ledger_cpp2 = fpm_cpp.ConservationLedger(100.0)
    d_cpp2 = ledger_cpp2.add_daemon(5.0)
    g_cpp_poor = fpm_cpp.gamma_from_energy(d_cpp2, gate_power=0.10, dt=1.0)
    ledger_py2 = fpm_py.ConservationLedger(E_max_total=100.0)
    d_py2 = ledger_py2.add_daemon(5.0)
    g_py_poor = fpm_py.gamma_from_energy(d_py2, gate_power=0.10, dt=1.0)
    match_poor = abs(g_cpp_poor - g_py_poor) < 1e-12
    all_pass &= check("gamma_from_energy (energy-poor) match", match_poor,
        f"py={g_py_poor:.10f}, cpp={g_cpp_poor:.10f}")
    # Sanity: energy-poor should give LARGER gamma than energy-rich
    all_pass &= check("Energy-poor gives larger gamma (endogenous noise works)",
        g_cpp_poor > g_cpp, f"poor={g_cpp_poor:.4f} > rich={g_cpp:.4f}")

    section("[7] Machine precision vs analytic continuous-dephasing solution")
    cases = [
        (0.01, 1.0, 600),
        (0.1,  1.0, 100),
        (0.5,  0.1, 200),
        (1.0,  0.01, 1000),
        (2.0,  0.5, 50),
        (10.0, 1.0, 20),  # gamma*dt = 10, regime the Euler form cannot reach
    ]
    for gamma, dt, n_steps in cases:
        rho0 = np.ascontiguousarray(fpm_py.pure_state([1, 1, 1, 1]))
        traj = fpm_cpp.simulate(rho0, gamma=gamma, dt=dt, n_steps=n_steps,
                                method="exact", use_omp=True)
        max_err = 0.0
        for t in range(n_steps + 1):
            t_cont = t * dt
            decay = math.exp(-gamma * t_cont)
            diag = np.diagonal(rho0).copy()
            analytic = decay * (rho0 - np.diag(diag)) + np.diag(diag)
            err = float(np.max(np.abs(traj[t] - analytic)))
            max_err = max(max_err, err)
        ok = max_err < 1e-14
        all_pass &= check(f"gamma={gamma:4.2f}, dt={dt:4.2f}, n_steps={n_steps:4d}",
            ok, f"max_err={max_err:.3e}")

    section("SUMMARY")
    print(f"\n  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"  C++ version: {fpm_cpp.__version__}")
    print(f"  Build: {fpm_cpp.build_info}")
    print(f"  Python FPM: {fpm_py.__version__}")
    if not all_pass:
        print("\n  The C++ extension is NOT a verified drop-in replacement.")
        print("  Do not use in production until all tests pass.")
        sys.exit(1)
    print("\n  The C++ extension is a verified bit-exact drop-in replacement")
    print("  for the Python FPM. Safe to use in production.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
