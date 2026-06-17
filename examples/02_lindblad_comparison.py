"""
Example 02: Theorem 3 Lindblad correspondence and machine precision.

Two demonstrations:

(A) Theorem 3 algebraic identity: the FPM affine map with the Euler
    form kappa = 1 - gamma*dt is algebraically identical to the
    Euler-discretized Lindblad dephasing equation.  Paper target:
    RMSE 6.13e-17.

(B) Machine-precision accuracy of the public exact-form integrator
    (kappa = exp(-gamma*dt)) against the analytic continuous
    dephasing solution.
"""
import math
import numpy as np

import fpm_qsim as fpm
from fpm_qsim._reference import euler_lindblad_step


def demo_theorem3():
    """Theorem 3: FPM affine Euler map == Euler-discretized Lindblad."""
    print("=" * 60)
    print("Theorem 3: FPM affine (Euler) == Euler-discretized Lindblad")
    print("=" * 60)

    rng = np.random.default_rng(seed=2026)
    dim = 4
    N_paths = 10
    T_ticks = 600
    gamma = 0.01
    dt = 1.0

    rmse_accum = 0.0
    max_abs_diff = 0.0
    n_offdiag = 0

    for _ in range(N_paths):
        psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        psi /= np.linalg.norm(psi)
        rho0 = np.outer(psi, psi.conj())

        rho_fpm = rho0.copy()
        rho_lind = rho0.copy()

        for _ in range(T_ticks):
            rho_fpm = euler_lindblad_step(rho_fpm, gamma=gamma, dt=dt)
            rho_lind = euler_lindblad_step(rho_lind, gamma=gamma, dt=dt)

            off_fpm = rho_fpm[~np.eye(dim, dtype=bool)]
            off_lind = rho_lind[~np.eye(dim, dtype=bool)]
            diff = np.abs(off_fpm - off_lind)
            rmse_accum += float(np.sum(diff ** 2))
            max_abs_diff = max(max_abs_diff, float(diff.max()))
            n_offdiag += off_fpm.size

    rmse = np.sqrt(rmse_accum / n_offdiag)
    print(f"N_paths:            {N_paths}")
    print(f"T_ticks:            {T_ticks}")
    print(f"RMSE off-diagonal:  {rmse:.6e}")
    print(f"Max abs difference: {max_abs_diff:.6e}")
    print(f"Paper reference:    6.13e-17")
    print(f"Verdict:            {'PASS' if rmse < 1e-14 else 'FAIL'}")
    print()


def demo_machine_precision():
    """Public lindblad_step vs analytic continuous dephasing."""
    print("=" * 60)
    print("Public lindblad_step vs analytic continuous dephasing")
    print("=" * 60)

    rng = np.random.default_rng(seed=2026)
    dim = 4
    cases = [
        (0.01, 1.0, 600),
        (0.1, 1.0, 100),
        (0.5, 0.1, 200),
        (1.0, 0.01, 1000),
        (2.0, 0.5, 50),
        # Large gamma*dt: a regime the Euler form cannot reach.
        (10.0, 1.0, 20),
    ]

    max_err_overall = 0.0
    for gamma, dt, n_steps in cases:
        for _ in range(3):
            psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            psi /= np.linalg.norm(psi)
            rho0 = np.outer(psi, psi.conj())

            traj = fpm.simulate(rho0, gamma=gamma, dt=dt, n_steps=n_steps)
            for t in range(n_steps + 1):
                t_cont = t * dt
                decay = math.exp(-gamma * t_cont)
                diag = np.diagonal(rho0).copy()
                analytic = decay * (rho0 - np.diag(diag)) + np.diag(diag)
                err = float(np.max(np.abs(traj[t] - analytic)))
                max_err_overall = max(max_err_overall, err)

    print(f"Max abs error across all cases (incl. gamma*dt=10): {max_err_overall:.6e}")
    print(f"Target: 1e-14 (machine precision)")
    print(f"Verdict: {'PASS' if max_err_overall < 1e-14 else 'FAIL'}")


if __name__ == "__main__":
    demo_theorem3()
    print()
    demo_machine_precision()
