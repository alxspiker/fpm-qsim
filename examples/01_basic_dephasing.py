"""
Example 01: Basic dephasing dynamics.

A single qubit starts in the |+> state (maximal coherence) and
undergoes pure dephasing.  We track the off-diagonal coherence
amplitude over time and verify that it decays as exp(-gamma*t)
(machine precision, since v0.1.1 uses the exact continuous form).
"""
import math
import numpy as np

import fpm_qsim as fpm


def main():
    # |+><+| = (1/2) [[1, 1], [1, 1]]
    rho0 = fpm.pure_state([1, 1])
    gamma = 0.05
    dt = 1.0
    n_steps = 100

    traj = fpm.simulate(rho0, gamma=gamma, dt=dt, n_steps=n_steps)

    # Compare to the analytic decay exp(-gamma*t).
    print(f"{'tick':>6}  {'|rho_01|':>12}  {'analytic':>12}  {'rel_err':>10}")
    for t in [0, 10, 20, 50, 100]:
        actual = abs(traj[t, 0, 1])
        analytic = 0.5 * math.exp(-gamma * t * dt)
        rel_err = abs(actual - analytic) / max(analytic, 1e-30)
        print(f"{t:>6}  {actual:>12.6e}  {analytic:>12.6e}  {rel_err:>10.2e}")

    # Sanity check: final state is nearly maximally mixed.
    rho_final = traj[-1]
    print(f"\nFinal diagonal: [{rho_final[0,0].real:.4f}, {rho_final[1,1].real:.4f}]")
    print(f"Final coherence: {abs(rho_final[0,1]):.3e}")
    print(f"Expected coherence: {0.5 * math.exp(-gamma * n_steps * dt):.3e}")


if __name__ == "__main__":
    main()
