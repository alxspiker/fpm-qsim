"""
Example 04: Multi-qubit dephasing and partial trace.

Simulate dephasing on a 2-qubit system and extract the reduced
single-qubit state via partial trace.
"""
import numpy as np

import fpm_qsim as fpm


def main():
    # Build a 2-qubit Bell state (|00> + |11>)/sqrt(2).
    # |00> = [1, 0, 0, 0], |11> = [0, 0, 0, 1]
    bell = fpm.pure_state([1, 0, 0, 1])
    print(f"Initial Bell state, shape={bell.shape}, trace={np.trace(bell).real:.4f}")

    # Dephasing on the full 4x4 system.
    gamma = 0.02
    traj = fpm.simulate(bell, gamma=gamma, dt=1.0, n_steps=100)

    print(f"\n{'tick':>6}  {'|rho_01|':>10}  {'|rho_03|':>10}  "
          f"{'Tr(rho_A)':>10}")
    for t in [0, 25, 50, 100]:
        rho_t = traj[t]
        # Partial trace out qubit B (index 1) -> keep qubit A (index 0).
        rho_A = fpm.partial_trace(rho_t, keep=[0], dims=[2, 2])
        print(f"{t:>6}  {abs(rho_t[0,1]):>10.4e}  {abs(rho_t[0,3]):>10.4e}  "
              f"{np.trace(rho_A).real:>10.4f}")

    print("\nNote: the (0,3) coherence of the Bell state decays as the")
    print("system dephases.  After 100 steps it should be near zero.")
    print(f"Final |rho_03| = {abs(traj[-1, 0, 3]):.3e}")


if __name__ == "__main__":
    main()
