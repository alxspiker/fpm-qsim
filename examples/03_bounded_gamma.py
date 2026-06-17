"""
Example 03: Bounded gamma and the falsification ceiling.

The FPM framework derives a hard ceiling on the dephasing rate:

    gamma_max = 31.8738...

derived from the finite-lag theorem.  Observations of gamma > 32.0
would falsify the framework.  The CERN muon sits at gamma = 29.3,
safely below the ceiling.

This example shows how to use the `bounded=True` flag and what
happens when an observation would falsify the framework.
"""
import numpy as np

import fpm_qsim as fpm


def main():
    print("=" * 60)
    print("FPM Bounded Gamma Ceiling")
    print("=" * 60)
    print(f"  gamma_max (derived):              {fpm.GAMMA_MAX:.6f}")
    print(f"  falsification threshold:          {fpm.FALSIFICATION_THRESHOLD}")
    print(f"  CERN muon gamma:                  29.3  (in range)")
    print(f"  margin to falsification:          {fpm.FALSIFICATION_THRESHOLD - 29.3:.2f}")
    print()

    # Case 1: ordinary dephasing — well below the ceiling.
    rho = fpm.pure_state([1, 1])
    rho1 = fpm.lindblad_step(rho, gamma=1.0, dt=0.1, bounded=True)
    print(f"gamma=1.0, dt=0.1:  |rho_01| = {abs(rho1[0,1]):.6f}  (unclipped)")

    # Case 2: dephasing above the ceiling — clipped to gamma_max.
    rho2 = fpm.lindblad_step(rho, gamma=31.95, dt=0.01, bounded=True)
    print(f"gamma=31.95 (clipped to {fpm.GAMMA_MAX:.4f}), dt=0.01:  "
          f"|rho_01| = {abs(rho2[0,1]):.6f}")

    # Case 3: would-falsify observation.
    print()
    print("Now attempting gamma = 33.0 (would falsify FPM):")
    try:
        fpm.lindblad_step(rho, gamma=33.0, dt=0.01, bounded=True)
    except fpm.FalsificationError as e:
        print(f"  FalsificationError raised: {e}")
        print("  The framework refuses to silently clip an observation")
        print("  that would falsify it. Log the observation instead.")


if __name__ == "__main__":
    main()
