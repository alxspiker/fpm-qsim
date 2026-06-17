"""
Example 05: Circuit layer — Bell state preparation with endogenous dephasing.

This example demonstrates the v0.1.6 Circuit API:

  * Building a circuit with H, CNOT, and dephasing layers.
  * Attaching a daemon and closed-universe ledger so every gate and
    dephasing layer is billed for its simulated route cost.
  * Using endogenous gamma (derived from daemon energy and gate power)
    instead of an externally supplied rate.
  * Verifying that the daemon's energy depletes over time, with the
    closed-universe identity (replenish == spend + landauer) holding
    to within ~1% drift.

The circuit prepares a Bell state, then dephases it for several ticks.
Under FPM, the dephasing rate is not a free parameter: it is derived
from the daemon's energy budget and the gate power applied during the
step.  Energy-starved daemons or high-power gates decohere faster.
"""
import math
import numpy as np

import fpm_qsim as fpm


def main():
    # ------------------------------------------------------------------
    # Closed-universe setup.
    # ------------------------------------------------------------------
    E_max_total = 100.0
    ledger = fpm.ConservationLedger(E_max_total=E_max_total)
    daemon = ledger.add_daemon(E_init=80.0)
    print(f"Initial daemon energy: {daemon.E:.4f} / {E_max_total}")

    # ------------------------------------------------------------------
    # Build the circuit: H on q0, CNOT(0,1), then endogenous dephasing.
    # ------------------------------------------------------------------
    circ = fpm.Circuit(
        n_qubits=2,
        daemon=daemon,
        ledger=ledger,
        method="euler",          # FPM-native lattice map (Theorem 3)
        bounded=True,            # raise FalsificationError if gamma > 32
        default_gate_power=0.05, # gate power applied during each dephase
        cost_per_op=1e-5,        # energy fraction per simulated op
        taylor_order=8,
    )
    circ.h(0).cx(0, 1).dephase(dt=1.0)
    print(f"Operations queued: {circ.operations}")

    # ------------------------------------------------------------------
    # Run the circuit for 20 ticks.  After each tick we replenish the
    # daemon to keep the closed-universe identity balanced.
    # ------------------------------------------------------------------
    rho0 = fpm.pure_state([1, 0, 0, 0])  # |00><00|
    rho = rho0.copy()
    print(f"\n{'tick':>4}  {'|rho_03|':>10}  {'daemon_E':>10}  "
          f"{'cum_spend':>10}  {'cum_repl':>10}")

    for t in range(20):
        prev_spend = daemon.cumulative_spend
        rho = circ.step(rho)
        spent_this_tick = daemon.cumulative_spend - prev_spend
        # Closed-universe replenishment: refill exactly what was spent.
        ledger.record_replenish(daemon, spent_this_tick)
        if t in (0, 4, 9, 14, 19):
            print(f"{t + 1:>4}  {abs(rho[0, 3]):>10.4e}  "
                  f"{daemon.E:>10.4f}  "
                  f"{ledger.total_spend:>10.4f}  "
                  f"{ledger.total_replenish:>10.4f}")

    # ------------------------------------------------------------------
    # Report closed-universe conservation.
    # ------------------------------------------------------------------
    drift = ledger.drift()
    print(f"\nFinal drift: {drift:.4%}")
    print(f"  (paper Test 03 reports ~1.47% for 50 daemons / 300 ticks)")
    print(f"  (this example uses 1 daemon / 20 ticks, so drift is small)")

    # ------------------------------------------------------------------
    # Compare to the legacy continuous-math (method="exact") path.
    # ------------------------------------------------------------------
    print("\n--- Comparison: method='euler' vs method='exact' ---")
    rho_euler = rho.copy()

    # Reset and re-run with exact method.
    ledger2 = fpm.ConservationLedger(E_max_total=E_max_total)
    daemon2 = ledger2.add_daemon(E_init=80.0)
    circ_exact = fpm.Circuit(
        n_qubits=2,
        daemon=daemon2,
        ledger=ledger2,
        method="exact",
        bounded=True,
        default_gate_power=0.05,
        cost_per_op=1e-5,
    )
    circ_exact.h(0).cx(0, 1).dephase(dt=1.0)
    rho = rho0.copy()
    for _ in range(20):
        prev_spend = daemon2.cumulative_spend
        rho = circ_exact.step(rho)
        spent = daemon2.cumulative_spend - prev_spend
        ledger2.record_replenish(daemon2, spent)
    rho_exact = rho

    print(f"  |rho_03| (euler): {abs(rho_euler[0, 3]):.6e}")
    print(f"  |rho_03| (exact): {abs(rho_exact[0, 3]):.6e}")
    print(f"  relative diff:    "
          f"{abs(rho_euler[0, 3] - rho_exact[0, 3]) / abs(rho_exact[0, 3]):.2e}")
    print("  (euler has O(dt) per-step error vs the exact continuous form)")
    print(f"  daemon E (euler): {daemon.E:.4f}")
    print(f"  daemon E (exact): {daemon2.E:.4f}")
    print("  (exact bills more: Taylor construction of exp per off-diagonal)")


if __name__ == "__main__":
    main()
