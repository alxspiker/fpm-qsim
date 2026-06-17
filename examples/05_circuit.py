"""
Example 05: Circuit layer -- Bell state preparation with endogenous dephasing.

This example demonstrates the v0.1.6 Circuit API and the v0.1.7
``run_with_replenishment`` method:

  * Building a circuit with H, CNOT, and dephasing layers.
  * Attaching a daemon and closed-universe ledger so every gate and
    dephasing layer is billed for its simulated route cost.
  * Using endogenous gamma (derived from daemon energy and gate power)
    instead of an externally supplied rate.
  * Letting ``run_with_replenishment`` balance the closed-universe
    ledger automatically each tick, keeping the FPM conservation
    identity ``replenish == spend + landauer`` satisfied to ~0 drift.

The circuit prepares a Bell state, then dephases it for several ticks.
Under FPM, the dephasing rate is not a free parameter: it is derived
from the daemon's energy budget and the gate power applied during the
step.  Energy-starved daemons or high-power gates decohere faster.
"""
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
    # Run the circuit for 20 ticks with automatic replenishment.
    # ------------------------------------------------------------------
    print(f"\n{'tick':>4}  {'|rho_03|':>10}  {'daemon_E':>10}  "
          f"{'cum_spend':>10}  {'cum_repl':>10}")

    rho0 = fpm.pure_state([1, 0, 0, 0])  # |00><00|
    rho = rho0.copy()
    trajectory = [rho.copy()]
    checkpoints = {0, 4, 9, 14, 19}
    for t in range(20):
        rho = circ.run_with_replenishment(rho, n_steps=1, record=False)
        trajectory.append(rho.copy())
        if t not in checkpoints:
            continue
        print(f"{t + 1:>4}  {abs(rho[0, 3]):>10.4e}  "
              f"{daemon.E:>10.4f}  "
              f"{ledger.total_spend:>10.4f}  "
              f"{ledger.total_replenish:>10.4f}")
    traj = np.asarray(trajectory)

    # ------------------------------------------------------------------
    # Report closed-universe conservation.
    # ------------------------------------------------------------------
    drift = ledger.drift()
    print(f"\nFinal drift: {drift:.6e}")
    print(f"  (run_with_replenishment keeps drift at ~0 when no external")
    print(f"   Landauer is charged; paper Test 03 reports ~1.47% for 50")
    print(f"   daemons / 300 ticks.)")
    print(f"  Daemon energy: {daemon.E:.4f} / {E_max_total} (preserved")
    print(f"   by balanced replenishment)")

    # ------------------------------------------------------------------
    # Compare to the legacy continuous-math (method="exact") path.
    # ------------------------------------------------------------------
    print("\n--- Comparison: method='euler' vs method='exact' ---")

    # Euler (already done above).
    rho_euler = traj[-1].copy()

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
    traj_exact = circ_exact.run_with_replenishment(rho0, n_steps=20)
    rho_exact = traj_exact[-1]

    print(f"  |rho_03| (euler): {abs(rho_euler[0, 3]):.6e}")
    print(f"  |rho_03| (exact): {abs(rho_exact[0, 3]):.6e}")
    print(f"  relative diff:    "
          f"{abs(rho_euler[0, 3] - rho_exact[0, 3]) / max(abs(rho_exact[0, 3]), 1e-30):.2e}")
    print("  (euler has O(dt) per-step error vs the exact continuous form)")
    print(f"  daemon E (euler): {daemon.E:.4f}")
    print(f"  daemon E (exact): {daemon2.E:.4f}")
    print("  (exact bills more: Taylor construction of exp per off-diagonal)")
    print(f"  drift (euler): {ledger.drift():.2e}")
    print(f"  drift (exact): {ledger2.drift():.2e}")

    # ------------------------------------------------------------------
    # Show that run_with_replenishment matches the manual path.
    # ------------------------------------------------------------------
    print("\n--- Equivalence: run_with_replenishment vs manual loop ---")
    ledger3 = fpm.ConservationLedger(E_max_total=E_max_total)
    daemon3 = ledger3.add_daemon(E_init=80.0)
    circ3 = fpm.Circuit(
        n_qubits=2,
        daemon=daemon3,
        ledger=ledger3,
        method="euler",
        default_gate_power=0.05,
        cost_per_op=1e-5,
    )
    circ3.h(0).cx(0, 1).dephase(dt=1.0)
    rho = rho0.copy()
    for _ in range(20):
        prev_spend = daemon3.cumulative_spend
        prev_landauer = daemon3.cumulative_landauer
        rho = circ3.step(rho)
        delta = (
            (daemon3.cumulative_spend - prev_spend)
            + (daemon3.cumulative_landauer - prev_landauer)
        )
        ledger3.record_replenish(daemon3, delta)
    print(f"  Final rho matches: "
          f"{np.allclose(traj[-1], rho, atol=1e-14)}")
    print(f"  Spend  (auto/manual): "
          f"{ledger.total_spend:.6e} / {ledger3.total_spend:.6e}")
    print(f"  Repl.  (auto/manual): "
          f"{ledger.total_replenish:.6e} / {ledger3.total_replenish:.6e}")
    print(f"  Drift  (auto/manual): "
          f"{ledger.drift():.2e} / {ledger3.drift():.2e}")


if __name__ == "__main__":
    main()
