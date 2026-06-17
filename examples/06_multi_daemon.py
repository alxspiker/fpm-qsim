"""
Example 06: Multi-daemon circuits (v0.1.8).

This example demonstrates the v0.1.8 multi-daemon Circuit API:

  * Attaching a per-qubit DaemonState to each qubit via ``daemons=``.
  * Per-qubit endogenous dephasing: each qubit's dephasing rate is
    derived from its own daemon's energy budget, so an energy-rich
    qubit decoheres slowly while an energy-poor qubit decoheres fast.
  * Targeted dephasing: dephasing only specific qubits, leaving
    others untouched.
  * Closed-universe balancing across the daemon network: every daemon
    is replenished by what it spent each tick, keeping the network-
    wide identity ``total_replenish == total_spend + total_landauer``
    satisfied to ~0 drift.

This is the structural primitive for FPM network simulations and the
foundation for the planned ``fpm-bft``, ``fpm-fed``, and ``fpm-marl``
packages.
"""
import numpy as np

import fpm_qsim as fpm


def main():
    print("=" * 60)
    print("Multi-daemon circuit: per-qubit FPM network")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Closed-universe setup: one daemon per qubit.
    # ------------------------------------------------------------------
    E_max = 100.0
    ledger = fpm.ConservationLedger(E_max_total=E_max)
    # d0 owns qubit 0 (energy-rich), d1 owns qubit 1 (energy-poor).
    d0 = ledger.add_daemon(80.0)
    d1 = ledger.add_daemon(40.0)
    print(f"d0 (qubit 0): E = {d0.E:.2f} / {E_max}")
    print(f"d1 (qubit 1): E = {d1.E:.2f} / {E_max}")

    # ------------------------------------------------------------------
    # Build the multi-daemon circuit.
    # ------------------------------------------------------------------
    circ = fpm.Circuit(
        n_qubits=2,
        daemons=[d0, d1],
        ledger=ledger,
        method="euler",
        bounded=True,
        default_gate_power=0.05,
        cost_per_op=1e-5,
    )
    circ.h(0).cx(0, 1).dephase(dt=1.0)
    print(f"\nOperations queued: {circ.operations}")
    print(f"Multi-daemon mode: {circ.is_multi_daemon}")
    print(f"Daemons attached: {[d.index for d in circ.all_daemons]}")

    # ------------------------------------------------------------------
    # Run with automatic per-daemon replenishment.
    # ------------------------------------------------------------------
    print(f"\n{'tick':>4}  {'|rho_03|':>10}  {'d0.E':>8}  {'d1.E':>8}  "
          f"{'d0.spend':>10}  {'d1.spend':>10}  {'drift':>10}")

    rho = fpm.pure_state([1, 0, 0, 0])  # |00><00|
    trajectory = [rho.copy()]
    checkpoints = {0, 4, 9, 14, 19}
    for t in range(20):
        rho = circ.run_with_replenishment(rho, n_steps=1, record=False)
        trajectory.append(rho.copy())
        if t not in checkpoints:
            continue
        drift = ledger.drift()
        print(f"{t + 1:>4}  {abs(rho[0, 3]):>10.4e}  "
              f"{d0.E:>8.4f}  {d1.E:>8.4f}  "
              f"{d0.cumulative_spend:>10.4f}  "
              f"{d1.cumulative_spend:>10.4f}  "
              f"{drift:>10.2e}")
    traj = np.asarray(trajectory)

    print(f"\nNetwork-wide drift: {ledger.drift():.2e}")
    print(f"  (should be ~0 -- each daemon replenished for its own spend)")
    print(f"Total spend:   {ledger.total_spend:.4f}")
    print(f"Total replenish: {ledger.total_replenish:.4f}")
    print(f"  d0 paid: {d0.cumulative_spend:.4f} (owns q0: H + CNOT + dephase)")
    print(f"  d1 paid: {d1.cumulative_spend:.4f} (owns q1: CNOT + dephase)")

    # ------------------------------------------------------------------
    # Demonstrate per-qubit endogenous dephasing rates.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Per-qubit endogenous gamma: asymmetric dephasing")
    print("=" * 60)

    # Fresh daemons: d0 rich, d1 poor.  Apply only dephasing (no gates)
    # so we can isolate the per-qubit decay rates.
    ledger2 = fpm.ConservationLedger(E_max_total=E_max)
    d0_rich = ledger2.add_daemon(90.0)
    d1_poor = ledger2.add_daemon(10.0)
    circ2 = fpm.Circuit(
        n_qubits=2,
        daemons=[d0_rich, d1_poor],
        ledger=ledger2,
        method="euler",
        cost_per_op=0.0,  # no billing, isolate dephasing
        default_gate_power=0.1,
    )
    circ2.dephase(dt=1.0)

    # Start in a state with all coherences non-zero.
    rho0 = fpm.pure_state([1, 1, 1, 1])
    rho1 = circ2.step(rho0)

    # Compute the per-qubit gammas (d0 rich -> slow, d1 poor -> fast).
    gamma0 = fpm.gamma_from_energy(d0_rich, gate_power=0.1, dt=1.0, bounded=False)
    gamma1 = fpm.gamma_from_energy(d1_poor, gate_power=0.1, dt=1.0, bounded=False)
    print(f"d0 (rich, E=90): gamma = {gamma0:.4f}  (slow dephasing)")
    print(f"d1 (poor, E=10): gamma = {gamma1:.4f}  (fast dephasing)")
    print()
    print("Coherence decay after one dephasing step:")
    print(f"  |rho_01| (q1 differs): {abs(rho1[0, 1]):.4e}  (d1 poor -> small)")
    print(f"  |rho_02| (q0 differs): {abs(rho1[0, 2]):.4e}  (d0 rich -> large)")
    print(f"  |rho_03| (both differ): {abs(rho1[0, 3]):.4e}")
    print()
    print("Asymmetry ratio |rho_02| / |rho_01| = "
          f"{abs(rho1[0, 2]) / abs(rho1[0, 1]):.4f}")
    print("  (>1 means q0 dephased slower than q1, as expected)")

    # ------------------------------------------------------------------
    # Demonstrate targeted dephasing.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Targeted dephasing: only qubit 0 decoheres")
    print("=" * 60)

    ledger3 = fpm.ConservationLedger(E_max_total=E_max)
    d0 = ledger3.add_daemon(80.0)
    d1 = ledger3.add_daemon(80.0)
    circ3 = fpm.Circuit(
        n_qubits=2,
        daemons=[d0, d1],
        ledger=ledger3,
        method="euler",
        cost_per_op=1e-5,
        default_gate_power=0.1,
    )
    circ3.dephase(dt=1.0, targets=[0])  # only qubit 0

    rho0 = fpm.pure_state([1, 1, 1, 1])
    rho1 = circ3.step(rho0)
    print(f"  |rho_01| (q1 differs, q0 same): before={abs(rho0[0, 1]):.4e}, "
          f"after={abs(rho1[0, 1]):.4e}  (unchanged -- q0 not involved)")
    print(f"  |rho_02| (q0 differs):           before={abs(rho0[0, 2]):.4e}, "
          f"after={abs(rho1[0, 2]):.4e}  (decayed -- q0 targeted)")
    print(f"  |rho_03| (both differ):           before={abs(rho0[0, 3]):.4e}, "
          f"after={abs(rho1[0, 3]):.4e}  (decayed -- q0 component)")
    print()
    print(f"Billing: d0 paid {d0.cumulative_spend:.4f}, "
          f"d1 paid {d1.cumulative_spend:.4f}")
    print("  (d1 not billed -- its qubit wasn't targeted)")

    # ------------------------------------------------------------------
    # Three-qubit network.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Three-qubit FPM network")
    print("=" * 60)

    ledger4 = fpm.ConservationLedger(E_max_total=E_max)
    daemons = [ledger4.add_daemon(e) for e in (80.0, 60.0, 40.0)]
    circ4 = fpm.Circuit(
        n_qubits=3,
        daemons=daemons,
        ledger=ledger4,
        method="euler",
        cost_per_op=1e-5,
        default_gate_power=0.05,
    )
    circ4.h(0).cx(0, 1).cx(1, 2).dephase(dt=1.0)
    print(f"Daemons: {[d.E for d in daemons]}")
    rho0 = fpm.pure_state([1, 0, 0, 0, 0, 0, 0, 0])  # |000>
    traj = circ4.run_with_replenishment(rho0, n_steps=15)
    print(f"After 15 steps:")
    for i, d in enumerate(daemons):
        print(f"  d{i} (E_init={[80, 60, 40][i]}): E = {d.E:.4f}, "
              f"spend = {d.cumulative_spend:.4f}")
    print(f"  Network drift: {ledger4.drift():.2e}")
    print(f"  Trajectory shape: {traj.shape}")


if __name__ == "__main__":
    main()
