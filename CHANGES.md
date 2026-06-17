# CHANGES

## 0.1.8 (2026-06-18)

**Multi-daemon circuits: per-qubit FPM networks.**

- Added `daemons` parameter to `Circuit.__init__`.  When provided
  together with `ledger`, each qubit is backed by its own
  `DaemonState`, turning a multi-qubit circuit into a network of FPM
  daemons with the closed-universe identity holding across all of
  them.  Mutually exclusive with `daemon` (single-daemon mode).
- **Per-qubit billing**: single-qubit gates bill only the owning
  daemon.  Two-qubit gates split the bill 50/50 between the two
  target daemons.  `apply_unitary_full` bills all daemons equally.
- **Per-qubit endogenous dephasing**: in multi-daemon mode with
  endogenous γ (no explicit `gamma`), each qubit's dephasing rate is
  derived from its own daemon's energy budget via
  `gamma_from_energy`.  An energy-rich qubit decoheres slowly; an
  energy-poor qubit decoheres fast.  This is the structural primitive
  for FPM network simulations and the foundation for the planned
  `fpm-bft`, `fpm-fed`, and `fpm-marl` packages.
- Added `targets` keyword to `dephase()`: dephase only specific
  qubits, leaving others untouched.  In multi-daemon mode, only the
  named daemons are billed for the dephasing layer.  Use this to
  model isolated subsystems or targeted noise injection.
- `run_with_replenishment` now replenishes **every** daemon by
  exactly what it spent that tick (spend + landauer), keeping the
  network-wide identity `total_replenish == total_spend +
  total_landauer` satisfied to ~0 drift.
- New helper properties: `Circuit.is_multi_daemon`,
  `Circuit.all_daemons`, `Circuit.daemons`.
- New internal helpers: `_daemon_for_qubit(i)`,
  `_daemons_for_targets(targets)`, `_apply_per_qubit_dephase(rho, op)`,
  `_dephase_single_qubit(rho, qubit, dt, gate_power, load)`.
- `strang_step` updated to support multi-daemon mode: bills all
  daemons for the unitary halves, uses per-qubit endogenous γ for
  the dephasing layer when no explicit γ is given.
- 36 new tests in `tests/test_circuit.py` covering: construction
  validation, single/two/full-system gate billing, per-qubit
  endogenous γ (including direct-calculation equivalence), targeted
  dephasing, closed-universe balancing across multiple daemons,
  multi-daemon Strang splitting, three-qubit networks, energy-floor
  behavior, falsification via endogenous γ, and public API exposure.
  181 tests total, all passing.
- Added `examples/06_multi_daemon.py` demonstrating per-qubit
  dephasing, targeted dephasing, and a three-qubit FPM network with
  network-wide conservation.
- Backward compatible: all v0.1.7 single-daemon code continues to
  work unchanged.

## 0.1.7 (2026-06-18)

**`Circuit.run_with_replenishment` — automatic closed-universe balancing.**

- Added `Circuit.run_with_replenishment(rho0, n_steps, *, record=True)`.
  Same as `run()` but, after each `step()`, replenishes the daemon by
  exactly the energy debited that tick (spend + landauer), keeping
  the closed-universe identity `replenish == spend + landauer`
  satisfied to within energy-floor / energy-ceiling clipping.
- This is the FPM closed-universe conservation theorem (paper Test 03)
  made operational at the circuit level: callers no longer need to
  manually track `cumulative_spend` / `cumulative_landauer` deltas
  and call `record_replenish` themselves.
- Requires `daemon` and `ledger` to be attached to the circuit;
  raises `ValueError` otherwise (pointing users to `run()` for
  open-system simulations).
- Honest behavior at the boundaries:
  - If the daemon is near `E_max`, replenishment is capped at
    `E_max - E` and drift may grow. The framework is reporting that
    the configured `E_max_total` is too small to absorb the requested
    computation.
  - If the daemon is near the energy floor, spend is capped (existing
    behavior of `bill_compute_cost`) and replenishment restores the
    daemon.
  - Landauer debits charged externally between `step()` calls (via
    `ledger.record_landauer`) are NOT replenished by this method.
    The caller is responsible for those.
- 18 new tests in `tests/test_circuit.py` covering: trajectory
  shape, `record=False`, `n_steps=0`, daemon/ledger requirement,
  shape validation, drift-stays-at-zero, daemon-energy preservation,
  equivalence with manual replenishment loop, billing counter
  tracking, both `euler` and `exact` methods, external-landauer
  behavior, E_max clipping, floor clipping, density-matrix validity,
  input immutability, endogenous-gamma flow. 145 tests total, all
  passing.
- Updated `examples/05_circuit.py` to use `run_with_replenishment`
  instead of the manual replenishment loop.

## 0.1.6 (2026-06-18)

**Circuit layer: queue-based composition of unitary gates and FPM dephasing.**

- Added `Circuit` class (`fpm_qsim.circuit`) — a fluent queue builder
  for composing unitary gates with FPM dephasing layers under a single
  closed-universe ledger.
- Gate set: `h`, `x`, `y`, `z`, `s`, `t`, `cx`, `cz`, `swap`, and
  generic `u(theta, phi, lam, i)`. Plus `apply_unitary(U, targets)` for
  arbitrary k-qubit unitaries and `apply_unitary_full(U_full)` for
  pre-expanded operators (supports >10 qubits).
- Gates are applied via direct conjugation `U @ rho @ U^dagger`
  (standard quantum-circuit convention), not as Hamiltonian time
  evolution. This is the correct semantics for circuit gates and
  avoids the matrix-exp cost at application time.
- `dephase(gamma=None, *, dt=1.0, gate_power=None, load=None)` appends
  a dephasing layer. Supports both explicit gamma and endogenous
  gamma derived from the attached daemon (forwarded to
  `lindblad_step`).
- `step(rho)` applies the full queued sequence once. `run(rho0,
  n_steps, record=True)` repeats and returns the trajectory.
- `strang_step(rho, H, gamma, dt, ...)` provides the standard
  second-order Strang splitting (`U(dt/2) + dephase(dt) + U(dt/2)`)
  for Hamiltonian + dephasing dynamics. Uses `unitary_step` with
  `expm(-i H dt/2)` for the half-steps.
- **Closed-universe billing** (when `daemon` and `ledger` are
  attached): every gate and dephasing layer bills the ledger for its
  simulated construction cost:
  - Unitary gates: `N^2` scalar `exp` constructions (one per matrix
    element), each via a K-term Taylor series. `N = 2^n_qubits`.
  - Dephasing (`method="euler"`): 1 multiply + 1 addition per
    off-diagonal state variable (`N*(N-1)` total). The literal
    Theorem 3 lattice operation.
  - Dephasing (`method="exact"`): 1 scalar `exp` per off-diagonal
    state variable, each billed via K-term Taylor series. Oracle
    break, billed explicitly.
- Falsifiability ceiling is enforced when `bounded=True`: endogenous
  or explicit gammas exceeding 32.0 raise `FalsificationError`.
- 60 new tests in `tests/test_circuit.py` covering gate semantics,
  dephasing decay, billing math, Strang splitting, endogenous gamma,
  and falsification. 127 tests total, all passing.
- Added `examples/05_circuit.py` demonstrating Bell state preparation
  with endogenous dephasing and closed-universe billing.

## 0.1.5 (2026-06-17)

**Endogenous dephasing from daemon energy and gate load.**

- Added `gamma_from_energy(daemon, gate_power, load=None, dt=1.0)`.
  This derives `gamma` from the FPM contraction form
  `kappa_t = C_N * (1 + B_t)^(-3/4)` and
  `gamma_t = (1 - kappa_t) / dt`.
- The minimal gate-noise model uses
  `B_t = load + gate_power / energy_fraction`, so higher gate power,
  higher load, or lower daemon energy increases dephasing.
- Added `load` to `DaemonState` for storing the daemon's local load.
- `lindblad_step` and `simulate` now accept either explicit `gamma`
  or `daemon` + `gate_power` for first-class endogenous FPM noise.
- Exposed `gamma_from_energy` in the top-level public API.
- Kept raw `gamma=...` as the legacy explicit-rate path.

## 0.1.4 (2026-06-17)

**Ontological boundary between FPM-native and continuous-math maps.**

- Added `method` parameter to `lindblad_step` and `simulate`:
  - `method="euler"` (new): FPM-native lattice map with
    `kappa = 1 - gamma*dt`.  This is the literal Theorem 3
    identification.  Per state variable per step, the simulated
    route cost is exactly 1 multiply + 1 add — billable to
    `ConservationLedger.bill_compute_cost`.  Preserves closed-universe
    structural coherence.
  - `method="exact"` (default, backward compatible): `kappa = exp(-gamma*dt)`.
    Machine-precise continuous-dephasing solution.  Documented as a
    **non-physical oracle break** under FPM: the host's `np.exp` is
    handed to the simulated system without billing the simulated route
    cost.  Use for legacy-continuous-QM comparison only.
- Added `ConservationLedger.bill_compute_cost(daemon, n_multiplies, n_adds)`
  to debit simulated discrete computational work, enforcing closed-universe
  accounting.
- Added `exp_route_cost(taylor_order=8)` returning the
  `(n_multiplies, n_adds)` cost of constructing `exp(-gamma*dt)` via a
  finite K-term Taylor series under FPM's discrete action principle.
- Added `bill_exp_route_cost(ledger, daemon, taylor_order=8)` as a
  convenience wrapper.  Use after each `method="exact"` step to keep
  the closed-universe ledger balanced despite the continuous-math oracle.
- Updated README with the ontological distinction and usage examples
  for both methods.
- 57 tests, all passing.

## 0.1.3 (2026-06-17)

- Removed the `H` parameter from `lindblad_step`; Hamiltonian
  evolution now uses explicit `unitary_step` composition.
- Added `unitary_step` to the public API for exact Hamiltonian steps.
- Updated README benchmark framing: competitive pure-dephasing
  performance, not unique speed dominance.
- Added benchmark speed and accuracy tables to the README.

## 0.1.2 (2026-06-17)

- Clarified the PyPI installation command in the README.

## 0.1.1 (2026-06-17)

**Breaking change:** `lindblad_step` now uses the exact continuous
form `kappa = exp(-gamma*dt)` instead of the Euler form
`1 - gamma*dt`.  The public API now matches Kraus / matrix-exp /
QuTiP to machine precision (max abs error 1.1e-16 vs the analytic
continuous-dephasing solution).

- The Euler form (the literal Theorem 3 identification) is preserved
  in the private `fpm_qsim._reference.euler_lindblad_step` for
  Theorem 3 verification in the test suite.
- Added `kappa_exact(gamma, dt)` to the public API.
- Removed `reference_lindblad_step` from the public API.
- Added machine-precision regression test
  (`test_public_lindblad_step_machine_precision`).
- Added test confirming the exact form handles `gamma*dt > 1`
  without positivity violation (a regime the Euler form cannot
  reach).
- Updated README with honest speed and accuracy claims.
- 42 tests, all passing.

## 0.1.0 (2026-06-17)

Initial release.

- Core FPM affine map primitives (`fpm_affine_step`,
  `fpm_affine_trajectory`, `kappa_from_gamma`, `gamma_from_kappa`).
- Drop-in Lindblad dephasing API (`lindblad_step`, `simulate`).
- Bounded gamma form with falsifiable ceiling (`bounded_gamma`,
  `FalsificationError`).
- Density matrix utilities (`pure_state`, `maximally_mixed`,
  `partial_trace`, `is_density_matrix`, `trace_distance`,
  `fidelity`).
- Closed-universe conservation ledger (`ConservationLedger`,
  `DaemonState`).
- Tests reproducing paper Test 01, Test 02, Test 03, Test 07, and
  Test 09.
- Usage examples (basic dephasing, Lindblad correspondence,
  bounded gamma, multi-qubit partial trace).
