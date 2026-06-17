# fpm-qsim

**Drop-in Lindblad dephasing simulator backed by the Finite Possibility Mechanics affine map.**

`fpm-qsim` is a small, dependency-light Python package that lets you
simulate open-system quantum dephasing dynamics using the FPM affine
coherence map.  The map

```
c_{t+1} = kappa * c_t + nu
```

with `kappa ∈ [0, 1]` is the engine.  Theorem 3 of the FPM paper
identifies one particular choice — `kappa = 1 - gamma*dt` — with the
Euler-discretized Lindblad dephasing equation.  This package uses the
**exact continuous** form

```
kappa = exp(-gamma * dt)
```

which is also a valid FPM affine-map coefficient and which makes the
integrator **machine-precise** for pure dephasing: it reproduces the
analytic continuous-dephasing solution

```
rho(t) = exp(-gamma*t) * (rho_0 - diag(rho_0)) + diag(rho_0)
```

to machine precision (about `5e-16` in the benchmark), matching
Kraus and matrix-exponential references for pure dephasing.

The honest positioning is not "uniquely fastest."  `fpm-qsim` is the
reference implementation of the FPM affine-map primitives: it provides
competitive pure-dephasing simulation, a NumPy-only pure-dephasing
path, the FPM falsifiability ceiling, and the conservation-ledger
primitives needed by downstream FPM research.

As of v0.1.5, `gamma` no longer has to be a free external parameter:
`gamma_from_energy` can derive dephasing from a daemon's energy budget,
gate power, and local load.  This is the FPM-specific noise model:
high-power gates, higher load, or energy-starved daemons decohere
faster.

As of v0.1.6, the `Circuit` layer composes unitary gates with FPM
dephasing layers under a single closed-universe ledger, with automatic
billing of every simulated operation's route cost.

As of v0.1.7, `Circuit.run_with_replenishment` automates the closed-
universe balancing: after each step, the daemon is replenished by
exactly the energy debited that tick, keeping the conservation
identity `replenish == spend + landauer` satisfied to ~0 drift
(when no external Landauer is charged).

As of v0.1.8, the `daemons` parameter attaches a per-qubit
`DaemonState` to each qubit, turning a multi-qubit circuit into a
network of FPM daemons.  Each qubit's dephasing rate is derived from
its own daemon's energy budget, billing is routed to the owning
daemon(s), and `run_with_replenishment` keeps the network-wide
identity satisfied across all daemons.  This is the structural
primitive for the planned `fpm-bft`, `fpm-fed`, and `fpm-marl`
packages.

For speed, the important distinction is structural vs constant-factor:
pure dephasing can be implemented in `O(N^2)` per step by any
dephasing-aware method.  `fpm-qsim` is competitive with that
specialized baseline and much faster than a general matrix-exponential
Liouvillian on the same pure-dephasing problem.

---

## Installation

Install from PyPI:

```bash
pip install fpm-qsim
```

From source:

```bash
git clone https://github.com/alxspiker/fpm-qsim.git
cd fpm-qsim
pip install -e .
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.22.  SciPy is optional for
tests and for `unitary_step`.

---

## Quick start

```python
import math
import numpy as np
import fpm_qsim as fpm

# Start in the |+><+| state (maximal coherence).
rho0 = fpm.pure_state([1, 1])

# Apply one dephasing step.  Off-diagonal contracts by exp(-gamma*dt):
rho1 = fpm.lindblad_step(rho0, gamma=0.1, dt=1.0)
assert math.isclose(abs(rho1[0, 1]), math.exp(-0.1) * abs(rho0[0, 1]))

# Roll out a full trajectory.
traj = fpm.simulate(rho0, gamma=0.05, dt=1.0, n_steps=200)
assert traj.shape == (201, 2, 2)
```

### Drop-in replacement

If you have an existing Lindblad dephasing loop, swap one import:

```python
# Before
# from your_qsim_library import dephasing_step
# rho = dephasing_step(rho, c_ops=[sqrt(gamma / 2) * sigma_z], dt=0.1)

# After
from fpm_qsim import lindblad_step
rho = lindblad_step(rho, gamma=gamma, dt=0.1)
```

---

## Ontological distinction: `method="euler"` vs `method="exact"` (v0.1.4)

The FPM affine map `c_{t+1} = kappa * c_t + nu_t` constrains `kappa ∈ [0, 1]`
but does not mandate a specific form. v0.1.4 exposes two forms with
**strict ontological boundaries** between them:

### `method="euler"` — the FPM-native lattice map

```python
rho_next = fpm.lindblad_step(rho, gamma=0.05, dt=1.0, method="euler")
```

Uses `kappa = 1 - gamma*dt`. This is the literal Theorem 3
identification: the FPM affine map with this kappa is algebraically
equivalent to the Euler-discretized Lindblad dephasing equation.

**Under FPM's discrete action principle, this is the physically
realizable lattice operation.** Per state variable per step, the
simulated route cost is exactly **1 multiplication + 1 addition** —
both finite-integer operations on the discrete lattice. Billable to
the `ConservationLedger`:

```python
ledger.bill_compute_cost(daemon, n_multiplies=1, n_adds=1, cost_per_op=1e-5)
```

This preserves closed-universe structural coherence: every operation
performed inside the simulated universe is paid for by the daemon
that performed it. **Use this method when running FPM-aligned
simulations where the closed-universe ledger must remain balanced.**

Constraint: requires `0 ≤ gamma*dt ≤ 1` for the map to remain
contractive. Use `method="exact"` for unbounded `gamma*dt`.

### `method="exact"` — the legacy continuous-math oracle

```python
rho_next = fpm.lindblad_step(rho, gamma=0.05, dt=1.0, method="exact")  # default
```

Uses `kappa = exp(-gamma*dt)`. Machine-precise continuous-dephasing
solution: reproduces the analytic result to ~5e-16, matching Kraus /
matrix-exponential / QuTiP integrators.

**Ontological warning.** Under FPM, `exp` is a continuous-math
idealization. The simulated system cannot natively evaluate
`exp(-gamma*dt)`; doing so requires a discrete computational
construction (Taylor series, CORDIC, Padé approximant) whose
finite-integer route cost must be paid by the simulated daemons.
When `method="exact"` is used, the host evaluates `np.exp` and hands
the result to the simulated system as a **zero-cost oracle**,
deliberately breaking the FPM closed-universe ledger.

**Use this method only for legacy-continuous-QM comparison**, e.g.
when you need to benchmark FPM against standard quantum mechanics.
To preserve closed-universe accounting with this method, the caller
must explicitly bill the simulated construction cost:

```python
# After each method="exact" step, bill the simulated Taylor
# construction of exp(-gamma*dt) via a K-term series.
# cost_per_op is the energy fraction per simulated op; pick a value
# appropriate to your simulation's thermodynamic regime.
fpm.bill_exp_route_cost(ledger, daemon, taylor_order=8, cost_per_op=1e-5)
```

This bills `(2K multiplies + K additions)` per `exp` evaluation,
keeping the ledger closed despite the oracle injection. The
`taylor_order` parameter controls the simulated approximation depth
(default 8, giving ~1e-15 accuracy for `|gamma*dt| < 1`).

### Default

`method="exact"` is the default for backward compatibility and for
users who want machine-precision results without thinking about FPM
ontology. FPM-purist research should explicitly pass
`method="euler"`.

---

## Endogenous noise: deriving `gamma` from energy (v0.1.5)

Most Lindblad tools ask the caller to provide a dephasing rate:

```python
rho = fpm.lindblad_step(rho, gamma=0.05, dt=1.0)
```

That path still works.  The FPM-distinctive path derives `gamma` from
the daemon doing the work:

```python
ledger = fpm.ConservationLedger(E_max_total=100.0)
daemon = ledger.add_daemon(E_init=75.0)

gamma = fpm.gamma_from_energy(
    daemon,
    gate_power=0.20,
    load=0.10,
    dt=1.0,
)
rho = fpm.lindblad_step(rho, gamma=gamma, dt=1.0, method="euler")
```

For convenience, `lindblad_step` and `simulate` can derive `gamma`
directly:

```python
rho = fpm.lindblad_step(
    rho,
    dt=1.0,
    daemon=daemon,
    gate_power=0.20,
    load=0.10,
    method="euler",
)
```

The contraction model is

```
kappa_t = C_N * (1 + B_t)^(-3/4)
gamma_t = (1 - kappa_t) / dt
```

with the package's minimal gate-noise load

```
B_t = load + gate_power / energy_fraction
```

So dephasing increases when:

- gate power increases,
- local load increases,
- daemon energy fraction decreases.

Use this when you want noise to be endogenous to the simulated FPM
ledger rather than supplied as an external phenomenological rate.

---

## Circuit layer (v0.1.6)

`Circuit` composes unitary gates with FPM dephasing layers under a
single closed-universe ledger.  It removes the boilerplate of
manually composing `unitary_step` and `lindblad_step` (and the
associated ledger billing) for the common case of "apply these gates,
then dephase, repeat N times".

The circuit is a **queue of unitary gates and dephasing layers**.
Each call to `step()` applies the full queue once.  `run(rho0, n_steps)`
repeats the queue `n_steps` times and returns the trajectory.

### Minimal usage

```python
import fpm_qsim as fpm

circ = fpm.Circuit(2)
circ.h(0).cx(0, 1).dephase(gamma=0.05)

rho0 = fpm.pure_state([1, 0, 0, 0])  # |00><00|
traj = circ.run(rho0, n_steps=10)
# traj.shape == (11, 4, 4)
```

### FPM-aligned usage (closed-universe billing)

```python
ledger = fpm.ConservationLedger(E_max_total=100.0)
daemon = ledger.add_daemon(80.0)

circ = fpm.Circuit(
    2,
    daemon=daemon,
    ledger=ledger,
    method="euler",            # FPM-native lattice map
    bounded=True,              # raise on gamma > 32
    default_gate_power=0.05,   # endogenous-noise input
    cost_per_op=1e-5,          # energy fraction per simulated op
)
circ.h(0).cx(0, 1).dephase(dt=1.0)

rho = fpm.pure_state([1, 0, 0, 0])
for _ in range(20):
    prev_spend = daemon.cumulative_spend
    rho = circ.step(rho)
    spent = daemon.cumulative_spend - prev_spend
    ledger.record_replenish(daemon, spent)  # closed-universe identity
```

### Available gates

| Method | Gate |
|---|---|
| `h(i)`, `x(i)`, `y(i)`, `z(i)` | Single-qubit Pauli + Hadamard |
| `s(i)`, `t(i)` | Phase gates |
| `u(theta, phi, lam, i)` | General single-qubit unitary |
| `cx(control, target)` | CNOT |
| `cz(i, j)` | Controlled-Z |
| `swap(i, j)` | SWAP |
| `apply_unitary(U, targets)` | Arbitrary k-qubit unitary (k ≤ 10) |
| `apply_unitary_full(U_full)` | Pre-expanded full-Hilbert-space unitary (any n) |
| `dephase(gamma=None, *, dt=1.0, gate_power=None, load=None)` | FPM dephasing layer |

Gates are applied via **direct conjugation** `U @ rho @ U^dagger`
(standard quantum-circuit convention), not as Hamiltonian time
evolution.  For Hamiltonian dynamics, use `strang_step(rho, H, gamma, dt)`.

### Ontological billing

When `daemon` and `ledger` are attached, every operation bills the
closed-universe ledger for its simulated construction cost:

| Operation | Billed as |
|---|---|
| Unitary gate | `N^2` scalar `exp` constructions (one per matrix element), each via K-term Taylor series |
| `dephase(method="euler")` | 1 mul + 1 add per off-diagonal state var (`N*(N-1)` ops) — literal Theorem 3 lattice operation |
| `dephase(method="exact")` | 1 scalar `exp` per off-diagonal state var, each via K-term Taylor — oracle break, billed explicitly |

Billing is opt-in: if `ledger` is `None` or `daemon` is `None`, no
billing happens and the circuit behaves as a pure state-stepper.

### Strang splitting for Hamiltonian + dephasing

```python
H = np.array([[1, 1], [1, -1]], dtype=complex) * 0.5
rho = fpm.pure_state([1, 1])
circ = fpm.Circuit(1, daemon=daemon, ledger=ledger, method="euler")
rho = circ.strang_step(rho, H, gamma=0.05, dt=0.5)
# Implements: U(dt/2) -> dephase(dt) -> U(dt/2)  [O(dt^3) per-step error]
```

### Honest scope

The circuit layer implements only the gate set with a clean FPM
correspondence.  General dissipative channels (amplitude damping,
depolarizing, etc.) remain out of scope: they have no FPM
correspondence theorem.  Only the pure-dephasing affine map of
Theorem 3 is wired in via `dephase`.

### Closed-universe balancing (v0.1.7)

`run_with_replenishment(rho0, n_steps)` is the same as `run()` but,
after each `step()`, replenishes the daemon by exactly the energy
debited that tick (spend + landauer).  This keeps the closed-universe
identity `replenish == spend + landauer` satisfied to ~0 drift when
no external Landauer is charged.

```python
ledger = fpm.ConservationLedger(E_max_total=100.0)
daemon = ledger.add_daemon(80.0)
circ = fpm.Circuit(
    2, daemon=daemon, ledger=ledger, method="euler",
    default_gate_power=0.05, cost_per_op=1e-5,
)
circ.h(0).cx(0, 1).dephase(dt=1.0)

rho0 = fpm.pure_state([1, 0, 0, 0])
traj = circ.run_with_replenishment(rho0, n_steps=20)
# ledger.drift() is ~0 (no external Landauer).
# daemon.E is preserved (balanced replenishment).
```

Honest behavior at the boundaries:

- **Daemon near `E_max`**: replenishment is capped at `E_max - E`,
  drift may grow.  The framework is reporting that `E_max_total` is
  too small to absorb the requested computation.
- **Daemon near the energy floor**: spend is capped (existing
  behavior of `bill_compute_cost`), replenishment restores the
  daemon.
- **External Landauer**: debits charged via `ledger.record_landauer`
  between `step()` calls are NOT replenished by this method.  The
  caller is responsible for those.

Requires `daemon` and `ledger` to be attached.  Raises `ValueError`
otherwise, pointing users to plain `run()` for open-system
simulations.

### Multi-daemon circuits (v0.1.8)

The `daemons` parameter attaches a per-qubit `DaemonState` to each
qubit, turning a multi-qubit circuit into a network of FPM daemons.
Each qubit's dephasing rate is derived from its own daemon's energy
budget, billing is routed to the owning daemon(s), and
`run_with_replenishment` keeps the network-wide identity satisfied
across all daemons.

```python
ledger = fpm.ConservationLedger(E_max_total=100.0)
d0 = ledger.add_daemon(80.0)  # energy-rich -> slow dephasing
d1 = ledger.add_daemon(40.0)  # energy-poor -> fast dephasing
circ = fpm.Circuit(
    2, daemons=[d0, d1], ledger=ledger, method="euler",
    default_gate_power=0.05, cost_per_op=1e-5,
)
circ.h(0).cx(0, 1).dephase(dt=1.0)

rho0 = fpm.pure_state([1, 0, 0, 0])
traj = circ.run_with_replenishment(rho0, n_steps=20)
# d0 billed for H + half of CNOT + half of dephase
# d1 billed for half of CNOT + half of dephase
# Network-wide ledger.drift() is ~0.
```

**Per-qubit billing rules:**

| Operation | Billing in multi-daemon mode |
|---|---|
| Single-qubit gate on qubit `i` | `daemons[i]` pays the full bill |
| Two-qubit gate on qubits `i, j` | Split 50/50 between `daemons[i]` and `daemons[j]` |
| `apply_unitary_full` | All daemons billed equally |
| `dephase(targets=None)` | All daemons billed |
| `dephase(targets=[i, ...])` | Only the named daemons billed |

**Per-qubit endogenous γ:** when `daemons` is set and a dephasing
layer uses endogenous γ (no explicit `gamma`), each qubit's
dephasing rate is derived from its own daemon's energy budget via
`gamma_from_energy`.  An energy-rich qubit decoheres slowly; an
energy-poor qubit decoheres fast.  This is implemented via the
`_dephase_single_qubit` primitive, which contracts only the
off-diagonal elements where the target qubit's bit index differs
between row and column.

**Targeted dephasing:** `dephase(targets=[0])` dephases only qubit
0, leaving qubit 1's coherences untouched.  Useful for modeling
isolated subsystems or injecting noise into specific qubits.

**Backward compatibility:** all v0.1.7 single-daemon code continues
to work unchanged.  Use `daemon=` for single-daemon mode, `daemons=`
for per-qubit mode.  The two are mutually exclusive.

---

## What's distinctive about `fpm-qsim`

| Property | fpm-qsim | QuTiP | matrix-exp | Kraus |
|---|---|---|---|---|
| Dephasing accuracy | **~5e-16** | ~2e-7 in the benchmark | ~5e-16 | ~5e-16 |
| Pure-dephasing speed | **Competitive with dephasing-specialized O(N^2)** | General solver overhead | General or specialized | Single-qubit baseline |
| Dependencies | **NumPy for pure dephasing; SciPy only for `unitary_step`** | SciPy + Cython + ... | SciPy for general matrix-exp | NumPy |
| Falsifiability ceiling | **gamma_max = 31.87** | --- | --- | --- |
| Theorem-verified affine map | **Theorem 3** | --- | --- | --- |
| Closed-universe ledger | **Yes** | --- | --- | --- |
| Endogenous gamma from energy | **Yes (v0.1.5)** | --- | --- | --- |
| Circuit layer with billing | **Yes (v0.1.6)** | --- | --- | --- |
| Auto-balanced closed-universe runs | **Yes (v0.1.7)** | --- | --- | --- |
| Multi-daemon per-qubit networks | **Yes (v0.1.8)** | --- | --- | --- |
| Combined unitary + dephasing | **`Circuit` queue or `strang_step`** | Built in | Built in | Channel-specific |

The only Lindblad integrator in the Python ecosystem with a
**built-in falsifiability ceiling**: observations with `gamma > 32.0`
raise `FalsificationError` rather than silently producing unphysical
results.  This is the FPM finite-lag-ceiling theorem (paper Test 09)
made operational.

Equally important: this package is where the FPM theorem is made
operational.  `kappa_exact`, `kappa_from_gamma`, `bounded_gamma`,
`ConservationLedger`, and `DaemonState` are not generic convenience
functions; they are the public primitives for building FPM-aligned
simulators, auditors, and falsification tests.

---

## Benchmarks

The benchmark corrected three earlier overclaims:

- The large speedup against general matrix-exp is real, but structural:
  it comes from exploiting pure-dephasing structure, which any
  dephasing-aware implementation can also exploit.
- With the gamma convention corrected, fpm-qsim, matrix-exp, and Kraus
  all match the analytic pure-dephasing solution at machine precision.
- The old `H` parameter on `lindblad_step` was removed because it used
  a naive Euler unitary kick.  Use `unitary_step` and compose the
  splitting explicitly.

Speed results for 1,000 pure-dephasing steps:

| Qubits | Dim | fpm-qsim | general matrix-exp | dephasing-specialized matrix-exp | scipy `solve_ivp` | QuTiP | Kraus |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 3.38 ms | 1.88 ms | 1.18 ms | 13.28 ms | 10.44 ms | 5.47 ms |
| 2 | 4 | 3.31 ms | 1.98 ms | 1.45 ms | 13.76 ms | --- | --- |
| 3 | 8 | 3.39 ms | 15.03 ms | 1.86 ms | 28.58 ms | --- | --- |
| 4 | 16 | 4.03 ms | 136.08 ms | 3.19 ms | 147.54 ms | --- | --- |
| 5 | 32 | 7.28 ms | --- | 6.19 ms | --- | --- | --- |
| 6 | 64 | 17.30 ms | --- | 18.34 ms | --- | --- | --- |

At 4 qubits, `fpm-qsim` is about `34x` faster than the general
matrix-exp baseline, but about `25%` slower than a matrix-exp baseline
that is specialized for pure dephasing.  The fair claim is that
`fpm-qsim` is competitive with the best dephasing-specialized approach
while also carrying the FPM research API and falsifiability checks.

Accuracy results, measured as max absolute error vs the analytic
solution after 1,000 steps:

| Qubits | Dim | fpm-qsim | general matrix-exp | dephasing-specialized matrix-exp | scipy `solve_ivp` | QuTiP | Kraus |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 4.6e-16 | 4.6e-16 | 4.6e-16 | 4.00e-12 | 2.04e-07 | 4.6e-16 |
| 2 | 4 | 3.5e-16 | 3.5e-16 | 3.5e-16 | 1.57e-12 | --- | --- |
| 3 | 8 | 3.4e-16 | 3.4e-16 | 3.4e-16 | 1.33e-12 | --- | --- |
| 4 | 16 | 1.9e-16 | 1.9e-16 | 1.9e-16 | 1.16e-12 | --- | --- |
| 5 | 32 | 9.7e-17 | --- | 9.7e-17 | --- | --- | --- |
| 6 | 64 | 7.4e-17 | --- | 7.4e-17 | --- | --- | --- |

Benchmark configuration: `gamma = 0.02`, `dt = 1.0`, Haar-random pure
initial state, wall time reported as the minimum of three repeats.

---

## API reference

### Core FPM primitives (`fpm_qsim.core`)

| Symbol | Description |
|---|---|
| `GAMMA_MAX = 31.8738...` | Falsifiable Lorentz-factor ceiling derived from the finite-lag theorem. |
| `FALSIFICATION_THRESHOLD = 32.0` | Observations above this falsify FPM. |
| `ENERGY_FLOOR_FRACTION = 0.03138...` | v5.0 zero-energy floor (Test 07). |
| `ISOTROPIC_WEIGHT_LIMIT = 1/3` | Spectral-gap isotropic limit weight (Test 04). |
| `kappa_from_gamma(gamma, dt=1.0)` | Euler-form contraction coefficient `1 - gamma*dt` (Theorem 3 form). |
| `kappa_exact(gamma, dt=1.0)` | Exact continuous-form contraction coefficient `exp(-gamma*dt)`. Used by `lindblad_step` when `method="exact"`. |
| `gamma_from_kappa(kappa, dt=1.0)` | Inverse of `kappa_from_gamma`. |
| `gamma_from_energy(daemon, gate_power, load=None, dt=1.0)` | **v0.1.5.** Derive endogenous dephasing from daemon energy, gate power, and local load. |
| `fpm_affine_step(c, kappa, nu=0.0)` | One tick of the affine map `c_{t+1} = kappa*c_t + nu`. |
| `fpm_affine_trajectory(c0, kappa, nu=0.0, n_steps=1)` | Closed-form rollout of the affine map. |
| `bounded_gamma(gamma_raw, gamma_max=GAMMA_MAX)` | Clip a rate to the ceiling; raise if it would falsify. |
| `FalsificationError` | Raised when an observation would falsify FPM. |

### Lindblad-equivalent API (`fpm_qsim.lindblad`)

| Symbol | Description |
|---|---|
| `lindblad_step(rho, gamma=None, dt=1.0, *, daemon=None, gate_power=None, load=None, method="exact", bounded=False)` | Advance a density matrix by one FPM-affine dephasing step. Pass explicit `gamma` for legacy use, or pass `daemon` + `gate_power` to derive endogenous gamma. |
| `unitary_step(rho, H, dt=1.0)` | Apply one exact Hamiltonian step, `rho -> U rho U^dagger`, using a matrix exponential. |
| `simulate(rho0, gamma=None, dt=1.0, n_steps=1, *, daemon=None, gate_power=None, load=None, method="exact", bounded=False, record=True)` | Roll out a pure-dephasing trajectory. `method` and energy-derived gamma inputs are forwarded to `lindblad_step`. |

### State utilities (`fpm_qsim.states`)

| Symbol | Description |
|---|---|
| `basis_state(index, dim)` | Computational basis column vector. |
| `pure_state(amplitudes)` | Build `|psi><psi|` from unnormalized amplitudes. |
| `maximally_mixed(dim)` | `I_dim / dim`. |
| `partial_trace(rho, keep, dims)` | Partial trace over a tensor-product Hilbert space. |
| `is_density_matrix(rho, tol=1e-9)` | Validate Hermiticity, trace, positivity. |
| `trace_distance(rho, sigma)` | `0.5 * ||rho - sigma||_1`. |
| `fidelity(rho, sigma)` | Uhlmann fidelity. |

### Closed-universe conservation (`fpm_qsim.conservation`)

| Symbol | Description |
|---|---|
| `DaemonState` | Per-daemon bookkeeping (energy, coherence, local `load`, cumulative flows). |
| `ConservationLedger` | Closed-universe ledger; tracks `replenish == spend + landauer`. |
| `ConservationLedger.bill_compute_cost(daemon, n_multiplies, n_adds)` | **v0.1.4.** Bill a daemon for simulated discrete computational work. Use after `method="euler"` steps (1 mul + 1 add per state var) or any other simulated compute. |
| `exp_route_cost(taylor_order=8)` | **v0.1.4.** Return `(n_multiplies, n_adds)` cost of constructing `exp(-gamma*dt)` via a K-term Taylor series under FPM's discrete action principle. |
| `bill_exp_route_cost(ledger, daemon, taylor_order=8)` | **v0.1.4.** Convenience wrapper: bill the simulated Taylor construction of one `exp` evaluation. Use after each `method="exact"` step to keep the closed-universe ledger balanced despite the oracle. |

### Circuit layer (`fpm_qsim.circuit`) — v0.1.6, v0.1.7, v0.1.8

| Symbol | Description |
|---|---|
| `Circuit(n_qubits, *, daemon=None, daemons=None, ledger=None, method="exact", bounded=False, default_load=None, default_gate_power=None, cost_per_op=1e-5, taylor_order=8)` | Queue-based circuit composing unitary gates with FPM dephasing. Use `daemon=` for single-daemon mode (v0.1.6) or `daemons=` for per-qubit multi-daemon mode (v0.1.8). Auto-bills the ledger when daemon(s) and ledger are attached. |
| `Circuit.h(i)`, `.x(i)`, `.y(i)`, `.z(i)`, `.s(i)`, `.t(i)` | Single-qubit gates (fluent, return self). |
| `Circuit.u(theta, phi, lam, i)` | General single-qubit unitary. |
| `Circuit.cx(control, target)`, `.cz(i, j)`, `.swap(i, j)` | Two-qubit gates. |
| `Circuit.apply_unitary(U, targets)` | Append an arbitrary k-qubit unitary (k ≤ 10). |
| `Circuit.apply_unitary_full(U_full)` | Append a pre-expanded full-Hilbert-space unitary (any n). |
| `Circuit.dephase(gamma=None, *, dt=1.0, gate_power=None, load=None, targets=None)` | Append an FPM dephasing layer. Endogenous gamma derived from daemon if `gamma` is omitted. **v0.1.8:** `targets` restricts the layer to specific qubits (multi-daemon mode: only named daemons billed). |
| `Circuit.step(rho)` | Apply the full queued sequence once. Returns the next density matrix. |
| `Circuit.run(rho0, n_steps=1, *, record=True)` | Apply `step()` `n_steps` times. Returns trajectory of shape `(n_steps+1, dim, dim)` if `record=True`, else final state. |
| `Circuit.run_with_replenishment(rho0, n_steps=1, *, record=True)` | **v0.1.7.** Same as `run()` but replenishes each daemon by exactly the debited amount (spend + landauer) each tick, keeping the closed-universe ledger balanced. **v0.1.8:** In multi-daemon mode, replenishes every daemon. |
| `Circuit.strang_step(rho, H, gamma, dt, *, gate_power=None, load=None)` | One Strang-split round: `U(dt/2) + dephase(dt) + U(dt/2)`. Uses `expm(-i H dt/2)` for the half-steps. Bills three operations. **v0.1.8:** Supports multi-daemon mode with per-qubit endogenous γ. |
| `Circuit.reset()` | Clear the queue (does not reset billing counters). |
| `Circuit.reset_stats()` | Reset billing counters (does not clear the queue). |
| `Circuit.operations` | List of human-readable descriptions of queued ops. |
| `Circuit.gates_applied`, `.dephase_layers_applied` | Cumulative billing counters. |
| `Circuit.is_multi_daemon` | **v0.1.8.** `True` if constructed with `daemons=`. |
| `Circuit.all_daemons` | **v0.1.8.** List of all daemons (1 in single-daemon mode, n_qubits in multi-daemon mode, 0 if no daemon). |
| `Circuit.daemons` | **v0.1.8.** Tuple of per-qubit daemons, or `None` in single-daemon mode. |

---

## The math, in one page

The FPM coherence variable `c_t` evolves under the affine map

```
c_{t+1} = kappa * c_t + nu
```

where `kappa ∈ [0, 1]` is the contraction coefficient and `nu` is a
bounded innovation noise.  The FPM framework constrains `kappa` to
`[0, 1]` but does not mandate a specific form; different `kappa`
choices correspond to different numerical integrators of the same
physics:

| `kappa` choice | What it computes | Accuracy |
|---|---|---|
| `1 - gamma*dt` | Euler-discretized Lindblad dephasing | `O(dt)` per step |
| `exp(-gamma*dt)` (this package) | Exact continuous Lindblad dephasing | **machine precision** |

**Theorem 3 (Lindblad Correspondence).** The Euler form (`kappa = 1 -
gamma*dt`) is algebraically identical to the Euler discretization of
the Lindblad master equation for a dephasing channel with `H = 0`:

```
rho_{t+1} = (1 - gamma*dt) * rho_t + gamma*dt * diag(rho_t)
```

under the identification `gamma_t = (1 - kappa_t) / dt`.  Verified
numerically to RMSE 3.6 × 10⁻¹⁷ on off-diagonal density-matrix
elements over 600 ticks and 10 random paths (paper Test 02; reproduced
in `tests/test_lindblad_correspondence.py::test_theorem3_lindblad_correspondence`).

The exact form (`kappa = exp(-gamma*dt)`) is **also** a valid FPM
affine-map coefficient (in `[0, 1]` for all `gamma, dt >= 0`) and is
what the public `lindblad_step` uses.  It reproduces the analytic
continuous-dephasing solution to machine precision, matching Kraus and
matrix-exp references.  The Euler form remains available in
`fpm_qsim._reference.euler_lindblad_step` for Theorem 3 research.

### Hamiltonian evolution

`lindblad_step` is deliberately scoped to pure dephasing.  Earlier
versions accepted an `H` parameter and applied a naive Euler unitary
kick before dephasing.  That silently reintroduced splitting error for
`H != 0` and broke the machine-precision guarantee.

For combined unitary + dephasing dynamics, compose the exact
Hamiltonian step explicitly.  The recommended Strang splitting is:

```python
rho = fpm.unitary_step(rho, H, dt / 2)
rho = fpm.lindblad_step(rho, gamma=gamma, dt=dt)
rho = fpm.unitary_step(rho, H, dt / 2)
```

This has `O(dt^3)` per-step splitting error.  `unitary_step` requires
SciPy for the matrix exponential; the pure-dephasing path remains
NumPy-only.

### The falsifiable ceiling

The finite-lag ceiling theorem (paper Test 09) caps the physically
admissible gamma at

```
gamma_max = 31.8738...
```

The CERN muon (`gamma = 29.3`) sits below this ceiling.  Any
observation of `gamma > 32.0` falsifies the FPM framework.  The
package refuses to silently clip such observations; pass
`bounded=True` to `lindblad_step` and a `FalsificationError` will be
raised — log the observation instead.

---

## Reproducing the paper's tests

The `tests/` directory reproduces four of the paper's numerical
experiments and adds a machine-precision regression test:

| Test file | Paper test | Reference |
|---|---|---|
| `tests/test_lindblad_correspondence.py::test_theorem3_lindblad_correspondence` | Test 02 | RMSE 3.6e-17 |
| `tests/test_lindblad_correspondence.py::test_public_lindblad_step_machine_precision` | new in v0.1.1 | max err 1.1e-16 |
| `tests/test_dispersion_contraction.py` | Test 01 | D* = 1.8e-4 |
| `tests/test_conservation.py` | Test 03 | Drift < 5% |
| `tests/test_bounded_gamma.py` | Test 07 + Test 09 | gamma_max = 31.87 |

Run them with:

```bash
pip install -e ".[test]"
pytest -v
```

---

## Scope and honest limitations

This package implements pure dephasing channels with `H = 0`, the
regime where the FPM theorem provides an algebraic correspondence.

- `lindblad_step` does not accept `H`.  For `H != 0`, compose
  `unitary_step` and `lindblad_step` explicitly, preferably with
  Strang splitting.  Accuracy then depends on the caller's chosen
  splitting strategy.
- General Lindblad channels (amplitude damping, depolarizing, etc.)
  are not directly equivalent to the FPM affine map.  Future
  versions may extend the correspondence.
- The `ConservationLedger` is a faithful bookkeeping layer but does
  not, by itself, enforce the semantic-entropy conservation gate
  (paper Section 6.6).  Callers must implement that gate in their
  application logic.

---

## Changelog

### 0.1.8 (2026-06-18)

**Multi-daemon circuits: per-qubit FPM networks.**

- Added `daemons` parameter to `Circuit.__init__`: one `DaemonState`
  per qubit.  Per-qubit billing (single-qubit gate bills only owning
  daemon; two-qubit gate splits 50/50; `apply_unitary_full` bills all
  equally).  Per-qubit endogenous γ (each qubit's dephasing rate
  derived from its own daemon).  Added `targets` keyword to
  `dephase()` for targeted dephasing.  `run_with_replenishment`
  replenishes every daemon each tick, keeping the network-wide
  identity satisfied.
- 36 new tests in `tests/test_circuit.py`. 181 tests total, all passing.
- Added `examples/06_multi_daemon.py`.
- Backward compatible: all v0.1.7 single-daemon code unchanged.

### 0.1.7 (2026-06-18)

**`Circuit.run_with_replenishment` — automatic closed-universe balancing.**

- Added `Circuit.run_with_replenishment(rho0, n_steps, *, record=True)`.
  Same as `run()` but, after each `step()`, replenishes the daemon by
  exactly the energy debited that tick (spend + landauer), keeping the
  closed-universe identity `replenish == spend + landauer` satisfied
  to within energy-floor/ceiling clipping.
- Requires `daemon` and `ledger` attached; raises `ValueError`
  otherwise (points users to `run()` for open-system simulations).
- 18 new tests in `tests/test_circuit.py`. 145 tests total, all passing.
- Updated `examples/05_circuit.py` to use the new method.

### 0.1.6 (2026-06-18)

**Circuit layer: queue-based composition of unitary gates and FPM dephasing.**

- Added `Circuit` class for composing unitary gates (`h`, `x`, `y`, `z`,
  `s`, `t`, `cx`, `cz`, `swap`, `u`, `apply_unitary`, `apply_unitary_full`)
  with FPM dephasing layers under a single closed-universe ledger.
- Gates applied via direct conjugation `U @ rho @ U^dagger` (standard
  circuit convention). For Hamiltonian time evolution, use
  `strang_step(rho, H, gamma, dt)`.
- Auto-bills the ledger when `daemon` and `ledger` are attached:
  unitaries as `N^2` scalar `exp` Taylor constructions, dephasing per
  the circuit's `method` (1 mul + 1 add per state var for `euler`;
  scalar `exp` per state var for `exact`).
- 60 new tests in `tests/test_circuit.py`. 127 tests total, all passing.
- Added `examples/05_circuit.py`.

### 0.1.5 (2026-06-17)

- Added `gamma_from_energy(daemon, gate_power, load=None, dt=1.0)`,
  deriving dephasing from daemon energy, gate power, and local load.
- `lindblad_step` and `simulate` now accept either explicit `gamma`
  or `daemon` + `gate_power` for endogenous FPM noise.
- Kept raw `gamma=...` as the legacy explicit-rate path.

### 0.1.4 (2026-06-17)

**Ontological boundary between FPM-native and continuous-math maps.**

- Added `method` parameter to `lindblad_step` and `simulate`:
  - `method="euler"`: FPM-native lattice map (`kappa = 1 - gamma*dt`).
    The literal Theorem 3 affine map.  Per-step simulated route cost
    is 1 multiply + 1 add — billable to `ConservationLedger`.
  - `method="exact"` (default): `kappa = exp(-gamma*dt)`.  Machine-
    precise continuous-dephasing solution.  Documented as a non-physical
    oracle break under FPM.  Callers who need closed-universe accounting
    must explicitly bill the simulated construction cost via
    `bill_exp_route_cost`.
- Added `ConservationLedger.bill_compute_cost(daemon, n_multiplies, n_adds)`
  to debit simulated discrete computational work.
- Added `exp_route_cost(taylor_order=8)` and `bill_exp_route_cost(ledger,
  daemon, taylor_order=8)` to bill the simulated Taylor construction
  of `exp(-gamma*dt)` when using `method="exact"`.
- 57 tests, all passing.

### 0.1.3 (2026-06-17)

- Reframed the README around the audit-corrected benchmark:
  competitive pure-dephasing performance, not unique speed dominance.
- Added benchmark tables for speed and accuracy.
- Documented the removal of `H` from `lindblad_step` and the explicit
  `unitary_step` composition path for `H != 0`.

### 0.1.2 (2026-06-17)

- Clarified the PyPI installation command in the README.

### 0.1.1 (2026-06-17)

- **Breaking:** `lindblad_step` now uses the exact continuous form
  `kappa = exp(-gamma*dt)` instead of the Euler form `1 - gamma*dt`.
  The public API now matches Kraus / matrix-exp / QuTiP to machine
  precision (max abs error 1.1e-16 vs analytic).  The Euler form
  is preserved in the private `fpm_qsim._reference` module for
  Theorem 3 verification.
- Added `kappa_exact(gamma, dt)` to the public API.
- Removed `reference_lindblad_step` from the public API (moved to
  `fpm_qsim._reference.euler_lindblad_step`).
- Added machine-precision regression test.
- Added test confirming the exact form handles `gamma*dt > 1`
  without positivity violation (a regime the Euler form cannot
  reach).
- Updated README with honest speed and accuracy claims.

### 0.1.0 (2026-06-17)

Initial release.  Euler-form affine map, Lindblad-equivalent API,
bounded gamma falsifiability, density-matrix utilities, closed-
universe conservation ledger.

---

## Citation

If you use `fpm-qsim` in published research, please cite:

```bibtex
@misc{spiker2026fpm,
  title  = {Finite Possibility Mechanics: A Unified Information-Theoretic Framework},
  author = {Spiker, Alx},
  year   = {2026},
  note   = {See Theorem 3 (Lindblad Correspondence) and the Finite-Lag-Ceiling Theorem.}
}

@misc{fpm-qsim,
  title  = {fpm-qsim: Drop-in Lindblad dephasing simulator backed by the FPM affine map},
  author = {Spiker, Alx},
  year   = {2026},
  url    = {https://github.com/alxspiker/fpm-qsim},
}
```

---

## License

MIT.  See `LICENSE`.
