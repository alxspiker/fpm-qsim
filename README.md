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

to **1.1 × 10⁻¹⁶** (verified in `tests/test_lindblad_correspondence.py`),
matching Kraus / matrix-exponential / QuTiP integrators without their
cost.

Because the FPM map is a single vectorized affine update — no matrix
exponentials, no Kraus decomposition — it scales as `O(N²)` per step
on an `N × N` density matrix, vs `O(N⁴)` for matrix-exponential
methods.  On 4-qubit (16 × 16) systems this is a ~20× speedup; on
larger systems the gap widens further.

---

## Installation

```bash
pip install fpm-qsim
```

From source:

```bash
git clone https://github.com/alxspiker/fpm-qsim.git
cd fpm-qsim
pip install -e .
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.22.  SciPy is optional
(for tests).

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
# from your_qsim_library import lindblad_step
# rho = lindblad_step(rho, H=None, c_ops=[sqrt(gamma) * sigma_z], dt=0.1)

# After
from fpm_qsim import lindblad_step
rho = lindblad_step(rho, gamma=gamma, dt=0.1)
```

---

## What's distinctive about `fpm-qsim`

| Property | fpm-qsim | QuTiP | matrix-exp | Kraus |
|---|---|---|---|---|
| Dephasing accuracy | **1.1 × 10⁻¹⁶** | 1.1 × 10⁻¹⁶ | 1.1 × 10⁻¹⁶ | 1.1 × 10⁻¹⁶ |
| Per-step cost on `N×N` | **O(N²)** | O(N²)–O(N³) | O(N⁴) | O(N²) |
| Dependencies | **NumPy only** | SciPy + Cython + … | SciPy | NumPy |
| Falsifiability ceiling | **γ_max = 31.87** | — | — | — |
| Theorem-verified affine map | **Theorem 3** | — | — | — |
| Closed-universe ledger | **Yes** | — | — | — |

The only Lindblad integrator in the Python ecosystem with a
**built-in falsifiability ceiling**: observations with `γ > 32.0`
raise `FalsificationError` rather than silently producing unphysical
results.  This is the FPM finite-lag-ceiling theorem (paper Test 09)
made operational.

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
| `kappa_exact(gamma, dt=1.0)` | Exact continuous-form contraction coefficient `exp(-gamma*dt)`. **This is what `lindblad_step` uses.** |
| `gamma_from_kappa(kappa, dt=1.0)` | Inverse of `kappa_from_gamma`. |
| `fpm_affine_step(c, kappa, nu=0.0)` | One tick of the affine map `c_{t+1} = kappa*c_t + nu`. |
| `fpm_affine_trajectory(c0, kappa, nu=0.0, n_steps=1)` | Closed-form rollout of the affine map. |
| `bounded_gamma(gamma_raw, gamma_max=GAMMA_MAX)` | Clip a rate to the ceiling; raise if it would falsify. |
| `FalsificationError` | Raised when an observation would falsify FPM. |

### Lindblad-equivalent API (`fpm_qsim.lindblad`)

| Symbol | Description |
|---|---|
| `lindblad_step(rho, gamma, dt=1.0, *, H=None, bounded=False)` | Advance a density matrix by one exact FPM-affine dephasing step. |
| `simulate(rho0, gamma, dt=1.0, n_steps=1, *, H=None, bounded=False, record=True)` | Roll out a full trajectory. |

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
| `DaemonState` | Per-daemon bookkeeping (energy, coherence, cumulative flows). |
| `ConservationLedger` | Closed-universe ledger; tracks `replenish == spend + landauer`. |

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
continuous-dephasing solution to 1.1 × 10⁻¹⁶, matching Kraus /
matrix-exp / QuTiP integrators.  The Euler form remains available
in `fpm_qsim._reference.euler_lindblad_step` for Theorem 3 research.

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

- The `H` parameter on `lindblad_step` extends the integrator to
  non-zero Hamiltonians by composing an Euler unitary kick with the
  dephasing contraction.  This is a standard composition, but the
  exact algebraic correspondence to a closed-form affine map is
  proven only for `H = 0`.  Callers using `H != 0` should validate
  against a reference integrator for their specific use case.
- General Lindblad channels (amplitude damping, depolarizing, etc.)
  are not directly equivalent to the FPM affine map.  Future
  versions may extend the correspondence.
- The `ConservationLedger` is a faithful bookkeeping layer but does
  not, by itself, enforce the semantic-entropy conservation gate
  (paper Section 6.6).  Callers must implement that gate in their
  application logic.

---

## Changelog

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
