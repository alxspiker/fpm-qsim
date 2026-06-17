# CHANGES

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
