# fpm-qsim

C++ accelerated FPM quantum-simulation primitives exposed as a Python extension
module.

Install from PyPI:

```bash
pip install fpm-qsim
```

Use it from Python:

```python
import fpm_cpp as fpm
```

## Layout

```text
.
├── src/
│   ├── fpm_core.hpp          # Header-only C++17 FPM core
│   └── fpm_cpp_bindings.cpp  # pybind11 extension bindings
├── scripts/
│   └── build.sh              # Manual Linux/macOS/WSL build helper
├── tests/
│   ├── smoke_test.py         # Fast installed-wheel smoke test
│   └── equivalence_test.py   # Reference equivalence checks
├── pyproject.toml            # PyPI build metadata
├── setup.py                  # Extension build configuration
├── MANIFEST.in               # Source distribution contents
└── README.md
```

## Example

```python
import numpy as np
import fpm_cpp as fpm

rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)

rho1 = fpm.lindblad_step(
    rho0,
    gamma=0.1,
    dt=1.0,
    method="exact",
    use_omp=False,
)

traj = fpm.simulate(
    rho0,
    gamma=0.05,
    dt=1.0,
    n_steps=1000,
    method="exact",
    use_omp=True,
)

fpm.bounded_gamma(29.3)
```

## Build From Source

PyPI installs prebuilt wheels when available. If no wheel matches your platform,
pip builds from source and requires:

- Python 3.9+
- NumPy
- pybind11
- A C++17 compiler

Manual local build:

```bash
python -m pip install pybind11 numpy
./scripts/build.sh
python tests/smoke_test.py
```

On Windows, normal `pip install fpm-qsim` builds with MSVC when a matching wheel
is unavailable.

## API Surface

The extension module is `fpm_cpp`.

Core functions:

- `lindblad_step(rho, gamma, dt=1.0, method="exact", use_omp=True)`
- `simulate(rho0, gamma, dt=1.0, n_steps=1, method="exact", use_omp=True)`
- `kappa_from_gamma(gamma, dt=1.0)`
- `kappa_exact(gamma, dt=1.0)`
- `gamma_from_kappa(kappa, dt=1.0)`
- `bounded_gamma(gamma_raw, gamma_max=GAMMA_MAX)`
- `gamma_from_energy(daemon, gate_power, load=None, dt=1.0, C_N=1.0, bounded=True)`

State and accounting:

- `DaemonState`
- `ConservationLedger`
- `exp_route_cost(taylor_order=8)`
- `bill_exp_route_cost(ledger, daemon, taylor_order=8, cost_per_op=1e-5)`

Constants:

- `GAMMA_MAX`
- `FALSIFICATION_THRESHOLD`
- `ENERGY_FLOOR_FRACTION`
- `ISOTROPIC_WEIGHT_LIMIT`

## Verification

After installing or building locally:

```bash
python tests/smoke_test.py
```

`tests/equivalence_test.py` is retained for reference comparison workflows that
also have the old Python implementation available.

## License

MIT
