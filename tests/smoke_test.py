import numpy as np

import fpm_cpp as fpm


rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
rho1 = fpm.lindblad_step(rho0, gamma=0.02, dt=1.0, method="exact", use_omp=False)

assert rho1.shape == (2, 2)
assert np.allclose(np.diag(rho1), np.diag(rho0))
assert abs(rho1[0, 1]) < abs(rho0[0, 1])

try:
    fpm.bounded_gamma(40.0)
except fpm.FalsificationError:
    pass
else:
    raise AssertionError("bounded_gamma(40.0) should raise FalsificationError")

print(f"fpm_cpp {fpm.__version__} smoke test passed")
