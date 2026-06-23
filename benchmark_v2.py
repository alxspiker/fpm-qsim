#!/usr/bin/env python3
"""
FPM C++ vs Python vs Competitors — Head-to-Head Benchmark (v2)
================================================================

Adds the C++ accelerated FPM module to the existing 7-method comparison:
  1. FPM C++ (OpenMP)        <- NEW
  2. FPM C++ (serial)        <- NEW
  3. FPM Python (NumPy)
  4. QuTiP mesolve
  5. Qiskit Aer phase-damp
  6. matrix-exp (general)
  7. matrix-exp (specialized)
  8. Kraus (single-qubit)
  9. scipy.solve_ivp

Extends qubit range from 1-6 to 1-8 qubits (Hilbert dim 2-256) to
demonstrate the C++ advantage at scale.

Output:
  /home/z/my-project/work/fpm_cpp_analysis/benchmark_results_v2.json
  /home/z/my-project/work/fpm_cpp_analysis/benchmark_results_v2.csv
"""
from __future__ import annotations

import gc
import json
import math
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.linalg

# ---------- Add the C++ module to path ----------
sys.path.insert(0, "/home/z/my-project/work/fpm_cpp_analysis/fpm_cpp")

# ---------- Imports ----------
import fpm_qsim as fpm_py
import fpm_cpp          # C++ accelerated FPM

try:
    import qutip
    HAVE_QUTIP = True
except Exception:
    HAVE_QUTIP = False

try:
    import qiskit
    import qiskit_aer
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, phase_damping_error
    from qiskit.quantum_info import DensityMatrix, Statevector
    HAVE_QISKIT = True
except Exception:
    HAVE_QISKIT = False

# ---------- Output paths ----------
OUT_DIR = Path("/home/z/my-project/work/fpm_cpp_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = OUT_DIR / "benchmark_results_v2.json"
CSV_PATH  = OUT_DIR / "benchmark_results_v2.csv"

# ---------- Benchmark configuration ----------
GAMMA = 0.02
DT = 1.0
N_STEPS = 1000
N_REPEATS = 3
# Extended qubit range: 1-7 (Hilbert dim 2-128). 8 qubits would require 1 GB
# of trajectory buffer per repeat and push the matrix-exp baselines past 10 GB.
N_QUBITS_LIST = [1, 2, 3, 4, 5, 6, 7]
SEED = 2026

# Per-method qubit caps (to avoid spending 4+ hours on the slow baselines):
METHOD_QUBIT_CAPS = {
    "matrix-exp (general)":  6,   # 2 GB at 6q is the hard limit
    "scipy.solve_ivp":       5,   # ~5s at 5q, ~30s at 6q (skip 6q+)
    "QuTiP mesolve":         6,   # ~3.5 s at 6q (skip 7q)
    "Qiskit Aer phase-damp": 4,
}
# Per-method repeat override (use 1 repeat for the slow baselines to save time)
METHOD_REPEATS_OVERRIDE = {
    "matrix-exp (general)": 1,
    "scipy.solve_ivp":      1,
    "QuTiP mesolve":        1,
}


# --------------------------------------------------------------------------- #
# Reference utilities
# --------------------------------------------------------------------------- #

def haar_random_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def analytic_dephasing(rho0: np.ndarray, gamma: float, t: float) -> np.ndarray:
    decay = math.exp(-gamma * t)
    diag = np.diagonal(rho0).copy()
    return decay * (rho0 - np.diag(diag)) + np.diag(diag)


def max_abs_error(traj: np.ndarray, rho0: np.ndarray, gamma: float, dt: float) -> float:
    n_steps = traj.shape[0] - 1
    err = 0.0
    for t in range(n_steps + 1):
        analytic = analytic_dephasing(rho0, gamma, t * dt)
        err = max(err, float(np.max(np.abs(traj[t] - analytic))))
    return err


def max_abs_error_endpoints(traj, rho0, gamma, dt, n_steps):
    analytic_final = analytic_dephasing(rho0, gamma, n_steps * dt)
    err_t0 = float(np.max(np.abs(traj[0] - rho0)))
    err_tN = float(np.max(np.abs(traj[1] - analytic_final)))
    return max(err_t0, err_tN)


def min_eigenvalue(rho: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).min())


# --------------------------------------------------------------------------- #
# Method 1: FPM C++ (OpenMP) — NEW
# --------------------------------------------------------------------------- #

def bench_fpm_cpp_omp(rho0, n_steps):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    traj = fpm_cpp.simulate(rho0, gamma=GAMMA, dt=DT, n_steps=n_steps,
                            method="exact", use_omp=True)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return traj, (t1 - t0), peak / (1024 * 1024)


# --------------------------------------------------------------------------- #
# Method 2: FPM C++ (serial) — NEW
# --------------------------------------------------------------------------- #

def bench_fpm_cpp_serial(rho0, n_steps):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    traj = fpm_cpp.simulate(rho0, gamma=GAMMA, dt=DT, n_steps=n_steps,
                            method="exact", use_omp=False)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return traj, (t1 - t0), peak / (1024 * 1024)


# --------------------------------------------------------------------------- #
# Method 3: FPM Python (NumPy) — the original
# --------------------------------------------------------------------------- #

def bench_fpm_py(rho0, n_steps):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    traj = fpm_py.simulate(rho0, gamma=GAMMA, dt=DT, n_steps=n_steps, method="exact")
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return traj, (t1 - t0), peak / (1024 * 1024)


# --------------------------------------------------------------------------- #
# Method 4: QuTiP mesolve
# --------------------------------------------------------------------------- #

def bench_qutip(rho0, n_steps):
    if not HAVE_QUTIP:
        return None, float("nan"), 0.0
    dim = rho0.shape[0]
    n_qubits = int(round(math.log2(dim)))
    c_ops = []
    for k in range(n_qubits):
        op = qutip.tensor([qutip.sigmaz() if i == k else qutip.qeye(2) for i in range(n_qubits)])
        c_ops.append(np.sqrt(GAMMA / 2.0) * op)
    H = qutip.qzero([2] * n_qubits)
    rho0_q = qutip.Qobj(rho0, dims=[[2] * n_qubits, [2] * n_qubits])
    tlist = np.linspace(0.0, n_steps * DT, n_steps + 1)

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = qutip.mesolve(H, rho0_q, tlist, c_ops=c_ops, options={"atol": 1e-10, "rtol": 1e-8})
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    traj = np.stack([state.full() for state in result.states])
    return traj, (t1 - t0), peak / (1024 * 1024)


# --------------------------------------------------------------------------- #
# Method 5: Qiskit Aer phase-damping (per-gate Kraus)
# --------------------------------------------------------------------------- #

def bench_qiskit_aer(rho0, n_steps):
    if not HAVE_QISKIT:
        return None, float("nan"), 0.0
    dim = rho0.shape[0]
    n_qubits = int(round(math.log2(dim)))
    if n_qubits > 4:
        return None, float("nan"), 0.0
    b = 1.0 - math.exp(-2.0 * GAMMA * DT)
    from qiskit.quantum_info import DensityMatrix, Kraus
    kraus_ops = [
        np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - b)]], dtype=complex),
        np.array([[0.0, 0.0], [0.0, math.sqrt(b)]], dtype=complex),
    ]
    full_kraus_single = []
    for k in range(n_qubits):
        ops = []
        for ki in kraus_ops:
            factors = [np.eye(2, dtype=complex)] * n_qubits
            factors[k] = ki
            full = factors[0]
            for f in factors[1:]:
                full = np.kron(full, f)
            ops.append(full)
        full_kraus_single.append(ops)
    dm = DensityMatrix(rho0)
    err_objs = [Kraus(full_kraus_single[k]) for k in range(n_qubits)]
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        for k in range(n_qubits):
            dm = dm.evolve(err_objs[k])
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    final = np.array(dm.data)
    traj = np.stack([rho0, final])
    return traj, (t1 - t0), peak / (1024 * 1024)


# --------------------------------------------------------------------------- #
# Method 6: General matrix-exp Liouvillian
# --------------------------------------------------------------------------- #

def _liouvillian_dephasing(dim, gamma):
    diag_projector = np.zeros((dim * dim, dim * dim), dtype=np.complex128)
    for k in range(dim):
        e_k = np.zeros(dim); e_k[k] = 1.0
        diag_projector += np.kron(np.outer(e_k, e_k), np.outer(e_k, e_k))
    identity = np.eye(dim * dim, dtype=np.complex128)
    return -gamma * (identity - diag_projector)


def bench_matrix_exp_general(rho0, n_steps):
    dim = rho0.shape[0]
    L = _liouvillian_dephasing(dim, GAMMA)
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    U = scipy.linalg.expm(L * DT)
    vec_rho = rho0.reshape(-1, 1, order="C").astype(np.complex128)
    traj = np.empty((n_steps + 1, dim, dim), dtype=np.complex128)
    traj[0] = rho0
    for t in range(1, n_steps + 1):
        vec_rho = U @ vec_rho
        traj[t] = vec_rho.reshape(dim, dim, order="C")
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return traj, (t1 - t0), peak / (1024 * 1024)


# --------------------------------------------------------------------------- #
# Method 7: Dephasing-specialized matrix-exp
# --------------------------------------------------------------------------- #

def bench_matrix_exp_specialized(rho0, n_steps):
    dim = rho0.shape[0]
    decay = math.exp(-GAMMA * DT)
    diag0 = np.diagonal(rho0).copy()
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    rho = rho0.copy()
    traj = np.empty((n_steps + 1, dim, dim), dtype=np.complex128)
    traj[0] = rho0
    for t in range(1, n_steps + 1):
        rho = decay * rho
        np.fill_diagonal(rho, diag0)
        traj[t] = rho
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return traj, (t1 - t0), peak / (1024 * 1024)


# --------------------------------------------------------------------------- #
# Method 8: Kraus single-qubit baseline
# --------------------------------------------------------------------------- #

def bench_kraus(rho0, n_steps):
    dim = rho0.shape[0]
    if dim != 2:
        return None, float("nan"), 0.0
    p = 1.0 - math.exp(-2.0 * GAMMA * DT)
    K0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - p)]], dtype=complex)
    K1 = np.array([[0.0, 0.0], [0.0, math.sqrt(p)]], dtype=complex)
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    rho = rho0.copy()
    traj = np.empty((n_steps + 1, dim, dim), dtype=np.complex128)
    traj[0] = rho0
    for t in range(1, n_steps + 1):
        rho = K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T
        traj[t] = rho
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return traj, (t1 - t0), peak / (1024 * 1024)


# --------------------------------------------------------------------------- #
# Method 9: scipy.solve_ivp
# --------------------------------------------------------------------------- #

def bench_solve_ivp(rho0, n_steps):
    from scipy.integrate import solve_ivp
    dim = rho0.shape[0]
    L = _liouvillian_dephasing(dim, GAMMA)
    def rhs_real(t, y):
        y_c = y[:dim*dim] + 1j * y[dim*dim:]
        dy = L @ y_c
        return np.concatenate([dy.real, dy.imag])
    y0 = rho0.reshape(-1, order="C").astype(np.complex128)
    y0_real = np.concatenate([y0.real, y0.imag])
    t_span = (0.0, n_steps * DT)
    t_eval = np.linspace(0.0, n_steps * DT, n_steps + 1)
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    sol = solve_ivp(rhs_real, t_span, y0_real, t_eval=t_eval, method="RK45",
                    rtol=1e-8, atol=1e-10)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if not sol.success:
        return None, float("nan"), 0.0
    traj = np.empty((n_steps + 1, dim, dim), dtype=np.complex128)
    for k in range(n_steps + 1):
        y_c = sol.y[:dim*dim, k] + 1j * sol.y[dim*dim:, k]
        traj[k] = y_c.reshape(dim, dim, order="C")
    return traj, (t1 - t0), peak / (1024 * 1024)


# --------------------------------------------------------------------------- #
# Method registry
# --------------------------------------------------------------------------- #

METHODS = [
    ("FPM C++ (OpenMP)",         bench_fpm_cpp_omp,        True),
    ("FPM C++ (serial)",         bench_fpm_cpp_serial,     True),
    ("FPM Python (NumPy)",       bench_fpm_py,             True),
    ("QuTiP mesolve",            bench_qutip,              HAVE_QUTIP),
    ("Qiskit Aer phase-damp",    bench_qiskit_aer,         HAVE_QISKIT),
    ("matrix-exp (general)",     bench_matrix_exp_general, True),
    ("matrix-exp (specialized)", bench_matrix_exp_specialized,True),
    ("Kraus (single-qubit)",     bench_kraus,              True),
    ("scipy.solve_ivp",          bench_solve_ivp,          True),
]


def run_one(method_name, fn, available, rho0, n_steps, gamma, dt, n_qubits):
    # Apply per-method qubit cap
    cap = METHOD_QUBIT_CAPS.get(method_name, 99)
    if n_qubits > cap:
        return {"method": method_name, "available": False,
                "wall_time_s": None, "max_abs_error": None,
                "peak_memory_mb": None, "min_eigenvalue": None, "n_repeats": 0,
                "error": f"skipped (n_qubits={n_qubits} > cap={cap})"}
    if not available:
        return {"method": method_name, "available": False,
                "wall_time_s": None, "max_abs_error": None,
                "peak_memory_mb": None, "min_eigenvalue": None, "n_repeats": 0}
    best_time = float("inf")
    last_traj = None
    last_mem = 0.0
    repeats = METHOD_REPEATS_OVERRIDE.get(method_name, N_REPEATS)
    for _ in range(repeats):
        traj, wtime, mem = fn(rho0, n_steps)
        if traj is None:
            return {"method": method_name, "available": False,
                    "wall_time_s": None, "max_abs_error": None,
                    "peak_memory_mb": None, "min_eigenvalue": None, "n_repeats": 0,
                    "error": "method returned None"}
        if wtime < best_time:
            best_time = wtime
            last_traj = traj
            last_mem = mem
    if last_traj.shape[0] == 2:
        err = max_abs_error_endpoints(last_traj, rho0, gamma, dt, n_steps)
        error_mode = "endpoints_only"
    else:
        err = max_abs_error(last_traj, rho0, gamma, dt)
        error_mode = "full_trajectory"
    min_eig = min_eigenvalue(last_traj[-1])
    return {"method": method_name, "available": True,
            "wall_time_s": best_time, "max_abs_error": err,
            "error_mode": error_mode, "peak_memory_mb": last_mem,
            "min_eigenvalue": min_eig, "n_repeats": repeats}


def main():
    rng = np.random.default_rng(SEED)
    all_results = []
    print("=" * 100)
    print(f" FPM C++ vs Python vs Competitors — Head-to-Head Benchmark v2")
    print(f" gamma={GAMMA}, dt={DT}, n_steps={N_STEPS}, n_repeats={N_REPEATS}")
    print(f" fpm-qsim {fpm_py.__version__} (Python), {fpm_cpp.__version__} (C++)")
    print("=" * 100)
    for n_qubits in N_QUBITS_LIST:
        dim = 2 ** n_qubits
        rho0 = np.ascontiguousarray(haar_random_state(dim, rng), dtype=np.complex128)
        print(f"\n--- {n_qubits} qubits (dim={dim}) ---")
        for method_name, fn, available in METHODS:
            res = run_one(method_name, fn, available, rho0, N_STEPS, GAMMA, DT, n_qubits)
            res["n_qubits"] = n_qubits
            res["dim"] = dim
            all_results.append(res)
            if res["available"]:
                print(f"  {method_name:28s}  time={res['wall_time_s']*1000:9.2f} ms   "
                      f"err={res['max_abs_error']:.3e}   "
                      f"mem={res['peak_memory_mb']:7.2f} MB")
            else:
                print(f"  {method_name:28s}  [unavailable]")
    # Save
    with open(JSON_PATH, "w") as f:
        json.dump({
            "config": {
                "gamma": GAMMA, "dt": DT, "n_steps": N_STEPS,
                "n_repeats": N_REPEATS, "n_qubits_list": N_QUBITS_LIST,
                "seed": SEED,
                "fpm_py_version": fpm_py.__version__,
                "fpm_cpp_version": fpm_cpp.__version__,
                "fpm_cpp_build": fpm_cpp.build_info,
                "qutip_version": qutip.__version__ if HAVE_QUTIP else None,
                "qiskit_version": qiskit.__version__ if HAVE_QISKIT else None,
                "qiskit_aer_version": qiskit_aer.__version__ if HAVE_QISKIT else None,
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__ if hasattr(scipy, "__version__") else None,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nWrote JSON: {JSON_PATH}")
    import csv
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_qubits", "dim", "method", "available",
                    "wall_time_s", "wall_time_ms", "max_abs_error",
                    "peak_memory_mb", "min_eigenvalue"])
        for r in all_results:
            w.writerow([r["n_qubits"], r["dim"], r["method"], r["available"],
                        r["wall_time_s"], (r["wall_time_s"] or 0) * 1000,
                        r["max_abs_error"], r["peak_memory_mb"], r["min_eigenvalue"]])
    print(f"Wrote CSV:  {CSV_PATH}")


if __name__ == "__main__":
    main()
