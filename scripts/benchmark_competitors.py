"""Benchmark fpm_cpp against common quantum/scientific Python baselines.

Run from the repository root after installing optional competitors:

    python -m pip install scipy qutip qiskit qiskit-aer
    python scripts/benchmark_competitors.py

Outputs are written under scripts/benchmark_results/.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import fpm_cpp as fpm


RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"


def _optional_imports() -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for name in ("scipy", "qutip", "qiskit", "qiskit_aer"):
        try:
            modules[name] = __import__(name)
        except Exception as exc:  # pragma: no cover - diagnostic path
            modules[name] = exc
    return modules


def _bench(
    name: str,
    fn: Callable[[], np.ndarray],
    expected: complex,
    repeat: int,
) -> dict[str, float | str]:
    vals: list[float] = []
    out: np.ndarray | None = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        vals.append(time.perf_counter() - t0)
    assert out is not None
    final = out[0, 1]
    err = abs(final - expected)
    print(
        f"{name:28s} median_ms={statistics.median(vals) * 1e3:10.4f} "
        f"final_abs={abs(final):.12e} err={err:.3e}"
    )
    return {
        "name": name,
        "median_ms": statistics.median(vals) * 1e3,
        "min_ms": min(vals) * 1e3,
        "max_ms": max(vals) * 1e3,
        "final_abs_rho01": float(abs(final)),
        "abs_error": float(err),
    }


def _write_trajectory_chart(
    fpm_traj: np.ndarray,
    sqm_traj: np.ndarray,
    expected_traj: np.ndarray,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fpm_vs_sqm_trajectory.png"
    t = np.arange(len(expected_traj))
    fpm_abs = np.abs(fpm_traj[:, 0, 1])
    sqm_abs = np.abs(sqm_traj[:, 0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    axes[0].plot(t, fpm_abs, color="#1f5a8a", lw=1.8, label="FPM C++")
    axes[0].plot(t, expected_traj, color="#333333", ls="--", lw=1.0, label="analytic")
    axes[0].set_title("FPM C++ trajectory")
    axes[0].set_xlabel("tick")
    axes[0].set_ylabel("|rho01|")
    axes[0].set_yscale("log")
    axes[0].legend()

    axes[1].plot(t, sqm_abs, color="#8a4b1f", lw=1.8, label="SQM expm")
    axes[1].plot(t, expected_traj, color="#333333", ls="--", lw=1.0, label="analytic")
    axes[1].set_title("SQM matrix-exponential trajectory")
    axes[1].set_xlabel("tick")
    axes[1].set_yscale("log")
    axes[1].legend()

    fig.suptitle("Pure dephasing trajectory: FPM vs SQM equivalent")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def run_pure_dephasing(modules: dict[str, Any], repeat: int) -> dict[str, Any]:
    from scipy.linalg import expm
    from scipy.integrate import solve_ivp

    gamma = 0.02
    dt = 1.0
    n_steps = 1000
    rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    expected = 0.5 * math.exp(-gamma * dt * n_steps)
    expected_traj = np.array(
        [0.5 * math.exp(-gamma * dt * t) for t in range(n_steps + 1)],
        dtype=float,
    )
    liouvillian = np.diag([0.0, -gamma, -gamma, 0.0])
    u_step = expm(liouvillian * dt)
    u_total = expm(liouvillian * (dt * n_steps))

    def fpm_serial() -> np.ndarray:
        return fpm.simulate(
            rho0,
            gamma=gamma,
            dt=dt,
            n_steps=n_steps,
            method="exact",
            use_omp=False,
        )[-1]

    def fpm_omp() -> np.ndarray:
        return fpm.simulate(
            rho0,
            gamma=gamma,
            dt=dt,
            n_steps=n_steps,
            method="exact",
            use_omp=True,
        )[-1]

    def fpm_trajectory() -> np.ndarray:
        return fpm.simulate(
            rho0,
            gamma=gamma,
            dt=dt,
            n_steps=n_steps,
            method="exact",
            use_omp=False,
        )

    def sqm_trajectory() -> np.ndarray:
        out = np.empty((n_steps + 1, 2, 2), dtype=np.complex128)
        v = rho0.reshape(-1)
        out[0] = rho0
        for t in range(1, n_steps + 1):
            v = u_step @ v
            out[t] = v.reshape(2, 2)
        return out

    def scipy_matrix_exp_step() -> np.ndarray:
        v = rho0.reshape(-1)
        for _ in range(n_steps):
            v = u_step @ v
        return v.reshape(2, 2)

    def scipy_matrix_exp_total() -> np.ndarray:
        return (u_total @ rho0.reshape(-1)).reshape(2, 2)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        return liouvillian @ y

    def scipy_solve_ivp() -> np.ndarray:
        sol = solve_ivp(
            rhs,
            (0.0, dt * n_steps),
            rho0.reshape(-1),
            rtol=1e-9,
            atol=1e-12,
        )
        return sol.y[:, -1].reshape(2, 2)

    print("\nPURE DEPHASING 1Q, gamma=0.02, dt=1, n_steps=1000")
    print("expected |rho01|", f"{expected:.12e}")
    benchmarks: list[dict[str, Any]] = []
    benchmarks.append(_bench("FPM C++ serial", fpm_serial, expected, repeat))
    benchmarks.append(_bench("FPM C++ OpenMP", fpm_omp, expected, repeat))
    benchmarks.append(_bench("SciPy expm step loop", scipy_matrix_exp_step, expected, repeat))
    benchmarks.append(_bench("SciPy expm total", scipy_matrix_exp_total, expected, repeat))
    benchmarks.append(_bench("SciPy solve_ivp", scipy_solve_ivp, expected, repeat))

    qutip = modules.get("qutip")
    if not isinstance(qutip, Exception):
        sz = qutip.sigmaz()
        rho = qutip.Qobj(rho0)
        c_ops = [math.sqrt(gamma / 2.0) * sz]

        def qutip_default() -> np.ndarray:
            result = qutip.mesolve(
                0 * sz,
                rho,
                [0.0, dt * n_steps],
                c_ops=c_ops,
                e_ops=[],
            )
            return result.states[-1].full()

        def qutip_tight_tol() -> np.ndarray:
            result = qutip.mesolve(
                0 * sz,
                rho,
                [0.0, dt * n_steps],
                c_ops=c_ops,
                e_ops=[],
                options={"rtol": 1e-12, "atol": 1e-14, "nsteps": 100000},
            )
            return result.states[-1].full()

        benchmarks.append(_bench("QuTiP mesolve default", qutip_default, expected, repeat))
        benchmarks.append(_bench("QuTiP mesolve tight", qutip_tight_tol, expected, repeat))
    else:
        print("QuTiP unavailable:", qutip)

    fpm_traj = fpm_trajectory()
    sqm_traj = sqm_trajectory()
    chart_path = _write_trajectory_chart(fpm_traj, sqm_traj, expected_traj)
    print("trajectory chart", chart_path)

    return {
        "scenario": {
            "name": "pure_dephasing_1q",
            "gamma": gamma,
            "dt": dt,
            "n_steps": n_steps,
            "expected_final_abs_rho01": expected,
        },
        "benchmarks": benchmarks,
        "trajectory": {
            "chart": str(chart_path),
            "fpm_final_abs_rho01": float(abs(fpm_traj[-1, 0, 1])),
            "sqm_final_abs_rho01": float(abs(sqm_traj[-1, 0, 1])),
            "max_abs_delta_fpm_vs_sqm": float(
                np.max(np.abs(fpm_traj[:, 0, 1] - sqm_traj[:, 0, 1]))
            ),
            "max_abs_delta_fpm_vs_analytic": float(
                np.max(np.abs(np.abs(fpm_traj[:, 0, 1]) - expected_traj))
            ),
        },
    }


def run_qiskit_smoke(modules: dict[str, Any]) -> dict[str, Any]:
    print("\nQISKIT AER")
    qiskit = modules.get("qiskit")
    qiskit_aer = modules.get("qiskit_aer")
    if isinstance(qiskit, Exception) or isinstance(qiskit_aer, Exception):
        print("Qiskit/Aer unavailable:", qiskit, qiskit_aer)
        return {"available": False, "error": f"{qiskit!r} {qiskit_aer!r}"}

    try:
        from qiskit_aer.noise import phase_damping_error

        err = phase_damping_error(1.0 - math.exp(-0.02))
        print("phase_damping_error available yes")
        print("phase_damping_error type", type(err).__name__)
        return {"available": True, "phase_damping_error_type": type(err).__name__}
    except Exception as exc:  # pragma: no cover - diagnostic path
        print("phase_damping_error check failed", repr(exc))
        return {"available": False, "error": repr(exc)}


def run_arena_checks() -> dict[str, Any]:
    print("\nFPM NETWORK ARENA")
    arena_sizes = []
    for n in (1, 10, 1000, 100000):
        t0 = time.perf_counter()
        net = fpm.FpmNetwork(n, max(0, n // 2))
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        arena_sizes.append({
            "n": n,
            "arena_bytes": net.arena_bytes,
            "arena_aligned": bool(net.arena_aligned),
            "construct_ms": elapsed_ms,
        })
        print(
            f"n={n:6d} arena_bytes={net.arena_bytes:10d} "
            f"aligned={net.arena_aligned} construct_ms={elapsed_ms:8.3f}"
        )

    net = fpm.FpmNetwork(2)
    routing = [
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ]
    net.set_routing_tensor(0, routing)
    out = net.viscosity_update_from_routing(0)
    s9 = math.sqrt(sum(x * x for x in routing) / 9.0)
    k1 = abs(routing[0] + routing[4] + routing[8])
    c_n = 1.0 / ((1.0 + k1) ** 0.2 * (1.0 + s9) ** 1.8)

    print("routing S9 observed/expected", f"{out.S9:.12f}", f"{s9:.12f}")
    print("routing K1 observed/expected", f"{out.K1:.12f}", f"{k1:.12f}")
    print("routing C_N observed/expected", f"{out.C_N:.12f}", f"{c_n:.12f}")

    try:
        net.resolve_torsion_link(0, 1)
        print("zombie gate refusal FAILED")
    except RuntimeError as exc:
        print("zombie gate refusal OK", str(exc))

    net.set_energy(0, 0.1, 1.0)
    net.set_energy(1, 0.9, 1.0)
    net.set_mode(0, fpm.MODE_ZOMBIE)
    ledger = net.resolve_torsion_link(0, 1)
    print("resolve_torsion_link executed", ledger.joint_quantization_executed)
    print("pull_exhaust", f"{ledger.pull_exhaust:.12f}")
    print("partner_energy_after", f"{net.energy(1):.12f}")
    print("partner_mode_after", net.mode(1))
    print("total_pull_exhaust", f"{net.total_pull_exhaust:.12f}")
    return {
        "arena_sizes": arena_sizes,
        "routing": {
            "S9_observed": out.S9,
            "S9_expected": s9,
            "K1_observed": out.K1,
            "K1_expected": k1,
            "C_N_observed": out.C_N,
            "C_N_expected": c_n,
        },
        "zombie_gate": {
            "pull_exhaust": ledger.pull_exhaust,
            "partner_energy_after": net.energy(1),
            "partner_mode_after": net.mode(1),
            "total_pull_exhaust": net.total_pull_exhaust,
            "joint_quantization_executed": bool(ledger.joint_quantization_executed),
        },
    }


def write_json(results: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return out_path


def main() -> None:
    repeat = 5
    modules = _optional_imports()

    versions = {
        "fpm_cpp": fpm.__version__,
        "numpy": np.__version__,
    }
    print("VERSIONS")
    print("fpm_cpp", fpm.__version__)
    print("numpy", np.__version__)
    for name, mod in modules.items():
        if isinstance(mod, Exception):
            print(name, "unavailable", repr(mod))
            versions[name] = None
        else:
            version = getattr(mod, "__version__", "unknown")
            versions[name] = version
            print(name, version)

    if isinstance(modules.get("scipy"), Exception):
        raise SystemExit("SciPy is required for the baseline benchmark.")

    results = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": versions,
        "pure_dephasing": run_pure_dephasing(modules, repeat),
        "qiskit_aer": run_qiskit_smoke(modules),
        "arena": run_arena_checks(),
    }
    json_path = write_json(results)
    print("\nARTIFACTS")
    print("json", json_path)
    print("chart", results["pure_dephasing"]["trajectory"]["chart"])


if __name__ == "__main__":
    main()
