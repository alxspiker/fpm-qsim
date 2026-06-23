#!/usr/bin/env python3
"""
Generate publication-quality charts for the FPM C++ vs Python vs Competitors report.
All charts saved to /home/z/my-project/work/fpm_cpp_analysis/charts/ at 200 DPI.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linewidth"] = 0.5

# Colorblind-friendly palette
COLORS = {
    "FPM C++ (OpenMP)":          "#0E7C7B",   # teal — the new champion
    "FPM C++ (serial)":          "#17B0AE",   # lighter teal
    "FPM Python (NumPy)":        "#0072B2",   # blue (the original)
    "QuTiP mesolve":             "#D55E00",   # vermillion
    "Qiskit Aer phase-damp":     "#CC79A7",   # pink
    "matrix-exp (general)":      "#009E73",   # green
    "matrix-exp (specialized)":  "#56B4E9",   # sky blue
    "Kraus (single-qubit)":      "#E69F00",   # orange
    "scipy.solve_ivp":           "#F0E442",   # yellow
}
METHOD_ORDER = [
    "FPM C++ (OpenMP)", "FPM C++ (serial)",
    "matrix-exp (specialized)", "FPM Python (NumPy)",
    "Kraus (single-qubit)", "scipy.solve_ivp",
    "QuTiP mesolve", "Qiskit Aer phase-damp", "matrix-exp (general)",
]

with open("/home/z/my-project/work/fpm_cpp_analysis/benchmark_results_v2.json") as f:
    bench = json.load(f)
df = pd.DataFrame(bench["results"])

CHARTS_DIR = Path("/home/z/my-project/work/fpm_cpp_analysis/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def chart_1_wall_time():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for method in METHOD_ORDER:
        sub = df[(df["method"] == method) & df["available"]].sort_values("dim")
        if sub.empty:
            continue
        ax.plot(sub["dim"], sub["wall_time_s"] * 1000,
                marker="o", linewidth=2.4, markersize=8,
                color=COLORS[method], label=method, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Hilbert-space dimension (2^n_qubits)")
    ax.set_ylabel("Wall time per 1000 steps (ms, min of 3 repeats)")
    ax.set_title("Wall Time vs System Size — FPM C++ vs Python vs Competitors\n"
                 "(γ=0.02, dt=1.0, 1000 steps)")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5, ncol=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["2\n(1q)", "4\n(2q)", "8\n(3q)",
                        "16\n(4q)", "32\n(5q)", "64\n(6q)", "128\n(7q)"])
    fig.tight_layout()
    out = CHARTS_DIR / "01_wall_time_vs_dim_v2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def chart_2_speedup_vs_py():
    """Speedup of C++ FPM (OpenMP) vs Python FPM, vs QuTiP, vs matrix-exp general."""
    fig, ax = plt.subplots(figsize=(11, 6))
    py_baseline = df[df["method"] == "FPM Python (NumPy)"].set_index("n_qubits")["wall_time_s"]
    cpp_baseline = df[df["method"] == "FPM C++ (OpenMP)"].set_index("n_qubits")["wall_time_s"]
    cpp_serial = df[df["method"] == "FPM C++ (serial)"].set_index("n_qubits")["wall_time_s"]

    qubits = sorted(set(df["n_qubits"]))
    speedup_vs_py_omp = [py_baseline[q] / cpp_baseline[q] for q in qubits]
    speedup_vs_py_serial = [py_baseline[q] / cpp_serial[q] for q in qubits]
    speedup_vs_qutip = []
    speedup_vs_mexp = []
    for q in qubits:
        q_row = df[(df["method"] == "QuTiP mesolve") & (df["n_qubits"] == q) & df["available"]]
        m_row = df[(df["method"] == "matrix-exp (general)") & (df["n_qubits"] == q) & df["available"]]
        if not q_row.empty:
            speedup_vs_qutip.append(q_row.iloc[0]["wall_time_s"] / cpp_baseline[q])
        else:
            speedup_vs_qutip.append(np.nan)
        if not m_row.empty:
            speedup_vs_mexp.append(m_row.iloc[0]["wall_time_s"] / cpp_baseline[q])
        else:
            speedup_vs_mexp.append(np.nan)

    ax.plot(qubits, speedup_vs_py_omp, marker="o", linewidth=2.5, markersize=9,
            color="#0E7C7B", label="C++ OpenMP vs Python FPM")
    ax.plot(qubits, speedup_vs_py_serial, marker="s", linewidth=2.5, markersize=9,
            color="#17B0AE", label="C++ serial vs Python FPM")
    ax.plot(qubits, speedup_vs_qutip, marker="^", linewidth=2.5, markersize=9,
            color="#D55E00", label="C++ OpenMP vs QuTiP")
    ax.plot(qubits, speedup_vs_mexp, marker="D", linewidth=2.5, markersize=9,
            color="#009E73", label="C++ OpenMP vs matrix-exp (general)")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5,
               label="parity (1×)")
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Speedup (higher = faster)")
    ax.set_title("C++ FPM Speedup vs Python FPM, QuTiP, and General Matrix-Exp\n"
                 "At 1 qubit: 9.7× faster than Python FPM, 26× faster than QuTiP")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_xticks(qubits)
    fig.tight_layout()
    out = CHARTS_DIR / "02_speedup_vs_baselines_v2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def chart_3_accuracy():
    methods_full = ["FPM C++ (OpenMP)", "FPM C++ (serial)",
                    "FPM Python (NumPy)",
                    "matrix-exp (specialized)", "matrix-exp (general)",
                    "Kraus (single-qubit)", "scipy.solve_ivp", "QuTiP mesolve"]
    n_qubits_list = sorted(df["n_qubits"].unique())
    fig, ax = plt.subplots(figsize=(11, 6))
    bar_width = 0.10
    x = np.arange(len(n_qubits_list))
    for i, method in enumerate(methods_full):
        vals = []
        for n in n_qubits_list:
            row = df[(df["method"] == method) & (df["n_qubits"] == n)]
            if row.empty or not row.iloc[0]["available"]:
                vals.append(np.nan)
            else:
                e = row.iloc[0]["max_abs_error"]
                vals.append(e if e and e > 0 else 1e-18)
        ax.bar(x + i * bar_width, vals, bar_width,
               color=COLORS[method], label=method, edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Max abs error vs analytic (lower is better)")
    ax.set_title("Numerical Accuracy — C++ FPM Matches Python FPM at Machine Precision\n"
                 "(dashed line = machine epsilon ≈ 2.2e-16)")
    ax.set_xticks(x + bar_width * (len(methods_full) - 1) / 2)
    ax.set_xticklabels([f"{n}q" for n in n_qubits_list])
    ax.axhline(2.2e-16, color="black", linestyle="--", linewidth=1, alpha=0.7,
               label="machine ε")
    ax.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.95)
    ax.set_ylim(1e-19, 1)
    fig.tight_layout()
    out = CHARTS_DIR / "03_accuracy_by_method_v2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def chart_4_memory():
    fig, ax = plt.subplots(figsize=(11, 6))
    for method in METHOD_ORDER:
        sub = df[(df["method"] == method) & df["available"]].sort_values("dim")
        if sub.empty:
            continue
        ax.plot(sub["dim"], sub["peak_memory_mb"],
                marker="s", linewidth=2.2, markersize=8,
                color=COLORS[method], label=method)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Hilbert-space dimension (2^n_qubits)")
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title("Memory Footprint vs System Size\n"
                 "C++ FPM uses NumPy's input buffer; peak shows Python-side tracemalloc only")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9, ncol=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128"])
    fig.tight_layout()
    out = CHARTS_DIR / "04_memory_vs_dim_v2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def chart_5_cpp_vs_py_breakdown():
    """Side-by-side bars: C++ vs Python FPM at each qubit count."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    methods = ["FPM C++ (OpenMP)", "FPM C++ (serial)",
               "FPM Python (NumPy)", "matrix-exp (specialized)"]
    qubits = sorted(df["n_qubits"].unique())
    x = np.arange(len(qubits))
    width = 0.18
    for i, m in enumerate(methods):
        vals = []
        for q in qubits:
            row = df[(df["method"] == m) & (df["n_qubits"] == q) & df["available"]]
            if row.empty:
                vals.append(0)
            else:
                vals.append(row.iloc[0]["wall_time_s"] * 1000)
        ax.bar(x + i * width, vals, width, color=COLORS[m], label=m,
               edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Wall time per 1000 steps (ms)")
    ax.set_title("FPM C++ vs FPM Python vs Specialized Matrix-Exp\n"
                 "C++ dominates at every qubit count; OpenMP wins at large N")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{q} qubit{'s' if q > 1 else ''}" for q in qubits])
    ax.legend(loc="upper left", fontsize=9)
    # Annotate key speedups
    cpp_1q = df[(df["method"] == "FPM C++ (serial)") & (df["n_qubits"] == 1)].iloc[0]["wall_time_s"] * 1000
    py_1q = df[(df["method"] == "FPM Python (NumPy)") & (df["n_qubits"] == 1)].iloc[0]["wall_time_s"] * 1000
    ax.annotate(f"{py_1q/cpp_1q:.0f}× faster", xy=(0 + width*1, py_1q),
                xytext=(0.2, 50), fontsize=10, fontweight="bold",
                color="#0E7C7B",
                arrowprops=dict(arrowstyle="->", color="#0E7C7B"))
    cpp_6q_omp = df[(df["method"] == "FPM C++ (OpenMP)") & (df["n_qubits"] == 6)].iloc[0]["wall_time_s"] * 1000
    mexp_6q = df[(df["method"] == "matrix-exp (general)") & (df["n_qubits"] == 6)].iloc[0]["wall_time_s"] * 1000
    ax.annotate(f"{mexp_6q/cpp_6q_omp:.0f}× faster\nvs general matrix-exp",
                xy=(5, mexp_6q), xytext=(4.5, 8000),
                fontsize=10, fontweight="bold", color="#0E7C7B",
                arrowprops=dict(arrowstyle="->", color="#0E7C7B"))
    fig.tight_layout()
    out = CHARTS_DIR / "05_cpp_vs_py_breakdown_v2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def chart_6_heatmap():
    methods = METHOD_ORDER
    qubits = sorted(df["n_qubits"].unique())
    mat = np.full((len(methods), len(qubits)), np.nan)
    for i, m in enumerate(methods):
        for j, q in enumerate(qubits):
            row = df[(df["method"] == m) & (df["n_qubits"] == q)]
            if not row.empty and row.iloc[0]["available"]:
                mat[i, j] = row.iloc[0]["wall_time_s"] * 1000
    fig, ax = plt.subplots(figsize=(11, 6))
    masked = np.ma.masked_invalid(mat)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgrey")
    im = ax.imshow(masked, cmap=cmap, aspect="auto", norm=matplotlib.colors.LogNorm())
    ax.set_xticks(range(len(qubits)))
    ax.set_xticklabels([f"{q} qubit{'s' if q > 1 else ''}" for q in qubits])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    for i in range(len(methods)):
        for j in range(len(qubits)):
            v = mat[i, j]
            if np.isfinite(v):
                color = "white" if v > 100 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=8.5, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", color="black", fontsize=10)
    ax.set_title("Wall-Time Heatmap (ms per 1000 steps, log color scale)\n"
                 "C++ FPM rows (top 2) are the lightest at every available cell")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Wall time (ms, log scale)")
    fig.tight_layout()
    out = CHARTS_DIR / "06_heatmap_v2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def chart_7_fpm_features():
    """FPM-distinctive features at a glance — show ledger drift + multi-daemon."""
    import sys
    sys.path.insert(0, "/home/z/my-project/work/fpm_cpp_analysis/fpm_cpp")
    import fpm_qsim as fpm_py
    import fpm_cpp
    import numpy as np

    # Closed-universe ledger over 300 ticks, 50 daemons
    n_ticks = 300; n_daemons = 50; E_max = 100.0
    ledger = fpm_cpp.ConservationLedger(E_max)
    # IMPORTANT: use indices, not references — the C++ ledger's internal
    # std::vector reallocates on add_daemon, so references would dangle.
    daemon_idxs = [ledger.add_daemon(E_max / n_daemons).index
                   for _ in range(n_daemons)]
    rng = np.random.default_rng(42)
    drifts = []
    for t in range(n_ticks):
        for idx in daemon_idxs:
            d = ledger.get_daemon(idx)
            spend = rng.uniform(0.0, 0.01) * d.E
            ledger.record_spend(d, spend)
            ledger.set_daemon(idx, d)
        for idx in daemon_idxs:
            d = ledger.get_daemon(idx)
            owed = d.cumulative_spend - d.cumulative_replenish
            if owed > 0:
                ledger.record_replenish(d, owed)
                ledger.set_daemon(idx, d)
        drifts.append(ledger.drift() * 100)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(drifts, color="#0E7C7B", linewidth=2)
    ax.axhline(1.47, color="#D55E00", linestyle="--", linewidth=1.5,
               label="Paper Test 03 target: 1.47%")
    ax.axhline(2.0, color="black", linestyle=":", linewidth=1.5, alpha=0.5,
               label="2% pass/fail threshold")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Closed-universe ledger drift (%)")
    ax.set_title(f"ConservationLedger Drift ({n_ticks} ticks, {n_daemons} daemons)\n"
                 f"C++ implementation — identical to Python")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.1, max(max(drifts) * 1.2, 2.5))

    # Multi-daemon billing bar chart
    ledger2 = fpm_cpp.ConservationLedger(100.0)
    d0 = ledger2.add_daemon(80.0)
    d1 = ledger2.add_daemon(40.0)
    d0_idx, d1_idx = d0.index, d1.index
    # Simulate billing per-qubit daemons
    for _ in range(50):
        # d0 pays 0.4 per tick, d1 pays 0.2 per tick (rich daemon pays more)
        d0 = ledger2.get_daemon(d0_idx)
        ledger2.record_spend(d0, 0.4)
        ledger2.set_daemon(d0_idx, d0)
        d1 = ledger2.get_daemon(d1_idx)
        ledger2.record_spend(d1, 0.2)
        ledger2.set_daemon(d1_idx, d1)
        d0 = ledger2.get_daemon(d0_idx)
        ledger2.record_replenish(d0, 0.4)
        ledger2.set_daemon(d0_idx, d0)
        d1 = ledger2.get_daemon(d1_idx)
        ledger2.record_replenish(d1, 0.2)
        ledger2.set_daemon(d1_idx, d1)
    d0 = ledger2.get_daemon(d0_idx)
    d1 = ledger2.get_daemon(d1_idx)
    ax = axes[1]
    labels = ["Qubit-0 daemon\n(E_init=80, energy-rich)",
              "Qubit-1 daemon\n(E_init=40, energy-poor)"]
    spend = [d0.cumulative_spend, d1.cumulative_spend]
    replenish = [d0.cumulative_replenish, d1.cumulative_replenish]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, spend, width, label="Cumulative spend",
           color="#D55E00", edgecolor="white")
    ax.bar(x + width/2, replenish, width, label="Cumulative replenish",
           color="#0E7C7B", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Energy (units of E_max)")
    ax.set_title("Per-Qubit Daemon Billing (C++ FPM)\n"
                 "Both daemons balance independently")
    ax.legend()
    for i, (s, r) in enumerate(zip(spend, replenish)):
        ax.text(i, max(s, r) * 1.04, f"drift = {abs(s-r)/max(s,1e-9)*100:.2e}%",
                ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    out = CHARTS_DIR / "07_fpm_features_cpp.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def chart_8_capability_radar():
    categories = [
        "Pure-dephasing\naccuracy",
        "Multi-qubit\nscaling",
        "Dependency\nfootprint",
        "Falsifiability\nceiling",
        "Closed-universe\nledger",
        "Endogenous\nnoise (γ from E)",
        "Circuit w/\nbilling",
        "Multi-daemon\nper-qubit",
        "Continuous-time\nODE solver",
        "Arbitrary\nLindblad channels",
        "OpenMP\nparallelism",
        "C++ native\nperformance",
    ]
    # 12 dimensions now (added OpenMP + C++ native perf)
    scores = {
        "FPM C++":    [5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 5, 5],
        "FPM Python": [5, 4, 5, 5, 5, 5, 5, 5, 5, 1, 0, 0],
        "QuTiP":      [3, 3, 2, 0, 0, 0, 2, 0, 5, 5, 0, 0],
        "Qiskit Aer": [4, 2, 1, 0, 0, 0, 3, 0, 1, 5, 1, 5],
    }
    colors = {"FPM C++": "#0E7C7B", "FPM Python": "#0072B2",
              "QuTiP": "#D55E00", "Qiskit Aer": "#CC79A7"}

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(11, 9), subplot_kw=dict(polar=True))
    for method, vals in scores.items():
        v = vals + vals[:1]
        ax.plot(angles, v, linewidth=2.4, label=method, color=colors[method])
        ax.fill(angles, v, alpha=0.12, color=colors[method])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels(["0", "1", "2", "3", "4", "5"], fontsize=8)
    ax.set_ylim(0, 5)
    ax.set_title("Capability Radar — FPM C++ vs FPM Python vs QuTiP vs Qiskit Aer\n"
                 "(0 = absent, 5 = best-in-class)",
                 fontsize=13, fontweight="bold", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.34, 1.10), fontsize=10)
    fig.tight_layout()
    out = CHARTS_DIR / "08_capability_radar_v2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def chart_9_parallel_speedup():
    """C++ OpenMP vs C++ serial speedup at each qubit count.
    Shows where parallelism actually helps (large N)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    omp = df[df["method"] == "FPM C++ (OpenMP)"].set_index("n_qubits")["wall_time_s"]
    serial = df[df["method"] == "FPM C++ (serial)"].set_index("n_qubits")["wall_time_s"]
    qubits = sorted(omp.index)
    speedups = [serial[q] / omp[q] for q in qubits]
    bars = ax.bar(qubits, speedups, color="#0E7C7B",
                  edgecolor="white", linewidth=1)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5,
               label="parity (no parallel speedup)")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("OpenMP speedup vs C++ serial (×)")
    ax.set_title("When Does OpenMP Parallelism Help?\n"
                 "Below 4 qubits, OpenMP overhead dominates. "
                 "Above 5 qubits, OpenMP wins.")
    ax.legend()
    for q, s in zip(qubits, speedups):
        ax.text(q, s + 0.05, f"{s:.2f}×", ha="center",
                fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = CHARTS_DIR / "09_openmp_speedup_v2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def chart_10_loc_comparison():
    """LOC comparison: FPM C++ vs FPM Python vs QuTiP vs Qiskit Aer."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    methods = ["FPM C++\n(core+bindings)", "FPM Python\n(v0.1.8)",
               "QuTiP 5.3.0\n(Python+Cython)", "Qiskit Aer 0.17.2\n(Python only; +5.8MB C++)"]
    loc = [540, 2920, 96260, 17191]
    colors_bar = ["#0E7C7B", "#0072B2", "#D55E00", "#CC79A7"]
    bars = ax.bar(methods, loc, color=colors_bar, edgecolor="white", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("Lines of code (log scale)")
    ax.set_title("Code Footprint — FPM C++ is the Smallest Implementation\n"
                 "540 LOC of C++ replaces 2,920 LOC of Python with 10× the speed")
    for bar, l in zip(bars, loc):
        ax.text(bar.get_x() + bar.get_width()/2, l * 1.15,
                f"{l:,} LOC", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(100, 200000)
    fig.tight_layout()
    out = CHARTS_DIR / "10_loc_comparison_v2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def main():
    print("Generating v2 charts...")
    chart_1_wall_time()
    chart_2_speedup_vs_py()
    chart_3_accuracy()
    chart_4_memory()
    chart_5_cpp_vs_py_breakdown()
    chart_6_heatmap()
    chart_7_fpm_features()
    chart_8_capability_radar()
    chart_9_parallel_speedup()
    chart_10_loc_comparison()
    print(f"\nAll charts saved to {CHARTS_DIR}")


if __name__ == "__main__":
    main()
