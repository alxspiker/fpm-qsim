// fpm_core.hpp — Core FPM primitives in C++ (header-only)
// =========================================================
// C++ port of fpm_qsim.core.  Provides:
//   - Physical constants (GAMMA_MAX, FALSIFICATION_THRESHOLD, etc.)
//   - kappa_from_gamma, kappa_exact, gamma_from_kappa
//   - gamma_from_energy (endogenous noise)
//   - bounded_gamma + FalsificationError
//   - fpm_affine_step, fpm_affine_trajectory
//
// All routines operate on Eigen-free raw std::vector<std::complex<double>>
// for maximum portability and zero external dependencies beyond the C++ STL.
// The Python binding layer (fpm_cpp_bindings.cpp) converts NumPy <-> std::vector.

#pragma once

#include <complex>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstdint>
#include <limits>

#if defined(_MSC_VER)
#define FPM_RESTRICT __restrict
#else
#define FPM_RESTRICT __restrict__
#endif

namespace fpm_cpp {

// ---------------------------------------------------------------------------
// Physical constants (must match Python fpm_qsim.core exactly)
// ---------------------------------------------------------------------------

constexpr double GAMMA_MAX              = 31.873862947240752;
constexpr double FALSIFICATION_THRESHOLD = 32.0;
constexpr double ENERGY_FLOOR_FRACTION  = 0.03138766217547228;
constexpr double ISOTROPIC_WEIGHT_LIMIT = 1.0 / 3.0;

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// FalsificationError
// ---------------------------------------------------------------------------

class FalsificationError : public std::runtime_error {
public:
    explicit FalsificationError(const std::string& msg)
        : std::runtime_error(msg) {}
};

// ---------------------------------------------------------------------------
// Affine-map coefficient builders
// ---------------------------------------------------------------------------

inline double kappa_from_gamma(double gamma, double dt = 1.0) {
    double product = gamma * dt;
    if (product < 0.0 || product > 1.0) {
        throw std::invalid_argument(
            "gamma * dt = " + std::to_string(product) +
            " is outside [0, 1]; the affine map would be non-contractive.");
    }
    return 1.0 - product;
}

inline double kappa_exact(double gamma, double dt = 1.0) {
    if (gamma < 0.0) throw std::invalid_argument("gamma must be non-negative");
    if (dt < 0.0)    throw std::invalid_argument("dt must be non-negative");
    return std::exp(-gamma * dt);
}

inline double gamma_from_kappa(double kappa, double dt = 1.0) {
    if (!(kappa >= 0.0 && kappa <= 1.0))
        throw std::invalid_argument("kappa outside [0, 1]");
    if (dt <= 0.0) throw std::invalid_argument("dt must be positive");
    return (1.0 - kappa) / dt;
}

// ---------------------------------------------------------------------------
// bounded_gamma + FalsificationError
// ---------------------------------------------------------------------------

inline double bounded_gamma_scalar(double gamma_raw,
                                   double gamma_max = GAMMA_MAX) {
    if (gamma_raw > FALSIFICATION_THRESHOLD) {
        throw FalsificationError(
            "gamma = " + std::to_string(gamma_raw) +
            " exceeds the FPM falsification threshold " +
            std::to_string(FALSIFICATION_THRESHOLD) +
            ". This observation would falsify the framework.");
    }
    return std::min(gamma_raw, gamma_max);
}

// ---------------------------------------------------------------------------
// DaemonState (C++ mirror of fpm_qsim.conservation.DaemonState)
// ---------------------------------------------------------------------------

struct DaemonState {
    int    index               = 0;
    double E_max               = 0.0;
    double E                   = 0.0;
    Complex coherence          = {0.0, 0.0};
    double load                = 0.0;
    double cumulative_spend     = 0.0;
    double cumulative_replenish = 0.0;
    double cumulative_landauer  = 0.0;

    double energy_fraction() const {
        return (E_max > 0.0) ? (E / E_max) : 0.0;
    }
};

// ---------------------------------------------------------------------------
// gamma_from_energy (endogenous noise from daemon energy)
// ---------------------------------------------------------------------------

inline double gamma_from_energy(const DaemonState& daemon,
                                double gate_power,
                                double load = -1.0,    // -1 = use daemon.load
                                double dt = 1.0,
                                double C_N = 1.0,
                                bool bounded = true) {
    if (gate_power < 0.0)
        throw std::invalid_argument("gate_power must be non-negative");
    if (dt <= 0.0)
        throw std::invalid_argument("dt must be positive");
    if (!(C_N >= 0.0 && C_N <= 1.0))
        throw std::invalid_argument("C_N must be in [0, 1]");

    double eff_load = (load < 0.0) ? daemon.load : load;
    if (eff_load < 0.0)
        throw std::invalid_argument("load must be non-negative");

    double energy_fraction = daemon.energy_fraction();
    if (energy_fraction < 0.0)
        throw std::invalid_argument("energy_fraction must be non-negative");

    double eff_energy = std::max({energy_fraction,
                                  ENERGY_FLOOR_FRACTION,
                                  std::numeric_limits<double>::epsilon()});
    double eff_load_total = eff_load + gate_power / eff_energy;
    double kappa = C_N * std::pow(1.0 + eff_load_total, -0.75);
    kappa = std::min(1.0, std::max(0.0, kappa));
    double gamma = gamma_from_kappa(kappa, dt);
    return bounded ? bounded_gamma_scalar(gamma) : gamma;
}

// ---------------------------------------------------------------------------
// ConservationLedger (C++ mirror)
// ---------------------------------------------------------------------------

class ConservationLedger {
public:
    double E_max_total;
    std::vector<DaemonState> daemons;

    explicit ConservationLedger(double E_max_total_)
        : E_max_total(E_max_total_) {}

    DaemonState& add_daemon(double E_init) {
        if (E_init < 0.0 || E_init > E_max_total)
            throw std::invalid_argument(
                "E_init out of range [0, E_max_total]");
        DaemonState d;
        d.index = static_cast<int>(daemons.size());
        d.E_max = E_max_total;
        d.E     = E_init;
        daemons.push_back(d);
        return daemons.back();
    }

    void record_spend(DaemonState& d, double amount) {
        if (amount < 0.0) throw std::invalid_argument("spend amount must be >= 0");
        amount = std::min(amount, d.E);
        d.E -= amount;
        d.cumulative_spend += amount;
    }

    void record_replenish(DaemonState& d, double amount) {
        if (amount < 0.0) throw std::invalid_argument("replenish amount must be >= 0");
        amount = std::min(amount, d.E_max - d.E);
        d.E += amount;
        d.cumulative_replenish += amount;
    }

    void record_landauer(DaemonState& d, double bits_erased) {
        if (bits_erased < 0.0)
            throw std::invalid_argument("bits_erased must be >= 0");
        int N_bit_eq = std::max(1, static_cast<int>(daemons.size()));
        double debit = (bits_erased / N_bit_eq) * d.E_max;
        double floor = ENERGY_FLOOR_FRACTION * d.E_max;
        double new_E = std::max(floor, d.E - debit);
        double actual_debit = d.E - new_E;
        d.E = new_E;
        d.cumulative_landauer += actual_debit;
    }

    double bill_compute_cost(DaemonState& d,
                             int n_multiplies, int n_adds,
                             double cost_per_op) {
        if (n_multiplies < 0 || n_adds < 0)
            throw std::invalid_argument("op counts must be >= 0");
        int n_ops = n_multiplies + n_adds;
        if (n_ops == 0) return 0.0;
        if (cost_per_op < 0.0)
            throw std::invalid_argument("cost_per_op must be >= 0");
        double debit = n_ops * cost_per_op * d.E_max;
        double floor = ENERGY_FLOOR_FRACTION * d.E_max;
        double new_E = std::max(floor, d.E - debit);
        double actual_debit = d.E - new_E;
        d.E = new_E;
        d.cumulative_spend += actual_debit;
        return actual_debit;
    }

    double total_spend()      const { double s=0; for (auto& d: daemons) s+=d.cumulative_spend;      return s; }
    double total_replenish()  const { double s=0; for (auto& d: daemons) s+=d.cumulative_replenish;  return s; }
    double total_landauer()   const { double s=0; for (auto& d: daemons) s+=d.cumulative_landauer;   return s; }

    double drift() const {
        double rhs  = total_spend() + total_landauer();
        double denom = std::max(total_replenish(), 1e-12);
        return std::abs(total_replenish() - rhs) / denom;
    }
};

// ---------------------------------------------------------------------------
// exp_route_cost + bill_exp_route_cost (oracle-construction cost helpers)
// ---------------------------------------------------------------------------

inline std::pair<int,int> exp_route_cost(int taylor_order = 8) {
    if (taylor_order < 0)
        throw std::invalid_argument("taylor_order must be >= 0");
    int K = taylor_order;
    return {2 * K, K};  // (n_multiplies, n_adds)
}

inline double bill_exp_route_cost(ConservationLedger& ledger,
                                  DaemonState& daemon,
                                  int taylor_order = 8,
                                  double cost_per_op = 1e-5) {
    auto [n_mul, n_add] = exp_route_cost(taylor_order);
    return ledger.bill_compute_cost(daemon, n_mul, n_add, cost_per_op);
}

// ---------------------------------------------------------------------------
// The drop-in dephasing step (the hot path)
// ---------------------------------------------------------------------------
// Operates on a flat row-major N×N std::vector<Complex> buffer.
// Two specializations:
//   - lindblad_step_serial:  single-threaded
//   - lindblad_step_omp:     OpenMP-parallel off-diagonal contraction
//
// Both produce BIT-IDENTICAL output to the Python fpm_qsim.lindblad_step
// because the contraction is elementwise scalar multiplication: out[i,j] =
// kappa * rho[i,j] for i != j, and out[i,i] = rho[i,i] (diagonal fixed point).
//
// In OpenMP mode the parallel region uses static scheduling + N×N
// contiguous-memory access, so SIMD auto-vectorization is preserved.

enum class FpmMethod { Exact, Euler };

inline FpmMethod parse_method(const std::string& m) {
    if (m == "exact") return FpmMethod::Exact;
    if (m == "euler") return FpmMethod::Euler;
    throw std::invalid_argument("method must be 'exact' or 'euler'; got '" + m + "'");
}

// Compute kappa from gamma + dt + method.
inline double compute_kappa(double gamma, double dt, FpmMethod method) {
    if (gamma < 0.0) throw std::invalid_argument("gamma must be non-negative");
    if (dt    < 0.0) throw std::invalid_argument("dt must be non-negative");
    switch (method) {
        case FpmMethod::Exact:
            return std::exp(-gamma * dt);
        case FpmMethod::Euler: {
            double product = gamma * dt;
            if (product < 0.0 || product > 1.0)
                throw std::invalid_argument(
                    "method='euler' requires 0 <= gamma*dt <= 1; got " +
                    std::to_string(product));
            return 1.0 - product;
        }
    }
    throw std::invalid_argument("unreachable");
}

// Serial dephasing step.  `rho` is row-major N×N, modified in place is
// optional; returns a new vector by default.
inline std::vector<Complex> lindblad_step_serial(
    const std::vector<Complex>& rho, int N,
    double kappa)
{
    std::vector<Complex> out(rho);
    // Off-diagonal contraction: out[i,j] = kappa * rho[i,j] for i != j.
    // Diagonal untouched (it is the fixed point of the affine map).
    for (int i = 0; i < N; ++i) {
        const int row = i * N;
        for (int j = 0; j < N; ++j) {
            if (i != j) out[row + j] = kappa * rho[row + j];
        }
    }
    return out;
}

// OpenMP-parallel dephasing step.  Identical numerical output, parallel
// off-diagonal contraction.  Uses static scheduling to preserve SIMD.
inline std::vector<Complex> lindblad_step_omp(
    const std::vector<Complex>& rho, int N,
    double kappa)
{
    std::vector<Complex> out(rho);
    const Complex* FPM_RESTRICT src = rho.data();
    Complex*       FPM_RESTRICT dst = out.data();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const int row = i * N;
        // Inner loop is auto-vectorized: each iteration is an independent
        // scalar complex multiply (2 doubles).  The diagonal check costs
        // one branch per element but is predictable.
        for (int j = 0; j < N; ++j) {
            if (i != j) dst[row + j] = kappa * src[row + j];
            // else dst[row+j] already equals src[row+j] (copy)
        }
    }
    return out;
}

// Convenience: roll out a full trajectory.
// traj[t] is the state at time t (t in [0, n_steps]).
// traj has shape (n_steps+1) * N * N, row-major.
//
// Two-pass strategy to avoid dangling pointers to local vectors:
//   Pass 1: roll out the trajectory in a flat buffer, reading from the
//           previous slice of the same buffer (no temporary vector needed).
//   The dephasing step is applied in-place to slice t to produce slice t+1,
//   which is just `dst[i,j] = kappa * src[i,j]` for i != j and
//   `dst[i,j] = src[i,j]` for i == j.
inline std::vector<Complex> simulate_trajectory(
    const std::vector<Complex>& rho0, int N,
    double gamma, double dt, int n_steps,
    FpmMethod method, bool use_omp = true)
{
    double kappa = compute_kappa(gamma, dt, method);
    const int slice = N * N;
    std::vector<Complex> traj((n_steps + 1) * slice);

    // Copy rho0 into traj[0].
    std::copy(rho0.begin(), rho0.end(), traj.begin());

    // Roll out: read from traj[(t-1)*slice], write to traj[t*slice].
    // Both pointers are into the same std::vector — safe.
    for (int t = 1; t <= n_steps; ++t) {
        const Complex* FPM_RESTRICT src = traj.data() + (t - 1) * slice;
        Complex*       FPM_RESTRICT dst = traj.data() + t * slice;
        if (use_omp) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                const int row = i * N;
                for (int j = 0; j < N; ++j) {
                    dst[row + j] = (i == j) ? src[row + j] : kappa * src[row + j];
                }
            }
        } else {
            for (int i = 0; i < N; ++i) {
                const int row = i * N;
                for (int j = 0; j < N; ++j) {
                    dst[row + j] = (i == j) ? src[row + j] : kappa * src[row + j];
                }
            }
        }
    }
    return traj;
}

// ---------------------------------------------------------------------------
// Affine-map primitives (for the API surface; not the hot path)
// ---------------------------------------------------------------------------

inline std::vector<Complex> fpm_affine_step(const std::vector<Complex>& c,
                                            double kappa,
                                            Complex nu = {0.0, 0.0})
{
    std::vector<Complex> out(c.size());
    for (size_t i = 0; i < c.size(); ++i)
        out[i] = kappa * c[i] + nu;
    return out;
}

}  // namespace fpm_cpp
