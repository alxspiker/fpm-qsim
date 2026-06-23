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
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#if defined(_MSC_VER)
#include <malloc.h>
#endif

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

constexpr size_t FPM_ALIGNMENT = 64;
constexpr double FPM_PI = 3.141592653589793238462643383279502884;
constexpr uint8_t FPM_MODE_FLOW = 0;
constexpr uint8_t FPM_MODE_FATIGUE = 1;
constexpr uint8_t FPM_MODE_ZOMBIE = 2;

// ---------------------------------------------------------------------------
// C++17 aligned allocation + non-owning views
// ---------------------------------------------------------------------------

inline size_t align_up(size_t value, size_t alignment = FPM_ALIGNMENT) {
    return ((value + alignment - 1) / alignment) * alignment;
}

class AlignedBlock {
    void* ptr_ = nullptr;
    size_t bytes_ = 0;

public:
    AlignedBlock() = default;

    explicit AlignedBlock(size_t bytes)
        : bytes_(align_up(bytes)) {
        if (bytes_ == 0) return;
#if defined(_MSC_VER)
        ptr_ = _aligned_malloc(bytes_, FPM_ALIGNMENT);
        if (!ptr_) throw std::bad_alloc();
#else
        if (posix_memalign(&ptr_, FPM_ALIGNMENT, bytes_) != 0)
            throw std::bad_alloc();
#endif
        std::memset(ptr_, 0, bytes_);
    }

    ~AlignedBlock() {
#if defined(_MSC_VER)
        _aligned_free(ptr_);
#else
        std::free(ptr_);
#endif
    }

    AlignedBlock(const AlignedBlock&) = delete;
    AlignedBlock& operator=(const AlignedBlock&) = delete;

    AlignedBlock(AlignedBlock&& other) noexcept
        : ptr_(other.ptr_), bytes_(other.bytes_) {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }

    AlignedBlock& operator=(AlignedBlock&& other) noexcept {
        if (this == &other) return *this;
#if defined(_MSC_VER)
        _aligned_free(ptr_);
#else
        std::free(ptr_);
#endif
        ptr_ = other.ptr_;
        bytes_ = other.bytes_;
        other.ptr_ = nullptr;
        other.bytes_ = 0;
        return *this;
    }

    uint8_t* bytes() { return static_cast<uint8_t*>(ptr_); }
    const uint8_t* bytes() const { return static_cast<const uint8_t*>(ptr_); }
    size_t size_bytes() const { return bytes_; }
    uintptr_t address() const { return reinterpret_cast<uintptr_t>(ptr_); }
    bool aligned(size_t alignment = FPM_ALIGNMENT) const {
        return ptr_ == nullptr || (address() % alignment) == 0;
    }
};

template <typename T>
struct ArrayView {
    T* data_ptr = nullptr;
    size_t len = 0;

    ArrayView() = default;
    ArrayView(T* ptr, size_t length) : data_ptr(ptr), len(length) {}

    inline T& operator[](size_t index) { return data_ptr[index]; }
    inline const T& operator[](size_t index) const { return data_ptr[index]; }

    T* data() { return data_ptr; }
    const T* data() const { return data_ptr; }
    size_t size() const { return len; }
};

template <typename T>
inline ArrayView<T> map_view(uint8_t*& ptr, size_t count) {
    ptr = reinterpret_cast<uint8_t*>(align_up(reinterpret_cast<uintptr_t>(ptr)));
    T* typed = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * count;
    return ArrayView<T>(typed, count);
}

inline size_t fpm_network_arena_bytes(size_t n_daemons, size_t n_pairs) {
    size_t offset = 0;
    auto add = [&](size_t count, size_t elem_size) {
        offset = align_up(offset);
        offset += count * elem_size;
    };
    for (int i = 0; i < 7; ++i) add(n_daemons, sizeof(double));
    for (int i = 0; i < 3; ++i) add(n_daemons * 9, sizeof(double));
    add(n_daemons, sizeof(uint8_t));
    add(n_daemons, sizeof(int32_t));
    add(n_pairs * 2, sizeof(int32_t));
    return align_up(offset);
}

// ---------------------------------------------------------------------------
// SoA network arena. This owns all network physical state.
// ---------------------------------------------------------------------------

struct FpmNetworkArena {
    const size_t n_daemons;
    const size_t n_pairs;

    AlignedBlock memory_block;

    ArrayView<double> E;
    ArrayView<double> E_max;
    ArrayView<double> b;
    ArrayView<double> Omega;
    ArrayView<double> kappa;
    ArrayView<double> tau;
    ArrayView<double> pi;

    ArrayView<double> psi_re;
    ArrayView<double> psi_im;
    ArrayView<double> R;

    ArrayView<uint8_t> mode;
    ArrayView<int32_t> partner;
    ArrayView<int32_t> pairs;

    FpmNetworkArena(size_t daemons, size_t links)
        : n_daemons(daemons),
          n_pairs(links),
          memory_block(fpm_network_arena_bytes(daemons, links)) {
        if (daemons == 0)
            throw std::invalid_argument("n_daemons must be positive");
        uint8_t* ptr = memory_block.bytes();

        E = map_view<double>(ptr, n_daemons);
        E_max = map_view<double>(ptr, n_daemons);
        b = map_view<double>(ptr, n_daemons);
        Omega = map_view<double>(ptr, n_daemons);
        kappa = map_view<double>(ptr, n_daemons);
        tau = map_view<double>(ptr, n_daemons);
        pi = map_view<double>(ptr, n_daemons);

        psi_re = map_view<double>(ptr, n_daemons * 9);
        psi_im = map_view<double>(ptr, n_daemons * 9);
        R = map_view<double>(ptr, n_daemons * 9);

        mode = map_view<uint8_t>(ptr, n_daemons);
        partner = map_view<int32_t>(ptr, n_daemons);
        pairs = map_view<int32_t>(ptr, n_pairs * 2);

        for (size_t i = 0; i < n_daemons; ++i) {
            E_max[i] = 1.0;
            E[i] = 1.0;
            Omega[i] = 0.85;
            kappa[i] = 1.0;
            tau[i] = 0.5;
            pi[i] = 0.5;
            mode[i] = FPM_MODE_FLOW;
            partner[i] = -1;
            for (size_t j = 0; j < 9; ++j) {
                const size_t off = i * 9 + j;
                psi_re[off] = (j == 0) ? 1.0 : 0.0;
                psi_im[off] = 0.0;
                R[off] = (j == 0 || j == 4 || j == 8) ? 1.0 : 0.0;
            }
        }
        for (size_t i = 0; i < n_pairs * 2; ++i)
            pairs[i] = -1;
    }

    size_t size_bytes() const { return memory_block.size_bytes(); }
    bool aligned() const { return memory_block.aligned(); }

    void validate_index(size_t idx) const {
        if (idx >= n_daemons)
            throw std::out_of_range("daemon index out of range");
    }

    double* routing_tensor(size_t idx) {
        validate_index(idx);
        return R.data() + idx * 9;
    }

    const double* routing_tensor(size_t idx) const {
        validate_index(idx);
        return R.data() + idx * 9;
    }
};

class DaemonProxy {
    FpmNetworkArena* arena_ = nullptr;
    size_t idx_ = 0;

public:
    DaemonProxy(FpmNetworkArena& arena, size_t idx)
        : arena_(&arena), idx_(idx) {
        arena_->validate_index(idx_);
    }

    size_t index() const { return idx_; }
    double E() const { return arena_->E[idx_]; }
    double E_max() const { return arena_->E_max[idx_]; }
    uint8_t mode() const { return arena_->mode[idx_]; }
    double* R() { return arena_->routing_tensor(idx_); }
    const double* R() const { return arena_->routing_tensor(idx_); }

    void set_E(double value) { arena_->E[idx_] = value; }
    void deduct_E(double amount) { arena_->E[idx_] = std::max(0.0, arena_->E[idx_] - amount); }
};

// ---------------------------------------------------------------------------
// Private network math kernels operating on raw arena views.
// ---------------------------------------------------------------------------

inline double shear_aggregate(const double* R9) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < 9; ++i)
        sum_sq += R9[i] * R9[i];
    return std::sqrt(sum_sq / 9.0);
}

inline double trace_curvature(const double* R9) {
    return std::abs(R9[0] + R9[4] + R9[8]);
}

inline double mobility(double K1, double S9,
                       double alpha = 0.2,
                       double beta = 1.8,
                       double A = 1.0) {
    if (A < 0.0)
        throw std::invalid_argument("A must be non-negative");
    return A / (std::pow(1.0 + std::max(0.0, K1), alpha)
                * std::pow(1.0 + std::max(0.0, S9), beta));
}

inline std::vector<int64_t> largest_remainder_counts(
    const std::vector<double>& expected_counts,
    int64_t total_count) {
    if (total_count < 0)
        throw std::invalid_argument("total_count must be non-negative");
    std::vector<int64_t> counts(expected_counts.size(), 0);
    std::vector<std::pair<double, size_t>> remainders;
    remainders.reserve(expected_counts.size());

    int64_t floor_sum = 0;
    for (size_t i = 0; i < expected_counts.size(); ++i) {
        if (expected_counts[i] < 0.0)
            throw std::invalid_argument("expected counts must be non-negative");
        const double floored = std::floor(expected_counts[i]);
        counts[i] = static_cast<int64_t>(floored);
        floor_sum += counts[i];
        remainders.push_back({expected_counts[i] - floored, i});
    }

    int64_t remaining = total_count - floor_sum;
    if (remaining < 0)
        throw std::invalid_argument("floored counts exceed total_count");
    std::sort(remainders.begin(), remainders.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second < b.second;
              });
    for (int64_t k = 0; k < remaining && k < static_cast<int64_t>(remainders.size()); ++k)
        counts[remainders[static_cast<size_t>(k)].second] += 1;
    return counts;
}

inline double angular_delta(double a, double b) {
    double d = std::fmod(std::abs(a - b), FPM_PI);
    return std::min(d, FPM_PI - d);
}

inline double bell_local_torsion_correlation(double a, double b) {
    return 1.0 - 4.0 * angular_delta(a, b) / FPM_PI;
}

inline double geometric_torsion_correlation(double a, double b) {
    return -std::cos(2.0 * (a - b));
}

struct JointTorsionResult {
    std::vector<double> probabilities;
    std::vector<int64_t> counts;
    std::vector<double> quantized_probabilities;
    double fpm_correlation = 0.0;
    double tv_distance = 0.0;
};

inline JointTorsionResult joint_torsion_lrm_distribution(
    double a, double b, int64_t n_bit_eq = 1000000) {
    const double E_target = geometric_torsion_correlation(a, b);
    std::vector<double> p_joint = {
        (1.0 + E_target) / 4.0,
        (1.0 - E_target) / 4.0,
        (1.0 - E_target) / 4.0,
        (1.0 + E_target) / 4.0,
    };
    std::vector<double> expected;
    expected.reserve(p_joint.size());
    for (double p : p_joint)
        expected.push_back(p * static_cast<double>(n_bit_eq));
    std::vector<int64_t> counts = largest_remainder_counts(expected, n_bit_eq);

    std::vector<double> q;
    q.reserve(counts.size());
    for (int64_t c : counts)
        q.push_back(static_cast<double>(c) / static_cast<double>(n_bit_eq));

    JointTorsionResult out;
    out.probabilities = std::move(p_joint);
    out.counts = std::move(counts);
    out.quantized_probabilities = std::move(q);
    out.fpm_correlation = (
        out.quantized_probabilities[0] - out.quantized_probabilities[1]
        - out.quantized_probabilities[2] + out.quantized_probabilities[3]);
    double tv = 0.0;
    for (size_t i = 0; i < out.probabilities.size(); ++i)
        tv += std::abs(out.quantized_probabilities[i] - out.probabilities[i]);
    out.tv_distance = 0.5 * tv;
    return out;
}

struct ViscosityOutput {
    double Omega = 0.0;
    double kappa = 0.0;
    double C_N = 0.0;
    double S9 = 0.0;
    double K1 = 0.0;
};

struct LedgerOutput {
    double pull_exhaust = 0.0;
    double tv_distance = 0.0;
    bool joint_quantization_executed = false;
};

class FpmNetwork {
    FpmNetworkArena arena_;
    double total_pull_exhaust_ = 0.0;
    double total_route_spend_ = 0.0;
    int64_t n_bit_eq_ = 1000000;

public:
    FpmNetwork(size_t n_daemons, size_t n_pairs = 0)
        : arena_(n_daemons, n_pairs) {}

    FpmNetworkArena& arena() { return arena_; }
    const FpmNetworkArena& arena() const { return arena_; }

    size_t n_daemons() const { return arena_.n_daemons; }
    size_t n_pairs() const { return arena_.n_pairs; }
    size_t arena_bytes() const { return arena_.size_bytes(); }
    bool arena_aligned() const { return arena_.aligned(); }

    double energy(size_t idx) const {
        arena_.validate_index(idx);
        return arena_.E[idx];
    }

    uint8_t mode(size_t idx) const {
        arena_.validate_index(idx);
        return arena_.mode[idx];
    }

    void set_energy(size_t idx, double E, double E_max) {
        arena_.validate_index(idx);
        if (E_max <= 0.0)
            throw std::invalid_argument("E_max must be positive");
        if (E < 0.0 || E > E_max)
            throw std::invalid_argument("E must be in [0, E_max]");
        arena_.E[idx] = E;
        arena_.E_max[idx] = E_max;
    }

    void set_mode(size_t idx, uint8_t mode) {
        arena_.validate_index(idx);
        if (mode > FPM_MODE_ZOMBIE)
            throw std::invalid_argument("mode must be FLOW=0, FATIGUE=1, or ZOMBIE=2");
        arena_.mode[idx] = mode;
    }

    std::vector<double> routing_tensor(size_t idx) const {
        const double* R9 = arena_.routing_tensor(idx);
        return std::vector<double>(R9, R9 + 9);
    }

    void set_routing_tensor(size_t idx, const std::vector<double>& R9) {
        if (R9.size() != 9)
            throw std::invalid_argument("routing tensor must contain 9 values");
        double* dst = arena_.routing_tensor(idx);
        std::copy(R9.begin(), R9.end(), dst);
    }

    ViscosityOutput viscosity_update_from_routing(
        size_t idx,
        double B_load = 0.0,
        double alpha = 0.2,
        double beta = 1.8) {
        arena_.validate_index(idx);
        const double* R9 = arena_.routing_tensor(idx);
        ViscosityOutput out;
        out.S9 = shear_aggregate(R9);
        out.K1 = trace_curvature(R9);
        out.C_N = mobility(out.K1, out.S9, alpha, beta);
        const double depletion = std::pow(1.0 + std::max(0.0, B_load), -0.75);
        out.kappa = std::min(1.0, std::max(0.0, out.C_N * depletion));
        out.Omega = std::min(0.85, std::max(0.50, 1.0 - out.kappa));
        arena_.kappa[idx] = out.kappa;
        arena_.Omega[idx] = out.Omega;
        total_route_spend_ += 1e-5 * arena_.E_max[idx];
        arena_.E[idx] = std::max(ENERGY_FLOOR_FRACTION * arena_.E_max[idx],
                                 arena_.E[idx] - 1e-5 * arena_.E_max[idx]);
        return out;
    }

    void force_zombie_starvation(size_t idx) {
        arena_.validate_index(idx);
        const double threshold = 0.20 * arena_.E_max[idx];
        if (arena_.E[idx] > threshold) {
            total_pull_exhaust_ += arena_.E[idx] - threshold;
            arena_.E[idx] = threshold;
        }
        arena_.mode[idx] = FPM_MODE_ZOMBIE;
    }

    LedgerOutput resolve_torsion_link(size_t idx_a, size_t idx_b) {
        arena_.validate_index(idx_a);
        arena_.validate_index(idx_b);
        if (idx_a == idx_b)
            throw std::invalid_argument("torsion link requires two distinct daemons");
        if (arena_.mode[idx_a] != FPM_MODE_ZOMBIE)
            throw std::runtime_error("daemon A is not in ZOMBIE mode; starvation required");

        LedgerOutput out;
        const double zombie_threshold = 0.20 * arena_.E_max[idx_b];
        if (arena_.E[idx_b] > zombie_threshold) {
            out.pull_exhaust = arena_.E[idx_b] - zombie_threshold;
            total_pull_exhaust_ += out.pull_exhaust;
            arena_.E[idx_b] = zombie_threshold;
            arena_.mode[idx_b] = FPM_MODE_ZOMBIE;
        }

        const auto q = joint_torsion_lrm_distribution(0.0, FPM_PI / 8.0, n_bit_eq_);
        out.tv_distance = q.tv_distance;
        out.joint_quantization_executed = true;
        return out;
    }

    double total_pull_exhaust() const { return total_pull_exhaust_; }
    double total_route_spend() const { return total_route_spend_; }
};

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
