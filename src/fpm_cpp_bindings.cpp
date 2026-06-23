// fpm_cpp_bindings.cpp — pybind11 Python bindings for the C++ FPM library
// =======================================================================
//
// Build:
//   g++ -O3 -fopenmp -march=native -ffast-math -shared -std=c++17 \
//       -fPIC $(python -m pybind11 --includes) \
//       fpm_cpp_bindings.cpp -o fpm_cpp$(python3-config --extension-suffix) \
//       -fopenmp
//
// Exposes:
//   fpm_cpp.simulate_trajectory(rho0, gamma, dt, n_steps, method="exact",
//                               use_omp=True) -> np.ndarray  (n_steps+1, N, N)
//   fpm_cpp.lindblad_step(rho, gamma, dt, method="exact", use_omp=True)
//   fpm_cpp.kappa_exact, kappa_from_gamma, gamma_from_kappa
//   fpm_cpp.GAMMA_MAX, FALSIFICATION_THRESHOLD, ENERGY_FLOOR_FRACTION
//   fpm_cpp.bounded_gamma(gamma) -> float (raises FalsificationError on >32)
//   fpm_cpp.DaemonState, ConservationLedger

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "fpm_core.hpp"

namespace py = pybind11;
using namespace fpm_cpp;

#ifndef VERSION_INFO
#define VERSION_INFO "0.2.0"
#endif

// --------------------------------------------------------------------------
// NumPy <-> std::vector<Complex> converters
// --------------------------------------------------------------------------

// View a NumPy complex128 array as a flat std::vector<Complex> (zero-copy).
static std::vector<Complex> numpy_to_vec_complex(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> arr)
{
    auto buf = arr.request();
    if (buf.itemsize != sizeof(std::complex<double>))
        throw std::runtime_error("expected complex128 array");
    Complex* ptr = reinterpret_cast<Complex*>(buf.ptr);
    return std::vector<Complex>(ptr, ptr + buf.size);
}

// Wrap a std::vector<Complex> as a NumPy complex128 array.
// The vector is heap-allocated and owned by a py::capsule so NumPy can
// free it when the array is garbage-collected.
static py::array_t<std::complex<double>> vec_complex_to_numpy(
    std::vector<Complex> vec, std::vector<py::ssize_t> shape)
{
    // Compute total size from shape to verify.
    py::ssize_t total = 1;
    for (auto s : shape) total *= s;
    if (total != static_cast<py::ssize_t>(vec.size()))
        throw std::runtime_error("shape/size mismatch in vec_complex_to_numpy");

    auto* data = new std::vector<Complex>(std::move(vec));
    auto capsule = py::capsule(data, [](void* p) {
        delete reinterpret_cast<std::vector<Complex>*>(p);
    });
    // Build strides (row-major, element is sizeof(complex<double>) = 16 bytes).
    std::vector<py::ssize_t> strides(shape.size());
    strides[shape.size() - 1] = static_cast<py::ssize_t>(sizeof(std::complex<double>));
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * shape[i + 1];
    return py::array_t<std::complex<double>>(
        shape, strides,
        reinterpret_cast<std::complex<double>*>(data->data()),
        capsule
    );
}

// --------------------------------------------------------------------------
// Python-facing lindblad_step (one step, returns new array)
// --------------------------------------------------------------------------

static py::array_t<std::complex<double>> py_lindblad_step(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> rho_arr,
    double gamma, double dt,
    const std::string& method = "exact",
    bool use_omp = true)
{
    auto buf = rho_arr.request();
    if (buf.ndim != 2 || buf.shape[0] != buf.shape[1])
        throw std::invalid_argument("rho must be a square 2-D array");
    int N = static_cast<int>(buf.shape[0]);
    std::vector<Complex> rho = numpy_to_vec_complex(rho_arr);

    FpmMethod m = parse_method(method);
    double kappa = compute_kappa(gamma, dt, m);

    std::vector<Complex> next = use_omp
        ? lindblad_step_omp(rho, N, kappa)
        : lindblad_step_serial(rho, N, kappa);

    return vec_complex_to_numpy(std::move(next), {N, N});
}

// --------------------------------------------------------------------------
// Python-facing simulate_trajectory
// --------------------------------------------------------------------------

static py::array_t<std::complex<double>> py_simulate_trajectory(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> rho0_arr,
    double gamma, double dt, int n_steps,
    const std::string& method = "exact",
    bool use_omp = true)
{
    auto buf = rho0_arr.request();
    if (buf.ndim != 2 || buf.shape[0] != buf.shape[1])
        throw std::invalid_argument("rho0 must be a square 2-D array");
    int N = static_cast<int>(buf.shape[0]);
    std::vector<Complex> rho0 = numpy_to_vec_complex(rho0_arr);

    FpmMethod m = parse_method(method);
    std::vector<Complex> traj = simulate_trajectory(
        rho0, N, gamma, dt, n_steps, m, use_omp);

    return vec_complex_to_numpy(std::move(traj),
                                {n_steps + 1, N, N});
}

// --------------------------------------------------------------------------
// Python-facing DaemonState
// --------------------------------------------------------------------------

static py::class_<DaemonState> make_daemon_state(py::module_& m) {
    return py::class_<DaemonState>(m, "DaemonState")
        .def(py::init<>())
        .def_readwrite("index",               &DaemonState::index)
        .def_readwrite("E_max",               &DaemonState::E_max)
        .def_readwrite("E",                   &DaemonState::E)
        .def_readwrite("load",                &DaemonState::load)
        .def_readwrite("cumulative_spend",     &DaemonState::cumulative_spend)
        .def_readwrite("cumulative_replenish", &DaemonState::cumulative_replenish)
        .def_readwrite("cumulative_landauer",  &DaemonState::cumulative_landauer)
        .def_property_readonly("energy_fraction", &DaemonState::energy_fraction);
}

// --------------------------------------------------------------------------
// Python-facing ConservationLedger
// --------------------------------------------------------------------------

static py::class_<ConservationLedger> make_ledger(py::module_& m) {
    py::class_<ConservationLedger> ledger(m, "ConservationLedger");
    ledger.def(py::init<double>(), py::arg("E_max_total"))
        .def_readwrite("E_max_total", &ConservationLedger::E_max_total)
        .def_readwrite("daemons",     &ConservationLedger::daemons)
        // add_daemon returns a *copy* of the new DaemonState (not a reference)
        // to avoid dangling pointers when the internal std::vector reallocates.
        // The returned DaemonState has the correct index, which callers can use
        // to index back into ledger.daemons[i] for later updates.
        .def("add_daemon", [](ConservationLedger& l, double E_init) {
            // Take a copy AFTER add_daemon returns (the internal vector may
            // have reallocated, so we read from l.daemons.back()).
            l.add_daemon(E_init);
            return l.daemons.back();  // returns a copy
        }, py::arg("E_init"))
        // Index-based accessors avoid dangling references entirely.
        .def("get_daemon", [](ConservationLedger& l, int idx) -> DaemonState {
            return l.daemons.at(idx);
        }, py::arg("idx"))
        .def("set_daemon", [](ConservationLedger& l, int idx, const DaemonState& d) {
            l.daemons.at(idx) = d;
        }, py::arg("idx"), py::arg("daemon"))
        .def("record_spend",      &ConservationLedger::record_spend)
        .def("record_replenish",  &ConservationLedger::record_replenish)
        .def("record_landauer",   &ConservationLedger::record_landauer)
        .def("bill_compute_cost", &ConservationLedger::bill_compute_cost,
             py::arg("daemon"), py::arg("n_multiplies"), py::arg("n_adds"),
             py::arg("cost_per_op"))
        .def_property_readonly("total_spend",     &ConservationLedger::total_spend)
        .def_property_readonly("total_replenish", &ConservationLedger::total_replenish)
        .def_property_readonly("total_landauer",  &ConservationLedger::total_landauer)
        .def("drift", &ConservationLedger::drift);
    return ledger;
}

// --------------------------------------------------------------------------
// Python-facing SoA FpmNetwork
// --------------------------------------------------------------------------

static py::class_<FpmNetwork> make_fpm_network(py::module_& m) {
    py::class_<ViscosityOutput>(m, "ViscosityOutput")
        .def_readonly("Omega", &ViscosityOutput::Omega)
        .def_readonly("kappa", &ViscosityOutput::kappa)
        .def_readonly("C_N", &ViscosityOutput::C_N)
        .def_readonly("S9", &ViscosityOutput::S9)
        .def_readonly("K1", &ViscosityOutput::K1);

    py::class_<LedgerOutput>(m, "LedgerOutput")
        .def_readonly("pull_exhaust", &LedgerOutput::pull_exhaust)
        .def_readonly("tv_distance", &LedgerOutput::tv_distance)
        .def_readonly("joint_quantization_executed",
                      &LedgerOutput::joint_quantization_executed);

    return py::class_<FpmNetwork>(m, "FpmNetwork")
        .def(py::init<size_t, size_t>(),
             py::arg("n_daemons"), py::arg("n_pairs") = 0)
        .def_property_readonly("n_daemons", &FpmNetwork::n_daemons)
        .def_property_readonly("n_pairs", &FpmNetwork::n_pairs)
        .def_property_readonly("arena_bytes", &FpmNetwork::arena_bytes)
        .def_property_readonly("arena_aligned", &FpmNetwork::arena_aligned)
        .def("energy", &FpmNetwork::energy, py::arg("idx"))
        .def("mode", &FpmNetwork::mode, py::arg("idx"))
        .def("set_energy", &FpmNetwork::set_energy,
             py::arg("idx"), py::arg("E"), py::arg("E_max"))
        .def("set_mode", &FpmNetwork::set_mode,
             py::arg("idx"), py::arg("mode"))
        .def("routing_tensor", &FpmNetwork::routing_tensor, py::arg("idx"))
        .def("set_routing_tensor", &FpmNetwork::set_routing_tensor,
             py::arg("idx"), py::arg("R9"))
        .def("viscosity_update_from_routing",
             &FpmNetwork::viscosity_update_from_routing,
             py::arg("idx"), py::arg("B_load") = 0.0,
             py::arg("alpha") = 0.2, py::arg("beta") = 1.8)
        .def("force_zombie_starvation", &FpmNetwork::force_zombie_starvation,
             py::arg("idx"))
        .def("resolve_torsion_link", &FpmNetwork::resolve_torsion_link,
             py::arg("idx_a"), py::arg("idx_b"))
        .def_property_readonly("total_pull_exhaust",
                               &FpmNetwork::total_pull_exhaust)
        .def_property_readonly("total_route_spend",
                               &FpmNetwork::total_route_spend);
}

// --------------------------------------------------------------------------
// bounded_gamma (with FalsificationError bridge to Python)
// --------------------------------------------------------------------------

static double py_bounded_gamma(double gamma_raw,
                               double gamma_max = GAMMA_MAX) {
    return bounded_gamma_scalar(gamma_raw, gamma_max);
}

// --------------------------------------------------------------------------
// Module entry point
// --------------------------------------------------------------------------

PYBIND11_MODULE(fpm_cpp, m) {
    m.doc() = "C++ accelerated FPM (Finite Possibility Mechanics) primitives.";
    py::register_exception<FalsificationError>(m, "FalsificationError");

    // Constants
    m.attr("GAMMA_MAX")              = GAMMA_MAX;
    m.attr("FALSIFICATION_THRESHOLD") = FALSIFICATION_THRESHOLD;
    m.attr("ENERGY_FLOOR_FRACTION")  = ENERGY_FLOOR_FRACTION;
    m.attr("ISOTROPIC_WEIGHT_LIMIT") = ISOTROPIC_WEIGHT_LIMIT;
    m.attr("MODE_FLOW")              = FPM_MODE_FLOW;
    m.attr("MODE_FATIGUE")           = FPM_MODE_FATIGUE;
    m.attr("MODE_ZOMBIE")            = FPM_MODE_ZOMBIE;

    // Core functions
    m.def("kappa_from_gamma", &kappa_from_gamma,
          py::arg("gamma"), py::arg("dt") = 1.0);
    m.def("kappa_exact",     &kappa_exact,
          py::arg("gamma"), py::arg("dt") = 1.0);
    m.def("gamma_from_kappa", &gamma_from_kappa,
          py::arg("kappa"), py::arg("dt") = 1.0);
    m.def("bounded_gamma", &py_bounded_gamma,
          py::arg("gamma_raw"), py::arg("gamma_max") = GAMMA_MAX);

    // gamma_from_energy (takes a DaemonState)
    m.def("gamma_from_energy",
          [](const DaemonState& d, double gate_power,
             py::object load, double dt, double C_N, bool bounded) {
              double load_val = load.is_none() ? -1.0 : load.cast<double>();
              return gamma_from_energy(d, gate_power, load_val, dt, C_N, bounded);
          },
          py::arg("daemon"), py::arg("gate_power"),
          py::arg("load") = py::none(), py::arg("dt") = 1.0,
          py::arg("C_N") = 1.0, py::arg("bounded") = true);

    // Lindblad step + trajectory (the hot path)
    m.def("lindblad_step", &py_lindblad_step,
          py::arg("rho"), py::arg("gamma"), py::arg("dt") = 1.0,
          py::arg("method") = "exact", py::arg("use_omp") = true);
    m.def("simulate", &py_simulate_trajectory,
          py::arg("rho0"), py::arg("gamma"), py::arg("dt") = 1.0,
          py::arg("n_steps") = 1, py::arg("method") = "exact",
          py::arg("use_omp") = true);

    // DaemonState + Ledger
    make_daemon_state(m);
    make_ledger(m);
    make_fpm_network(m);

    // exp_route_cost + bill_exp_route_cost
    m.def("exp_route_cost", [](int taylor_order) {
        auto [n_mul, n_add] = exp_route_cost(taylor_order);
        return std::make_pair(n_mul, n_add);
    }, py::arg("taylor_order") = 8);
    m.def("bill_exp_route_cost", &bill_exp_route_cost,
          py::arg("ledger"), py::arg("daemon"),
          py::arg("taylor_order") = 8, py::arg("cost_per_op") = 1e-5);

    // Build info
    m.attr("__version__") = VERSION_INFO;
    m.attr("build_info")  = "C++17 + pybind11 3.x";
}
