// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak
//
// Thin pybind11 surface for HypercubeLCN. Ergonomics (shape checks, fit,
// pickle, docs) live in hypercube_lcn/__init__.py.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "../Core.h"
#include "../Training.h"

namespace py = pybind11;

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {

py::array_t<float> vector_to_array(const std::vector<float>& v)
{
    py::array_t<float> arr(v.size());
    if (!v.empty())
        std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
    return arr;
}

} // namespace

// Single de-templated binding. Hypercube dim is a runtime constructor
// argument, so one C++ type and one Python class serve every dimension.
PYBIND11_MODULE(_core, m)
{
    m.doc() = "HypercubeLCN: a locally connected network on a Boolean "
              "hypercube, every weight trained";
#ifndef HYPERCUBE_LCN_VERSION
#  error "HYPERCUBE_LCN_VERSION must be set by CMake from hypercube_lcn/_version.py"
#endif
    m.attr("__version__") = HYPERCUBE_LCN_VERSION;

    py::class_<Core>(m, "_Core")
        .def(py::init([](size_t dim, uint64_t seed, size_t z_max,
                         size_t gather_span, bool tanh_last) {
            CoreConfig cfg;
            cfg.dim         = dim;
            cfg.seed        = seed;
            cfg.z_max       = z_max;
            cfg.gather_span = gather_span;
            cfg.tanh_last   = tanh_last;
            return Core::Create(cfg);
        }),
            py::arg("dim"),
            py::arg("seed")        = 934791766227647176ULL,
            py::arg("z_max")       = 0ULL,
            py::arg("gather_span") = 2ULL,
            py::arg("tanh_last")   = false)

        .def("forward", [](Core& self, FloatArray x) {
            auto buf = x.request();
            if (static_cast<size_t>(buf.size) != self.N())
                throw std::invalid_argument(
                    "field size (" + std::to_string(buf.size)
                    + ") must equal N (" + std::to_string(self.N()) + ")");
            py::gil_scoped_release release;
            self.Forward(static_cast<const float*>(buf.ptr));
        }, py::arg("x"),
           "Run the forward pass on one length-N field. Read the result "
           "with output().")

        .def("output", [](const Core& self) {
            return vector_to_array(self.Output());
        }, "Output field (length N) of the most recent forward().")

        .def("weights", [](const Core& self) {
            return vector_to_array(self.Weights());
        }, "Copy of all trainable weights: depth, axis, tap, vertex.")

        .def("load_weights", [](Core& self, FloatArray w) {
            auto buf = w.request();
            self.LoadWeights(static_cast<const float*>(buf.ptr),
                             static_cast<size_t>(buf.size));
        }, py::arg("weights"),
           "Replace all weights; length must equal len(weights()).")

        .def_property_readonly("dim", &Core::Dim)
        .def_property_readonly("N", &Core::N)
        .def_property_readonly("seed", &Core::Seed)
        .def_property_readonly("z_max", &Core::ZMax)
        .def_property_readonly("gather_span", &Core::GatherSpan)
        .def_property_readonly("tanh_last", &Core::TanhLast)
        .def_property_readonly("num_weights", [](const Core& self) {
            return self.Weights().size();
        });

    // keep_alive<1, 2>: the Training instance pins its Core — the C++ side
    // holds a bare reference, so the Core must outlive the Training.
    py::class_<Training>(m, "_Training")
        .def(py::init([](Core& core, float lr, float lr_min_frac,
                         int lr_decay_epochs, bool restore_best,
                         float beta1, float beta2, float eps) {
            TrainingConfig cfg;
            cfg.lr              = lr;
            cfg.lr_min_frac     = lr_min_frac;
            cfg.lr_decay_epochs = lr_decay_epochs;
            cfg.restore_best    = restore_best;
            cfg.beta1           = beta1;
            cfg.beta2           = beta2;
            cfg.eps             = eps;
            return std::make_unique<Training>(core, cfg);
        }),
            py::arg("core"),
            py::arg("lr")              = 5e-3f,
            py::arg("lr_min_frac")     = 1.0f,
            py::arg("lr_decay_epochs") = 0,
            py::arg("restore_best")    = false,
            py::arg("beta1")           = 0.9f,
            py::arg("beta2")           = 0.999f,
            py::arg("eps")             = 1e-8f,
            py::keep_alive<1, 2>())

        .def("loss", [](Training& self, FloatArray target) {
            auto buf = target.request();
            py::gil_scoped_release release;
            return self.Loss(std::span<const float>{
                static_cast<const float*>(buf.ptr),
                static_cast<size_t>(buf.size)});
        }, py::arg("target"),
           "0.5 * SSE against the most recent forward(), and the gradient "
           "seed for backward(). len(target) == N is full-width; shorter is "
           "masked to vertices 0..len-1 (free hidden vertices).")

        .def("backward", [](Training& self) {
            py::gil_scoped_release release;
            self.Backward();
        }, "Backprop the most recent loss(), accumulating into the gradient. "
           "Consumes the loss seed: each loss() feeds exactly one backward().")

        .def("zero_grad", &Training::ZeroGrad,
             "Clear the accumulated gradient. Call at the start of each batch.")

        .def("adam", [](Training& self) {
            py::gil_scoped_release release;
            self.Adam();
        }, "One Adam step on the accumulated gradient.")

        .def("set_epoch", &Training::SetEpoch,
             py::arg("epoch"), py::arg("num_epochs") = 0,
             "Apply the cosine schedule for this epoch. Call once at the "
             "start of the epoch.")

        .def("observe", [](Training& self, float metric, int epoch) {
            // A new best copies the full weight vector — release the GIL
            // like the other weight-sized operations.
            py::gil_scoped_release release;
            self.Observe(metric, epoch);
        }, py::arg("metric"), py::arg("epoch"),
           "Snapshot weights on a new low metric (restore_best only; "
           "lower wins).")

        .def("restore_best", &Training::RestoreBest,
             "Write the best-metric snapshot back into the core (no-op if "
             "none).")

        .def("reset", &Training::Reset,
             "Forget Adam moments, step count, lr, and the best snapshot. "
             "Core weights are untouched.")

        .def("grad", [](const Training& self) {
            return vector_to_array(self.Grad());
        }, "Copy of the accumulated gradient, same layout as weights().")

        .def("accumulate_grad", [](Training& self, FloatArray g) {
            auto buf = g.request();
            py::gil_scoped_release release;
            self.AccumulateGrad(std::span<const float>{
                static_cast<const float*>(buf.ptr),
                static_cast<size_t>(buf.size)});
        }, py::arg("grad"),
           "Add an externally accumulated gradient (e.g. a worker's grad()) "
           "into this instance's gradient, elementwise. The data-parallel "
           "merge; length must equal len(grad()).")

        .def_property_readonly("lr", &Training::Lr)
        .def_property_readonly("has_best", &Training::HasBest)
        .def_property_readonly("best_epoch", &Training::BestEpoch)
        .def_property_readonly("best_metric", &Training::BestMetric)
        .def_property_readonly("restore_best_enabled", [](const Training& self) {
            return self.Config().restore_best;
        });
}
