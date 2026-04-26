#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"
#include "autograd.h"

namespace py = pybind11;

PYBIND11_MODULE(ttorch, m) {
    m.doc() = "TTorch — lightweight tensor and autograd library";

    py::class_<Tensor>(m, "Tensor")

        // ── Constructors ─────────────────────────────────────
        .def(py::init<const std::vector<int>&>(),
             py::arg("dims"))

        // ── Factory methods ──────────────────────────────────
        .def_static("zeros",
            [](std::vector<int> dims) {
                return Tensor::zeros(dims);
            }, py::arg("dims"))

        .def_static("ones",
            [](std::vector<int> dims) {
                return Tensor::ones(dims);
            }, py::arg("dims"))

        .def_static("custom",
            [](std::vector<int> dims, double val) {
                return Tensor::custom(dims, val);
            }, py::arg("dims"), py::arg("val"))

        .def_static("form",
            [](std::vector<int> dims, std::vector<double> data) {
                return Tensor::form(dims, data);
            }, py::arg("dims"), py::arg("data"))

        // ── Core fields ──────────────────────────────────────
        .def_readwrite("data",          &Tensor::data)
        .def_readwrite("shape",         &Tensor::shape)
        .def_readwrite("requires_grad", &Tensor::requires_grad)
        .def_readwrite("is_leaf",       &Tensor::is_leaf)

        // grad is a Tensor* — expose as optional Tensor
        .def_property("grad",
            [](Tensor& self) -> py::object {
                if (self.grad == nullptr) return py::none();
                return py::cast(*self.grad);
            },
            [](Tensor& self, py::object val) {
                if (val.is_none()) {
                    self.grad = nullptr;
                } else {
                    self.grad = new Tensor(val.cast<Tensor>());
                }
            })

        // ── Utilities ────────────────────────────────────────
        .def("print",   &Tensor::print)
        .def("dims",    &Tensor::dims)
        .def("at",      &Tensor::at,      py::arg("index"))
        .def("value",   &Tensor::value,   py::arg("data"))
        .def("ndim",    &Tensor::ndim)
        .def("size",    &Tensor::size,    py::arg("dim"))

        // ── Shape ops ────────────────────────────────────────
        .def("reshape", &Tensor::reshape, py::arg("newshape"))
        .def("view",    &Tensor::view,    py::arg("newshape"))
        .def("flatten", &Tensor::flatten)

        // ── Math ops ─────────────────────────────────────────
        .def("add",        &Tensor::add,        py::arg("n"))
        .def("add_int",    &Tensor::add_int,    py::arg("i"))
        .def("scale_int",  &Tensor::scale_int,  py::arg("i"))
        .def("multiply",   &Tensor::multiply,   py::arg("b"))
        .def("dot",        &Tensor::dot,        py::arg("b"))
        .def("transpose",  &Tensor::transpose)
        .def("T",          &Tensor::T)
        .def("sum",        &Tensor::sum,        py::arg("axis") = 0)
        .def("mean",       &Tensor::mean,       py::arg("axis") = 0)

        // ── Autograd ─────────────────────────────────────────
        .def("backward",   &Tensor::backward)
        .def("zero_grad",  &Tensor::zero_grad)

        // ── Python repr ──────────────────────────────────────
        .def("__repr__", [](Tensor& self) {
            std::string s = "Tensor(shape=[";
            for (int i = 0; i < (int)self.shape.size(); i++) {
                s += std::to_string(self.shape[i]);
                if (i < (int)self.shape.size() - 1) s += ", ";
            }
            s += "], requires_grad=" + std::string(self.requires_grad ? "True" : "False") + ")";
            return s;
        });

    // ── Free functions ────────────────────────────────────────
    m.def("relu",    &relu,    py::arg("x"));
    m.def("sigmoid", &sigmoid, py::arg("x"));
}
