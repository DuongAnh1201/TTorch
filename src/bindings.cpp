#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(ttorch, m) {
    m.doc() = "TTorch Python bindings";
}
