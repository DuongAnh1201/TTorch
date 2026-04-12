# my_cpp_lib

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![C++](https://img.shields.io/badge/C++-17-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build](https://img.shields.io/badge/build-CMake-orange.svg)

`my_cpp_lib` is a **lightweight tensor and automatic differentiation library written in C++ with Python bindings**.

The project explores the internal architecture of modern deep learning frameworks by implementing core components such as:

* tensor operations
* automatic differentiation (autograd)
* Python bindings
* modular compute kernels

The long-term goal is to build a **minimal deep learning backend similar to PyTorch**, while keeping the codebase simple and educational.

---

# Key Features

* ⚡ **High-performance C++ core**
* 🧠 **Automatic differentiation engine**
* 🐍 **Python bindings for easy use**
* 🧱 **Modular tensor operation system**
* 🛠 **CMake-based build system**
* 📦 **Python package distribution via `pyproject.toml`**

---

# Project Structure

```id="struct01"
my_cpp_lib/
│
├── pyproject.toml
├── CMakeLists.txt
├── README.md
├── LICENSE
│
├── src/
│   ├── tensor.cpp
│   ├── autograd.cpp
│   └── bindings.cpp
│
├── include/
│   └── tensor.h
│
├── python/
│   └── my_cpp_lib/
│       └── __init__.py
│
└── tests/
```

### Directory Overview

| Directory        | Purpose                    |
| ---------------- | -------------------------- |
| `src/`           | C++ source implementation  |
| `include/`       | Public C++ headers         |
| `python/`        | Python interface package   |
| `tests/`         | Unit tests                 |
| `pyproject.toml` | Python build configuration |
| `CMakeLists.txt` | C++ build configuration    |

---

# Installation

## Install from Source

Clone the repository:

```id="install01"
git clone https://github.com/yourusername/my_cpp_lib.git
cd my_cpp_lib
```

Install build tools:

```id="install02"
pip install build
```

Build the Python package:

```id="install03"
python -m build
```

Install the generated wheel:

```id="install04"
pip install dist/*.whl
```

---

# Quick Example

```cpp
#include "tensor.h"

// Create tensors
Tensor a = Tensor::form({2, 3}, {1, 2, 3, 4, 5, 6});
Tensor b = Tensor::ones({2, 3});

// Element-wise operations
Tensor c = a.add(b);        // element-wise addition
Tensor d = a.multiply(b);   // element-wise multiplication
Tensor e = a.add_int(10.0); // broadcast scalar addition
Tensor f = a.scale_int(2.0); // broadcast scalar multiply

// Matrix operations
Tensor t  = a.transpose();  // or a.T()
Tensor ab = a.dot(a.T());   // matrix multiplication → (2,2)

// Reductions
Tensor s = a.sum(0);   // sum along axis 0 → shape (2,)
Tensor m = a.mean(1);  // mean along axis 1 → shape (3,)

// Shape manipulation
Tensor flat = a.flatten();          // → shape (6,)
Tensor r    = a.reshape({3, 2});    // → shape (3,2)

// Print
a.print();
```

---

# Development

### Build with CMake

```id="dev01"
mkdir build
cd build
cmake ..
make
```

### Run tests

```id="dev02"
pytest
```

---

# Architecture Overview

The library follows a layered architecture similar to modern deep learning frameworks:

```id="arch01"
Python API
     ↓
Python Bindings (pybind11)
     ↓
C++ Core Library
     ↓
Tensor + Autograd Engine
     ↓
CPU / Future GPU Kernels
```

---

# Tensor API

### Static Constructors

| Method | Description |
|---|---|
| `Tensor::zeros({rows, cols})` | Tensor filled with 0.0 |
| `Tensor::ones({rows, cols})` | Tensor filled with 1.0 |
| `Tensor::custom({rows, cols}, val)` | Tensor filled with a custom scalar |
| `Tensor::form({rows, cols}, data)` | Tensor from a `vector<double>` |

### Shape & Access

| Method | Description |
|---|---|
| `dims()` | Returns shape as `vector<int>` |
| `at({i, j})` | Element access by index |
| `value(data)` | Set tensor data from `vector<double>` |
| `reshape(newshape)` | Returns reshaped tensor (same total elements) |
| `view(newshape)` | Alias for reshape, also prints the result |
| `flatten()` | Returns 1D tensor |

### Math Operations

| Method | Description |
|---|---|
| `add(Tensor)` | Element-wise addition (same shape) |
| `add_int(double)` | Add scalar to every element |
| `scale_int(double)` | Multiply every element by scalar |
| `multiply(Tensor)` | Element-wise multiplication (same shape) |
| `dot(Tensor)` | Matrix multiplication — supports 1D and 2D tensors |
| `transpose()` / `T()` | Transpose a 2D tensor |
| `sum(axis)` | Sum along axis (1D or 2D) |
| `mean(axis)` | Mean along axis (1D or 2D) |

### Display

| Method | Description |
|---|---|
| `print()` | Pretty-print the tensor with nested brackets |

---

# Roadmap

Planned development milestones:

* [x] Core tensor data structure
* [x] Tensor math operations (add, multiply, dot, transpose, sum, mean)
* [ ] Autograd computation graph
* [ ] Backpropagation engine
* [ ] Neural network modules
* [ ] Optimizers
* [ ] GPU backend support

---

# Dependencies

Core dependencies:

* **C++17**
* **CMake**
* **Python 3.8+**
* **pybind11**

Additional dependencies may be introduced as the project evolves.

---

# Contributing

Contributions are welcome.

Steps to contribute:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request

Please ensure that tests pass before submitting contributions.

---

# License

This project is licensed under the terms described in the `LICENSE` file.

---

# Status

⚠️ This project is currently **experimental and under active development**.

APIs and internal design may change as the project evolves.

---

# Inspiration

This project is inspired by the architecture of modern ML frameworks:

* PyTorch
* TensorFlow
* tinygrad
* NumPy
