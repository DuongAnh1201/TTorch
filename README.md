# TTorch

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![C++](https://img.shields.io/badge/C++-17-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build](https://img.shields.io/badge/build-CMake-orange.svg)

`TTorch` is a **lightweight tensor and automatic differentiation library written in C++ with Python bindings**.

The project explores the internal architecture of modern deep learning frameworks by implementing core components such as:

* tensor operations
* automatic differentiation (autograd)
* Python bindings
* modular compute kernels

The long-term goal is to build a **minimal deep learning backend similar to PyTorch**, while keeping the codebase simple and educational.

---

# Key Features

* **High-performance C++ core**
* **Automatic differentiation engine**
* **Python bindings for easy use**
* **Modular tensor operation system**
* **CMake-based build system**
* **Python package distribution via `pyproject.toml`**

---

# Project Structure

```
TTorch/
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
│   ├── tensor.h
│   └── autograd.h
│
├── python/
│   └── my_cpp_lib/
│       └── __init__.py
│
└── tests/
    └── test.cpp
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

```bash
git clone https://github.com/DuongAnh1212/TTorch.git
cd TTorch
```

Install build tools:

```bash
pip install build
```

Build the Python package:

```bash
python -m build
```

Install the generated wheel:

```bash
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
Tensor c = a.add(b);         // element-wise addition
Tensor d = a.multiply(b);    // element-wise multiplication
Tensor e = a.add_int(10.0);  // broadcast scalar addition
Tensor f = a.scale_int(2.0); // broadcast scalar multiply

// Matrix operations
Tensor t  = a.transpose();   // or a.T()
Tensor ab = a.dot(a.T());    // matrix multiplication → (2,2)

// Reductions
Tensor s = a.sum(0);   // sum along axis 0 → shape (3,)
Tensor m = a.mean(1);  // mean along axis 1 → shape (2,)

// Shape manipulation
Tensor flat = a.flatten();        // → shape (6,)
Tensor r    = a.reshape({3, 2});  // → shape (3,2)

// Print
a.print();
```

---

# Autograd Example

```cpp
#include "tensor.h"
#include "autograd.h"

Tensor x = Tensor::form({2, 2}, {1, 2, 3, 4});
x.requires_grad = true;

Tensor y = relu(x);   // forward pass with grad tracking
y.backward();         // backpropagate gradients

x.grad->print();      // gradient of x
x.zero_grad();        // reset gradients
```

---

# Development

### Build with CMake

```bash
mkdir build
cd build
cmake ..
make
```

### Run tests

```bash
pytest
```

---

# Architecture Overview

The library follows a layered architecture similar to modern deep learning frameworks:

```
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
| `ndim()` | Returns the number of dimensions |
| `size(int dim)` | Returns the size of a given dimension |
| `at({i, j})` | Element access by index |
| `value(data)` | Set tensor data from `vector<double>` |
| `slice(v, start, end)` | Extract a sub-range from a flat data vector |
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

### Autograd

| Method | Description |
|---|---|
| `backward()` | Backpropagate gradients from this tensor |
| `zero_grad()` | Reset accumulated gradient to zero |

### Display

| Method | Description |
|---|---|
| `print()` | Pretty-print the tensor with nested brackets |

---

# Autograd API

TTorch implements a dynamic computation graph. Each operation attaches a `GradFn` to the output tensor, enabling automatic gradient computation via `backward()`.

### Gradient Functions

| GradFn | Forward op | Backward rule |
|---|---|---|
| `AddBackward` | `add(a, b)` | grad flows equally to `a` and `b` |
| `AddScalarBackward` | `add_int(a, s)` | grad flows to `a` unchanged |
| `MulBackward` | `multiply(a, b)` | `grad_a = grad * b`, `grad_b = grad * a` |
| `ScaleBackward` | `scale_int(a, s)` | `grad_a = grad * s` |
| `DotBackward` | `dot(a, b)` | `grad_a = grad @ b.T`, `grad_b = a.T @ grad` |
| `TransposeBackward` | `transpose(a)` | `grad_a = grad.transpose()` |
| `FlattenBackward` | `flatten(a)` | `grad_a = grad.reshape(original_shape)` |
| `SumBackward` | `sum(a, axis)` | broadcast grad back to original shape |
| `MeanBackward` | `mean(a, axis)` | `grad / N`, broadcast back to original shape |
| `ReLUBackward` | `relu(a)` | `grad_a = grad * (a > 0)` |
| `SigmoidBackward` | `sigmoid(a)` | `grad_a = grad * sigmoid(a) * (1 - sigmoid(a))` |

### Activation Functions

| Function | Description |
|---|---|
| `relu(Tensor& x)` | Element-wise ReLU with gradient tracking |
| `sigmoid(Tensor& x)` | Element-wise sigmoid with gradient tracking |

---

# Roadmap

Planned development milestones:

* [x] Core tensor data structure
* [x] Tensor math operations (add, multiply, dot, transpose, sum, mean)
* [x] Autograd computation graph (GradFn architecture)
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

This project is currently **experimental and under active development**.

APIs and internal design may change as the project evolves.

---

# Inspiration

This project is inspired by the architecture of modern ML frameworks:

* PyTorch
* TensorFlow
* tinygrad
* NumPy
