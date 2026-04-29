# TTorch

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![C++](https://img.shields.io/badge/C++-17-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-0.1.2-orange.svg)

`TTorch` is a **lightweight tensor and automatic differentiation library written in C++ with Python bindings**.

The project explores the internal architecture of modern deep learning frameworks by implementing core components from scratch: tensor operations, a dynamic computation graph, backpropagation, and Python bindings via pybind11.

---

## Features

- **C++17 core** — tensors, math ops, shape manipulation
- **Dynamic autograd engine** — computation graph built during the forward pass, traversed in reverse on `backward()`
- **Activation functions** — `relu`, `sigmoid` with gradient tracking
- **Python bindings** — full API exposed via pybind11
- **scikit-build-core** build backend — `pip install` works out of the box
- **CI/CD** — wheels built via cibuildwheel and published to PyPI on push to `main`

---

## Project Structure

```
TTorch/
├── pyproject.toml        # Python build config (scikit-build-core + pybind11)
├── CMakeLists.txt        # C++ build config
├── README.md
├── AUTOGRAD.md           # Deep-dive into the autograd design
├── LICENSE
│
├── include/
│   ├── tensor.h          # Tensor class declaration + autograd fields
│   └── autograd.h        # GradFn base class + all GradFn subclasses
│
├── src/
│   ├── tensor.cpp        # Tensor ops + forward graph building
│   ├── autograd.cpp      # Backward engine + gradient rules
│   └── bindings.cpp      # pybind11 Python bindings
│
├── ttorch/
│   └── __init__.py       # Python package entry point
│
└── tests/
    └── test.cpp          # C++ unit tests
```

---

## Installation

### From PyPI (recommended)

```bash
pip install ttorch
```

### From Source

```bash
git clone https://github.com/DuongAnh1212/TTorch.git
cd TTorch
pip install .
```

Or build a wheel manually:

```bash
pip install build
python -m build
pip install dist/*.whl
```

**Requirements:** Python 3.11+, a C++17-capable compiler, CMake.

---

## Quick Start

### Python

```python
import ttorch

# Create tensors
a = ttorch.Tensor.form([2, 3], [1, 2, 3, 4, 5, 6])
b = ttorch.Tensor.ones([2, 3])

# Math ops
c = a.add(b)          # element-wise addition
d = a.multiply(b)     # element-wise multiplication
e = a.dot(a.T())      # matrix multiply → shape (2, 2)
s = a.sum(0)          # sum along axis 0 → shape (3,)

# Autograd
a.requires_grad = True
y = ttorch.relu(a)
y.backward()
print(a.grad)         # gradient of a
a.zero_grad()
```

### C++

```cpp
#include "tensor.h"
#include "autograd.h"

Tensor a = Tensor::form({2, 3}, {1, 2, 3, 4, 5, 6});
a.requires_grad = true;

Tensor y = relu(a);
y.backward();

a.grad->print();  // gradient of a
a.zero_grad();
```

---

## Tensor API

### Factory Methods

| Method | Description |
|---|---|
| `Tensor::zeros(dims)` | Tensor filled with 0.0 |
| `Tensor::ones(dims)` | Tensor filled with 1.0 |
| `Tensor::custom(dims, val)` | Tensor filled with a scalar |
| `Tensor::form(dims, data)` | Tensor from a `vector<double>` |

### Shape & Access

| Method | Description |
|---|---|
| `dims()` | Returns shape as `vector<int>` |
| `ndim()` | Number of dimensions |
| `size(dim)` | Size of a given dimension |
| `at(index)` | Element access by multi-index |
| `reshape(newshape)` | Returns reshaped tensor |
| `view(newshape)` | Alias for reshape |
| `flatten()` | Returns 1D tensor |

### Math Operations

| Method | Description |
|---|---|
| `add(Tensor)` | Element-wise addition |
| `add_int(double)` | Scalar broadcast addition |
| `scale_int(double)` | Scalar broadcast multiply |
| `multiply(Tensor)` | Element-wise multiplication |
| `dot(Tensor)` | Matrix multiplication (1D and 2D) |
| `transpose()` / `T()` | Transpose a 2D tensor |
| `sum(axis)` | Sum along axis |
| `mean(axis)` | Mean along axis |

### Autograd

| Method | Description |
|---|---|
| `backward()` | Backpropagate gradients through the computation graph |
| `zero_grad()` | Reset accumulated gradient to zero |

---

## Autograd API

TTorch implements a dynamic computation graph — the graph is built automatically during the forward pass as a chain of `GradFn` pointers.

### Gradient Functions

| GradFn | Forward op | Backward rule |
|---|---|---|
| `AddBackward` | `add(a, b)` | grad flows to `a` and `b` unchanged |
| `AddScalarBackward` | `add_int(a, s)` | grad flows to `a` unchanged |
| `MulBackward` | `multiply(a, b)` | `grad_a = grad * b`, `grad_b = grad * a` |
| `ScaleBackward` | `scale_int(a, s)` | `grad_a = grad * s` |
| `DotBackward` | `dot(a, b)` | `grad_a = grad @ b.T`, `grad_b = a.T @ grad` |
| `TransposeBackward` | `transpose(a)` | `grad_a = grad.T` |
| `FlattenBackward` | `flatten(a)` | `grad_a = grad.reshape(original_shape)` |
| `SumBackward` | `sum(a, axis)` | broadcast grad back to original shape |
| `MeanBackward` | `mean(a, axis)` | broadcast `grad / n` back to original shape |
| `ReLUBackward` | `relu(a)` | `grad_a = grad * (a > 0)` |
| `SigmoidBackward` | `sigmoid(a)` | `grad_a = grad * sigmoid(a) * (1 - sigmoid(a))` |

### Activation Functions

| Function | Description |
|---|---|
| `relu(Tensor& x)` | Element-wise ReLU with gradient tracking |
| `sigmoid(Tensor& x)` | Element-wise sigmoid with gradient tracking |

---

## Architecture

```
Python API (ttorch)
      ↓
pybind11 bindings  (src/bindings.cpp)
      ↓
C++ Core
  ├── Tensor + math ops   (include/tensor.h, src/tensor.cpp)
  └── Autograd engine     (include/autograd.h, src/autograd.cpp)
```

The backward engine uses topological sort to visit nodes in reverse dependency order, accumulating gradients into leaf tensors. See [AUTOGRAD.md](AUTOGRAD.md) for a detailed design walkthrough with worked examples.

---

## Build from C++ Only

```bash
mkdir build && cd build
cmake ..
make
```

---

## Roadmap

- [x] Core tensor data structure
- [x] Tensor math operations (add, multiply, dot, transpose, sum, mean)
- [x] Autograd computation graph (GradFn architecture)
- [x] Backpropagation engine (`backward`, `zero_grad`)
- [x] Activation functions (relu, sigmoid)
- [x] Python bindings (pybind11 via scikit-build-core)
- [x] PyPI publishing via CI/CD (cibuildwheel + GitHub Actions)
- [ ] Neural network modules (Linear, loss functions)
- [ ] Optimizers (SGD, Adam)
- [ ] GPU backend

---

## Dependencies

| Dependency | Purpose |
|---|---|
| C++17 | Core library |
| CMake | Build system |
| Python 3.11+ | Python interface |
| pybind11 | Python bindings |
| scikit-build-core | Python build backend |

---
## Fix Small Information
The last number of the patch will increment in terms of changing small information and does not harm to the code.
## Minor changes
The minor number will increment if a new interface, API, or type is introduced into the public interface of TTorch. 

## Patch Changes
The patch number will increment if:
      1. Bug fixes that align code to the behevior of an API, improves performance or improves code size efficiently

For now, you cannot expect ABI or API stability with anything in the **update_method** branch.

## License

MIT — see [LICENSE](LICENSE).

---

## Inspiration

- [PyTorch](https://pytorch.org/)
- [tinygrad](https://github.com/tinygrad/tinygrad)
- [micrograd](https://github.com/karpathy/micrograd)
