#pragma once
#include "tensor.h"
#include <vector>

// ── Base class for all gradient functions ────────────────────
// Every op (add, dot, etc.) creates one of these and attaches
// it to the output tensor as output.grad_fn
struct GradFn {
    std::vector<Tensor*> inputs; // tensors that were used in the forward pass
    virtual void backward(Tensor& grad) = 0;
    virtual ~GradFn() = default;
};

// ── One GradFn per operation ─────────────────────────────────

// add(a, b) → grad flows equally to both a and b
struct AddBackward : GradFn {
    void backward(Tensor& grad) override;
};

// add_int(a, scalar) → grad flows to a unchanged
struct AddScalarBackward : GradFn {
    void backward(Tensor& grad) override;
};

// multiply(a, b) element-wise → grad_a = grad * b, grad_b = grad * a
struct MulBackward : GradFn {
    Tensor saved_a, saved_b;
    void backward(Tensor& grad) override;
};

// scale_int(a, scalar) → grad_a = grad * scalar
struct ScaleBackward : GradFn {
    double scalar;
    void backward(Tensor& grad) override;
};

// dot(a, b) matrix multiply → grad_a = grad @ b.T, grad_b = a.T @ grad
struct DotBackward : GradFn {
    Tensor saved_a, saved_b;
    void backward(Tensor& grad) override;
};

// transpose(a) → grad_a = grad.transpose()
struct TransposeBackward : GradFn {
    void backward(Tensor& grad) override;
};

// flatten(a) → grad_a = grad.reshape(original_shape)
struct FlattenBackward : GradFn {
    std::vector<int> original_shape;
    void backward(Tensor& grad) override;
};

// sum(a, axis) → grad_a = broadcast grad back to original shape
struct SumBackward : GradFn {
    std::vector<int> original_shape;
    int axis;
    void backward(Tensor& grad) override;
};

// mean(a, axis) → grad_a = grad / N, broadcast back to original shape
struct MeanBackward : GradFn {
    std::vector<int> original_shape;
    int axis;
    int n; // number of elements averaged over
    void backward(Tensor& grad) override;
};

// relu(a) → grad_a = grad * (a > 0)
struct ReLUBackward : GradFn {
    Tensor saved_input; // need to know which elements were positive
    void backward(Tensor& grad) override;
};

// sigmoid(a) → grad_a = grad * sigmoid(a) * (1 - sigmoid(a))
struct SigmoidBackward : GradFn {
    Tensor saved_output; // sigmoid output is reused in backward
    void backward(Tensor& grad) override;
};

//Softmax(a)
struct SoftmaxBackward : GradFn{
    void backward(Tensor& grad) override;
};

// ── Free function ops ─────────────────────────────────────────
// These wrap the Tensor methods and attach the correct GradFn
Tensor relu(Tensor& x);
Tensor sigmoid(Tensor& x);
Tensor softmax(Tensor& x);