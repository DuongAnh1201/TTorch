#include "tensor.h"
#include "autograd.h"
#include <iostream>
using namespace std;

// ── Private helpers ──────────────────────────────────────────

int Tensor::numel() {
    int total = 1;
    for (int d : shape) total *= d;
    return total;
}

int Tensor::ndim() {
    return shape.size();
}

int Tensor::size(int dim) {
    if (dim >= ndim())
        throw invalid_argument("out of index");
    return shape[dim];
}

void Tensor::printRecursive(vector<double> data, vector<int> sh) {
    if (sh.size() == 1) {
        cout << "[";
        for (int i = 0; i < sh[0]; i++) {
            cout << data[i];
            if (i < sh[0] - 1) cout << ",";
        }
        cout << "]";
    } else {
        int chunk = data.size() / sh[0];
        cout << "[";
        vector<int> rest_shape(sh.begin() + 1, sh.end());
        for (int i = 0; i < sh[0]; i++) {
            vector<double> chunk_data = slice(data, i * chunk, i * chunk + chunk);
            printRecursive(chunk_data, rest_shape);
            if (i < sh[0] - 1) cout << "\n";
        }
        cout << "]";
    }
}

// ── Constructors ─────────────────────────────────────────────

Tensor::Tensor(const initializer_list<int>& dims) {
    shape = dims;
    int total = 1;
    for (int d : shape) total *= d;
    data.assign(total, 0.0);
}

Tensor::Tensor(const vector<int>& dims) {
    shape = dims;
    int total = 1;
    for (int d : shape) total *= d;
    data.assign(total, 0.0);
}

// ── Factory methods ──────────────────────────────────────────

Tensor Tensor::zeros(initializer_list<int> dims) {
    Tensor t;
    t.shape = dims;
    t.data.assign(t.numel(), 0.0);
    return t;
}

Tensor Tensor::ones(initializer_list<int> dims) {
    Tensor t;
    t.shape = dims;
    t.data.assign(t.numel(), 1.0);
    return t;
}

Tensor Tensor::custom(initializer_list<int> dims, double val) {
    Tensor t;
    t.shape = dims;
    t.data.assign(t.numel(), val);
    return t;
}

Tensor Tensor::form(initializer_list<int> dims, vector<double> d) {
    Tensor t = Tensor::zeros(dims);
    if (d.size() != t.data.size())
        throw invalid_argument("data size is not compatible");
    t.data = d;
    return t;
}

Tensor Tensor::form(const vector<int>& dims, vector<double> d) {
    Tensor t(dims);
    if (d.size() != t.data.size())
        throw invalid_argument("data size is not compatible");
    t.data = d;
    return t;
}

// ── Utilities ────────────────────────────────────────────────

void Tensor::print() {
    printRecursive(data, shape);
    cout << endl;
}

vector<int> Tensor::dims() {
    return shape;
}

vector<double> Tensor::value(vector<double> a) {
    if (a.size() != data.size())
        throw invalid_argument("size of argument is not compatible");
    for (int i = 0; i < (int)data.size(); i++)
        data[i] = a[i];
    return data;
}

double Tensor::at(vector<int> index) {
    if (index.size() != shape.size())
        throw invalid_argument("index is not compatible with tensor");
    if (index.size() == 1)
        return data[index[0]];
    return data[index[0] * shape[1] + index[1]];
}

vector<double> Tensor::slice(vector<double>& v, int start, int end) {
    return vector<double>(v.begin() + start, v.begin() + end);
}

// ── Shape ops ────────────────────────────────────────────────

Tensor Tensor::reshape(vector<int> newshape) {
    int total = 1;
    for (int d : newshape) total *= d;
    if (total != (int)data.size())
        throw invalid_argument("invalid reshape");
    Tensor t(newshape);
    t.data = data;
    return t;
}

Tensor Tensor::view(vector<int> newshape) {
    int total = 1;
    for (int d : newshape) total *= d;
    if ((int)data.size() != total)
        throw invalid_argument("invalid view");
    return reshape(newshape);
}

Tensor Tensor::flatten() {
    int total = 1;
    for (int d : shape) total *= d;
    Tensor result({total});
    result.data = data;
    return result;
}

// ── Math ops ─────────────────────────────────────────────────

Tensor Tensor::add(Tensor& n) {
    if (shape != n.shape)
        throw invalid_argument("shapes are not compatible for add");
    Tensor result(shape);
    for (int i = 0; i < (int)result.data.size(); i++)
        result.data[i] = data[i] + n.data[i];
    //Build graph - attach GradFn
    if (requires_grad || n.requires_grad) 
    {
        result.requires_grad = true;
        result.is_leaf = false;
        auto* fn = new AddBackward();
        fn->inputs = {this, &n}; //connect to inputs
        result.grad_fn = fn;
    }
    return result;
}

Tensor Tensor::add_int(double i) {
    Tensor result;
    result.shape = shape;
    result.data = data;
    for (int j = 0; j < (int)result.data.size(); j++)
        result.data[j] += i;
    // Build graph - attach GradFn
    if (requires_grad)
    {
        result.requires_grad = true;
        result.is_leaf = false;
        auto* fn = new AddScalarBackward();
        fn->inputs = {this};
        result.grad_fn = fn;
    }
    return result;
}

Tensor Tensor::scale_int(double i) {
    Tensor result;
    result.shape = shape;
    result.data = data;
    for (int j = 0; j < (int)result.data.size(); j++)
        result.data[j] *= i;
    //Build graph - attach GradFn
    if(requires_grad)
    {
        result.requires_grad = true;
        result.is_leaf = false;
        auto* fn = new ScaleBackward();
        fn->scalar = i;  // save scalar — needed for grad_a = grad * scalar
        fn->inputs = {this};
        result.grad_fn = fn;
    }
    return result;
}

Tensor Tensor::multiply(Tensor &b) {
    if (shape != b.shape)
        throw invalid_argument("shapes are not compatible for multiply");
    Tensor result(shape);
    for (int i = 0; i < (int)data.size(); i++)
        result.data[i] = data[i] * b.data[i];
    // Build graph - attach GradFn
    if (requires_grad || b.requires_grad)
    {
        result.requires_grad = true;
        result.is_leaf = false;
        auto* fn = new MulBackward();
        fn->saved_a = *this;  // save a — needed for grad_b = grad * a
        fn->saved_b = b;      // save b — needed for grad_a = grad * b
        fn->inputs = {this, &b};
        result.grad_fn = fn;
    }
    return result;
}

Tensor Tensor::dot(Tensor &b) {
    if (shape.size() != 2 && shape.size() != 1)
        throw invalid_argument("only accept 1D or 2D tensors");
    if (b.shape.size() != 2 && b.shape.size() != 1)
        throw invalid_argument("only accept 1D or 2D tensors");

    // save original tensors before shape mutation
    Tensor saved_a = *this;
    Tensor saved_b = b;

    if (shape.size() == 1) shape = vector<int>{(int)data.size(), 1};
    if (b.shape.size() == 1) b.shape = vector<int>{(int)b.data.size(), 1};

    int row_a = shape[0], col_a = shape[1];
    int row_b = b.shape[0], col_b = b.shape[1];

    if (col_a != row_b)
        throw invalid_argument("incompatible shapes for dot product");

    Tensor result({row_a, col_b});
    for (int i = 0; i < row_a; i++)
        for (int j = 0; j < col_b; j++)
            for (int k = 0; k < col_a; k++)
                result.data[i * col_b + j] += data[col_a * i + k] * b.data[col_b * k + j];
    if (requires_grad || b.requires_grad)
    {
        result.requires_grad = true;
        result.is_leaf = false;
        auto* fn = new DotBackward();
        fn->saved_a = saved_a;  // needed for grad_b = a.T @ grad
        fn->saved_b = saved_b;  // needed for grad_a = grad @ b.T
        fn->inputs = {this, &b};
        result.grad_fn = fn;
    }
    return result;
}

Tensor Tensor::transpose() {
    if (shape.size() != 2)
        throw invalid_argument("only accept 2D matrix");
    int rows = shape[0], cols = shape[1];
    Tensor result({cols, rows});
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            result.data[c * rows + r] = data[r * cols + c];
    if (requires_grad)
    {
        result.requires_grad = true;
        result.is_leaf = false;
        auto* fn = new TransposeBackward();
        fn->inputs = {this};
        result.grad_fn = fn;
    }
    return result;
}

Tensor Tensor::T() {
    return transpose();
}

Tensor Tensor::sum(int axis) {
    vector<int> original = shape;
    Tensor result;

    // Forward computation
    if (shape.size() == 1) {
        result = Tensor({1});
        result.data[0] = 0;
        for (double v : data) result.data[0] += v;
    } else if (shape.size() == 2) {
        int rows = shape[0], cols = shape[1];
        if (axis == 0) {
            result = Tensor::zeros({rows});
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    result.data[r] += data[r * cols + c];
        } else if (axis == 1) {
            result = Tensor::zeros({cols});
            for (int c = 0; c < cols; c++)
                for (int r = 0; r < rows; r++)
                    result.data[c] += data[r * cols + c];
        } else {
            throw invalid_argument("sum only supports 1D and 2D tensors");
        }
    } else {
        throw invalid_argument("sum only supports 1D and 2D tensors");
    }

    // Build graph - attach GradFn (same for all cases)
    if (requires_grad) {
        result.requires_grad = true;
        result.is_leaf = false;
        auto* fn = new SumBackward();
        fn->original_shape = original;
        fn->axis = axis;
        fn->inputs = {this};
        result.grad_fn = fn;
    }

    return result;
}

Tensor Tensor::mean(int axis) {
    vector<int> original = shape;
    Tensor result;
    int n;

    // Forward computation
    if (shape.size() == 1) {
        result = Tensor({1});           // fix: must initialize result with shape
        Tensor m = sum();
        result.data[0] = m.data[0] / numel();
        n = numel();
    } else if (shape.size() == 2) {
        int rows = shape[0], cols = shape[1];
        if (axis == 0) {
            result = Tensor::zeros({rows});
            Tensor m = sum(0);
            for (int r = 0; r < rows; r++)
                result.data[r] = m.data[r] / cols;
            n = cols;
        } else if (axis == 1) {
            result = Tensor::zeros({cols});
            Tensor m = sum(1);
            for (int c = 0; c < cols; c++)
                result.data[c] = m.data[c] / rows;
            n = rows;
        } else {
            throw invalid_argument("mean only supports 1D and 2D tensors");
        }
    } else {
        throw invalid_argument("mean only supports 1D and 2D tensors");
    }

    // Build graph - attach GradFn
    if (requires_grad) {
        result.requires_grad = true;
        result.is_leaf = false;
        auto* fn = new MeanBackward();
        fn->original_shape = original;
        fn->axis = axis;
        fn->n = n;
        fn->inputs = {this};
        result.grad_fn = fn;
    }

    return result;  // fix: was missing
}
