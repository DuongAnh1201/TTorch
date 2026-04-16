#include "tensor.h"
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

Tensor Tensor::add(Tensor n) {
    if (shape != n.shape)
        throw invalid_argument("shapes are not compatible for add");
    Tensor result(shape);
    for (int i = 0; i < (int)result.data.size(); i++)
        result.data[i] = data[i] + n.data[i];
    return result;
}

Tensor Tensor::add_int(double i) {
    Tensor t;
    t.shape = shape;
    t.data = data;
    for (int j = 0; j < (int)t.data.size(); j++)
        t.data[j] += i;
    return t;
}

Tensor Tensor::scale_int(double i) {
    Tensor t;
    t.shape = shape;
    t.data = data;
    for (int j = 0; j < (int)t.data.size(); j++)
        t.data[j] *= i;
    return t;
}

Tensor Tensor::multiply(Tensor b) {
    if (shape != b.shape)
        throw invalid_argument("shapes are not compatible for multiply");
    Tensor result(shape);
    for (int i = 0; i < (int)data.size(); i++)
        result.data[i] = data[i] * b.data[i];
    return result;
}

Tensor Tensor::dot(Tensor b) {
    if (shape.size() != 2 && shape.size() != 1)
        throw invalid_argument("only accept 1D or 2D tensors");
    if (b.shape.size() != 2 && b.shape.size() != 1)
        throw invalid_argument("only accept 1D or 2D tensors");

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
    return result;
}

Tensor Tensor::T() {
    return transpose();
}

Tensor Tensor::sum(int axis) {
    if (shape.size() == 1) {
        Tensor result({1});
        result.data[0] = 0;
        for (double v : data) result.data[0] += v;
        return result;
    }
    if (shape.size() == 2) {
        int rows = shape[0], cols = shape[1];
        if (axis == 0) {
            Tensor result = Tensor::zeros({rows});
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    result.data[r] += data[r * cols + c];
            return result;
        } else if (axis == 1) {
            Tensor result = Tensor::zeros({cols});
            for (int c = 0; c < cols; c++)
                for (int r = 0; r < rows; r++)
                    result.data[c] += data[r * cols + c];
            return result;
        }
    }
    throw invalid_argument("sum only supports 1D and 2D tensors");
}

Tensor Tensor::mean(int axis) {
    if (shape.size() == 1) {
        Tensor result({1});
        Tensor m = sum();
        result.data[0] = m.data[0] / numel();
        return result;
    }
    if (shape.size() == 2) {
        int rows = shape[0], cols = shape[1];
        if (axis == 0) {
            Tensor m = sum(0);
            Tensor result = Tensor::zeros({rows});
            for (int r = 0; r < rows; r++)
                result.data[r] = m.data[r] / cols;
            return result;
        }
        if (axis == 1) {
            Tensor m = sum(1);
            Tensor result = Tensor::zeros({cols});
            for (int c = 0; c < cols; c++)
                result.data[c] = m.data[c] / rows;
            return result;
        }
    }
    throw invalid_argument("mean only supports 1D and 2D tensors");
}
