#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>

struct GradFn; // forward declare so Tensor can hold a pointer to it

class Tensor {
private:
    int numel();
    void printRecursive(std::vector<double> data, std::vector<int> sh);

public:
    // --- Core data ---
    std::vector<int>    shape;
    std::vector<double> data;

    // --- Autograd fields (set by autograd, not tensor logic) ---
    bool     requires_grad = false;
    Tensor*  grad          = nullptr;
    GradFn*  grad_fn       = nullptr;
    bool     is_leaf       = true;

    // --- Constructors ---
    Tensor() = default;
    Tensor(const std::initializer_list<int>& dims);
    Tensor(const std::vector<int>& dims);

    // --- Factory methods ---
    static Tensor zeros(std::initializer_list<int> dims);
    static Tensor ones(std::initializer_list<int> dims);
    static Tensor custom(std::initializer_list<int> dims, double val);
    static Tensor form(std::initializer_list<int> dims, std::vector<double> data);
    static Tensor form(const std::vector<int>& dims, std::vector<double> data);

    // --- Utilities ---
    void              print();
    std::vector<int>  dims();
    std::vector<double> value(std::vector<double> a);
    double            at(std::vector<int> index);
    std::vector<double> slice(std::vector<double>& v, int start, int end);
    int               ndim();
    int               size(int dim);

    // --- Shape ops ---
    Tensor reshape(std::vector<int> newshape);
    Tensor view(std::vector<int> newshape);
    Tensor flatten();

    // --- Math ops ---
    Tensor add(Tensor& n);
    Tensor add_int(double i);
    Tensor scale_int(double i);
    Tensor multiply(Tensor& b);
    Tensor dot(Tensor& b);
    Tensor transpose();
    Tensor T();
    Tensor sum(int axis = 0);
    Tensor mean(int axis = 0);

    // --- Autograd helpers (implemented in autograd.cpp) ---
    void backward();
    void zero_grad();
};
