// tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>

class Tensor
{
public:
    std::vector<int> shape;
    std::vector<double> tensor;

    Tensor(const std::initializer_list<int>& dims);
    Tensor(const std::vector<int>& dims);

    std::vector<int> dims();
    void zeros();
    void ones();
    void value(const std::vector<double>& a);

    void reshape(const std::vector<int>& newshape);
    Tensor view(const std::vector<int>& newshape);
    void print();

    Tensor add(const Tensor& other);
    Tensor transpose();
    Tensor dot(const Tensor& b);

private:
    void printRecursive(const std::vector<double>& data,
                        const std::vector<int>& sh);
    std::vector<double> slice(const std::vector<double>& v, int start, int end);
};

#endif

int main()
{
    Tensor m({2, 3});
    Tensor n({3,2});
    m.value({1,2,3,4,5,6});
    n.value({1,4,2,5,3,6});
    Tensor k = m.dot(n);
    k.print();
}