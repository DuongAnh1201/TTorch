#include<iostream>
#include "nn.h"
#include "tensor.h"
#include "autograd.h"

using namespace std;

Linear::Linear(int in_features, int out_features) {
    W = Tensor::custom({in_features, out_features}, 0.01);
    b = Tensor::custom({out_features}, 0.01);
    W.requires_grad = true;
    b.requires_grad = true;
}

Tensor Linear::forward(Tensor& x) {
    return x.dot(W).add(b);
}

Tensor mse_loss(Tensor& pred, Tensor& target)
{
    auto diff = pred.subtract(target);
    auto sq = diff.multiply(diff);
    auto mean = sq.mean();
    return mean;
}

