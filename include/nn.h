#pragma once
#include "tensor.h"
#include "autograd.h"

struct Linear {
    Tensor W;   // weights [in_features, out_features]
    Tensor b;   // bias    [out_features]

    Linear(int in_features, int out_features);
    Tensor forward(Tensor& x);
};

Tensor mse_loss(Tensor& pred, Tensor& target);
