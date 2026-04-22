// autograd.cpp - placeholder
#include<iostream>
#include "tensor.h"
#include "autograd.h"

using namespace std;

//Core Infrastructure
//Helper function
static void accum_grad(Tensor* t, const Tensor& g)
{
    if(!t->requires_grad)
    {
        return;
    }
    if(!t->grad)
    {
        t->grad = new Tensor(t->shape);
        for (double &v: t->grad->data){v = 0.0;}
    }
    for(int i = 0; i< (int)t->grad->data.size(); i ++)
    {
        t->grad->data[i] += g.data[i];
    }
}

// ── AddBackward ──────────────────────────────────────────────
// forward:  c = a + b
// backward: grad flows to both a and b unchanged
void AddBackward::backward(Tensor& g)
{
    accum_grad(inputs[0], g);
    accum_grad(inputs[1], g);
}

void AddScalarBackward::backward(Tensor& g)
{
    accum_grad(inputs[0], g);
}


//Gradient Functions

//Backward Engine

//Activation Functions

// 
