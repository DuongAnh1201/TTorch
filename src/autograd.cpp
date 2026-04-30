// autograd.cpp - placeholder
#include<iostream>
#include "tensor.h"
#include "autograd.h"
#include<cmath>

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

void Tensor::backward()
{

}

void Tensor::zero_grad()
{
    if (grad) {
        for (double& v : grad->data) v = 0.0;
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

void MulBackward::backward(Tensor& g)
{
    Tensor grad_a(saved_a.shape);
    for(int i = 0; i<(int) g.data.size(); i++)
    {
        grad_a.data[i] = g.data[i] * saved_b.data[i];
    }
    accum_grad(inputs[0], grad_a);
    
    Tensor grad_b(saved_b.shape);
    for(int i=0; i<(int) g.data.size(); i++)
    {
        grad_b.data[i] = g.data[i]*saved_a.data[i];
    }
    accum_grad(inputs[1], grad_b);
}

void ScaleBackward::backward(Tensor& g)
{
    Tensor grad_a(g.shape);
    for(int i = 0; i<(int)g.data.size(); i++)
    {
        grad_a.data[i] = g.data[i] * scalar;
    }
    accum_grad(inputs[0], grad_a);
    
}

void DotBackward::backward(Tensor& g)
{
    Tensor grad_a(saved_a.shape);
    Tensor b_T = saved_b.T();
    grad_a = g.dot(b_T);
    accum_grad(inputs[0], grad_a);
    
    Tensor grad_b(saved_b.shape);
    Tensor a_T = saved_a.T();
    grad_b = a_T.dot(g);
    accum_grad(inputs[1], grad_b);
}

void TransposeBackward::backward(Tensor& g)
{
    Tensor grad_T = g.T();
    accum_grad(inputs[0], grad_T);
}

void FlattenBackward::backward(Tensor& g)
{
    Tensor grad_a;
    grad_a = g.reshape(original_shape);
    accum_grad(inputs[0],grad_a);
}

static Tensor broadcast(const Tensor& grad, const vector<int>& original_shape, int axis)
{
    Tensor result(original_shape);
    int rows = original_shape[0];
    int cols = original_shape[1];
    if(axis == 0)
    {
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.data[r * cols + c] = grad.data[r];

    }
    else
    {
        for (int r =0; r<rows; r++)
        {
            for (int c = 0; c<cols; c++)
            {
                result.data[r*cols+c] = grad.data[c];
            }
        }
    }
    return result;
}
void SumBackward::backward(Tensor& g)
{
    if (original_shape.size() == 1)
    {
        Tensor grad_a(original_shape);
        for (double& v: grad_a.data) v = g.data[0];
        accum_grad(inputs[0], grad_a);
    }
    else if (original_shape.size() == 2) {
        Tensor grad_a = broadcast(g, original_shape, axis);
        accum_grad(inputs[0], grad_a);
    }
}

void MeanBackward::backward(Tensor& g)
{
    if(original_shape.size() == 1)
    {
        Tensor grad_a(original_shape);
        for (double& v:grad_a.data) v = g.data[0]/n;
        accum_grad(inputs[0], grad_a);
    }
    else if(original_shape.size() == 2)
    {
        Tensor grad_a = broadcast(g, original_shape, axis);
        for (int i = 0; i < (int)grad_a.data.size(); i++)
        {
            grad_a.data[i] = grad_a.data[i] / n;
        }
        accum_grad(inputs[0], grad_a);
        
    }
}

void SigmoidBackward::backward(Tensor& g)
{
    Tensor grad_a(g.shape);
    for (int i=0; i<(int)g.data.size(); i++)
    {
        grad_a.data[i] = g.data[i]*saved_output.data[i]*(1-saved_output.data[i]);
    }
    accum_grad(inputs[0], grad_a);
}

void ReLUBackward::backward(Tensor& g)
{
    Tensor grad_a(g.shape);
    for(int i=0; i<(int)g.data.size(); i++)
    {
        if (saved_input.data[i]>0){
            grad_a.data[i] = g.data[i];
        }
        else
        {
            grad_a.data[i] = 0;
        }
    }
    accum_grad(inputs[0], grad_a);
}

Tensor sigmoid(Tensor& x)
{
    Tensor result(x.shape);
    for (int i = 0; i< (int)x.data.size(); i++)
    {
        result.data[i] = 1/(1+exp(-x.data[i]));
    }
    
    if (x.requires_grad)
    {
        result.requires_grad = true;
        result.is_leaf = false;
        auto* fn = new SigmoidBackward();
        fn->saved_output = result;
        fn->inputs = {&x};
        result.grad_fn = fn;
    }
    return result;
}


Tensor relu(Tensor& x)
{
    Tensor result(x.shape);
    for (int i = 0; i<(int)x.data.size(); i++)
    {
        result.data[i] = max(0.0, x.data[i]);
    }
    
    if (x.requires_grad)
    {
        result.requires_grad = true;
        result.is_leaf = false;
        auto* fn = new ReLUBackward();
        fn->saved_input = x;
        fn->inputs = {&x};
        result.grad_fn = fn;
    }
    return result;
}

Tensor softmax(Tensor& x)
{
    double denor= 0;
    vector<double> data;
    for (int i = 0; i < (int)x.data.size(); i++){
        double m = exp(x.data[i]);
        denor += m;
        data.push_back(m);
    }
    Tensor result(x.shape);
    for (int i = 0; i<(int)x.data.size(); i++){
        result.data[i] = data[i]/denor;
    }

    if(x.requires_grad){
        result.requires_grad = true;
        result.is_leaf = false;
        auto* fn = new SoftmaxBackward();
         
    }




}