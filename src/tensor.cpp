#include<iostream>
#include<Eigen/Dense>
using namespace std;
using namespace Eigen;

class Tensor
{
    public:
        vector<int> shape;
        vector<double> tensor;
        Tensor(const initializer_list<int>& dims)
        {
            shape = dims;
            int total = 1;
            for (int d: shape)
            {
                total *= d;
            }
            tensor.resize(total);
            for (int a = 0; a<tensor.size(); a++)
            {
                tensor[a] = 0.0;
            } 
        };
        vector<int> dims()
        {
            vector<int> A(shape.size());
            for (int i=0; i<shape.size(); i++)
            {
                A[i] = shape[i];
            };
            return A;
        };
        
        vector<double> zeros()
        {
           for (int i=0; i<tensor.size(); i++)
           {
               tensor[i] = 0.0;
           };
           return tensor;
        }
        vector<double> ones()
        {   
            for (int i=0; i<tensor.size(); i++)
            {
                tensor[i] = 1.0;
            };
            return tensor;
        };
        
        vector<double> tensor(int j)
        {
            for (int i=0; i<tensor.size(); i++)
            {
                tensor[i] = j;
            };
            return tensor;
        };
        //Matrix manipulation
        vector<int> reshape(vector<int> newshape)
        {
            int total = 1;
            for (int d: newshape)
            {
                total*= d;
            }
            if (total != tensor.size())
            {
                throw invalid_argument("Invalid Reshape");
            };
            shape = newshape;
            return shape;
        };
        
        int view(vector<int> a)
        {
            if (a.size()<shape.size() || a.size()>shape.size())
            {
                throw invalid_argument("Invalid Argument");
            };
            for (int i=0; i<shape.size(); i++)
            {
                if(a[i]>tensor[i])
                {
                    throw invalid_argument("Invalid Argument");
                };
                
                
            }
                
            
        };
        
        //Matrix math
        //Matrix plus one int
        vector<double> add_int(double i)
        {
            for (int j =0; j<tensor.size(); j++)
            {
                tensor[j] += i;
            };
            return tensor;
        }
        
        //Matrix subtract one int
        vector<double> sub_int(double i)
        {
            for (int j=0; j<tensor.size(); j++)
            {
                tensor[j] -= i;
            };
            return tensor;
        }
        
        //Matrix with one scaler
        vector<double> scale_int(double i)
        {
            for (int j=0; j<tensor.size(); j++)
            {
                tensor[j] *= i;
            };
            return tensor;
        }
        
        Tensor<double> add(Tensor m, Tensor n)
        {   
            Tensor k;
            if (m.shape != n.shape)
            {
                throw invalid_argument("Invalid Argument");
            }
            else{
                for (int i=0; i<=tensor.size(); i++)
                {
                    k[i] = m[i]+n[i];
                }
                return k;
        }
        
        
};



int main()
{
    Tensor t({2, 3});
    vector s = t.dims();
    vector<double> m = t.zeros();
    
    return 0;
}
