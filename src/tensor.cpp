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
            for (int a = 0; a<(int)tensor.size(); a++)
            {
                tensor[a] = 0.0;
            }
        };

        Tensor(const vector<int>& dims)
        {
            shape = dims;
            int total = 1;
            for (int d: shape)
            {
                total *= d;
            }
            tensor.resize(total);
            for (int a = 0; a<(int)tensor.size(); a++)
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
        
        //Reshape a tensor without copying data - returns a new tensor with a different shape
        Tensor view(vector<int> newshape)
        {
            int total_new = 1;
            for (int i = 0; i<newshape.size(); i++)
            {
                total_new *= newshape[i];
            }
    
            if (tensor.size()!= total_new)
            {
                throw invalid_argument("Invalid newshape");
            }
            vector<int> d = reshape(newshape);
            printRecursive(tensor, newshape);
            return d;
            
        };
        //slicing helper
        vector<double> slice(vector<double>& v, int start, int end)
            {
                return vector<double>(v.begin() + start, v.begin() + end);
            }
        void printRecursive(vector<double> data, vector<int> sh)
            {
                if (sh.size()==1)
                {
                    cout<< "[";
                    for (int i=0; i<sh[0];i++)
                    {
                        cout<<data[i];
                        if (i<sh[0]-1) cout<<",";
                        
                    }
                    cout<<"]";
                }
                else
                {
                    int chunk = data.size() / sh[0];
                    cout<< "[";
                    vector<int> rest_shape(sh.begin()+1, sh.end());
                    for(int i=0; i<sh[0]; i++)
                    {
                        vector<double> chunk_data = slice(data, i*chunk, i*chunk+chunk);
                        printRecursive(chunk_data, rest_shape);
                        if(i<sh[0]-1) cout<< "\n";
                        
                    }
                    cout <<"]";
                }
            }
        
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
        
        //Add two tensor
        vector<double> add(Tensor m, Tensor n)
        {   
            vector<double> k;
            if (m.shape != n.shape)
            {
                throw invalid_argument("Invalid Argument");
            }
            else{
                for (int i=0; i<=tensor.size(); i++)
                {
                    k[i] = m.tensor[i]+n.tensor[i];
                }
                return k;
        }
        
        
        //transpose
        //suppose we are using 2D matrix.
        vector<double> transpose(Tensor a)
        {
            if (a.shape.size() !=2)
            {
                throw invalid_argument("Only accept 2D matrix");
            }
            else
            {
                int rows = a.shape[0];
                int cols = a.shape[1];
                vector<double> result(rows*cols);
                for (int r = 0; r< rows; r++)
                {
                    for (int c = 0; c< cols; c++)
                    {
                        result[c*rows + r] = a.tensor[r*cols+c]
                    }
                }
                return result;
            }
            
        }
        //dot product
        
        //determinant
        
        
                        
        
        
        
};



int main()
{
    Tensor t({2, 3});
    t.view({3,2});
    
}
