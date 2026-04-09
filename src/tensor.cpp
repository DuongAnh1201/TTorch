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
        
        vector<double> custom(int j)
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
        
        void print()
        {
            printRecursive(tensor, shape);
            cout << endl;
        }
        
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
        
        //Create Tensor with custom value
        vector<double> value(vector<double> a)
        {
            if (a.size() != tensor.size())
            {
                throw invalid_argument("The size of the argument is not compatiple");
            }
            else
            {
                for (int i=0; i<tensor.size(); i++)
                {
                    tensor[i] = a[i];
                }
            }
            return tensor;
        }
        
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
        //at method
        double at(vector<int> index){
            if(index.size() != shape.size()){
                throw invalid_argument("The index is not compatible to the tensor");
            }
            if(index.size() == 1){
                return tensor[index[0]];
            }
            else{
                return tensor[index[0]*shape[1]+index[1]];
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
        Tensor add(Tensor n)
        {   
            vector<double> k;
            if (shape != n.shape())
            {
                throw invalid_argument("Two tensors are not compatible to add");
            }
            else{
                Tensor result(shape);
                for (int i=0; i<=tensor.size(); i++)
                {
                    result.tensor[i] = tensor[i]+n.tensor[i];
                }
                return k;
            }
        };
        
        
        //transpose
        //suppose we are using 2D matrix.
        Tensor transpose()
        {
            if (shape.size() !=2)
            {
                throw invalid_argument("Only accept 2D matrix");
            }
            else
            {
                int rows = shape[0];
                int cols = shape[1];
                Tensor result({rows,cols});
                for (int r = 0; r< rows; r++)
                {
                    for (int c = 0; c< cols; c++)
                    {
                        result.tensor[c*rows + r] = tensor[r*cols+c];
                    }
                }
                return result;
            }  
        }
        //dot product
        //Suppose that 2x2D matrix only
        Tensor dot(Tensor b)
        {
            int row_a = shape[0];
            int col_a = shape[1];
            int row_b = b.shape[0];
            int col_b = b.shape[1];
            if (col_a!=row_b)
            {
                throw invalid_argument("Two tensors are not compatiple to do the dot product");
            }
            else
            {
                Tensor result({row_a,col_b});
                for(int i=0; i<row_a; i++)
                {
                    for(int j=0; j<col_b; j++)
                    {
                        for(int k=0; k<col_a; k++)
                        {
                            result.tensor[i*col_b + j] += tensor[col_a*i+k]*b.tensor[col_b*k+j]; 
                        }
                    }
                }
                return result;
                
            }
        }
    
        //Multiply two tensor
        Tensor multiply(Tensor b){
            if (shape != b.shape()){
                throw invalid_argument("Two tensors are not compatible to multiply");
            }
            else{
                Tensor result({shape[0], shape[1]});
                for (int i=0; i<tensor.size(); i++){
                    result.tensor[i] = tensor[i] * b.tensor[i];
                }
                return result;

                }
                }
        
        //determinant
        
        
                        
        
        
        
};




int main()
{
    Tensor m({2, 3});
    Tensor n({3,2});
    m.value({1,2,3,4,5,6});
    n.value({1,4,2,5,3,6});
    Tensor k = m.dot(n);
    k.print();
}
