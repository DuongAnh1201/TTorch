#include<iostream>
//#include<Eigen/Dense>
using namespace std;
// using namespace Eigen;

class Tensor
{
    private:
        Tensor() {}
        int numel()
        {
            int total = 1;
            for(int d: shape) total *=d;
            return total;
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
            
        int ndim()
        {
            return shape.size();
        }
        
        int size(int dim)
        {
            if (dim>= ndim())
            {
                throw invalid_argument("out of index");
            }
            else
            {
                return shape[dim];
            }
        }
        
    public:
        vector<int> shape;
        vector<double> data;
        Tensor(const initializer_list<int>& dims)
        {
            shape = dims;
            int total = 1;
            for (int d: shape)
            {
                total *= d;
            }
            data.resize(total);
            for (int a = 0; a<(int)data.size(); a++)
            {
                data[a] = 0.0;
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
            data.resize(total);
            for (int a = 0; a<(int)data.size(); a++)
            {
                data[a] = 0.0;
            }
        };
        
        // Tensor const tensor(vector<double> a)
        // {   
        //     if (data.size()!=a.size())
        //     {
        //         throw invalid_argument("the argument is not compatible with the initial tensor");
        //     }
        //     Tensor result(shape);
        //     for (int i = 0; i<a.size(); i++)
        //     {
        //         result.data[i] = a[i];
        //     }
        //     return result;
        // }
        vector<int> dims()
        {
            return shape;
        };
        
        static Tensor zeros(initializer_list<int> dims)
        {
            Tensor t;
            t.shape =dims;
            
           t.data.assign(t.numel(), 0.0);
           return t;
        }
        static Tensor ones(initializer_list<int> dims)
        {   
            Tensor t;
            t.shape = dims;
            
            t.data.assign(t.numel(), 1.0);
            return t;
        };
        
        static Tensor custom(initializer_list<int>dims,double j)
        {
            Tensor t;
            t.shape = dims;
            t.data.assign(t.numel(), j);
            return t;
        };
        
        static Tensor form(initializer_list<int> dims, vector<double> data)
        {
            Tensor t = Tensor::zeros(dims);
            if (data.size() != t.data.size())
            {
                throw invalid_argument("data's size is not compatible");
            }
            t.data = data;
            return t;
        }

        static Tensor form(const vector<int>& dims, vector<double> data)
        {
            Tensor t(dims);
            if (data.size() != t.data.size())
            {
                throw invalid_argument("data's size is not compatible");
            }
            t.data = data;
            return t;
        }
        //Matrix manipulation
        Tensor reshape(vector<int> newshape)
        {
            int total = 1;
            for (int d: newshape)
            {
                total*= d;
            }
            if (total != data.size())
            {
                throw invalid_argument("Invalid Reshape");
            };
            Tensor new_tensor(newshape);
            new_tensor.data = data; 
            return new_tensor;
        };
        
        void print()
        {
            printRecursive(data, shape);
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
    
            if (data.size()!= total_new)
            {
                throw invalid_argument("Invalid newshape");
            }
            Tensor d = reshape(newshape);
            printRecursive(d.data, newshape);  
            return d;  
        };
        
        //Create Tensor with custom value
        vector<double> value(vector<double> a)
        {
            if (a.size() != data.size())
            {
                throw invalid_argument("The size of the argument is not compatiple");
            }
            else
            {
                for (int i=0; i<data.size(); i++)
                {
                    data[i] = a[i];
                }
            }
            return data;
        }
        
        //slicing helper
        vector<double> slice(vector<double>& v, int start, int end)
            {
                return vector<double>(v.begin() + start, v.begin() + end);
            }
        
        //at method
        double at(vector<int> index){
            if(index.size() != shape.size()){
                throw invalid_argument("The index is not compatible to the tensor");
            }
            if(index.size() == 1){
                return data[index[0]];
            }
            else{
                return data[index[0]*shape[1]+index[1]];
            }
        }
        //Matrix math

       
        //Matrix plus one int

        Tensor add_int(double i)
        {
            Tensor t;
            t.shape = shape;
            t.data = data;
            for (int j=0; j<(int)t.data.size(); j++)
            {
                t.data[j] += i;
            };
            return t;
        }
        
        //Matrix with one scaler
        Tensor scale_int(double i)
        {
            Tensor t;
            t.shape = shape;
            t.data = data;
            for (int j=0; j<(int)t.data.size(); j++)
            {
                t.data[j] *= i;
            };
            return t;
        }
        
        //Add two tensor
        Tensor add(Tensor n)
        {   
            if (shape != n.shape)
            {
                throw invalid_argument("Two tensors are not compatible to add");
            }
            else{
                Tensor result(shape);
                for (int i=0; i<(int)result.data.size(); i++)
                {
                    result.data[i] = data[i]+n.data[i];
                }
                return result;
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
                Tensor result({cols,rows});
                for (int r = 0; r< rows; r++)
                {
                    for (int c = 0; c< cols; c++)
                    {
                        result.data[c*rows + r] = data[r*cols+c];
                    }
                }
                return result;
            }  
        }
        Tensor T()
        {
            if (shape.size()!=2)
            {
                throw invalid_argument("Only accept 2D matrix");
            }
            else
            {
                int rows = shape[0];
                int cols = shape[1];
                Tensor result({cols, rows});
                for (int r = 0; r< rows; r++)
                {
                    for (int c=0; c< rows; c++)
                    {
                        result.data[c*rows+r] = data[r*cols+c];
                    }
                }
                return result;
                
            }
        }
        //dot product
        //Suppose that 2x2D matrix only
        Tensor dot(Tensor b)
        {
            if (shape.size() !=2 && shape.size() != 1)
            {
                throw invalid_argument("Only accept 1D-2D matrix");
            }
            if (b.shape.size() != 2 && b.shape.size() != 1)
            {
                throw invalid_argument("Only accept 1D-2D matrix");
            }
            else
            {
                if (shape.size() == 1)
                {
                    int n = (int)data.size();
                    shape = vector<int> {n,1};
                }
                if (b.shape.size() == 1)
                {
                    int m = (int) data.size();
                    b.shape = vector<int> {m,1};
                }
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
                                result.data[i*col_b + j] += data[col_a*i+k]*b.data[col_b*k+j]; 
                            }
                        }
                    }
                    return result;
                }
            }
        }
    
        //Multiply two tensor
        Tensor multiply(Tensor b){
            if (shape != b.shape){
                throw invalid_argument("Two tensors are not compatible to multiply");
            }
            else{
                Tensor result(shape);
                for (int i=0; i<data.size(); i++){
                    result.data[i] = data[i] * b.data[i];
                }
                return result;

                }
                }
        
        //determinant
        
        //flatten
        Tensor flatten()
        {
            int total = 1;
            for(int d: shape)
            {
                total *=d;
            }
            Tensor result({total});
            result.data = data; //transfer the data
            return result;
        }
        //sum
        Tensor sum(int axis = 0)
        {
            if(shape.size() == 1)
            {
                if(axis == 0)
                {
                    Tensor result({1});
                    result.data[0] = 0;
                    for (int i: data)
                    {
                        result.data[0] += i;
                        return result;
                    }
                }
            }
            else
            {
                if (shape.size() == 2)
                {
                    int rows = shape[0];
                    int cols = shape[1];
                    
                    if(axis == 0)
                    {
                        Tensor result = Tensor::zeros({rows});
                        for (int r = 0; r <rows; r++)
                        {
                            for (int c = 0; c<cols; c++)
                            {
                                result.data[r] += data[r*cols + c];
                            }
                        }
                        return result;
                    }
                    else if(axis == 1)
                    {
                        Tensor result = Tensor::zeros({cols});
                        for (int c = 0; c<cols; c++)
                        {
                            for (int r= 0; r<rows; r++)
                            {
                                result.data[c] += data[r*cols + c];
                            }
                        }
                        return result;
                    }
                    else
                    {
                        throw invalid_argument("Can't compute for larger then 2D matrix");
                    }
                }
            }
        }
        
        

        //mean
        Tensor mean(int axis = 0)
        {
            if(shape.size() == 1)
            {
                Tensor result({1});
                Tensor m = sum();
                int num = numel();
                result.data[0] = m.data[0]/num;
                return result;
            }
            else
            {
                if (shape.size() == 2)
                {
                
                    int rows = shape[0];
                    int cols = shape[1];
                    if(axis == 0){ 
                        Tensor m = sum();
                        Tensor result = Tensor::zeros({rows});
                    
                        for (int r = 0; r<rows; r++)
                        {
                            result.data[r] = m.data[r]/cols;
                        }
                        return result;
                    }
                    if(axis == 1)
                    {
                        Tensor m = sum(1);
                        Tensor result = Tensor::zeros({cols});
                        for(int c = 0; c<cols; c++)
                        {
                            result.data[c] = m.data[c]/rows;
                        }
                        return result;
                    }
                }
                
            }
        }
        //max
        
                        
        
};




int main()
{
    Tensor m({2, 3});
    Tensor n({3,2});
    Tensor q({2, 3});
    m.value({1,2,3,4,5,6});
    m.print();
    n.value({1,4,2,5,3,6});
    q.value({1,2,3,4,5,6});
    Tensor k= m.add(q);
    k.print();
    Tensor j = k.flatten();
    j.print();
    Tensor a = m.transpose();
    a.print();
    Tensor b = a.sum(1);
    b.print();
    // Tensor k = m.dot(n);
    // k.print();
    // k.view({1,4});
    // Tensor q = k.reshape({1,4});
    // q.print();
}
