#ifndef Matrix_h
#define Matrix_h

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

template <typename T>
class Matrix
{
public:
    Matrix(int m, int n, T value=0) : m(m), n(n)
    {
        matrix.resize(m, std::vector<T>(n, value));
    }
    
    Matrix(std::initializer_list<std::vector<T>> matrix) : matrix(matrix)
    {
        m = matrix.size();
        
        if (m > 0)
        {
            n = matrix[0].size();
            
            for (const auto& line : matrix)
                assert(line.size() == n);
        }
    }
    
    class MatrixRow
    {
    public:
        MatrixRow(std::vector<T>& row) : row(row) {}
        
        T& operator[](int j) const
        {
            return row.at(j);
        }
        
        void swap(MatrixRow&& other)
        {
            assert(row.size() == other.row.size());
            
            auto it = row.begin();
            auto it2 = other.row.begin();
            
            while (it != row.end())
            {
                T tmp = *it;
                
                *it = *it2;
                *it2 = tmp;
                
                it++;
                it2++;
            }
        }
        
        void divideRow(T n)
        {
            for (auto it = row.begin(); it != row.end(); it++)
                *it = *it / n;
        }
        
        void addMultipleRow(T multiple, MatrixRow&& other)
        {
            assert(row.size() == other.row.size());
            
            auto it = row.begin();
            auto it2 = other.row.begin();
            
            while (it != row.end())
            {
                *it = *it + (*it2 * multiple);
                
                it++;
                it2++;
            }
        }
        
    private:
        std::vector<T>& row;
    };
    
    MatrixRow operator[](int i)
    {
        return MatrixRow(matrix[i]);
    }
    
    Matrix<T> operator+(const Matrix<T>& other)
    {
        assert(m == other.m && n == other.n);
        
        Matrix<T> ret(other);
        
        int i = 0;
        for (const auto& line : matrix)
        {
            int j = 0;
            for (const auto& value : line)
            {
                ret[i][j] += value;
                
                j++;
            }
            i++;
        }
        
        return ret;
    }
    
    Matrix<T> operator*(Matrix<T>& other)
    {
        assert(n == other.m);
        
        Matrix<T> ret(m, other.n);
        
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < other.n; j++)
            {
                int sum = 0;
                
                for (int k = 0; k < n; k++)
                {
                    T a = matrix[i][k];
                    T b = other[k][j];
                    
                    sum += a * b;
                }
                
                ret[i][j] = sum;
            }
        }
        
        return ret;
    }
    
    Matrix<T> concat(const Matrix<T>& other)
    {
        assert(m == other.m);
        
        Matrix<T> ret(m, n + other.n);
        
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                ret[i][j] = matrix[i][j];
        
        for (int i = 0; i < m; i++)
            for (int j = 0; j < other.n; j++)
                ret[i][n + j] = other.matrix[i][j];
        
        return ret;
    }
    
    void print(bool fraction=false)
    {
        int max = 3;
        std::stringstream ss;
        
        for (const auto& line : matrix)
        {
            for (const auto& value : line)
            {
                ss.str(std::string());
                
                ss << value;
                
                ss.seekg(0, std::ios::end);
                
                max = std::max<int>(max, int(ss.tellg()));
            }
        }
        
        for (const auto& line : matrix)
        {
            std::cout << std::left << std::setfill(' ');
            
            for (const auto& value : line)
                std::cout << std::setw(max + 2) << value << ' ';
            
            std::cout << std::endl;
        }
    }
    
    int getRows()
    {
        return m;
    }
    
    int getColumns()
    {
        return n;
    }
    
private:
    std::vector<std::vector<T>> matrix;
    
    int m;
    int n;
};


#endif /* Matrix_h */
