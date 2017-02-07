/*
 LilMatrix_sparse.hpp
 SEI 2017
 Oren Wright
 
 */

#ifndef GB_SEQUENTIAL_LILMATRIX_HPP
#define GB_SEQUENTIAL_LILMATRIX_HPP

#include <iostream>
#include <vector>
#include <typeinfo>

namespace GraphBLAS
{
    template<typename scalarT, typename... tagsT>
    class LilMatrix_sparse
    {
    private:
        IndexType m;    // Number of rows
        IndexType n;    // Number of columns
        IndexType nnz;  // Number of non-zero values
        
        // List-of-lists storage (LIL)
        std::vector<std::vector<std::tuple<IndexType, scalarT>>> lil;
        
    public:
        typedef scalarT ScalarType;
        
        // Constructor
        LilMatrix(IndexType m_in,
                  IndexType n_in)
        : m(m_in), n(n_in)
        {
            lil.resize(m);
            nnz = 0;
        }
        
        // Constructor - copy
        LilMatrix(LilMatrix<scalarT> const &lil_in)
        {
            if (this != &lil_in)
            {
                m = lil_in.m;
                n = lil_in.n;
                nnz = lil_in.nnz;
                lil = lil_in.lil;
            }
        }
        
        // Constructor - from dense matrix
        LilMatrix(std::vector<std::vector<scalarT>> const &val): m(val.size()), n(val[0].size())
        {
            lil.resize(m);
            nnz = 0;
            for (IndexType ii = 0; ii < m; ii++)
            {
                for (IndexType jj = 0; jj < n; jj++)
                {
                    if (val[ii][jj] != zero)
                    {
                        lil[ii].push_back(std::make_tuple(jj, val[ii][jj]));
                        nnz = nnz + 1;
                    }
                }
            }
        }
        
        // Destructor
        ~LilMatrix()
        {}
        
        // Number of rows
        void nrows(IndexType &m_out)
        {
            m_out = m;
        }
        
        // Number of columns
        void ncols(IndexType &n_out)
        {
            n_out = n;
        }
        
        // Number of non-zeroes
        IndexType get_nnz() const
        {
            return nnz;
        }
        
        // Get value at index
        scalarT get_value_at(IndexType irow,
                             IndexType icol) const
        {
            if (irow >= m || icol >= n)
            {
                throw DimensionException("get_value_at: index out of bounds");
            }
            IndexType ind;
            scalarT val;
            
            for (auto tupl : lil[irow])		// Range-based loop, access by value
            {
                std::tie(ind, val) = tupl;
                if (ind == icol)
                {
                    return val;
                }
            }
            throw DimensionException("get_value_at: no entry at index");
        }
        
        // Set value at index
        void set_value_at(IndexType irow, IndexType icol, scalarT const &val)
        {
            if (irow >= m || icol >= n)
            {
                throw DimensionException("set_value_at: index out of bounds");
            }
            
            if (lil[irow].is_empty())
            {
                lil[irow].push_back(std::make_tuple(icol, val));
            }
            else
            {
                typename std::vector<std::tuple<IndexType, scalarT>>::iterator it;
                for (it = lil[irow].begin(); it != lil[irow].end(); it++)
                {
                    if (std::get<0>(*it) == icol)
                    {
                        it = lil[irow].erase(it);
                        lil[irow].insert(it, std::make_tuple(icol, val));
                        return;
                    }
                    else if (std::get<0>(*it) > icol)
                    {
                        lil[irow].insert(it, std::make_tuple(icol, val));
                        return;
                    }
                }
                lil[irow].push_back(std::make_tuple(icol, val));
            }
        }
        
        // Get column indices for a given row
        void getColumnIndices(IndexType irow, IndexArrayType &v) const
        {
            if (irow >= m)
            {
                throw DimensionException("getColumnIndices: index out of bounds");
            }
            
            if (!lil[irow].is_empty())
            {
                IndexType ind;
                scalarT val;
                
                for (auto tupl : lil[irow])
                {
                    std::tie(ind, val) = tupl;
                    v.push_back(ind);
                }
            }
        }
        
        // Get row indices for a given column
        void getRowIndices(IndexType icol, IndexArrayType &v) const
        {
            if (icol >= n)
            {
                throw DimensionException("getRowIndices: index out of bounds");
            }
            
            IndexType ind;
            scalarT val;
            
            for (IndexType ii = 0; ii < m; ii++)
            {
                if (!lil[ii].is_empty())
                {
                    for (auto tupl : lil[ii])
                    {
                        std::tie(ind, val) = tupl;
                        if (ind == icol)
                        {
                            v.push_back(ii);
                            break;
                        }
                        if (ind > icol)
                        {
                            break;
                        }
                    }
                }
            }
        }
        
        // output specific to the storage layout of this type of matrix
        void print_info(std::ostream &os) const
        {
            os << "LilMatrix_sparse<" << typeid(scalarT).name() << ">" << std::endl;
            os << "dimensions: " << m << " x " << n
            << std::endl;
            os << "num nonzeros = " << nnz << std::endl;
            for (IndexType row = 0; row < lil.size(); ++row)
            {
                os << row << " :";
                for (auto it = lil[row].begin();
                     it != lil[row].end();
                     ++it)
                {
                    os << " " << std::get<0>(*it)
                    << ":" << std::get<1>(*it);
                }
                os << std::endl;
            }
        }
        
        friend std::ostream &operator<<(std::ostream             &os,
                                        LilMatrix<scalarT> const &mat)
        {
            mat.print_info(os);
            return os;
        }
    };

}

#endif // GB_SEQUENTIAL_LILMATRIX_HPP
