/*
 * Copyright (c) 2017 Carnegie Mellon University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY EXPRESSLY DISCLAIMS TO THE FULLEST EXTENT PERMITTED BY 
 * LAW ALL EXPRESS, IMPLIED, AND STATUTORY WARRANTIES, INCLUDING, WITHOUT 
 * LIMITATION, THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
 * PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */

#ifndef GB_SEQUENTIAL_LILSPARSEMATRIX_HPP
#define GB_SEQUENTIAL_LILSPARSEMATRIX_HPP

#include <iostream>
#include <vector>
#include <typeinfo>

namespace GraphBLAS
{
    template<typename ScalarT, typename... TagsT>
    class LilSparseMatrix
    {
    public:
        typedef ScalarT ScalarType;
        
        // Constructor
        LilSparseMatrix(IndexType num_rows,
                        IndexType num_cols)
        : m_num_rows(num_rows), m_num_cols(num_cols)
        {
            m_data.resize(m_num_rows);
            m_nnz = 0;
        }
        
        // Constructor - copy
        LilSparseMatrix(LilSparseMatrix<ScalarT> const &rhs)
        {
            if (this != &rhs)
            {
                m_num_rows = rhs.m_num_rows;
                m_num_cols = rhs.m_num_cols;
                m_nnz = rhs.m_nnz;
                m_data = rhs.m_data;
            }
        }
        
        // Constructor - from dense matrix
        LilSparseMatrix(std::vector<std::vector<ScalarT>> const &val)
            : m_num_rows(val.size()),
              m_num_cols(val[0].size())
        {
            m_data.resize(m_num_rows);
            m_nnz = 0;
            for (IndexType ii = 0; ii < m_num_rows; ii++)
            {
                for (IndexType jj = 0; jj < m_num_cols; jj++)
                {
                    m_data[ii].push_back(std::make_tuple(jj, val[ii][jj]));
                    m_nnz = m_nnz + 1;
                }
            }
        }
        
        // Constructor - from dense matrix, without implied zeros
        LilSparseMatrix(std::vector<std::vector<ScalarT>> const &val,
                        ScalarT const zero)
            : m_num_rows(val.size()),
              m_num_cols(val[0].size())
        {
            m_data.resize(m_num_rows);
            m_nnz = 0;
            for (IndexType ii = 0; ii < m_num_rows; ii++)
            {
                for (IndexType jj = 0; jj < m_num_cols; jj++)
                {
                    if (val[ii][jj] != zero)
                    {
                        m_data[ii].push_back(std::make_tuple(jj, val[ii][jj]));
                        m_nnz = m_nnz + 1;
                    }
                }
            }
        }
        
        // Destructor
        ~LilSparseMatrix()
        {}
        
        // Number of rows
        void nrows(IndexType &num_rows) const
        {
            num_rows = m_num_rows;
        }
        
        // Number of columns
        void ncols(IndexType &num_cols) const
        {
            num_cols = m_num_cols;
        }
        
        // Number of non-zeroes
        IndexType nnz() const
        {
            return m_nnz;
        }
        
        // Get value at index
        ScalarT get_value_at(IndexType irow,
                             IndexType icol) const
        {
            if (irow >= m_num_rows || icol >= m_num_cols)
            {
                throw DimensionException("get_value_at: index out of bounds");
            }
            IndexType ind;
            ScalarT val;
            
            for (auto tupl : m_data[irow])		// Range-based loop, access by value
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
        void set_value_at(IndexType irow, IndexType icol, ScalarT const &val)
        {
            if (irow >= m_num_rows || icol >= m_num_cols)
            {
                throw DimensionException("set_value_at: index out of bounds");
            }
            
            if (m_data[irow].empty())
            {
                m_data[irow].push_back(std::make_tuple(icol, val));
                m_nnz = m_nnz + 1;
            }
            else
            {
                typename std::vector<std::tuple<IndexType, ScalarT>>::iterator it;
                for (it = m_data[irow].begin(); it != m_data[irow].end(); it++)
                {
                    if (std::get<0>(*it) == icol)
                    {
                        it = m_data[irow].erase(it);
                        m_data[irow].insert(it, std::make_tuple(icol, val));
                        return;
                    }
                    else if (std::get<0>(*it) > icol)
                    {
                        m_data[irow].insert(it, std::make_tuple(icol, val));
                        m_nnz = m_nnz + 1;
                        return;
                    }
                }
                m_data[irow].push_back(std::make_tuple(icol, val));
                m_nnz = m_nnz + 1;
            }
        }
        
        // Get column indices for a given row
        void getColumnIndices(IndexType irow, IndexArrayType &v) const
        {
            if (irow >= m_num_rows)
            {
                throw DimensionException("getColumnIndices: index out of bounds");
            }
            
            if (!m_data[irow].empty())
            {
                IndexType ind;
                ScalarT val;
                v.resize(0);
                
                for (auto tupl : m_data[irow])
                {
                    std::tie(ind, val) = tupl;
                    v.push_back(ind);
                }
            }
        }
        
        // Get row indices for a given column
        void getRowIndices(IndexType icol, IndexArrayType &v) const
        {
            if (icol >= m_num_cols)
            {
                throw DimensionException("getRowIndices: index out of bounds");
            }
            
            IndexType ind;
            ScalarT val;
            v.resize(0);
            
            for (IndexType ii = 0; ii < m_num_rows; ii++)
            {
                if (!m_data[ii].empty())
                {
                    for (auto tupl : m_data[ii])
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
        
        // EQUALITY OPERATORS
        /**
         * @brief Equality testing for LilMatrix.
         * @param rhs The right hand side of the equality operation.
         * @return If this LilMatrix and rhs are identical.
         */
        bool operator==(LilSparseMatrix<ScalarT> const &rhs) const
        {
            if ((m_num_rows != rhs.m_num_rows) ||
                (m_num_cols != rhs.m_num_cols) ||
                (m_nnz != rhs.m_nnz))
            {
                return false;
            }
            IndexArrayType thisIndex;
            IndexArrayType rhsIndex;
            for (IndexType ii = 0; ii < m_num_rows; ii++)
            {
                getColumnIndices(ii, thisIndex);
                rhs.getColumnIndices(ii, rhsIndex);
                if (thisIndex != rhsIndex)
                {
                    return false;
                }
                else
                {
                    if (thisIndex.empty())
                    {
                        for (IndexType jj = 0; jj < thisIndex.size(); jj++)
                        {
                            if (get_value_at(ii, thisIndex[jj]) != rhs.get_value_at(ii, thisIndex[jj]))
                            {
                                return false;
                            }
                        }
                    }
                }
            }
            return true;
        }
        
        /**
         * @brief Inequality testing for LilMatrix.
         * @param rhs The right hand side of the inequality operation.
         * @return If this LilMatrix and rhs are not identical.
         */
        bool operator!=(LilSparseMatrix<ScalarT> const &rhs) const
        {
            return !(*this == rhs);
        }
        
        // output specific to the storage layout of this type of matrix
        void print_info(std::ostream &os) const
        {
            os << "LilSparseMatrix<" << typeid(ScalarT).name() << ">" << std::endl;
            os << "dimensions: " << m_num_rows << " x " << m_num_cols
            << std::endl;
            os << "num nonzeros = " << m_nnz << std::endl;
            for (IndexType row = 0; row < m_data.size(); ++row)
            {
                os << row << " :";
                for (auto it = m_data[row].begin();
                     it != m_data[row].end();
                     ++it)
                {
                    os << " " << std::get<0>(*it)
                    << ":" << std::get<1>(*it);
                }
                os << std::endl;
            }
        }
        
        friend std::ostream &operator<<(std::ostream             &os,
                                        LilSparseMatrix<ScalarT> const &mat)
        {
            mat.print_info(os);
            return os;
        }
    
    private:
        IndexType m_num_rows;    // Number of rows
        IndexType m_num_cols;    // Number of columns
        IndexType m_nnz;  // Number of non-zero values
        
        // List-of-lists storage (LIL)
        std::vector<std::vector<std::tuple<IndexType, ScalarT>>> m_data;
    };
    

}

#endif // GB_SEQUENTIAL_LILSPARSEMATRIX_HPP
