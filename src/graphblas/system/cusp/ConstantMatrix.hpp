/*
 * Copyright (c) 2015 Carnegie Mellon University and The Trustees of Indiana
 * University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY AND THE TRUSTEES OF INDIANA UNIVERSITY EXPRESSLY DISCLAIM
 * TO THE FULLEST EXTENT PERMITTED BY LAW ALL EXPRESS, IMPLIED, AND STATUTORY
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */

#ifndef GB_CUSP_CONSTANTMATRIX_HPP
#define GB_CUSP_CONSTANTMATRIX_HPP

#include <iostream>
#include <vector>
#include "detail/utility.inl"

namespace graphblas
{
    namespace backend{
    namespace detail{
        template <typename T>
        class ConstantVector : public cusp::array1d<T, cusp::device_memory>
        {
            __host__  ConstantVector ()
            {
            }

            __host__  ConstantVector (T n, const T &value)
            {
            }

            __host__  ConstantVector (const ConstantVector &v)
            {
            }
            template <typename Num>
            T& operator[](const Num n)
            {
                return this->data[0];
            }

            template <typename Num>
            T& operator[](const Num n) const
            {
                return this->data[0];
            }
        }
    }
    }
    /// @todo Should we support assignment by row?

    /**
     * @brief Class representing a list of lists format sparse matrix.
     */
    template<typename ScalarT>
    class ConstantMatrix : public graphblas::backend::Matrix<ScalarT>
    {
    private:
        typedef typename graphblas::backend::Matrix<ScalarT> ParentMatrixT;

    public:

        /**
         * @brief Construct a matrix whose entries are all same value
         *        with the given shape.
         *
         * @param[in] num_rows  Number of rows in the matrix
         * @param[in] num_cols  Number of cols in the matrix
         * @param[in] value     The value to "fill" the matrix with.
         */
        ConstantMatrix(IndexType      num_rows,
                       IndexType      num_cols,
                       ScalarT const &value)
            : ParentMatrixT(num_rows, num_cols, 1),
              m_value(value)
        {
        }

        ~ConstantMatrix()
        {
        }

    private:
        ScalarT   m_value;
    };
} // graphblas

#endif // GB_SEQUENTIAL_CONSTANTMATRIX_HPP
