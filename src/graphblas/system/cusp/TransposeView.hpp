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


#ifndef GB_CUSP_TRANSPOSE_VIEW_HPP
#define GB_CUSP_TRANSPOSE_VIEW_HPP

#include <graphblas/system/cusp/Matrix.hpp>

namespace graphblas
{
namespace backend
{
    /**
     * @brief View a matrix as if it were transposed.
     *
     * @tparam MatrixT         Inherits from the backend matrix.
     */
    template<typename MatrixT>
    class TransposeView : public graphblas::backend::Matrix<typename MatrixT::ScalarType>
    {
    private:
        typedef typename graphblas::backend::Matrix<typename MatrixT::ScalarType> ParentMatrixT;
    public:
        typedef typename MatrixT::ScalarType ScalarType;

        // CONSTRUCTORS

        TransposeView(MatrixT const &matrix)
            : ParentMatrixT(matrix)
        {
            thrust::swap(this->row_indices, this->column_indices);
            thrust::swap(this->num_rows, this->num_cols);
            this->sort_by_row_and_column();
        }
    };

} //backend
} // graphblas

#endif
