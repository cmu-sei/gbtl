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


#pragma once
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
    class TransposeView
    {
    private:
        typedef typename graphblas::backend::Matrix<typename MatrixT::ScalarType> ParentMatrixT;
    public:
        typedef typename MatrixT::ScalarType ScalarType;

        // CONSTRUCTORS

        TransposeView(MatrixT const &matrix)
        {
        }
    };

} //backend
} // graphblas
