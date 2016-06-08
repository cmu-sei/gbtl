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

#include <cusp/transpose.h>
#include "./merge.inl"
#include "../TransposeView.hpp"


namespace graphblas
{
namespace backend
{
    template<typename AMatrixT,
             typename CMatrixT>
    inline void transpose(const AMatrixT &a,
                    CMatrixT       &c)
    {
        /*
           c.resize(a.num_cols, a.num_rows, a.num_entries);
           c.row_indices = a.column_indices;
           c.column_indices = a.row_indices;
           c.values = a.values;
           */
        ::cusp::transpose<AMatrixT, CMatrixT>(a, c);
    }

    template<typename MatrixT>
    inline TransposeView<MatrixT> transpose(MatrixT const &a)
    {
        return TransposeView<MatrixT>(a);
    }
}
} // graphblas
