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

#include <cstddef>
#include <graphblas/system/sequential/LilMatrix.hpp>

namespace graphblas
{
    namespace backend
    {
        template<typename ScalarT, typename... TagsT>
        class Matrix : public LilMatrix<ScalarT>
        {
        public:
            typedef ScalarT ScalarType;

            // construct an empty matrix of fixed dimensions
            Matrix(IndexType   num_rows,
                   IndexType   num_cols,
                   ScalarT const &zero = static_cast<ScalarT>(0))
                : LilMatrix<ScalarT>(num_rows, num_cols, zero)
            {
            }

            // construct a matrix from dense data.
            Matrix(std::vector<std::vector<ScalarT> > const &values,
                   ScalarT const &zero = static_cast<ScalarT>(0))
                : LilMatrix<ScalarT>(values, zero)
            {
            }

            // copy construct
            Matrix(Matrix const &rhs)
                : LilMatrix<ScalarT>(rhs)
            {
            }

            //default constructor for constmat
            Matrix(): LilMatrix<ScalarT>(1,1,0) {}

            ~Matrix()
            {
            }

            void print_info(std::ostream &os) const
            {
                os << "Sequential Backend:" << std::endl;
                LilMatrix<ScalarT>::print_info(os);
            }
        };
    }
}
