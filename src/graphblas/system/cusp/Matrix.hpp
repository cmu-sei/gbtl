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
#include <iostream>

#include <graphblas/detail/config.hpp>
#include <cusp/coo_matrix.h>

namespace graphblas
{
    namespace backend
    {
        //ignoring all tags here, coo matrices only.
        template<typename ScalarT, typename... TagsT>
        class Matrix : public cusp::coo_matrix <IndexType, ScalarT, cusp::device_memory>
        {
        private:
            typedef typename cusp::coo_matrix <IndexType, ScalarT, cusp::device_memory> ParentMatrixT;
        public:
            typedef ScalarT ScalarType;
            ScalarType zero_value;
            //Matrix() = delete;
            Matrix(): ParentMatrixT(){};

            typedef typename ::cusp::device_memory MemorySpace;
            Matrix(const IndexType rows, const IndexType columns)
                : ParentMatrixT(rows, columns, 0) {}

            //use parent copy constructor:
            template<typename MatrixT>
            Matrix(const MatrixT &matrix)
                : ParentMatrixT(matrix) {}

            //use parent constructor with num vals:
            template<typename RowIndexType,
                     typename ColumnIndexType,
                     typename ValueType >
            Matrix(const RowIndexType rows, const ColumnIndexType columns, const ValueType num_vals)
                : ParentMatrixT(rows, columns, num_vals) {}

            //set:
            template <typename ZeroT>
            void set_zero(ZeroT z){
                this->zero_value = static_cast<ScalarType>(z);
            }

            IndexType get_nnz() const{
                return static_cast<IndexType>(this->num_entries);
            }

            ScalarT get_zero() const{
                return this->zero_value;
            }

            template <typename T1, typename T2>
            void get_shape(T1 &t1, T2 &t2) const{
                t1 = this->num_rows;
                t2 = this->num_cols;
            }

        };
    } // backend
} // graphblas
