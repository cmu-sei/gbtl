/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
 */

/**
 * Implementation of all sparse reduce variants for the sequential (CPU) backend.
 */

#ifndef GB_SEQUENTIAL_SPARSE_REDUCE_HPP
#define GB_SEQUENTIAL_SPARSE_REDUCE_HPP

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <iostream>
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"

//****************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        //********************************************************************
        /// Implementation of 4.3.9.1 reduce: Standard Matrix to Vector variant
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  // monoid or binary op only
                 typename AMatrixT>
        inline Info reduce(WVectorT        &w,
                           MaskT     const &mask,
                           AccumT           accum,
                           BinaryOpT        op,
                           AMatrixT  const &A,
                           bool             replace_flag = false)
        {
            // =================================================================
            // Do the basic reduction work with the binary op
            typedef typename BinaryOpT::result_type D3ScalarType;
            typedef typename AMatrixT::ScalarType AScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> >  ARowType;

            std::vector<std::tuple<IndexType, D3ScalarType> > t;

            if (A.nvals() > 0)
            {
                for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
                {
                    /// @todo Can't be a reference because A might be transpose
                    /// view.  Need to specialize on TransposeView and getCol()
                    ARowType const A_row(A.getRow(row_idx));

                    /// @todo There is something hinky with domains here.  How
                    /// does one perform the reduction in A domain but produce
                    /// partial results in D3(op)?
                    D3ScalarType t_val;
                    if (reduction(t_val, A_row, op))
                    {
                        t.push_back(std::make_tuple(row_idx, t_val));
                    }
                }
            }

            // =================================================================
            // Accumulate into Z
            // Type generator for z: D3(accum), or D(w) if no accum.
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                D3ScalarType,
                typename AccumT::result_type>::type  ZScalarType;
            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_opt_accum_1D(z, w, t, accum);

            // =================================================================
            // Copy Z into the final output, w, considering mask and replace
            write_with_opt_mask_1D(w, z, mask, replace_flag);
            return SUCCESS;
        }

        //********************************************************************
        /// Implementation of 4.3.9.2 reduce: Vector to scalar variant
        template<typename ValueT,
                 typename AccumT,
                 typename MonoidT, // monoid only
                 typename UVectorT>
        inline Info reduce_vector_to_scalar(ValueT         &val,
                                            AccumT          accum,
                                            MonoidT         op,
                                            UVectorT const &u)
        {
            // =================================================================
            // Do the basic reduction work with the monoid
            typedef typename MonoidT::result_type D3ScalarType;
            typedef typename UVectorT::ScalarType UScalarType;
            typedef std::vector<std::tuple<IndexType,UScalarType> >  UColType;

            D3ScalarType t = op.identity();

            if (u.nvals() > 0)
            {
                UColType const u_col(u.getContents());

                reduction(t, u_col, op);
            }

            // =================================================================
            // Accumulate into Z
            /// @todo Do we need a type generator for z: D(w) if no accum,
            /// or D3(accum). I think that D(z) := D(val) should be equivalent, but
            /// still need to work the proof.
            ValueT z;
            opt_accum_scalar(z, val, t, accum);

            // Copy Z into the final output
            val = z;
            return SUCCESS;
        }

        //********************************************************************
        /// Implementation of 4.3.9.3 reduce: Matrix to scalar variant
        template<typename ValueT,
                 typename AccumT,
                 typename MonoidT, // monoid only
                 typename AMatrixT>
        inline Info reduce_matrix_to_scalar(ValueT         &val,
                                            AccumT          accum,
                                            MonoidT         op,
                                            AMatrixT const &A)
        {
            // =================================================================
            // Do the basic reduction work with the monoid
            typedef typename MonoidT::result_type D3ScalarType;
            typedef typename AMatrixT::ScalarType AScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> >  ARowType;

            D3ScalarType t = op.identity();

            if (A.nvals() > 0)
            {
                for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
                {
                    /// @todo Can't be a reference because A might be transpose
                    /// view.  Need to specialize on TransposeView and getCol()
                    ARowType const A_row(A.getRow(row_idx));

                    /// @todo There is something hinky with domains here.  How
                    /// does one perform the reduction in A domain but produce
                    /// partial results in D3(op)?
                    D3ScalarType tmp;
                    if (reduction(tmp, A_row, op))
                    {
                        t = op(t, tmp); // reduce each row
                    }
                }
            }

            // =================================================================
            // Accumulate into Z
            /// @todo Do we need a type generator for z: D(w) if no accum,
            /// or D3(accum). I think that D(z) := D(val) should be equivalent, but
            /// still need to work the proof.
            ValueT z;
            opt_accum_scalar(z, val, t, accum);

            // Copy Z into the final output
            val = z;
            return SUCCESS;
        }

    } // backend
} // GraphBLAS

#endif
