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
#include <vector>
#include <graphblas/platforms/sequential/BitmapSparseVector.hpp>

namespace GraphBLAS
{
    namespace backend
    {
        //**********************************************************************
        /// @note ignoring all tags here, there is currently only one
        ///       implementation of vector: dense+bitmap.
        template<typename ScalarT, typename... TagsT>
        class Vector : public BitmapSparseVector<ScalarT>
        {
        private:
            typedef BitmapSparseVector<ScalarT> ParentVectorType;

        public:
            typedef ScalarT ScalarType;

            Vector() = delete;

            Vector(IndexType nsize) : ParentVectorType(nsize) {}

            Vector(IndexType const &nsize, ScalarT const &value)
                : ParentVectorType(nsize, value) {}

            Vector(std::vector<ScalarT> const &values)
                : ParentVectorType(values) {}

            Vector(std::vector<ScalarT> const &values, ScalarT const &zero)
                : ParentVectorType(values, zero) {}

            ~Vector() {}  // virtual?

            // necessary?
            bool operator==(Vector const &rhs) const
            {
                return BitmapSparseVector<ScalarT>::operator==(rhs);
            }

            // necessary?
            bool operator!=(Vector const &rhs) const
            {
                return BitmapSparseVector<ScalarT>::operator!=(rhs);
            }

            void printInfo(std::ostream &os) const
            {
                os << "Sequential Backend:" << std::endl;
                BitmapSparseVector<ScalarT>::printInfo(os);
            }
        };
    }
}
