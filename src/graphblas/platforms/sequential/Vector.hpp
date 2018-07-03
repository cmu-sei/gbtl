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
