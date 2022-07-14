/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors.
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
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DM20-0442
 */

#pragma once

#include <cstddef>
#include <iostream>

#include <graphblas/detail/config.hpp>
#include <vector>
#include <graphblas/platforms/optimized_sequential/BitmapSparseVector.hpp>

namespace grb
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
            using ParentVectorType = BitmapSparseVector<ScalarT>;

        public:
            using ScalarType = ScalarT;

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
                return ParentVectorType::operator==(rhs);
            }

            // necessary?
            bool operator!=(Vector const &rhs) const
            {
                return ParentVectorType::operator!=(rhs);
            }

            void printInfo(std::ostream &os) const
            {
                os << "Optimized Sequential Backend: ";
                ParentVectorType::printInfo(os);
            }
        };
    }
}
