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
#include <graphblas/platforms/sequential/LilSparseMatrix.hpp>

//****************************************************************************

namespace grb
{
    namespace backend
    {
        //********************************************************************
        template<typename ScalarT, typename... TagsT>
        class Matrix : public LilSparseMatrix<ScalarT>
        {
        private:
            using ParentMatrixType = LilSparseMatrix<ScalarT>;

        public:
            using ScalarType = ScalarT;

            // construct an empty matrix of fixed dimensions
            Matrix(IndexType   num_rows,
                   IndexType   num_cols)
                : ParentMatrixType(num_rows, num_cols)
            {
            }

            // copy construct
            Matrix(Matrix const &rhs)
                : ParentMatrixType(rhs)
            {
            }

            // construct a dense matrix from dense data.
            Matrix(std::vector<std::vector<ScalarT> > const &values)
                : ParentMatrixType(values)
            {
            }

            // construct a sparse matrix from dense data and a zero val.
            Matrix(std::vector<std::vector<ScalarT> > const &values,
                   ScalarT                                   zero)
                : ParentMatrixType(values, zero)
            {
            }

            ~Matrix() {}  // virtual?

            // necessary?
            bool operator==(Matrix const &rhs) const
            {
                return ParentMatrixType::operator==(rhs);
            }

            // necessary?
            bool operator!=(Matrix const &rhs) const
            {
                return ParentMatrixType::operator!=(rhs);
            }

            void printInfo(std::ostream &os) const
            {
                os << "Sequential Backend: ";
                ParentMatrixType::printInfo(os);
            }
        };
    }
}
