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
#include <graphblas/Matrix.hpp>

#define GB_INCLUDE_BACKEND_TRANSPOSE_VIEW 1
#include <backend_include.hpp>

//****************************************************************************
//****************************************************************************


namespace grb
{    //************************************************************************
    template<typename MatrixT>
    class TransposeView
    {
    public:
        using ScalarType = typename MatrixT::ScalarType;

        TransposeView(MatrixT const &mat)
            : m_mat(mat)
        {
        }

        IndexType nrows() const { return m_mat.ncols(); }
        IndexType ncols() const { return m_mat.nrows(); }

        void printInfo(std::ostream &os) const
        {
            os << "TransposeView of: ";
            m_mat.printInfo(os);
        }

        friend std::ostream &operator<<(std::ostream        &os,
                                        TransposeView const &mat)
        {
            os << "TransposeView of: ";
            os << mat.m_mat;
            return os;
        }

        MatrixT const &m_mat;

    };

    //************************************************************************
    template <class ViewT,
              typename std::enable_if_t<is_transpose_v<ViewT>, int> = 0>
    decltype(auto)
    get_internal_matrix(ViewT const &view)
    {
        return TransposeView(get_internal_matrix(view.m_mat));
    }
} // end namespace grb
