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

//****************************************************************************
//****************************************************************************

namespace grb
{
    //************************************************************************
    template<typename MatrixT>
    class MatrixStructureView
    {
    public:
        using ScalarType = bool;

        MatrixStructureView(MatrixT const &mat)
            : m_mat(mat)
        {
        }

        IndexType nrows() const { return m_mat.nrows(); }
        IndexType ncols() const { return m_mat.ncols(); }
        IndexType nvals() const { return m_mat.nvals(); }

        void printInfo(std::ostream &os) const
        {
            os << "MatrixStructureView of: ";
            m_mat.printInfo(os);
        }

        friend std::ostream &operator<<(std::ostream               &os,
                                        MatrixStructureView const &mat)
        {
            os << "MatrixStructureView of: ";
            os << mat.m_mat;
            return os;
        }

        MatrixT const &m_mat;
    };

    //************************************************************************
    template <class ViewT,
              typename std::enable_if_t<is_structure_v<ViewT>, int> = 0>
    decltype(auto)
    get_internal_matrix(ViewT const &view)
    {
        return MatrixStructureView(get_internal_matrix(view.m_mat));
    }


    //************************************************************************
    //************************************************************************

    //************************************************************************
    template<typename VectorT>
    class VectorStructureView
    {
    public:
        using ScalarType = bool;

        VectorStructureView(VectorT const &vec)
            : m_vec(vec)
        {
        }

        IndexType size()  const { return m_vec.size(); }
        IndexType nvals() const { return m_vec.nvals(); }

        void printInfo(std::ostream &os) const
        {
            os << "VectorStructureView of: ";
            m_vec.printInfo(os);
        }

        friend std::ostream &operator<<(std::ostream               &os,
                                        VectorStructureView const &vec)
        {
            os << "VectorStructureView of: ";
            os << vec.m_vec;
            return os;
        }

        VectorT const &m_vec;
    };

    //************************************************************************
    template <class ViewT,
              typename std::enable_if_t<is_structure_v<ViewT>, int> = 0>
    decltype(auto)
    get_internal_vector(ViewT const &view)
    {
        return VectorStructureView(get_internal_vector(view.m_vec));
    }
}
