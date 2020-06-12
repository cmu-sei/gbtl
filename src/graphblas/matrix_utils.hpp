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

#include <functional>
#include <vector>

#include <graphblas/graphblas.hpp>

//****************************************************************************
// Miscellaneous matrix convenience functions
//****************************************************************************
namespace grb
{
    //************************************************************************
    /**
     * @brief Constuct and return a matrix with elements on the diagonal.
     *
     * @param[in] v    The elements to put on the diagonal of a matrix.
     */
    template<typename MatrixT, typename VectorT>
    MatrixT diag(VectorT const &v)
    {
        IndexArrayType indices(v.nvals());
        std::vector<typename VectorT::ScalarType> vals(v.nvals());
        v.extractTuples(indices.begin(), vals.begin());

        //populate diagnals:
        MatrixT diag(v.size(), v.size());
        diag.build(indices.begin(), indices.begin(), vals.begin(), vals.size());
        return diag;
    }

    //************************************************************************
    /**
     * @brief Construct and retrun a scaled identity matrix of the given size.
     *
     * @param[in] mat_size The size of the identiy matrix to construct.
     * @param[in] val      The value to store along the diagonal
     */
    template<typename MatrixT>
    MatrixT scaled_identity(IndexType                    mat_size,
                            typename MatrixT::ScalarType val =
                            static_cast<typename MatrixT::ScalarType>(1))
    {
        using T = typename MatrixT::ScalarType;
        IndexArrayType indices;
        std::vector<T> vals(mat_size, val);
        for (IndexType ix = 0; ix < mat_size; ++ix)
        {
            indices.push_back(ix);
        }

        MatrixT identity(mat_size, mat_size);
        identity.build(indices, indices, vals);
        //grb::print_matrix(std::cerr, identity, "SCALED IDENTITY");
        return identity;
    }


    //************************************************************************
    /**
     * @brief Split a matrix into its lower and upper triangular portions
     *        Diagonal entries got to L.
     *
     * @param[in]  A  The matrix to split
     * @param[out] L  The lower triangular portion with the diagonal
     * @param[out] U  The upper triangular portion (no diagonal).
     */
    template<typename MatrixT>
    void split(MatrixT const &A, MatrixT &L, MatrixT &U)
    {
        /// @todo assert A, L, and U are same size.

        using T = typename MatrixT::ScalarType;

        grb::IndexType nvals(A.nvals());

        grb::IndexArrayType i(nvals), j(nvals);
        std::vector<T> v(nvals);
        A.extractTuples(i, j, v);

        IndexArrayType iL,jL, iU,jU;
        std::vector<T> vL, vU;

        for (grb::IndexType idx = 0; idx < nvals; ++idx)
        {
            if (i[idx] < j[idx])
            {
                iU.push_back(i[idx]);
                jU.push_back(j[idx]);
                vU.push_back(v[idx]);
            }
            else
            {
                iL.push_back(i[idx]);
                jL.push_back(j[idx]);
                vL.push_back(v[idx]);
            }
        }
        L.build(iL.begin(), jL.begin(), vL.begin(), vL.size());
        U.build(iU.begin(), jU.begin(), vU.begin(), vU.size());
    }

    //************************************************************************
    /**
     * @brief Normalize the rows of a matrix
     *
     * @param[in,out] A Matrix to normalize (in place)
     *
     */
    template<typename MatrixT>
    void normalize_rows(MatrixT &A)
    {
        using T = typename MatrixT::ScalarType;

        grb::Vector<T> w(A.nrows());
        grb::reduce(w,
                    grb::NoMask(), grb::NoAccumulate(),
                    grb::Plus<T>(),
                    A);
        grb::apply(w,
                   grb::NoMask(), grb::NoAccumulate(),
                   grb::MultiplicativeInverse<T>(),
                   w);

        IndexArrayType indices(w.nvals());
        std::vector<typename MatrixT::ScalarType> vals(w.nvals());
        w.extractTuples(indices.begin(), vals.begin());

        //populate diagnal:
        MatrixT Adiag(w.size(), w.size());
        Adiag.build(indices.begin(), indices.begin(), vals.begin(), vals.size());

        //Perform matrix multiply to scale rows
        grb::mxm(A,
                 grb::NoMask(), grb::NoAccumulate(),
                 grb::ArithmeticSemiring<T>(),
                 Adiag, A);
    }


    //************************************************************************
    /**
     * @brief Normalize the columns of a matrix
     *
     * @param[in,out] A Matrix to normalize (in place)
     *
     */
    template<typename MatrixT>
    void normalize_cols(MatrixT &A)
    {
        using T = typename MatrixT::ScalarType;

        grb::Vector<T> w(A.nrows());
        grb::reduce(w,
                    grb::NoMask(), grb::NoAccumulate(),
                    grb::Plus<T>(),
                    grb::transpose(A));
        grb::apply(w,
                   grb::NoMask(), grb::NoAccumulate(),
                   grb::MultiplicativeInverse<T>(),
                   w);

        IndexArrayType indices(w.nvals());
        std::vector<typename MatrixT::ScalarType> vals(w.nvals());
        w.extractTuples(indices.begin(), vals.begin());

        //populate diagnal:
        MatrixT Adiag(w.size(), w.size());
        Adiag.build(indices.begin(), indices.begin(), vals.begin(), vals.size());

        //Perform matrix multiply to scale rows
        grb::mxm(A,
                 grb::NoMask(), grb::NoAccumulate(),
                 grb::ArithmeticSemiring<T>(),
                 A, Adiag);
    }
}
