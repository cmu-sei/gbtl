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

#ifndef GB_MATRIX_UTILS_HPP
#define GB_MATRIX_UTILS_HPP

#include <functional>
#include <vector>

#include <graphblas/graphblas.hpp>

#define GB_INCLUDE_BACKEND_UTILITY 1
#include <graphblas/backend_include.hpp>

//****************************************************************************
// Miscellaneous matrix convenience functions
//****************************************************************************
namespace GraphBLAS
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
        //GraphBLAS::print_matrix(std::cerr, identity, "SCALED IDENTITY");
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

        GraphBLAS::IndexType rows(A.nrows());
        GraphBLAS::IndexType cols(A.ncols());
        GraphBLAS::IndexType nvals(A.nvals());

        GraphBLAS::IndexArrayType i(nvals), j(nvals);
        std::vector<T> v(nvals);
        A.extractTuples(i, j, v);

        IndexArrayType iL,jL, iU,jU;
        std::vector<T> vL, vU;

        for (GraphBLAS::IndexType idx = 0; idx < nvals; ++idx)
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

        GraphBLAS::Vector<T> w(A.nrows());
        GraphBLAS::reduce(w,
                          GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                          GraphBLAS::Plus<T>(),
                          A);
        GraphBLAS::apply(w,
                         GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                         GraphBLAS::MultiplicativeInverse<T>(),
                         w);

        IndexArrayType indices(w.nvals());
        std::vector<typename MatrixT::ScalarType> vals(w.nvals());
        w.extractTuples(indices.begin(), vals.begin());

        //populate diagnal:
        MatrixT Adiag(w.size(), w.size());
        Adiag.build(indices.begin(), indices.begin(), vals.begin(), vals.size());

        //Perform matrix multiply to scale rows
        GraphBLAS::mxm(A,
                       GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::ArithmeticSemiring<T>(),
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

        GraphBLAS::Vector<T> w(A.nrows());
        GraphBLAS::reduce(w,
                          GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                          GraphBLAS::Plus<T>(),
                          GraphBLAS::transpose(A));
        GraphBLAS::apply(w,
                         GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                         GraphBLAS::MultiplicativeInverse<T>(),
                         w);

        IndexArrayType indices(w.nvals());
        std::vector<typename MatrixT::ScalarType> vals(w.nvals());
        w.extractTuples(indices.begin(), vals.begin());

        //populate diagnal:
        MatrixT Adiag(w.size(), w.size());
        Adiag.build(indices.begin(), indices.begin(), vals.begin(), vals.size());

        //Perform matrix multiply to scale rows
        GraphBLAS::mxm(A,
                       GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::ArithmeticSemiring<T>(),
                       A, Adiag);
    }
}

#endif // GB_MATRIX_UTILS_HPP
