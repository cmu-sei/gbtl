/*
 * Copyright (c) 2017 Carnegie Mellon University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY EXPRESSLY DISCLAIMS
 * TO THE FULLEST EXTENT PERMITTED BY LAW ALL EXPRESS, IMPLIED, AND STATUTORY
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */


#ifndef ALGORITHMS_K_TRUSS_HPP
#define ALGORITHMS_K_TRUSS_HPP

#include <iostream>

#define GB_DEBUG
#include <graphblas/graphblas.hpp>
#include <graphblas/linalg_utils.hpp>

//****************************************************************************
namespace
{
    template <typename D1, typename D2=D1, typename D3=D1>
    struct FirstEqualsTwo
    {
        typedef D3 result_type;
        D3 operator()(D1 lhs, D2 rhs)
        {
            result_type res(0);
            if (lhs == static_cast<D1>(2))
                res = static_cast<D3>(rhs);
            return res;
        }
    };

    GEN_GRAPHBLAS_SEMIRING(Support2Semiring,
                           GraphBLAS::PlusMonoid, FirstEqualsTwo)

    //************************************************************************
    template <typename T, typename LessT = std::less<T>>
    struct SupportTest
    {
        typedef bool result_type;

        T const m_threshold;

        SupportTest(T threshold) : m_threshold(threshold) {}

        bool operator()(T val)
        {
            return LessT()(val, m_threshold);
        }
    };
}

//****************************************************************************
namespace algorithms
{
    //************************************************************************
    template<typename EMatrixT> //, typename AMatrixT>
    void k_truss(EMatrixT             &trusses,
                 EMatrixT const       &E,        // incidence array
                 GraphBLAS::IndexType  k_size)
    {
        typedef typename EMatrixT::ScalarType EdgeType;

        GraphBLAS::print_matrix(std::cout, E, "incidence");

        GraphBLAS::IndexType num_vertices(E.ncols());
        GraphBLAS::IndexType num_edges(E.nrows());

        // Build a mask for the diagonal of A
        GraphBLAS::Matrix<bool> DiagMask(num_vertices, num_vertices);
        GraphBLAS::IndexArrayType I_n;
        std::vector<bool> v_n(num_vertices, true);
        I_n.reserve(num_vertices);
        for (GraphBLAS::IndexType ix = 0; ix < num_vertices; ++ix)
        {
            I_n.push_back(ix);
        }
        // use build or assigne
        DiagMask.build(I_n, I_n, v_n);
        GraphBLAS::print_matrix(std::cout, DiagMask, "Diag(N)");

        // 1. Compute the degree of each vertex from the incidence matrix
        // via column-wise reduction (row reduce of transpose).
        //GraphBLAS::Vector<EdgeType> d(num_vertices); //degree
        //GraphBLAS::reduce(d, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        //                  GraphBLAS::PlusMonoid<EdgeType>(),
        //                  GraphBLAS::transpose(E));
        //d.printInfo(std::cout);

        // 2. Compute the adjacency matrix: A<!Diag> = E'*E
        GraphBLAS::Matrix<EdgeType> A(num_vertices, num_vertices);
        GraphBLAS::mxm(A,
                       GraphBLAS::complement(DiagMask),
                       GraphBLAS::NoAccumulate(),
                       GraphBLAS::ArithmeticSemiring<EdgeType>(),
                       GraphBLAS::transpose(E), E, true);
        GraphBLAS::print_matrix(std::cout, A, "adjacencies");

        // 3. Compute the support for each edge:
        // R = E*A
        // s = (R==2)*1
        GraphBLAS::Matrix<EdgeType> R(num_edges, num_vertices);
        GraphBLAS::mxm(R, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::ArithmeticSemiring<EdgeType>(),
                       E, A);
        GraphBLAS::print_matrix(std::cout, R, "R");

        GraphBLAS::Vector<EdgeType> OnesN(num_vertices);
        GraphBLAS::assign_constant(OnesN,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   static_cast<EdgeType>(1), I_n, true);
        GraphBLAS::Vector<EdgeType> s(num_edges);
        GraphBLAS::mxv(s, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       Support2Semiring<EdgeType>(),
                       R, OnesN, true);
        s.printInfo(std::cout);

        // 4. Determine edges which lack enough support for k-truss
        GraphBLAS::Vector<bool> x(num_edges);
        GraphBLAS::apply(x, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                         SupportTest<EdgeType>(k_size - 2),
                         s, true);
        GraphBLAS::apply(x, x, GraphBLAS::NoAccumulate(),
                         GraphBLAS::Identity<EdgeType>(),
                         x, true);
        x.printInfo(std::cout);

        if (x.nvals() > 0)
        {
            // Get the indices of 'falses' in x
            GraphBLAS::IndexArrayType x_indices(x.nvals());
            GraphBLAS::IndexArrayType x_vals(x.nvals());
            x.extractTuples(x_indices.begin(), x_vals.begin());
            std::cout << "x_indices: ";
            for (auto ix : x_indices)
            {
                std::cout << " " << ix;
            }
            std::cout << std::endl;

            // Get the indices of 'trues' in x
            GraphBLAS::Vector<bool> xNot(num_edges);
            GraphBLAS::apply(
                xNot, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                SupportTest<EdgeType, std::greater_equal<EdgeType>>(k_size - 2),
                s, true);
            GraphBLAS::apply(xNot, xNot, GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<EdgeType>(),
                             xNot, true);
            xNot.printInfo(std::cout);

            GraphBLAS::IndexArrayType xNot_indices(xNot.nvals());
            GraphBLAS::IndexArrayType xNot_vals(xNot.nvals());
            xNot.extractTuples(xNot_indices.begin(), xNot_vals.begin());
            std::cout << "xNot_indices: ";
            for (auto ix : xNot_indices)
            {
                std::cout << " " << ix;
            }
            std::cout << std::endl;

            do
            {
                GraphBLAS::IndexType nedges(xNot_indices.size());

                GraphBLAS::Matrix<EdgeType> Ex(x_indices.size(),
                                               num_vertices);
                GraphBLAS::extract(Ex,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   E,
                                   x_indices,
                                   I_n, // GrB_ALL
                                   true);
                GraphBLAS::print_matrix(std::cout, Ex, "Ex");

                GraphBLAS::Matrix<EdgeType> Enew(nedges,
                                                 num_vertices);
                GraphBLAS::extract(Enew,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   E,
                                   xNot_indices,
                                   I_n, // GrB_ALL
                                   true);
                GraphBLAS::print_matrix(std::cout, Enew, "Enew");

                GraphBLAS::Matrix<EdgeType> Rnew(nedges,
                                                 num_vertices);
                GraphBLAS::extract(Rnew,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   R,
                                   xNot_indices,
                                   I_n, // GrB_ALL
                                   true);
                GraphBLAS::print_matrix(std::cout, Rnew, "Rnew");

                GraphBLAS::Matrix<EdgeType> ExT_Ex(num_vertices,
                                                   num_vertices);
                GraphBLAS::mxm(ExT_Ex,
                               GraphBLAS::complement(DiagMask),
                               GraphBLAS::NoAccumulate(),
                               GraphBLAS::ArithmeticSemiring<EdgeType>(),
                               GraphBLAS::transpose(Ex), Ex, true);
                GraphBLAS::print_matrix(std::cout, ExT_Ex, "Ex'*Ex - diag");

                GraphBLAS::mxm(Rnew,
                               GraphBLAS::NoMask(),
                               GraphBLAS::Minus<EdgeType>(),
                               GraphBLAS::ArithmeticSemiring<EdgeType>(),
                               Enew, ExT_Ex, true);
                GraphBLAS::print_matrix(std::cout, Rnew, "R -= E*[Ex'*Ex - diag]");

                GraphBLAS::Vector<EdgeType> s(nedges);
                GraphBLAS::mxv(s, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                               Support2Semiring<EdgeType>(),
                               Rnew, OnesN, true);
                s.printInfo(std::cout);

                // 4. Determine edges which lack enough support for k-truss
                GraphBLAS::Vector<bool> x(nedges);
                GraphBLAS::apply(x, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                 SupportTest<EdgeType>(k_size - 2),
                                 s, true);
                GraphBLAS::apply(x, x, GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Identity<EdgeType>(),
                                 x, true);
                x.printInfo(std::cout);

            } while (false); //new_x.nvals());
        }

        // "assign" results to Eout
    }
}

#endif
