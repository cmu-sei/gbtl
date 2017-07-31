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
#include <memory>

#define GB_DEBUG
#include <graphblas/graphblas.hpp>

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
    template<typename EMatrixT>
    EMatrixT k_truss(EMatrixT const       &Ein,        // incidence array
                     GraphBLAS::IndexType  k_size)
    {
        typedef typename EMatrixT::ScalarType EdgeType;

        //GraphBLAS::print_matrix(std::cout, Ein, "incidence");

        GraphBLAS::IndexType num_vertices(Ein.ncols());
        GraphBLAS::IndexType num_edges(Ein.nrows());

        // Build a mask for the diagonal of A
        GraphBLAS::Matrix<bool> DiagMask(num_vertices, num_vertices);
        GraphBLAS::IndexArrayType I_n;
        std::vector<bool> v_n(num_vertices, true);
        I_n.reserve(num_vertices);
        for (GraphBLAS::IndexType ix = 0; ix < num_vertices; ++ix)
        {
            I_n.push_back(ix);
        }
        // use build or assign
        DiagMask.build(I_n, I_n, v_n);
        //GraphBLAS::print_matrix(std::cout, DiagMask, "Diag(N)");

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
                       GraphBLAS::transpose(Ein), Ein, true);
        //GraphBLAS::print_matrix(std::cout, A, "adjacencies");

        // 3. Compute the support for each edge:
        // R = E*A
        // s = (R==2)*1
        //GraphBLAS::Matrix<EdgeType> R(num_edges, num_vertices);
        auto E = std::make_shared<EMatrixT>(Ein);
        auto R = std::make_shared<EMatrixT>(num_edges, num_vertices);
        GraphBLAS::mxm(*R, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::ArithmeticSemiring<EdgeType>(),
                       *E, A);
        //GraphBLAS::print_matrix(std::cout, *R, "R");

        GraphBLAS::Vector<EdgeType> OnesN(num_vertices);
        GraphBLAS::assign_constant(OnesN,
                                   GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   static_cast<EdgeType>(1), I_n, true);
        auto s = std::make_shared<GraphBLAS::Vector<EdgeType>>(num_edges);
        GraphBLAS::mxv(*s, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       Support2Semiring<EdgeType>(),
                       *R, OnesN, true);
        //GraphBLAS::print_vector(std::cout, *s, "edge support");

        // 4. Determine edges which lack enough support for k-truss
        // x = find(s < k-2)
        auto x = std::make_shared<GraphBLAS::Vector<bool>>(num_edges);
        GraphBLAS::apply(*x, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                         SupportTest<EdgeType>(k_size - 2),
                         *s, true);
        GraphBLAS::apply(*x, *x, GraphBLAS::NoAccumulate(),
                         GraphBLAS::Identity<EdgeType>(),
                         *x, true);
        //GraphBLAS::print_vector(std::cout, *x, "edges lacking support");

        while (x->nvals() > 0)
        {
            //std::cout << "============= Iteration: |x| = " << x->nvals()
            //          << std::endl;

            // Step 0a: Get the indices of 'falses' in x
            GraphBLAS::IndexArrayType x_indices(x->nvals());
            GraphBLAS::IndexArrayType x_vals(x->nvals());
            x->extractTuples(x_indices.begin(), x_vals.begin());

            //std::cout << "x_indices: ";
            //for (auto ix : x_indices) std::cout << " " << ix;
            //std::cout << std::endl;

            // Step 0b: Get the indices of 'trues' in x
            GraphBLAS::Vector<bool> xc(num_edges);
            GraphBLAS::apply(
                xc, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                SupportTest<EdgeType, std::greater_equal<EdgeType>>(k_size - 2),
                *s, true);
            // masked no-op to get rid of stored falses.
            GraphBLAS::apply(xc, xc, GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<EdgeType>(),
                             xc, true);
            //GraphBLAS::print_vector(std::cout, xc, "complement(x)");

            GraphBLAS::IndexArrayType xc_indices(xc.nvals());
            std::vector<bool>         xc_vals(xc.nvals());
            xc.extractTuples(xc_indices.begin(), xc_vals.begin());

            //std::cout << "xc_indices: ";
            //for (auto ix : xc_indices) std::cout << " " << ix;
            //std::cout << std::endl;

            // Step 1: extract the edges that lack support
            // Ex = E(x,:)
            GraphBLAS::Matrix<EdgeType> Ex(x_indices.size(), num_vertices);
            GraphBLAS::extract(Ex,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               *E,
                               x_indices,
                               GraphBLAS::GrB_ALL,
                               true);
            //GraphBLAS::print_matrix(std::cout, Ex, "Ex");

            // Step 1b: extract the edges that are left
            // E := E(xc,:)
            num_edges = xc_indices.size();
            auto Enew = std::make_shared<EMatrixT>(num_edges, num_vertices);
            GraphBLAS::extract(*Enew,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               *E,
                               xc_indices,
                               GraphBLAS::GrB_ALL,
                               true);
            //GraphBLAS::print_matrix(std::cout, *Enew, "Enew");
            E = Enew;
            if (num_edges == 0)
            {
                break;
            }

            // R := R(xc,:)
            auto Rnew = std::make_shared<EMatrixT>(num_edges, num_vertices);
            GraphBLAS::extract(*Rnew,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               *R,
                               xc_indices,
                               GraphBLAS::GrB_ALL,
                               true);
            //GraphBLAS::print_matrix(std::cout, *Rnew, "Rnew");
            R = Rnew;

            // R := R - E[Ex'*Ex - diag(dx)]
            //
            // ExT_Ex<Diag> = Ex'*Ex
            // ExT_Ex = Ex'*Ex - diag(Ex'*Ex)
            GraphBLAS::Matrix<EdgeType> ExT_Ex(num_vertices, num_vertices);
            GraphBLAS::mxm(ExT_Ex,
                           GraphBLAS::complement(DiagMask),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::ArithmeticSemiring<EdgeType>(),
                           GraphBLAS::transpose(Ex), Ex, true);
            //GraphBLAS::print_matrix(std::cout, ExT_Ex, "Ex'*Ex - diag");

            // R -= E(Ex'*Ex)
            GraphBLAS::mxm(*R,
                           GraphBLAS::NoMask(),
                           GraphBLAS::Minus<EdgeType>(),
                           GraphBLAS::ArithmeticSemiring<EdgeType>(),
                           *Enew, ExT_Ex, true);
            //GraphBLAS::print_matrix(std::cout, *R, "R -= E*[Ex'*Ex - diag]");

            s = std::make_shared<GraphBLAS::Vector<EdgeType>>(num_edges);
            GraphBLAS::mxv(*s, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           Support2Semiring<EdgeType>(),
                           *Rnew, OnesN, true);
            //GraphBLAS::print_vector(std::cout, *s, "support");

            // 4. Determine edges which lack enough support for k-truss
            x = std::make_shared<GraphBLAS::Vector<bool>>(num_edges);
            GraphBLAS::apply(*x,
                             GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             SupportTest<EdgeType>(k_size - 2),
                             *s, true);
            //GraphBLAS::print_vector(std::cout, *x, "new x");
            GraphBLAS::apply(*x, *x, GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<EdgeType>(),
                             *x, true);
            //GraphBLAS::print_vector(std::cout, *x, "new x (masked noop)");
        }

        // return incidence matrix containing all edges in k-trusses
        return *E;
    }
}

#endif
