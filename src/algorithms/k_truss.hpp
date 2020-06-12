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

#include <iostream>
#include <memory>

#include <graphblas/graphblas.hpp>

//****************************************************************************
namespace
{
    template <typename D1, typename D2=D1, typename D3=D1>
    struct FirstEqualsTwo
    {
        D3 operator()(D1 lhs, D2 rhs)
        {
            D3 res(0);
            if (lhs == static_cast<D1>(2))
                res = static_cast<D3>(rhs);
            return res;
        }
    };

    GEN_GRAPHBLAS_SEMIRING(Support2Semiring,
                           grb::PlusMonoid, FirstEqualsTwo)

    //************************************************************************
    template <typename T, typename LessT = std::less<T>>
    struct SupportTest
    {
        T const m_threshold;

        SupportTest(T threshold) : m_threshold(threshold) {}

        bool operator()(T val)
        {
            return LessT()(val, m_threshold);  // val < threshold
        }
    };

    //************************************************************************
    template <typename T, typename LessT = std::less<T>>
    struct SupportMinTest
    {
        T const m_threshold;

        SupportMinTest(T threshold) : m_threshold(threshold) {}

        bool operator()(T val)
        {
            return !LessT()(val, m_threshold);  // val >= threshold
        }
    };
}

//****************************************************************************
namespace algorithms
{
    //************************************************************************
    template<typename EMatrixT>
    EMatrixT k_truss(EMatrixT const &Ein,        // incidence array
                     grb::IndexType  k_size)
    {
        using EdgeType = typename EMatrixT::ScalarType;

        //grb::print_matrix(std::cout, Ein, "incidence");

        grb::IndexType num_vertices(Ein.ncols());
        grb::IndexType num_edges(Ein.nrows());

        // Build a mask for the diagonal of A
        grb::IndexArrayType I_n;
        grb::Matrix<bool> DiagMask(num_vertices, num_vertices);
        for (grb::IndexType ix = 0; ix < num_vertices; ++ix)
        {
            I_n.push_back(ix);
            DiagMask.setElement(ix, ix, true);
        }
        //grb::print_matrix(std::cout, DiagMask, "Diag(N)");

        // 1. Compute the degree of each vertex from the incidence matrix
        // via column-wise reduction (row reduce of transpose).
        //grb::Vector<EdgeType> d(num_vertices); //degree
        //grb::reduce(d, grb::NoMask(), grb::NoAccumulate(),
        //            grb::PlusMonoid<EdgeType>(),
        //            grb::transpose(E));
        //d.printInfo(std::cout);

        // 2. Compute the adjacency matrix: A<!Diag> = E'*E
        grb::Matrix<EdgeType> A(num_vertices, num_vertices);
        grb::mxm(A,
                 grb::complement(DiagMask),
                 grb::NoAccumulate(),
                 grb::ArithmeticSemiring<EdgeType>(),
                 grb::transpose(Ein), Ein, grb::REPLACE);
        //grb::print_matrix(std::cout, A, "adjacencies");

        // 3. Compute the support for each edge:
        // R = E*A
        // s = (R==2)*1
        //grb::Matrix<EdgeType> R(num_edges, num_vertices);
        auto E = std::make_shared<EMatrixT>(Ein);
        auto R = std::make_shared<EMatrixT>(num_edges, num_vertices);
        grb::mxm(*R, grb::NoMask(), grb::NoAccumulate(),
                 grb::ArithmeticSemiring<EdgeType>(),
                 *E, A);
        //grb::print_matrix(std::cout, *R, "R");

        grb::Vector<EdgeType> OnesN(num_vertices);
        grb::assign(OnesN,
                    grb::NoMask(),
                    grb::NoAccumulate(),
                    static_cast<EdgeType>(1), I_n, grb::REPLACE);
        auto s = std::make_shared<grb::Vector<EdgeType>>(num_edges);
        grb::mxv(*s, grb::NoMask(), grb::NoAccumulate(),
                 Support2Semiring<EdgeType>(),
                 *R, OnesN, grb::REPLACE);
        //grb::print_vector(std::cout, *s, "edge support");

        // 4. Determine edges which lack enough support for k-truss
        // x = find(s < k-2)
        auto x = std::make_shared<grb::Vector<bool>>(num_edges);
        grb::apply(*x, grb::NoMask(), grb::NoAccumulate(),
                   SupportTest<EdgeType>(k_size - 2),
                   *s, grb::REPLACE);
        grb::apply(*x, *x, grb::NoAccumulate(),
                   grb::Identity<EdgeType>(),
                   *x, grb::REPLACE);
        //grb::print_vector(std::cout, *x, "edges lacking support");

        while (x->nvals() > 0)
        {
            //std::cout << "============= Iteration: |x| = " << x->nvals()
            //          << std::endl;

            // Step 0a: Get the indices of 'falses' in x
            grb::IndexArrayType x_indices(x->nvals());
            grb::IndexArrayType x_vals(x->nvals());
            x->extractTuples(x_indices.begin(), x_vals.begin());

            //std::cout << "x_indices: ";
            //for (auto ix : x_indices) std::cout << " " << ix;
            //std::cout << std::endl;

            // Step 0b: Get the indices of 'trues' in x
            grb::Vector<bool> xc(num_edges);
            grb::apply(
                xc, grb::NoMask(), grb::NoAccumulate(),
                SupportTest<EdgeType, std::greater_equal<EdgeType>>(k_size - 2),
                *s, grb::REPLACE);
            // masked no-op to get rid of stored falses.
            grb::apply(xc, xc, grb::NoAccumulate(),
                       grb::Identity<EdgeType>(),
                       xc, grb::REPLACE);
            //grb::print_vector(std::cout, xc, "complement(x)");

            grb::IndexArrayType xc_indices(xc.nvals());
            std::vector<bool>         xc_vals(xc.nvals());
            xc.extractTuples(xc_indices.begin(), xc_vals.begin());

            //std::cout << "xc_indices: ";
            //for (auto ix : xc_indices) std::cout << " " << ix;
            //std::cout << std::endl;

            // Step 1: extract the edges that lack support
            // Ex = E(x,:)
            grb::Matrix<EdgeType> Ex(x_indices.size(), num_vertices);
            grb::extract(Ex,
                         grb::NoMask(),
                         grb::NoAccumulate(),
                         *E,
                         x_indices,
                         grb::AllIndices(),
                         grb::REPLACE);
            //grb::print_matrix(std::cout, Ex, "Ex");

            // Step 1b: extract the edges that are left
            // E := E(xc,:)
            num_edges = xc_indices.size();
            auto Enew = std::make_shared<EMatrixT>(num_edges, num_vertices);
            grb::extract(*Enew,
                         grb::NoMask(),
                         grb::NoAccumulate(),
                         *E,
                         xc_indices,
                         grb::AllIndices(),
                         grb::REPLACE);
            //grb::print_matrix(std::cout, *Enew, "Enew");
            E = Enew;
            if (num_edges == 0)
            {
                break;
            }

            // R := R(xc,:)
            auto Rnew = std::make_shared<EMatrixT>(num_edges, num_vertices);
            grb::extract(*Rnew,
                         grb::NoMask(),
                         grb::NoAccumulate(),
                         *R,
                         xc_indices,
                         grb::AllIndices(),
                         grb::REPLACE);
            //grb::print_matrix(std::cout, *Rnew, "Rnew");
            R = Rnew;

            // R := R - E[Ex'*Ex - diag(dx)]
            //
            // ExT_Ex<Diag> = Ex'*Ex
            // ExT_Ex = Ex'*Ex - diag(Ex'*Ex)
            grb::Matrix<EdgeType> ExT_Ex(num_vertices, num_vertices);
            grb::mxm(ExT_Ex,
                     grb::complement(DiagMask),
                     grb::NoAccumulate(),
                     grb::ArithmeticSemiring<EdgeType>(),
                     grb::transpose(Ex), Ex, grb::REPLACE);
            //grb::print_matrix(std::cout, ExT_Ex, "Ex'*Ex - diag");

            // R -= E(Ex'*Ex)
            grb::mxm(*R,
                     grb::NoMask(),
                     grb::Minus<EdgeType>(),
                     grb::ArithmeticSemiring<EdgeType>(),
                     *Enew, ExT_Ex, grb::REPLACE);
            //grb::print_matrix(std::cout, *R, "R -= E*[Ex'*Ex - diag]");

            s = std::make_shared<grb::Vector<EdgeType>>(num_edges);
            grb::mxv(*s, grb::NoMask(), grb::NoAccumulate(),
                     Support2Semiring<EdgeType>(),
                     *Rnew, OnesN, grb::REPLACE);
            //grb::print_vector(std::cout, *s, "support");

            // 4. Determine edges which lack enough support for k-truss
            x = std::make_shared<grb::Vector<bool>>(num_edges);
            grb::apply(*x,
                       grb::NoMask(), grb::NoAccumulate(),
                       SupportTest<EdgeType>(k_size - 2),
                       *s, grb::REPLACE);
            //grb::print_vector(std::cout, *x, "new x");
            grb::apply(*x, *x, grb::NoAccumulate(),
                       grb::Identity<EdgeType>(),
                       *x, grb::REPLACE);
            //grb::print_vector(std::cout, *x, "new x (masked noop)");
        }

        // return incidence matrix containing all edges in k-trusses
        return *E;
    }

    //************************************************************************
    template<typename AMatrixT>
    AMatrixT k_truss2(AMatrixT const &Ain,   // undirected adjacency matrix
                      grb::IndexType  k_size)
    {
        using AType = typename AMatrixT::ScalarType;

        //grb::print_matrix(std::cout, Ain, "adjacency");

        grb::IndexType num_vertices(Ain.ncols());
        // TODO: assert square

        AMatrixT A(Ain);
        grb::IndexType num_edges, new_num_edges(A.nvals());

        grb::Matrix<bool> Mask(num_vertices, num_vertices);
        grb::Matrix<grb::IndexType> Support(num_vertices, num_vertices);

        do
        {
            num_edges = new_num_edges;

            // Compute the support of each edge
            // S<A,-> = (A' +.* A) = (A' * A) .* A
            grb::mxm(Support, A, grb::NoAccumulate(),
                     grb::ArithmeticSemiring<AType>(),
                     grb::transpose(A), A, grb::REPLACE);
            //grb::print_matrix(std::cout, Support, "Support");

            // Keep all edges with enough support
            // Mask = S .>= (k - 2)
            grb::apply(Mask,
                       grb::NoMask(), grb::NoAccumulate(),
                       SupportMinTest<AType>(k_size - 2),
                       Support, grb::REPLACE);

            // Annihilate all edges lacking support: A<A> = A or  A = A .* M
            grb::apply(A, Mask, grb::NoAccumulate(),
                       grb::Identity<AType>(), A, grb::REPLACE);
            new_num_edges = A.nvals();

        } while (new_num_edges != num_edges);  // loop while edges are being removed

        return A;
    }

}
