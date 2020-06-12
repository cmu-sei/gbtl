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

#include <vector>
#include <random>

#include <graphblas/graphblas.hpp>

//****************************************************************************
// The following are hacks because algorithm assumes access to implicit zero
// should return a value of zero.
//
namespace
{
    template <typename T>
    T at(grb::Vector<T> const &v,
         grb::IndexType        idx)
    {
        return (v.hasElement(idx) ? v.extractElement(idx) : T(0));
    }

    template <typename T, typename... TagsT>
    T at(grb::Matrix<T, TagsT...> const &A,
         grb::IndexType                  idx,
         grb::IndexType                  idy)
    {
        return (A.hasElement(idx, idy) ? A.extractElement(idx, idy) : T(0));
    }
}

//****************************************************************************
template <typename MatrixT, typename VectorT>
static void push(MatrixT  const &C,
                 MatrixT        &F,
                 VectorT        &excess,
                 grb::IndexType  u,
                 grb::IndexType  v)
{
    using T = typename MatrixT::ScalarType;
    T a = at(excess, u);
    T b = at(C, u, v) - at(F, u, v);
    T send = std::min(a, b);

    F.setElement(u, v, at(F, u, v) + send);
    F.setElement(v, u, at(F, v, u) - send);

    excess.setElement(u, at(excess,u) - send);
    excess.setElement(v, at(excess,v) + send);
}

//****************************************************************************
template <typename MatrixT, typename VectorT>
static void relabel(MatrixT  const &C,
                    MatrixT  const &F,
                    VectorT        &height,
                    grb::IndexType  u)
{
    using T = typename MatrixT::ScalarType;
    grb::IndexType num_nodes(C.nrows());

    T min_height = std::numeric_limits<T>::max();
    for (grb::IndexType v = 0; v < num_nodes; ++v)
    {
        if ((at(C,u, v) - at(F,u, v)) > 0)
        {
            T a = at(height,v);
            min_height = std::min(min_height, a);
            height.setElement(u, min_height + 1);
        }
    }
}

//****************************************************************************
/// @todo Does seen have a different scalar type (IndexType)
template <typename MatrixT, typename VectorT>
static void discharge(MatrixT  const &C,
                      MatrixT        &F,
                      VectorT        &excess,
                      VectorT        &height,
                      VectorT        &seen,
                      grb::IndexType  u)
{
    grb::IndexType num_nodes(C.nrows());

    while (at(excess,u) > 0)
    {
        if (at(seen,u) < num_nodes)
        {
            grb::IndexType v = at(seen,u);
            if (((at(C,u, v) - at(F,u, v)) > 0) &&
                (at(height,u) > at(height,v)))
            {
                push(C, F, excess, u, v);
            }
            else
            {
                seen.setElement(u, at(seen,u) + 1);
            }
        }
        else
        {
            relabel(C, F, height, u);
            seen.setElement(u, 0);
        }
    }
}

//****************************************************************************
//****************************************************************************
namespace algorithms
{
    /**
     * @brief Compute the maximum flow through a graph given the capacities of
     *        the edges of that graph using a push-relabel algorithm
     *
     * <ul>
     * <li>Using a push-relabel removes the need to calculate augmented paths
     * (as in Ford Fulkerson)</li>
     * <li>Better lent itself to an implementation using GraphBLAS
     * functions.</li>
     * </ul>
     * <p>In this implementation, a preflow, which is an assignment of flows to
     * edges allowing for more incoming flow than outcoming flow while
     * maintaining capacity constraints, is maintained.</p>
     * <ul>
     * <li>Use the operation <code>push</code> to "push" flow through the flow
     * network.</li>
     * <li><code>relabel</code> maintains the residual network (the <i>residual
     * network</i> is the network at the current iteration in the
     * algorithm).</li>
     * <li>These two operations can only be performed on active vertices at
     * each iteration, where an <i>active</i> vertex is a vertex such that
     * all edges in the residual graph that contain it only have positive
     * excess flows.</li>
     * </ul>
     *
     * @param[in]  capacity  The capacity matrix of the edges of the graph to
     *                       compute the max flow through.
     * @param[in]  source    The vertex to use as the source.
     * @param[in]  sink      The vertex to use as the sink.
     *
     * @return The value of the maximum flow that can be pushed through the
     *         source.
     *
     * @note This is not written in the philosphy of GraphBLAS and needs to
     *       be replaced or removed.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType maxflow_push_relabel(MatrixT  const &capacity,
                                                      grb::IndexType  source,
                                                      grb::IndexType  sink)
    {
        using T = typename MatrixT::ScalarType;
        grb::IndexType rows(capacity.nrows());
        grb::IndexType cols(capacity.ncols());

        grb::IndexType num_nodes = rows;

        MatrixT flow(rows,cols);

        grb::Vector<T> height(num_nodes);
        grb::Vector<T> excess(num_nodes);
        grb::Vector<T> seen(num_nodes);

        std::vector<grb::IndexType> list;

        for (grb::IndexType i = 0; i < num_nodes; ++i)
        {
            if ((i != source) && (i != sink))
            {
                list.push_back(i);
            }
        }

        height.setElement(source, num_nodes);
        excess.setElement(source,
                          std::numeric_limits<T>::max());
        for (grb::IndexType i = 0; i < num_nodes; ++i)
        {
            push(capacity, flow, excess, source, i);
        }

        //grb::print_matrix(std::cerr, flow, "\nFLOW");

        grb::IndexType  p = 0;
        while (p < (num_nodes - 2))
        {
            grb::IndexType u = list[p];
            T old_height = at(height, u);
            discharge(capacity, flow, excess, height, seen, u);
            //grb::print_matrix(std::cerr, flow, "\nFLOW after discharge");

            if (at(height,u) > old_height)
            {
                grb::IndexType t = list[p];

                list.erase(list.begin() + p);
                list.insert(list.begin() + 0, t);

                p = 0;
            }
            else
            {
                p += 1;
            }
        }

        //T maxflow = static_cast<T>(0);
        //for (grb::IndexType i = 0; i < num_nodes; ++i)
        //{
        //    maxflow += flow.extractElement(source, i);
        //}
        //grb::print_matrix(std::cerr, flow, "\nFlow");
        grb::Vector<T> flows(rows);
        grb::reduce(flows, grb::NoMask(), grb::NoAccumulate(),
                    grb::Plus<T>(), flow);
        T maxflow = at(flows,source);

        //grb::print_matrix(std::cerr, flow, "Final FLOW:");
        return maxflow;
    }

    //************************************************************************
    //************************************************************************
    template<typename MatrixT>
    bool maxflow_bfs(
        MatrixT                     const &graph,
        grb::IndexType                     source,
        grb::IndexType                     sink,
        grb::Vector<grb::IndexType> const &index_ramp,
        grb::Matrix<bool>                 &M)
    {
        using T = typename MatrixT::ScalarType;
        grb::Vector<grb::IndexType> parent_list(graph.nrows());
        parent_list.setElement(source, source);

        grb::Vector<grb::IndexType> wavefront(graph.ncols());
        wavefront.setElement(source, 1ul);

        while ((!parent_list.hasElement(sink)) && (wavefront.nvals() > 0))
        {
            // convert all stored values to their column index
            grb::eWiseMult(wavefront,
                           grb::NoMask(), grb::NoAccumulate(),
                           grb::First<grb::IndexType>(),
                           index_ramp, wavefront);

            // First because we are left multiplying wavefront rows
            // Masking out the parent list ensures wavefront values do not
            // overlap values already stored in the parent list
            grb::vxm(wavefront,
                     grb::complement(grb::structure(parent_list)),
                     grb::NoAccumulate(),
                     grb::MinFirstSemiring<grb::IndexType,T,grb::IndexType>(),
                     wavefront, graph, grb::REPLACE);

            grb::apply(parent_list,
                       grb::NoMask(),
                       grb::Plus<grb::IndexType>(),
                       grb::Identity<grb::IndexType>(),
                       wavefront);
        }

        if (!parent_list.hasElement(sink))
        {
            return false;
        }

        // Extract path from source to sink from parent list (reverse traverse)
        // build a mask
        M.clear();
        grb::IndexType curr_vertex(sink);
        while (curr_vertex != source)
        {
            grb::IndexType parent(parent_list.extractElement(curr_vertex));
            M.setElement(parent, curr_vertex, true);
            curr_vertex = parent;
        }

        return true;
    }

    //************************************************************************
    template<typename MatrixT>
    typename MatrixT::ScalarType maxflow_ford_fulk(MatrixT  const &graph,
                                                   grb::IndexType  source,
                                                   grb::IndexType  sink)
    {
        using T = typename MatrixT::ScalarType;
        grb::IndexType num_nodes(graph.nrows());

        // @todo assert graph matrix is square

        // create index ramp for index_of() functionality
        grb::Vector<grb::IndexType> index_ramp(num_nodes);
        {
            std::vector<grb::IndexType> idx(num_nodes);
            for (grb::IndexType i = 0; i < num_nodes; ++i)
            {
                idx[i] = i;
            }

            index_ramp.build(idx.begin(), idx.begin(), num_nodes);
        }

        grb::Matrix<bool> M(num_nodes, num_nodes);
        MatrixT F(num_nodes, num_nodes);
        MatrixT R = graph;
        grb::Vector<grb::IndexType> parent_list(num_nodes);
        T max_flow(0);

        //grb::print_matrix(std::cerr, F, "F");
        //grb::print_matrix(std::cerr, R, "R");
        //grb::print_matrix(std::cerr, graph, "graph");

        grb::IndexType count(0);
        while (maxflow_bfs(R, source, sink, index_ramp, M) &&
               count++ < graph.nvals())
        {
            //std::cerr << "----------- Iteration: " << count << " -----------\n";
            //grb::print_matrix(std::cerr, M, "M (path)");

            // gamma = min(M.*R)
            T gamma;
            grb::Matrix<T> G(num_nodes, num_nodes);
            grb::eWiseMult(G, grb::NoMask(), grb::NoAccumulate(),
                           grb::Times<T>(), M, R);
            grb::reduce(gamma, grb::NoAccumulate(),
                        grb::MinMonoid<T>(), G);
            //grb::print_matrix(std::cerr, G, "M.*R");
            //std::cerr << "gamma = min(M.*R) = " << gamma << std::endl;

            // GM = gamma*M
            MatrixT GM(num_nodes, num_nodes);
            grb::apply(GM, grb::NoMask(), grb::NoAccumulate(),
                       std::bind(grb::Times<T>(),
                                 std::placeholders::_1,
                                 gamma),
                       M);
            //grb::print_matrix(std::cerr, GM, "gamma*M");

            // F += gamma(M - M')
            //   += gamma*M + (-gamma*M')

            // GM := GM + (-GM')
            grb::apply(GM, grb::NoMask(), grb::Plus<T>(),
                       grb::AdditiveInverse<T>(),
                       grb::transpose(GM));
            // F := F + GM
            grb::eWiseAdd(F, grb::NoMask(),
                          grb::NoAccumulate(),
                          grb::Plus<T>(), F, GM, grb::REPLACE);
            //grb::print_matrix(std::cerr, GM, "gamma(M - M')");
            //grb::print_matrix(std::cerr, F, "F = F + gamma(M - M')");

            // R = graph - F

            // mF := (-F)
            MatrixT mF(num_nodes, num_nodes);
            grb::apply(mF, grb::NoMask(), grb::NoAccumulate(),
                       grb::AdditiveInverse<T>(), F);
            // R := graph + (-F)
            grb::eWiseAdd(R, grb::NoMask(), grb::NoAccumulate(),
                          grb::Plus<T>(), graph, mF, grb::REPLACE);
            // Clear the zero's
            grb::apply(R, R, grb::NoAccumulate(),
                       grb::Identity<T>(), R, grb::REPLACE);
            //grb::print_matrix(std::cerr, R, "R = graph - F");
        }

        //grb::print_matrix(std::cerr, F, "Final FLOW:");
        grb::Vector<T> sink_edges(num_nodes);
        grb::extract(sink_edges,
                     grb::NoMask(), grb::NoAccumulate(),
                     F, grb::AllIndices(), sink);
        grb::reduce(max_flow, grb::NoAccumulate(),
                    grb::PlusMonoid<T>(), sink_edges);
        //std::cerr << "Max flow = " << max_flow << std::endl;
        return max_flow;
    }

} // algorithms
