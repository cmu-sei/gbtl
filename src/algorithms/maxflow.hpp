/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
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
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
 */

#ifndef MAXFLOW_HPP
#define MAXFLOW_HPP

#include <vector>
#include <random>

#include <graphblas/graphblas.hpp>
#include <algorithms/bfs.hpp>

//****************************************************************************
// The following are hacks because algorithm assumes access to implicit zero
// should return a value of zero.
//
namespace
{
    template <typename T>
    T at(GraphBLAS::Vector<T> const &v, GraphBLAS::IndexType idx)
    {
        return (v.hasElement(idx) ? v.extractElement(idx) : T(0));
    }

    template <typename T, typename... TagsT>
    T at(GraphBLAS::Matrix<T, TagsT...> const &A, GraphBLAS::IndexType idx, GraphBLAS::IndexType idy)
    {
        return (A.hasElement(idx, idy) ? A.extractElement(idx, idy) : T(0));
    }
}

//****************************************************************************
template <typename MatrixT, typename VectorT>
static void push(MatrixT const &C,
                 MatrixT       &F,
                 VectorT       &excess,
                 GraphBLAS::IndexType u,
                 GraphBLAS::IndexType v)
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
static void relabel(MatrixT const &C,
                    MatrixT const &F,
                    VectorT       &height,
                    GraphBLAS::IndexType u)
{
    using T = typename MatrixT::ScalarType;
    GraphBLAS::IndexType num_nodes(C.nrows());
    GraphBLAS::IndexType cols(C.ncols());

    T min_height = std::numeric_limits<T>::max();
    for (GraphBLAS::IndexType v = 0; v < num_nodes; ++v)
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
static void discharge(MatrixT const &C,
                      MatrixT       &F,
                      VectorT       &excess,
                      VectorT       &height,
                      VectorT       &seen,
                      GraphBLAS::IndexType u)
{
    GraphBLAS::IndexType num_nodes(C.nrows());
    GraphBLAS::IndexType cols(C.ncols());

    while (at(excess,u) > 0)
    {
        if (at(seen,u) < num_nodes)
        {
            GraphBLAS::IndexType v = at(seen,u);
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
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType maxflow(MatrixT const        &capacity,
                                         GraphBLAS::IndexType  source,
                                         GraphBLAS::IndexType  sink)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType rows(capacity.nrows());
        GraphBLAS::IndexType cols(capacity.ncols());

        GraphBLAS::IndexType num_nodes = rows;

        MatrixT flow(rows,cols);

        //MatrixT height(1, num_nodes);
        //MatrixT excess(1, num_nodes);
        //MatrixT seen(1, num_nodes);
        GraphBLAS::Vector<T> height(num_nodes);
        GraphBLAS::Vector<T> excess(num_nodes);
        GraphBLAS::Vector<T> seen(num_nodes);

        std::vector<GraphBLAS::IndexType> list;

        for (GraphBLAS::IndexType i = 0; i < num_nodes; ++i)
        {
            if ((i != source) && (i != sink))
            {
                list.push_back(i);
            }
        }

        height.setElement(source, num_nodes);
        excess.setElement(source,
                          std::numeric_limits<T>::max());
        for (GraphBLAS::IndexType i = 0; i < num_nodes; ++i)
        {
            push(capacity, flow, excess, source, i);
        }

        //GraphBLAS::print_matrix(std::cerr, flow, "\nFLOW");

        GraphBLAS::IndexType  p = 0;
        while (p < (num_nodes - 2))
        {
            GraphBLAS::IndexType u = list[p];
            T old_height = at(height, u);
            discharge(capacity, flow, excess, height, seen, u);
            //GraphBLAS::print_matrix(std::cerr, flow, "\nFLOW after discharge");

            if (at(height,u) > old_height)
            {
                GraphBLAS::IndexType t = list[p];

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
        //for (GraphBLAS::IndexType i = 0; i < num_nodes; ++i)
        //{
        //    maxflow += flow.extractElement(source, i);
        //}
        //GraphBLAS::print_matrix(std::cerr, flow, "\nFlow");
        GraphBLAS::Vector<T> flows(rows);
        GraphBLAS::reduce(flows, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                          GraphBLAS::Plus<T>(), flow);
        T maxflow = at(flows,source);

        GraphBLAS::print_matrix(std::cerr, flow, "Final FLOW:");
        return maxflow;
    }

    //************************************************************************
    //************************************************************************
    template<typename MatrixT>
    bool maxflow_bfs(MatrixT const           &graph,
                     GraphBLAS::IndexType     source,
                     GraphBLAS::IndexType     sink,
                     GraphBLAS::Matrix<bool> &M)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::Vector<GraphBLAS::IndexType> parent_list(graph.nrows());
        parent_list.setElement(source, source + 1ul);

        GraphBLAS::Vector<GraphBLAS::IndexType> wavefront(graph.ncols());
        wavefront.setElement(source, 1ul);

        while ((!parent_list.hasElement(sink)) && (wavefront.nvals() > 0))
        {
            // convert all stored values to their 1-based column index
            index_of_1based(wavefront);

            // Select1st because we are left multiplying wavefront rows
            // Masking out the parent list ensures wavefront values do not
            // overlap values already stored in the parent list
            GraphBLAS::vxm(wavefront,
                           GraphBLAS::complement(parent_list),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::MinSelect1stSemiring<GraphBLAS::IndexType,T,GraphBLAS::IndexType>(),
                           wavefront, graph, true);

            GraphBLAS::apply(parent_list,
                             GraphBLAS::NoMask(),
                             GraphBLAS::Plus<GraphBLAS::IndexType>(),
                             GraphBLAS::Identity<GraphBLAS::IndexType>(),
                             wavefront);
        }

        if (!parent_list.hasElement(sink))
        {
            return false;
        }

        // Extract path from source to sink from parent list (reverse traverse)
        // build a mask
        M.clear();
        GraphBLAS::IndexType curr_vertex(sink);
        while (curr_vertex != source)
        {
            GraphBLAS::IndexType parent(parent_list.extractElement(curr_vertex) - 1);
            M.setElement(parent, curr_vertex, true);
            curr_vertex = parent;
        }

        return true;
    }

    //************************************************************************
    template<typename MatrixT>
    typename MatrixT::ScalarType maxflow_ford_fulk(MatrixT const        &graph,
                                                   GraphBLAS::IndexType  source,
                                                   GraphBLAS::IndexType  sink)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType num_nodes(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());

        // assert num_nodes == cols

        GraphBLAS::Matrix<bool> M(num_nodes, num_nodes);
        MatrixT F(num_nodes, num_nodes);
        MatrixT R = graph;
        GraphBLAS::Vector<GraphBLAS::IndexType> parent_list(num_nodes);
        T max_flow(0);

        //GraphBLAS::print_matrix(std::cerr, F, "F");
        //GraphBLAS::print_matrix(std::cerr, R, "R");
        //GraphBLAS::print_matrix(std::cerr, graph, "graph");

        GraphBLAS::IndexType count(0);
        while (maxflow_bfs(R, source, sink, M) &&
               count++ < graph.nvals())
        {
            //std::cerr << "----------- Iteration: " << count << " -----------\n";
            //GraphBLAS::print_matrix(std::cerr, M, "M (path)");

            // gamma = min(M.*R)
            T gamma;
            GraphBLAS::Matrix<T> G(num_nodes, num_nodes);
            GraphBLAS::eWiseMult(G, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<T>(), M, R);
            GraphBLAS::reduce(gamma, GraphBLAS::NoAccumulate(),
                              GraphBLAS::MinMonoid<T>(), G);
            //GraphBLAS::print_matrix(std::cerr, G, "M.*R");
            //std::cerr << "gamma = min(M.*R) = " << gamma << std::endl;

            // GM = gamma*M
            MatrixT GM(num_nodes, num_nodes);
            GraphBLAS::BinaryOp_Bind2nd<T, GraphBLAS::Times<T> > apply_gamma(gamma);
            GraphBLAS::apply(GM, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             apply_gamma, M);
            //GraphBLAS::print_matrix(std::cerr, GM, "gamma*M");

            // F += gamma(M - M')
            //   += gamma*M + (-gamma*M')

            // GM := GM + (-GM')
            GraphBLAS::apply(GM, GraphBLAS::NoMask(), GraphBLAS::Plus<T>(),
                             GraphBLAS::AdditiveInverse<T>(),
                             GraphBLAS::transpose(GM));
            // F := F + GM
            GraphBLAS::eWiseAdd(F, GraphBLAS::NoMask(),
                                GraphBLAS::NoAccumulate(),
                                GraphBLAS::Plus<T>(), F, GM, true);
            //GraphBLAS::print_matrix(std::cerr, GM, "gamma(M - M')");
            //GraphBLAS::print_matrix(std::cerr, F, "F = F + gamma(M - M')");

            // R = graph - F

            // mF := (-F)
            MatrixT mF(num_nodes, num_nodes);
            GraphBLAS::apply(mF, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             GraphBLAS::AdditiveInverse<T>(), F);
            // R := graph + (-F)
            GraphBLAS::eWiseAdd(R, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                GraphBLAS::Plus<T>(), graph, mF, true);
            // Clear the zero's
            GraphBLAS::apply(R, R, GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<T>(), R, true);
            //GraphBLAS::print_matrix(std::cerr, R, "R = graph - F");
        }

        GraphBLAS::print_matrix(std::cerr, F, "Final FLOW:");
        GraphBLAS::Vector<T> sink_edges(num_nodes);
        GraphBLAS::extract(sink_edges,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           F, GraphBLAS::AllIndices(), sink);
        GraphBLAS::reduce(max_flow, GraphBLAS::NoAccumulate(),
                          GraphBLAS::PlusMonoid<T>(), sink_edges);
        std::cerr << "Max flow = " << max_flow << std::endl;
        return max_flow;
    }

    //************************************************************************
    template<typename MatrixT>
    typename MatrixT::ScalarType maxflow_ford_fulk2(MatrixT const        &graph,
                                                    GraphBLAS::IndexType  source,
                                                    GraphBLAS::IndexType  sink)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType num_nodes(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());

        // assert num_nodes == cols

        GraphBLAS::Matrix<bool> M(num_nodes, num_nodes);
        MatrixT F(num_nodes, num_nodes);
        MatrixT G(num_nodes, num_nodes);
        MatrixT GM(num_nodes, num_nodes);
        MatrixT R = graph;
        T max_flow(0);

        //GraphBLAS::print_matrix(std::cerr, R, "R");
        //GraphBLAS::print_matrix(std::cerr, F, "F");

        GraphBLAS::IndexType count(0);
        while (maxflow_bfs(R, source, sink, M) &&
               count++ < graph.nvals())
        {
            //std::cerr << "----------- Iteration: " << count << " -----------\n";
            //GraphBLAS::print_matrix(std::cerr, M, "M (path)");

            // gamma = min(M.*R)
            T gamma;
            GraphBLAS::eWiseMult(G, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                 GraphBLAS::Times<T>(), M, R);
            GraphBLAS::reduce(gamma, GraphBLAS::NoAccumulate(),
                              GraphBLAS::MinMonoid<T>(), G);
            //GraphBLAS::print_matrix(std::cerr, G, "M.*R");
            //std::cerr << "gamma = min(M.*R) = " << gamma << std::endl;

            // GM = gamma*M
            GraphBLAS::BinaryOp_Bind2nd<T, GraphBLAS::Times<T> > apply_gamma(gamma);
            GraphBLAS::apply(GM, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             apply_gamma, M);
            //GraphBLAS::print_matrix(std::cerr, GM, "gamma*M");

            // F += gamma*M
            GraphBLAS::eWiseAdd(F, GraphBLAS::NoMask(),
                                GraphBLAS::NoAccumulate(),
                                GraphBLAS::Plus<T>(), F, GM, true);
            //GraphBLAS::print_matrix(std::cerr, F, "F += gamma*M");

            // R += (-gamma*M)
            GraphBLAS::apply(R, GraphBLAS::NoMask(), GraphBLAS::Plus<T>(),
                             GraphBLAS::AdditiveInverse<T>(), GM);
            //GraphBLAS::print_matrix(std::cerr, R, "R += -(gamma*M)");

            // Clear the zero's
            GraphBLAS::apply(R, R, GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<T>(), R, true);
            //GraphBLAS::print_matrix(std::cerr, R, "R<R> = R (annihilate zeros)");
        }

        GraphBLAS::print_matrix(std::cerr, F, "Final FLOW:");
        GraphBLAS::Vector<T> sink_edges(num_nodes);
        GraphBLAS::extract(sink_edges,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           F, GraphBLAS::AllIndices(), sink);
        GraphBLAS::reduce(max_flow, GraphBLAS::NoAccumulate(),
                          GraphBLAS::PlusMonoid<T>(), sink_edges);
        std::cerr << "Max flow = " << max_flow << std::endl;
        return max_flow;
    }

} // algorithms

#endif // MAXFLOW_HPP
