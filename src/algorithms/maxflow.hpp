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

#ifndef MAXFLOW_HPP
#define MAXFLOW_HPP

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

        GraphBLAS::print_matrix(std::cerr, flow, "\nFLOW");

        GraphBLAS::IndexType  p = 0;
        while (p < (num_nodes - 2))
        {
            GraphBLAS::IndexType u = list[p];
            T old_height = at(height, u);
            discharge(capacity, flow, excess, height, seen, u);
            GraphBLAS::print_matrix(std::cerr, flow, "\nFLOW after discharge");
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
        GraphBLAS::print_matrix(std::cerr, flow, "\nFlow");
        GraphBLAS::Vector<T> flows(rows);
        GraphBLAS::reduce(flows, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                          GraphBLAS::Plus<T>(), flow);
        T maxflow = at(flows,source);

        return maxflow;
    }

//****************************************************************************
    template<typename MatrixT>
    bool bfs_wrapper(MatrixT const        &graph,
                     GraphBLAS::IndexType  source,
                     GraphBLAS::IndexType  sink
                     GraphBLAS::Vector<GraphBLAS::IndexType> &parent_list)
    {
        GraphBLAS::Vector<bool> wavefront(graph.ncols());
        wavefront.setElement(source, true);
        parent_list.clear();

        algorithms::bfs(graph, wavefront, parent_list);

        return parent_list.hasElement(sink);
    }

    template<typename MatrixT>
    typename MatrixT::ScalarType maxflow_ford_fulk(MatrixT const        &graph,
                                                   GraphBLAS::IndexType  source,
                                                   GraphBLAS::IndexType  sink)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType num_nodes(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());

        // assert num_nodes == cols

        MatrixT rGraph = graph;
        GraphBLAS::Vector<IndexType> parent_list(num_nodes);
        T max_flow(0);

        while (bfs_wrapper(rGraph, source, sink, parent_list))
        {
            // There exists a path to sink from source
            GraphBLAS::IndexType u = parent_list.getElement(sink);
            T path_flow = rGraph.getElement(u, sink);

            for (GraphBLAS::IndexType v = sink;
                 v != source;
                 v = parent_list.getElement(v))
            {
                GraphBLAS::IndexType u = parent_list.getElement(v);
            }
        }
    }

} // algorithms

#endif // MAXFLOW_HPP
