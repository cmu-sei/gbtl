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
template <typename MatrixT>
static void push(MatrixT const &C,
                 MatrixT       &F,
                 MatrixT       &excess,
                 graphblas::IndexType u,
                 graphblas::IndexType v)
{
    using T = typename MatrixT::ScalarType;
    T a = excess.extractElement(0, u);
    T b = C.extractElement(u, v) - F.extractElement(u, v);
    T send = std::min(a, b);

    F.setElement(u, v, F.extractElement(u, v) + send);
    F.setElement(v, u, F.extractElement(v, u) - send);

    excess.setElement(0, u, excess.extractElement(0, u) - send);
    excess.setElement(0, v, excess.extractElement(0, v) + send);
}

//****************************************************************************
template <typename MatrixT>
static void relabel(MatrixT const &C,
                    MatrixT const &F,
                    MatrixT       &height,
                    graphblas::IndexType u)
{
    using T = typename MatrixT::ScalarType;
    graphblas::IndexType num_nodes, cols;
    C.get_shape(num_nodes, cols);

    T min_height = std::numeric_limits<T>::max();
    for (graphblas::IndexType v = 0; v < num_nodes; ++v)
    {
        if ((C.extractElement(u, v) - F.extractElement(u, v)) > 0)
        {
            T a = height.extractElement(0, v);
            min_height = std::min(min_height, a);
            height.setElement(0, u, min_height + 1);
        }
    }
}

//****************************************************************************
/// @todo Does seen have a different scalar type (IndexType)
template <typename MatrixT>
static void discharge(MatrixT const &C,
                      MatrixT       &F,
                      MatrixT       &excess,
                      MatrixT       &height,
                      MatrixT       &seen,
                      graphblas::IndexType u)
{
    graphblas::IndexType num_nodes, cols;
    C.get_shape(num_nodes, cols);

    while (excess.extractElement(0, u) > 0)
    {
        if (seen.extractElement(0, u) < num_nodes)
        {
            graphblas::IndexType v = seen.extractElement(0, u);
            if (((C.extractElement(u, v) - F.extractElement(u, v)) > 0) &&
                (height.extractElement(0, u) > height.extractElement(0, v)))
            {
                push(C, F, excess, u, v);
            }
            else
            {
                seen.setElement(0, u, seen.extractElement(0, u) + 1);
            }
        }
        else
        {
            relabel(C, F, height, u);
            seen.setElement(0, u, 0);
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
     * each iteration, where an <i>active</i> vertex is a a vertex such that
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
                                         graphblas::IndexType  source,
                                         graphblas::IndexType  sink)
    {
        using T = typename MatrixT::ScalarType;
        graphblas::IndexType rows, cols;
        capacity.get_shape(rows,cols);
        graphblas::IndexType num_nodes = rows;

        MatrixT flow(rows,cols);

        MatrixT height(1, num_nodes);
        MatrixT excess(1, num_nodes);
        MatrixT seen(1, num_nodes);

        std::vector<graphblas::IndexType> list;

        for (graphblas::IndexType i = 0; i < num_nodes; ++i)
        {
            if ((i != source) && (i != sink))
            {
                list.push_back(i);
            }
        }

        height.setElement(0, source, num_nodes);
        excess.setElement(0, source,
                            std::numeric_limits<T>::max());
        for (graphblas::IndexType i = 0; i < num_nodes; ++i)
        {
            push(capacity, flow, excess, source, i);
        }

        graphblas::IndexType  p = 0;
        while (p < (num_nodes - 2))
        {
            graphblas::IndexType u = list[p];
            T old_height = height.extractElement(0, u);
            discharge(capacity, flow, excess, height, seen, u);
            if (height.extractElement(0, u) > old_height)
            {
                graphblas::IndexType t = list[p];

                list.erase(list.begin() + p);
                list.insert(list.begin() + 0, t);

                p = 0;
            }
            else
            {
                p += 1;
            }
        }

        T maxflow = static_cast<T>(0);
        for (graphblas::IndexType i = 0; i < num_nodes; ++i)
        {
            maxflow += flow.extractElement(source, i);
        }

        return maxflow;
    }
} // algorithms

#endif // MAXFLOW_HPP
