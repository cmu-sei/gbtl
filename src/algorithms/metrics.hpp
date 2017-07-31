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

#ifndef ALGORITHMS_METRICS_HPP
#define ALGORITHMS_METRICS_HPP

#include <iostream>

#define GB_DEBUG
#include <graphblas/graphblas.hpp>
#include <algorithms/sssp.hpp>


namespace algorithms
{
    /**
     * @brief Compute the in-degree of a vertex in the given graph.
     *
     * @param[in] graph  The graph to compute the in-degree of a vertex in.
     * @param[in] vid    The vertex to compute the in-degree of.
     *
     * @return The in-degree of vertex vid (G(:,vid).nvals())
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType vertex_in_degree(MatrixT const        &graph,
                                                  GraphBLAS::IndexType  vid)
    {
        using T = typename MatrixT::ScalarType;

        if (vid >= graph.ncols())
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::Vector<T> in_edges(graph.nrows());
        GraphBLAS::extract(in_edges,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           graph,
                           GraphBLAS::GrB_ALL,
                           vid);

        return in_edges.nvals();
    }


    /**
     * @brief Compute the out-degree of a vertex in the given graph.
     *
     * @param[in] graph  The graph to compute the out-degree of a vertex in.
     * @param[in] vid    The vertex to compute the out-degree of.
     *
     * @return The out-degree of vertex vid (G(vid,:).nvals())
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType vertex_out_degree(MatrixT const        &graph,
                                                   GraphBLAS::IndexType  vid)
    {
        using T = typename MatrixT::ScalarType;

        if (vid >= graph.nrows())
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::Vector<T> out_edges(graph.nrows());
        GraphBLAS::extract(out_edges,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           GraphBLAS::transpose(graph),
                           GraphBLAS::GrB_ALL,
                           vid);

        return out_edges.nvals();
    }


    /**
     * @brief Compute the degree (in and out) of a vertex in the given graph.
     *
     * @param[in] graph The graph to compute the degree of a vertex in.
     * @param[in] vid   The vertex to compute the degree of.
     *
     * @return The degree of vertex vid.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType vertex_degree(MatrixT const        &graph,
                                               GraphBLAS::IndexType  vid)
    {
        return vertex_in_degree(graph, vid) + vertex_out_degree(graph, vid);
    }


    /**
     * @brief Compute the distances from a given starting vertex to all other
     * vertices in the graph.
     *
     * The distance between two vertices u and v is the length of the shortest
     * path between u and v.
     *
     * @param[in]  graph   The graph to compute the distances in.
     * @param[in]  sid     The starting vertex.
     * @param[out] result  The distances from s to all other
     *                     vertices in graph.
     *
     * @note Uses SSSP algorithm to compute the result
     */
    template<typename MatrixT, typename VectorT>
    void graph_distance(MatrixT const        &graph,
                        GraphBLAS::IndexType  sid,
                        VectorT              &result)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());

        if ((rows != cols) || (rows != result.size()))
        {
            throw GraphBLAS::DimensionException();
        }

        result.clear();
        result.setElement(sid, static_cast<T>(0));

        sssp(graph, result);
    }


    /**
     * @brief Compute the graph distance matrix for a graph.
     *
     * The distance matrix of a graph in a graph is a matrix
     * where the i,j entry is the distance from vertex i to vertex j
     *
     * @param[in]  graph   The graph to compute the distance matrix of.
     * @param[out] result  The graph distance matrix.
     */
    template<typename MatrixT>
    void graph_distance_matrix(MatrixT const &graph, MatrixT &result)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());

        if ((rows != cols) ||
            (result.nrows() != rows) ||
            (result.ncols() != cols))
        {
            throw GraphBLAS::DimensionException();
        }

        result.clear();
        result = GraphBLAS::scaled_identity<MatrixT>(
            rows,
            static_cast<T>(0));

        batch_sssp(graph, result);
    }

    /**
     * @brief Calculate the vertex eccentricity for a vertex in the given
     *        graph.
     *
     * Vertex eccentricity is the length of the longest shortest path
     * from a vertex (vid) to any other vertex, or:
     *
     * <center><p>\f$\epsilon(v) = \max\limits_{w \in V} d(v, w)\f$</p></center>
     *
     * @param[in] graph The graph to calculate the vertex eccentricity of the
     *                  given vertex in.
     * @param[in] vid   The vertex to calculate the vertex eccentricity of.
     *
     * @return The vertex eccentricity of vid in graph.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType vertex_eccentricity(
        MatrixT const        &graph,
        GraphBLAS::IndexType  vid)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());

        if (rows != cols)
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::Vector<T> result(rows);
        result.setElement(vid, static_cast<T>(0));
        sssp(graph, result);

        T max_sp(0);
        GraphBLAS::reduce(max_sp, GraphBLAS::NoAccumulate(),
                          GraphBLAS::MaxMonoid<T>(),
                          result);

        return max_sp;
    }


    /**
     * @brief Compute the radius of the given graph.
     *
     * The radius of a graph is the minimum eccentricity of
     * any vertex in that graph.
     *
     * @param[in] graph The graph to compute the radius of.
     *
     * @return The radius of graph.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType graph_radius(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;

        MatrixT result(graph.nrows(), graph.ncols());
        graph_distance_matrix(graph, result);
        //print_matrix(std::cout, result, "GRAPH DISTANCE MATRIX");

        // Max Reduce across the rows for eccentricity
        GraphBLAS::Vector<T> eccentricity(graph.nrows());
        GraphBLAS::reduce(eccentricity,
                          GraphBLAS::NoMask(),
                          GraphBLAS::NoAccumulate(),
                          GraphBLAS::Max<T>(),
                          result);

        // Min Reduce the maximums to get the radius scalar
        T radius(0);
        GraphBLAS::reduce(radius, GraphBLAS::NoAccumulate(),
                          GraphBLAS::MinMonoid<T>(),
                          eccentricity);

        return radius;
    }


    /**
     * @brief Compute the diameter of the given graph.
     *
     * The diameter of a graph is the maximum eccentricity of
     * any vertex in that graph.
     *
     * @param[in]  graph  The graph to compute the diameter of.
     *
     * @return The diameter of graph.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType graph_diameter(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;

        MatrixT result(graph.nrows(), graph.ncols());
        graph_distance_matrix(graph, result);
        //print_matrix(std::cout, result, "GRAPH DISTANCE MATRIX");

        // Max Reduce to scalar the distance matrix
        T diameter(0);
        GraphBLAS::reduce(diameter, GraphBLAS::NoAccumulate(),
                          GraphBLAS::MaxMonoid<T>(),
                          result);
        return diameter;
    }


    /**
     * @brief Compute the closeness centrality of a vertex in the given graph.
     *
     * The closeness centrality of a vertex attempts to measure how
     * central a vertex is in a given graph.  A concept that is related
     * to closeness centrality is the "far-ness" of a vertex, which is the
     * sum of the distances to that vertex from all other vertices.  Given a
     * vertex, v, the far-ness of v would then be calculated as
     *
     * \f$\delta(u, v) = \sum\limits_{u \in V}d(u, v)\f$.
     *
     * The closeness centrality of v would then be defined as
     *
     * \f$C(x)=\frac{1}{\delta(u,v)}=\frac{1}{\sum\limits_{u \in V}d(u,v)}\f$.
     *
     * @param[in]  graph  The graph to compute the closeness centrality of a
     *                    vertex in.
     * @param[in]  vid    The vertex to compute the closeness centrality of.
     *
     * @return The closeness centrality of vid
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType closeness_centrality(
        MatrixT const        &graph,
        GraphBLAS::IndexType  vid)
    {
        using T = typename MatrixT::ScalarType;

        GraphBLAS::Vector<T> distances(graph.nrows());
        graph_distance(graph, vid, distances);

        // Accumulate the distances
        T sum(0);
        GraphBLAS::reduce(sum, GraphBLAS::NoAccumulate(),
                          GraphBLAS::PlusMonoid<T>(),
                          distances);
        return sum;
    }

} // algorithms

#endif // METRICS_HPP
