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
#include <graphblas/linalg_utils.hpp>
#include <algorithms/sssp.hpp>


namespace algorithms
{
    /**
     * @brief Calculate the number of vertices in the graph.
     *
     * @todo This is taken from the number of rows in the matrix.  This needs
     *       work, the matrix dimension may not be indicative of the number
     *       of vertices in use.
     *
     * @param[in] graph The graph to calculate the number of vertices in.
     *
     * @return The number of vertices in <code>graph</code>.
     */
    template<typename MatrixT>
    int vertex_count(MatrixT const &graph)
    {
        graphblas::IndexType rows, cols;
        graph.get_shape(rows,cols);
        return rows;
    }


    /**
     * @brief Calculate the number of edges in the graph.
     *
     * @param[in] graph The graph to calculate the number of vertices in.
     *
     * @return The number of edges in <code>graph</code>.
     */
    template<typename MatrixT>
    graphblas::IndexType edge_count(MatrixT const &graph)
    {
        return graph.get_nnz();
    }


    /**
     * @brief Compute the in-degree of a vertex in the given graph.
     *
     * @param[in] graph  The graph to compute the in-degree of a vertex in.
     * @param[in] vid    The vertex to compute the in-degree of.
     *
     * @return The in-degree of vertex vid
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType vertex_in_degree(MatrixT const        &graph,
                                                  graphblas::IndexType  vid)
    {
        graphblas::IndexType rows, cols;
        graph.get_shape(rows, cols);

        if (vid >= rows)
        {
            throw graphblas::DimensionException();
        }

        graphblas::IndexType in_degree = 0;
        for (graphblas::IndexType i = 0; i < cols; ++i)
        {
            if (graph.extractElement(vid, i) != graph.get_zero())
            {
                ++in_degree;
            }
        }
        return in_degree;
    }


    /**
     * @brief Compute the out-degree of a vertex in the given graph.
     *
     * @param[in] graph  The graph to compute the out-degree of a vertex in.
     * @param[in] vid    The vertex to compute the out-degree of.
     *
     * @return The out-degree of vertex vid
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType vertex_out_degree(MatrixT const        &graph,
                                                   graphblas::IndexType  vid)
    {
        graphblas::IndexType rows, cols;
        graph.get_shape(rows, cols);

        if (vid >= cols)
        {
            throw graphblas::DimensionException();
        }

        graphblas::IndexType out_degree = 0;
        for (graphblas::IndexType i = 0; i < rows; ++i)
        {
            if (graph.extractElement(i, vid) != graph.get_zero())
            {
                out_degree++;
            }
        }
        return out_degree;
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
                                               graphblas::IndexType  vid)
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
     * @param[out] result  The distance from s to all other
     *                     vertices in graph.
     *
     * @note Uses SSSP algorithm to compute the result
     */
    template<typename MatrixT>
    void graph_distance(MatrixT const        &graph,
                        graphblas::IndexType  sid,
                        MatrixT              &result)
    {
        using T = typename MatrixT::ScalarType;
        graphblas::IndexType rows, cols;
        graph.get_shape(rows, cols);

        if (rows != cols)
        {
            throw graphblas::DimensionException();
        }

        MatrixT start(rows, rows, std::numeric_limits<T>::max());
        start.setElement(sid, sid, static_cast<T>(0));

        sssp(graph, start, result);
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
        graphblas::IndexType rows, cols;
        graph.get_shape(rows, cols);

        if (rows != cols)
        {
            throw graphblas::DimensionException();
        }

        MatrixT start = graphblas::identity<MatrixT>(
            rows,
            std::numeric_limits<T>::max(),
            static_cast<T>(0));

        sssp(graph, start, result);
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
        graphblas::IndexType  vid)
    {
        using T = typename MatrixT::ScalarType;
        graphblas::IndexType rows, cols;
        graph.get_shape(rows, cols);

        if (rows != cols)
        {
            throw graphblas::DimensionException();
        }

        MatrixT start(rows, rows,
                      std::numeric_limits<T>::max());
        start.setElement(vid, vid, static_cast<T>(0));

        MatrixT result(rows, rows);
        sssp(graph, start, result);

        T current_max = result.extractElement(vid, 0);
        for (graphblas::IndexType i = 1; i < rows; ++i)
        {
            current_max = std::max(current_max,
                                   result.extractElement(vid, i));
        }
        return current_max;
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
        graphblas::IndexType rows, cols;
        graph.get_shape(rows, cols);

        T current_min = vertex_eccentricity(graph, 0);
        for (graphblas::IndexType i = 1; i < rows; ++i)
        {
            current_min = std::min(current_min,
                                   vertex_eccentricity(graph, i));
        }
        return current_min;
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
        graphblas::IndexType rows, cols;
        graph.get_shape(rows, cols);

        T current_max = vertex_eccentricity(graph, 0);
        for (graphblas::IndexType i = 1; i < rows; i++)
        {
            current_max = std::max(current_max,
                                   vertex_eccentricity(graph, i));
        }
        return current_max;
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
        graphblas::IndexType  vid)
    {
        using T = typename MatrixT::ScalarType;
        graphblas::IndexType rows, cols;
        graph.get_shape(rows, cols);

        MatrixT result(rows, cols);
        graph_distance(graph, vid, result);

        T sum = result.extractElement(vid, 0);
        for (graphblas::IndexType i = 1; i < cols; ++i)
        {
            sum += result.extractElement(vid, i);
        }
        return sum;
    }

    /**
     * @brief Compute the degree centrality of a vertex in the given graph.
     *
     * Given a vertex, vid, its degree centrality is its degree.
     *
     * @param[in]  graph  The graph containing vid
     * @param[in]  vid    The vertex to compute the degree centrality of.
     *
     * @return The degree centrality of vid
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType degree_centrality(
        MatrixT const        &graph,
        graphblas::IndexType  vid)
    {
        return vertex_degree(graph, vid);
    }

    /**
     * @brief Compute the number of triangles in a given graph.
     *
     * This function expects an undirected graph.  If it is desired
     * to count the number of edges in a digraph, just pass in the
     * digraph, and then multiply the resulting number of triangles by 2.
     *
     * Given the adjacency matrix of a graph, the idea behind the
     * triangle counting algorithm used is as follows:
     * <ol>
     * <li>First, split \f$A\f$ into lower and upper triangular matrices such
     * that \f$A = L + U\f$.</li>
     *
     * <li>Because the multiplication of \f$L\f$ and \f$U\f$ counts all the
     * wedges \f$(i, j, k)\f$, where \f$j\f$ is the vetex with the lowest
     * degree, we compute the matrix \f$B = LU\f$.</li>
     *
     * <li>Finally, to determine whether the wedges close or not, we compute
     * \f$C = A \circ B\f$.</li>
     *
     * <li>The final number of triangles is then
     * \f$\sum\limits_i^N\sum\limits_j^N C_{ij}\f$.</li>
     * </ol>
     *
     * However, when implementing the algorithm, various optimizations were
     * made to improve performance.
     *
     * @param[in]  graph  The graph to compute the number of triangles in.
     *
     * @return The number of triangles in graph.
     */
    template<typename MatrixT>
    typename MatrixT::ScalarType triangle_count(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;
        graphblas::IndexType rows, cols;
        graph.get_shape(rows, cols);

        MatrixT L(rows, cols), U(rows, cols);
        graphblas::split(graph, L, U);

        MatrixT B(rows, cols);
        graphblas::mxm(L, U, B);

        MatrixT C(rows, cols);
        graphblas::ewisemult(graph, B, C);

        T sum = 0;
        for (graphblas::IndexType i = 0; i< rows; ++i)
        {
            for (graphblas::IndexType j = 0; j < cols; ++j)
            {
                sum = sum + C.extractElement(i, j);
            }
        }
        return sum / static_cast<T>(2);
    }
} // algorithms

#endif // METRICS_HPP
