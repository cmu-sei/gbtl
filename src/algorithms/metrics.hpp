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
            if (graph.get_value_at(vid, i) != graph.get_zero())
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
            if (graph.get_value_at(i, vid) != graph.get_zero())
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
        start.set_value_at(sid, sid, static_cast<T>(0));

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
        start.set_value_at(vid, vid, static_cast<T>(0));

        MatrixT result(rows, rows);
        sssp(graph, start, result);

        T current_max = result.get_value_at(vid, 0);
        for (graphblas::IndexType i = 1; i < rows; ++i)
        {
            current_max = std::max(current_max,
                                   result.get_value_at(vid, i));
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

        T sum = result.get_value_at(vid, 0);
        for (graphblas::IndexType i = 1; i < cols; ++i)
        {
            sum += result.get_value_at(vid, i);
        }
        return sum;
    }

    /**
     * @brief Compute the vertex betweenness centrality of all vertices in the
     *        given graph.
     *
     * The betweenness centrality of a vertex measures the number of
     * times a vertex acts as a bridge along the shortest path between two
     * vertices.
     *
     * \f$C_b(v)=\sum\limits_{s \\neq v \in V}\sum\limits_{t \\neq v \in V}\\frac{\sigma_{st}(v)}{\sigma_{st}}\f$
     *
     * @param[in]  graph  The graph to compute the betweenness centrality of.
     *
     * @return The betweenness centrality of all vertices in the graph.
     */
    template<typename MatrixT>
    std::vector<typename MatrixT::ScalarType>
    vertex_betweenness_centrality(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;
        graphblas::IndexType num_nodes, cols, depth;
        graph.get_shape(num_nodes, cols);
        if (num_nodes != cols)
        {
            throw graphblas::DimensionException();
        }

        MatrixT bc_mat(1, num_nodes);
        for (graphblas::IndexType i = 0; i < num_nodes; ++i)
        {
            depth = 0;
            MatrixT search(num_nodes, num_nodes);
            MatrixT fringe(1, num_nodes);
            /// @todo replaced with extract?
            for (graphblas::IndexType k = 0; k < num_nodes; ++k)
            {
                fringe.set_value_at(0, k, graph.get_value_at(i, k));
            }

            MatrixT n_shortest_paths(1, num_nodes);
            n_shortest_paths.set_value_at(0, i, static_cast<T>(1));

            while(fringe.get_nnz() != 0)
            {
                depth = depth + 1;
                graphblas::ewiseadd(n_shortest_paths,fringe,n_shortest_paths);

                for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                {
                    search.set_value_at(
                        depth, k,
                        (fringe.get_value_at(0, k) != 0));  // cast<T>?
                }

                MatrixT not_in_sps = graphblas::fill<MatrixT>(1, 1, num_nodes);
                //for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                //{
                //    not_in_sps.set_value_at(0, k, 1);
                //}

                for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                {
                    if (n_shortest_paths.get_value_at(0, k) != 0)
                    {
                        not_in_sps.set_value_at(0, k, 0); // get_zero()
                    }
                }
                graphblas::mxm(fringe, graph, fringe);
                graphblas::ewisemult(fringe, not_in_sps, fringe);
            }

            MatrixT update(1, num_nodes);
            MatrixT ones = graphblas::fill<MatrixT>(1, 1, num_nodes);

            while (depth >= 2)
            {
                MatrixT n_shortest_paths_inv(1, num_nodes);
                graphblas::apply(n_shortest_paths,
                                 n_shortest_paths_inv,
                                 graphblas::math::inverse<T>);

                MatrixT weights(num_nodes, 1);
                MatrixT temp(1, num_nodes);

                graphblas::ewiseadd(ones, update, temp);
                graphblas::ewisemult(temp, n_shortest_paths_inv, temp);

                for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                {
                    //weights[k][0] = search[depth][k] * temp[0][k];
                    weights.set_value_at(
                        k, 0,
                        search.get_value_at(depth, k) *
                        temp.get_value_at(0, k));
                }

                mxm(graph, weights, weights);

                for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                {
                    //temp[0][k] = search[depth-1][k] * n_shortest_paths[0][k];
                    temp.set_value_at(
                        0, k,
                        search.get_value_at(depth - 1, k) *
                        n_shortest_paths.get_value_at(0, k));
                }

                graphblas::ewisemult(weights,
                                     //graphblas::TransposeView<MatrixT>(temp),
                                     graphblas::transpose(temp),
                                     weights);

                graphblas::ewiseadd(update,
                                    //graphblas::TransposeView<MatrixT>(weights),
                                    graphblas::transpose(weights),
                                    update);
                depth = depth - 1;
            }

            graphblas::ewiseadd(bc_mat, update, bc_mat);
        }

        std::vector <typename MatrixT::ScalarType> betweenness_centrality;
        for (graphblas::IndexType k = 0; k<num_nodes;k++)
        {
            betweenness_centrality.push_back(bc_mat.get_value_at(0, k));
        }

        return betweenness_centrality;
    }

    /**
     * @brief Compute the edge betweenness centrality of all vertices in the
     *        given graph.
     *
     * The betweenness centrality of a vertex measures the number of
     * times an edge acts as a bridge along the shortest path between two
     * vertices.  Formally stated:
     *
     * \f$C_b(v)=\sum\limits_{s \\neq v \in V}\sum\limits_{t \\neq v \in V}\\frac{\sigma_{st}(v)}{\sigma_{st}}\f$
     *
     * @param[in] graph The graph to compute the edge betweenness centrality
     *
     * @return The betweenness centrality of all vertices in the graph
     */
    template<typename MatrixT>
    MatrixT edge_betweenness_centrality(MatrixT const &graph)
    {
        using T = typename MatrixT::ScalarType;
        graphblas::IndexType num_nodes, cols, depth;
        graph.get_shape(num_nodes, cols);

        if (num_nodes != cols)
        {
            throw graphblas::DimensionException();
        }

        MatrixT score(num_nodes, num_nodes);

        for (graphblas::IndexType root = 0; root < num_nodes; root++)
        {
            depth = 0;
            MatrixT search(num_nodes, num_nodes);
            MatrixT fringe (1,num_nodes);
            for (graphblas::IndexType k = 0; k < num_nodes; ++k)
            {
                //fringe[0][k] = graph[root][k];
                fringe.set_value_at(0, k,
                                    graph.get_value_at(root, k));
            }
            MatrixT n_shortest_paths(1, num_nodes);
            //n_shortest_paths[0][root] = 1;
            n_shortest_paths.set_value_at(0, root, 1);

            MatrixT update(num_nodes, num_nodes);
            MatrixT flow(1, num_nodes);
            //search[depth][root] = 1;
            search.set_value_at(depth, root, 1);

            while (fringe.get_nnz() != 0)
            {
                depth = depth + 1;
                graphblas::ewiseadd(n_shortest_paths,
                                    fringe,
                                    n_shortest_paths);
                for (graphblas::IndexType i = 0; i < num_nodes; ++i)
                {
                    //search[depth][i] = (fringe[0][i] != 0);
                    search.set_value_at(depth, i,
                                        (fringe.get_value_at(0, i) != 0));
                }

                MatrixT not_in_sps = graphblas::fill<MatrixT>(1, 1, num_nodes);
                //for (graphblas::IndexType k = 0; k < num_nodes; k++)
                //{
                //    not_in_sps[0][k] = 1;
                //}

                for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                {
                    if (n_shortest_paths.get_value_at(0, k) != 0)
                    {
                        not_in_sps.set_value_at(0, k, 0);  // get_zero()?
                    }
                }

                graphblas::mxm(fringe, graph, fringe);
                graphblas::ewisemult(fringe, not_in_sps, fringe);
            }

            while (depth >= 1)
            {
                MatrixT n_shortest_paths_inv(1, num_nodes);
                graphblas::apply(n_shortest_paths,
                                 n_shortest_paths_inv,
                                 graphblas::math::inverse<T>);
                MatrixT weights(1, num_nodes);

                for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                {
                    //weights[0][k] =
                    //    search[depth][k] * n_shortest_paths_inv[0][k];
                    weights.set_value_at(
                        0, k,
                        search.get_value_at(depth, k) *
                        n_shortest_paths_inv.get_value_at(0, k));
                }

                graphblas::ewisemult(weights, flow, weights);

                for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                {
                    //weights[0][k] = weights[0][k] + search[depth][k];
                    weights.set_value_at(
                        0, k,
                        weights.get_value_at(0, k) +
                        search.get_value_at(depth, k));
                }


                for (graphblas::IndexType i = 0; i < num_nodes; i++)
                {
                    for (graphblas::IndexType k=0;k<num_nodes; k++)
                    {
                        //update[i][k] = graph[i][k] * weights[0][k];
                        update.set_value_at(
                            i, k,
                            graph.get_value_at(i, k) *
                            weights.get_value_at(0, k));
                    }
                }

                for (graphblas::IndexType k=0;k<num_nodes; k++)
                {
                    //weights[0][k] =
                    //    search[depth-1][k] * n_shortest_paths[0][k];
                    weights.set_value_at(
                        0, k,
                        search.get_value_at(depth - 1, k) *
                        n_shortest_paths.get_value_at(0, k));
                }

                for (graphblas::IndexType i = 0; i < num_nodes; ++i)
                {
                    for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                    {
                        //update[k][i] = weights[0][k] * update[k][i];
                        update.set_value_at(
                            k, i,
                            weights.get_value_at(0, k) *
                            update.get_value_at(k, i));
                    }
                }
                graphblas::ewiseadd(score, update, score);

                MatrixT temp(num_nodes, 1);
                row_reduce(update, temp);
                for (graphblas::IndexType k = 0; k < num_nodes; ++k)
                {
                    //flow[0][k] = temp[k][0];
                    flow.set_value_at(0, k, temp.get_value_at(k, 0));
                }

                depth = depth - 1;
            }
        }
        return score;
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
                sum = sum + C.get_value_at(i, j);
            }
        }
        return sum / static_cast<T>(2);
    }
} // algorithms

#endif // METRICS_HPP
