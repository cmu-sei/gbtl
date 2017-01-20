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

#ifndef ALGORITHMS_BC_HPP
#define ALGORITHMS_BC_HPP

#include <iostream>

#define GB_DEBUG
#include <graphblas/graphblas.hpp>
#include <graphblas/linalg_utils.hpp>


namespace algorithms
{
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

} // algorithms

#endif // METRICS_HPP
