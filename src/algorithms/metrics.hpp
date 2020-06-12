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

#include <graphblas/graphblas.hpp>
#include <algorithms/sssp.hpp>

//****************************************************************************
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
    typename MatrixT::ScalarType vertex_in_degree(MatrixT const  &graph,
                                                  grb::IndexType  vid)
    {
        using T = typename MatrixT::ScalarType;

        if (vid >= graph.ncols())
        {
            throw grb::DimensionException();
        }

        grb::Vector<T> in_edges(graph.nrows());
        grb::extract(in_edges,
                     grb::NoMask(), grb::NoAccumulate(),
                     graph,
                     grb::AllIndices(),
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
    typename MatrixT::ScalarType vertex_out_degree(MatrixT const  &graph,
                                                   grb::IndexType  vid)
    {
        using T = typename MatrixT::ScalarType;

        if (vid >= graph.nrows())
        {
            throw grb::DimensionException();
        }

        grb::Vector<T> out_edges(graph.nrows());
        grb::extract(out_edges,
                     grb::NoMask(), grb::NoAccumulate(),
                     grb::transpose(graph),
                     grb::AllIndices(),
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
    typename MatrixT::ScalarType vertex_degree(MatrixT const  &graph,
                                               grb::IndexType  vid)
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
    void graph_distance(MatrixT const  &graph,
                        grb::IndexType  sid,
                        VectorT        &result)
    {
        using T = typename MatrixT::ScalarType;
        grb::IndexType rows(graph.nrows());
        grb::IndexType cols(graph.ncols());

        if ((rows != cols) || (rows != result.size()))
        {
            throw grb::DimensionException();
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
        grb::IndexType rows(graph.nrows());
        grb::IndexType cols(graph.ncols());

        if ((rows != cols) ||
            (result.nrows() != rows) ||
            (result.ncols() != cols))
        {
            throw grb::DimensionException();
        }

        result.clear();
        result = grb::scaled_identity<MatrixT>(
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
    typename MatrixT::ScalarType vertex_eccentricity(MatrixT const  &graph,
                                                     grb::IndexType  vid)
    {
        using T = typename MatrixT::ScalarType;
        grb::IndexType rows(graph.nrows());
        grb::IndexType cols(graph.ncols());

        if (rows != cols)
        {
            throw grb::DimensionException();
        }

        grb::Vector<T> result(rows);
        result.setElement(vid, static_cast<T>(0));
        sssp(graph, result);

        T max_sp(0);
        grb::reduce(max_sp, grb::NoAccumulate(),
                    grb::MaxMonoid<T>(),
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
        grb::Vector<T> eccentricity(graph.nrows());
        grb::reduce(eccentricity,
                    grb::NoMask(),
                    grb::NoAccumulate(),
                    grb::Max<T>(),
                    result);

        // Min Reduce the maximums to get the radius scalar
        T radius(0);
        grb::reduce(radius, grb::NoAccumulate(),
                    grb::MinMonoid<T>(),
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
        grb::reduce(diameter, grb::NoAccumulate(),
                    grb::MaxMonoid<T>(),
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
    typename MatrixT::ScalarType closeness_centrality(MatrixT const  &graph,
                                                      grb::IndexType  vid)
    {
        using T = typename MatrixT::ScalarType;

        grb::Vector<T> distances(graph.nrows());
        graph_distance(graph, vid, distances);

        // Accumulate the distances
        T sum(0);
        grb::reduce(sum, grb::NoAccumulate(),
                    grb::PlusMonoid<T>(),
                    distances);
        return sum;
    }

} // algorithms
