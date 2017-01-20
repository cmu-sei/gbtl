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

#ifndef ALGORITHMS_MIS_HPP
#define ALGORITHMS_MIS_HPP

#include <vector>
#include <random>

#include <graphblas/graphblas.hpp>

//****************************************************************************
namespace
{
    //************************************************************************
    std::default_random_engine             generator;
    std::uniform_real_distribution<double> distribution;

    // Return a random value that is scaled by the inverse of the degree.
    class SetRandom
    {
    public:
        typedef double result_type;
        SetRandom() {}

        __device__ __host__ inline result_type operator()(
            double candidate_flag,
            double degree)
        {
            double prob = 0.0;
            if (candidate_flag != 0.0)
            {
                // set probability of selecting node based on 1/degree
                prob =
                    (0.0001 + distribution(generator))/(1.0 + 2.0 * degree);
            }
            return prob;
        }
    };

    //************************************************************************
    // select source node if its probability is greater than max neighbor
    /// @todo Consider moving this to algebra.hpp
    template <typename T>
    class GreaterThan
    {
    public:
        typedef T result_type;

        __device__ __host__ inline result_type operator()(T const &source,
                                                          T const &max_neighbor)
        {
            if (source == 0)
                return 0;
            else if (max_neighbor == 0)
                return 1;
            else
                return ((source > max_neighbor) ?
                        static_cast<T>(1) :
                        static_cast<T>(0));
        }
    };
}

//****************************************************************************
namespace algorithms
{
    /**
     * @brief Convert the selection flags returned by mis into a set of
     *        vertex ID's (based on index position).
     */
    template <typename MatrixT>
    graphblas::IndexArrayType get_vertex_IDs(MatrixT const &independent_set)
    {
        graphblas::IndexArrayType ans;
        graphblas::IndexType rows, cols;
        independent_set.get_shape(rows, cols);
        for (graphblas::IndexType i = 0; i < rows; ++i)
        {
            if (independent_set.get_value_at(i, 0) != independent_set.get_zero())
            {
                ans.push_back(i);
            }
        }
        return ans;
    }

    /**
     * @brief Compute the Maximal Independent Set for a given graph.
     *
     * A variant of Luby's randomized algorithm (Luby 1985) is used:
     *  - A random number scaled by the inverse of the node's degree is assigned
     *    to each candidate node in the graph (algorithm starts with all nodes
     *    in the candidate set).
     *  - Each node's neighbor with the maximum random value is determined.
     *  - If the node has a larger random number than any of its neighbors,
     *    it is added to the independent set (IS).  This has the tendency to
     *    prefer selecting lower degree nodes.
     *  - Each selected node and all of their neighbors are removed from
     *    candidate set
     *  - The process is repeated
     *
     * @note Because of the random component of this algorithm, the MIS
     *       calculated across various calls to <code>mis</code> may vary.
     *
     * @param[in]  graph            NxN adjacency matrix of graph to compute
     *                              the maximal independent set on. This
     *                              must be an unweighted and undirected graph
     *                              (meaning the matrix is symmetric).  The
     *                              structural zero needs to be '0' and edges
     *                              are indicated by '1' to support use of the
     *                              Arithmetic semiring.
     * @param[out] independent_set  Nx1 vector of flags, '1' indicates vertex
     *                              is selected.  Must be empty on call.
     * @param[in]  seed             The seed for the random number generator
     *
     */
    template <typename MatrixT>
    void mis(MatrixT const &graph,
             MatrixT       &independent_set,
             double         seed = 0)
    {
        graphblas::IndexType rows, cols, r, c;
        graph.get_shape(rows, cols);
        independent_set.get_shape(r, c);
        if ((rows != cols) || (rows != r))
        {
            throw graphblas::DimensionException();
        }

        //graphblas::print_matrix(std::cout, graph, "Graph");

        generator.seed(seed);

        typedef graphblas::Matrix<double,
                                  graphblas::DirectedMatrixTag> RealMatrix;
        using T = typename MatrixT::ScalarType;

        // This will hold the set (non-zero implies part of the set
        //RealMatrix independent_set(rows, 1);
        RealMatrix neighbor_max(rows, 1);
        RealMatrix new_members(rows, 1);
        RealMatrix new_neighbors(rows, 1);
        RealMatrix prob(rows, 1);

        RealMatrix candidates(graphblas::fill<RealMatrix>(1.0, rows, 1, 0));

        // Compute the degree of each node, add 1 to prevent divide by zero
        // on isolated nodes.
        RealMatrix degrees(candidates);
        graphblas::row_reduce(
            graph,
            degrees,
            graphblas::math::Plus<T>());
        //graphblas::print_matrix(std::cout, degrees, "degrees");

        while (candidates.get_nnz() > 0)
        {
            //std::cout << "************* ITERATION ************* nnz = "
            //          << candidates.get_nnz() << std::endl;
            //graphblas::print_matrix(std::cout, candidates, "candidates");

            // assign new random values to all non-zero elements (ensures
            // that any ties that may occur between neighbors will eventually
            // be broken.
            graphblas::ewisemult(candidates, degrees, prob, SetRandom());
            //graphblas::print_matrix(std::cout, prob, "prob");

            // find the neighbor of each source node with the max random number
            graphblas::mxm(graph, prob, neighbor_max,
                           graphblas::MaxSelect2ndSemiring<double>());
            //graphblas::print_matrix(std::cout, neighbor_max, "neighbor_max");

            // Select source node if its probability is > neighbor_max
            graphblas::ewiseadd(prob, neighbor_max, new_members,
                                GreaterThan<double>());
            //graphblas::print_matrix(std::cout, new_members, "new_members");

            // Add new members to independent set.
            graphblas::ewiseadd(independent_set, new_members, independent_set,
                                graphblas::math::OrFn<double>());
            //graphblas::print_matrix(std::cout, independent_set, "IS");

            // Zero out candidates of new_members selected for independent set
            graphblas::ewisemult(
                graphblas::negate(new_members), candidates, candidates);
            //graphblas::print_matrix(std::cout, candidates,
            //                        "candidates (sans new_members)");

            if (candidates.get_nnz() == 0)
            {
                break;
            }

            // Neighbors of new members can also be removed
            graphblas::mxm(graph, new_members, new_neighbors,
                           graphblas::MaxSelect2ndSemiring<double>());
            //graphblas::print_matrix(std::cout, new_neighbors,
            //                        "new_member neighbors");

            // Zero out candidates of new member neighbors
            graphblas::ewisemult(
                graphblas::negate(new_neighbors), candidates, candidates);
            //graphblas::print_matrix(std::cout, candidates,
            //                        "candidates (sans new_members' neighbors)");
        }

    }

//****************************************************************************

    // Return a random value that is scaled by the inverse of the degree.
    class SetRandom2
    {
    public:
        typedef float result_type;
        SetRandom2() {}

        __device__ __host__ inline float operator()(
            uint32_t candidate,
            uint32_t degree)
        {
            float prob = 0.f;
            if (candidate)
            {
                prob =
                    (0.0001f + distribution(generator))/(1.f + 2.f * degree);
            }
            return prob;
        }
    };

    template <typename LhsT, typename RhsT, typename ResultT>
    class MaxSelect2ndTest
    {
    public:
        //typedef ScalarT ScalarType;
        typedef ResultT result_type;

        //template<typename LhsT, typename RhsT>
        __host__ __device__ result_type add(result_type const &a,
                                            result_type const &b) const
        {
            //annihilator_max
            if (a == static_cast<result_type>(0))
            {
                return b;
            }
            else if (b == static_cast<result_type>(0))
            {
                return a;
            }
            else
            {
                return (a > b) ? a : b;
            }
        }

        //template<typename LhsT, typename RhsT>
        __host__ __device__ result_type mult(LhsT first,
                                             RhsT second) const
        {
            // select2ndZero
            return (first != static_cast<LhsT>(0)) ?
                    static_cast<result_type>(second) : zero();
        }

        __host__ __device__ result_type zero() const
        {
            return static_cast<result_type>(0);
        }

        __host__ __device__ result_type one() const
        {
            return static_cast<result_type>(1);
        }
    };


    /**
     * @brief Compute the Maximal Independent Set for a given graph.
     *
     * A variant of Luby's randomized algorithm (Luby 1985) is used:
     *  - A random number scaled by the inverse of the node's degree is assigned
     *    to each candidate node in the graph (algorithm starts with all nodes
     *    in the candidate set).
     *  - Each node's neighbor with the maximum random value is determined.
     *  - If the node has a larger random number than any of its neighbors,
     *    it is added to the independent set (IS).  This has the tendency to
     *    prefer selecting lower degree nodes.
     *  - Each selected node and all of their neighbors are removed from
     *    candidate set
     *  - The process is repeated
     *
     * @note Because of the random component of this algorithm, the MIS
     *       calculated across various calls to <code>mis</code> may vary.
     *
     * @param[in]  graph            NxN adjacency matrix of graph to compute
     *                              the maximal independent set on. This
     *                              must be an unweighted and undirected graph
     *                              (meaning the matrix is symmetric).  The
     *                              structural zero needs to be '0' and edges
     *                              are indicated by '1' to support use of the
     *                              Arithmetic semiring.
     * @param[out] independent_set  Nx1 vector of flags, '1' indicates vertex
     *                              is selected.  Must be empty on call.
     * @param[in]  seed             The seed for the random number generator
     *
     */
    template <typename MatrixT>
    void mis2(MatrixT const &graph,
              MatrixT       &independent_set,
              double         seed = 0)
    {
        graphblas::IndexType rows, cols, r, c;
        graph.get_shape(rows, cols);
        independent_set.get_shape(r, c);
        if ((rows != cols) || (rows != r))
        {
            throw graphblas::DimensionException();
        }

        graphblas::print_matrix(std::cout, graph, "Graph");

        generator.seed(seed);

        typedef graphblas::Matrix<uint32_t,
                                  graphblas::DirectedMatrixTag> IntMatrix;
        typedef graphblas::Matrix<float,
                                  graphblas::DirectedMatrixTag> RealMatrix;
        using T = typename MatrixT::ScalarType;

        // This will hold the set (non-zero implies part of the set)
        //IntMatrix independent_set(rows, 1);
        RealMatrix prob(rows, 1);
        RealMatrix neighbor_max(rows, 1);

        IntMatrix new_members(rows, 1);
        IntMatrix new_neighbors(rows, 1);
        IntMatrix candidates(graphblas::fill<IntMatrix>(1, rows, 1, 0));
        IntMatrix degrees(rows, 1);

        // Compute the degree of each node, add 1 to prevent divide by zero
        // on isolated nodes.
        graphblas::row_reduce(
            graph,
            degrees,
            graphblas::math::Plus<T>());
        graphblas::print_matrix(std::cout, degrees, "degrees");

        MaxSelect2ndTest<uint32_t, float, float> maxSelect2ndIFF;

        while (candidates.get_nnz() > 0)
        {
            std::cout << "************* ITERATION ************* nnz = "
                      << candidates.get_nnz() << std::endl;
            graphblas::print_matrix(std::cout, candidates, "candidates");

            // assign new random values to all non-zero elements (ensures
            // that any ties that may occur between neighbors will eventually
            // be broken.
            // Can be replaced with apply masked by candidates.
            graphblas::ewisemult(candidates, degrees, prob, SetRandom2());
            graphblas::print_matrix(std::cout, prob, "prob");

            // find the neighbor of each source node with the max random number
            graphblas::mxv(graph, prob, neighbor_max, maxSelect2ndIFF);
            graphblas::print_matrix(std::cout, neighbor_max, "neighbor_max");

            // Select source node if its probability is > neighbor_max
            graphblas::ewiseadd(prob, neighbor_max, new_members,
                                GreaterThan<double>());
            graphblas::print_matrix(std::cout, new_members, "new_members");

            // Add new members to independent set.
            graphblas::ewiseadd(independent_set, new_members, independent_set,
                                graphblas::math::OrFn<double>());
            graphblas::print_matrix(std::cout, independent_set, "IS");

            // Zero out candidates of new_members selected for independent set
            graphblas::ewisemult(
                graphblas::negate(new_members), candidates, candidates);
            graphblas::print_matrix(std::cout, candidates,
                                    "candidates (sans new_members)");

            if (candidates.get_nnz() == 0)
            {
                break;
            }

            // Neighbors of new members can also be removed
            graphblas::mxm(graph, new_members, new_neighbors, maxSelect2ndIFF);
            graphblas::print_matrix(std::cout, new_neighbors,
                                    "new_member neighbors");

            // Zero out candidates of new member neighbors
            graphblas::ewisemult(
                graphblas::negate(new_neighbors), candidates, candidates);
            graphblas::print_matrix(std::cout, candidates,
                                    "candidates (sans new_members' neighbors)");
        }

    }
} // algorithms

#endif // MIS_HPP
