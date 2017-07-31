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
    template <typename T=float>
    class SetRandom
    {
    public:
        typedef T result_type;
        SetRandom() {}

        template <typename DegreeT>
        __device__ __host__ inline result_type operator()(
            bool,   // candidate_flag,
            DegreeT degree)
        {
            return static_cast<T>(0.0001 +
                                  distribution(generator)/(1. + 2.*degree));
        }
    };

    //************************************************************************
    // select source node if its probability is greater than max neighbor
    /// @todo Consider moving this to algebra.hpp
    // template <typename T>
    // class GreaterThan
    // {
    // public:
    //     typedef T result_type;

    //     __device__ __host__ inline result_type operator()(T const &source,
    //                                                       T const &max_neighbor)
    //     {
    //         if (source == 0)
    //             return 0;
    //         else if (max_neighbor == 0)
    //             return 1;
    //         else
    //             return ((source > max_neighbor) ?
    //                     static_cast<T>(1) :
    //                     static_cast<T>(0));
    //     }
    // };
}

//****************************************************************************
namespace algorithms
{
    /**
     * @brief Convert the selection flags returned by MIS into a set of
     *        vertex ID's (based on index position).
     *
     * @param[in] independent_set  Assumes a stored value in this vector
     *                             implies the corresponding vertex is selected.
     */
    template <typename VectorT>
    GraphBLAS::IndexArrayType get_vertex_IDs(VectorT const &independent_set)
    {
        using T = typename VectorT::ScalarType;
        GraphBLAS::IndexType set_size(independent_set.nvals());
        GraphBLAS::IndexArrayType ans(set_size);
        std::vector<T> vals(set_size);

        independent_set.extractTuples(ans.begin(), vals.begin());

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
     * @param[out] independent_set  N-vector of flags, 'true' indicates vertex
     *                              is selected.  Must be empty on call.
     * @param[in]  seed             The seed for the random number generator
     *
     */
    template <typename MatrixT>
    void mis(MatrixT const            &graph,
             GraphBLAS::Vector<bool>  &independent_set,
             double                    seed = 0)
    {
        GraphBLAS::IndexType num_vertices(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());
        GraphBLAS::IndexType r(independent_set.size());

        if ((num_vertices != cols) || (num_vertices != r))
        {
            throw GraphBLAS::DimensionException();
        }

        //GraphBLAS::print_matrix(std::cout, graph, "Graph");

        generator.seed(seed);

        using T = typename MatrixT::ScalarType;
        using RealT = float;
        typedef GraphBLAS::Vector<RealT> RealVector;
        typedef GraphBLAS::Vector<bool> BoolVector;

        RealVector prob(num_vertices);
        RealVector neighbor_max(num_vertices);
        BoolVector new_members(num_vertices);
        BoolVector new_neighbors(num_vertices);

        // Compute the degree of each node,
        GraphBLAS::Vector<RealT> degrees(num_vertices);
        GraphBLAS::reduce(degrees,
                          GraphBLAS::NoMask(), GraphBLAS::Plus<RealT>(),
                          GraphBLAS::PlusMonoid<RealT>(),
                          graph);
        //GraphBLAS::print_vector(std::cout, degrees, "degrees");

        // Start with all vertices except isolated ones as candidates (NOTE: dense!)
        BoolVector candidates(num_vertices);
        GraphBLAS::assign_constant(candidates,
                                   degrees,
                                   GraphBLAS::NoAccumulate(),
                                   true, GraphBLAS::GrB_ALL, true);

        // Courtesy of Tim Davis: singletons are not candidates.  Add to iset
        GraphBLAS::assign_constant(independent_set,
                                   GraphBLAS::complement(degrees),
                                   GraphBLAS::NoAccumulate(),
                                   true, GraphBLAS::GrB_ALL, true);

        while (candidates.nvals() > 0)
        {
            //std::cout << "************* ITERATION ************* nnz = "
            //          << candidates.nvals() << std::endl;
            //GraphBLAS::print_vector(std::cout, candidates, "candidates");

            // assign new random values to all non-zero elements (ensures
            // that any ties that may occur between neighbors will eventually
            // be broken.
            GraphBLAS::eWiseMult(prob,
                                 GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                 SetRandom<RealT>(),
                                 candidates, degrees);
            //GraphBLAS::print_vector(std::cout, prob, "prob");

            // find the neighbor of each source node with the max random number
            GraphBLAS::mxv(neighbor_max,
                           candidates, GraphBLAS::NoAccumulate(),
                           GraphBLAS::MaxSelect2ndSemiring<RealT>(),
                           graph, prob, true);
            //GraphBLAS::print_vector(std::cout, neighbor_max, "neighbor_max");

            // Select source node if its probability is > neighbor_max
            GraphBLAS::eWiseAdd(new_members,
                                GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                GraphBLAS::GreaterThan<RealT>(),
                                prob, neighbor_max);
            // Note the result above IS DENSE, I can add the following apply (or
            // many other operations to make it sparse (just add itself as the
            // mask).
            GraphBLAS::apply(new_members,
                             new_members, GraphBLAS::NoAccumulate(),
                             GraphBLAS::Identity<bool>(),
                             new_members, true);
            //GraphBLAS::print_vector(std::cout, new_members, "new_members");

            // Add new members to independent set.
            GraphBLAS::eWiseAdd(independent_set,
                                GraphBLAS::NoMask(), //new_members, //
                                GraphBLAS::NoAccumulate(),
                                GraphBLAS::LogicalOr<bool>(),
                                independent_set, new_members);
            //GraphBLAS::print_vector(std::cout, independent_set, "IS");

            // Remove new_members selected for independent set from candidates
            GraphBLAS::eWiseMult(candidates,
                                 GraphBLAS::complement(new_members),
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::LogicalAnd<bool>(),
                                 candidates, candidates, true);
            //GraphBLAS::print_vector(std::cout, candidates,
            //                        "candidates (sans new_members)");

            if (candidates.nvals() == 0)
            {
                break;
            }

            // Neighbors of new members can also be removed
            GraphBLAS::mxv(new_neighbors,
                           candidates, GraphBLAS::NoAccumulate(),
                           GraphBLAS::LogicalSemiring<bool>(),
                           graph, new_members);  // REPLACE doesn't seem needed
            //GraphBLAS::print_vector(std::cout, new_neighbors,
            //                        "new_member neighbors");

            // Zero out candidates of new member neighbors
            GraphBLAS::eWiseMult(candidates,
                                 GraphBLAS::complement(new_neighbors),
                                 GraphBLAS::NoAccumulate(),
                                 GraphBLAS::LogicalAnd<bool>(),
                                 candidates, candidates, true);
            //GraphBLAS::print_vector(std::cout, candidates,
            //                        "candidates (sans new_members' neighbors)");
        }

    }

} // algorithms

#endif // MIS_HPP
