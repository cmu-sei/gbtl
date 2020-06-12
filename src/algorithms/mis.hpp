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

#include <vector>
#include <random>

#include <graphblas/graphblas.hpp>

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
    grb::IndexArrayType get_vertex_IDs(VectorT const &independent_set)
    {
        using T = typename VectorT::ScalarType;
        grb::IndexType set_size(independent_set.nvals());
        grb::IndexArrayType ans(set_size);
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
    void mis(MatrixT const      &graph,
             grb::Vector<bool>  &independent_set,
             double              seed = 0)
    {
        std::default_random_engine             generator;
        std::uniform_real_distribution<double> distribution;
        generator.seed(seed);

        grb::IndexType num_vertices(graph.nrows());
        grb::IndexType cols(graph.ncols());
        grb::IndexType r(independent_set.size());

        if ((num_vertices != cols) || (num_vertices != r))
        {
            throw grb::DimensionException();
        }

        //grb::print_matrix(std::cout, graph, "Graph");

        using RealT = float;
        using RealVector = grb::Vector<RealT>;
        using BoolVector = grb::Vector<bool>;

        RealVector prob(num_vertices);
        RealVector neighbor_max(num_vertices);
        BoolVector new_members(num_vertices);
        BoolVector new_neighbors(num_vertices);

        // Compute the degree of each node,
        grb::Vector<RealT> degrees(num_vertices);
        grb::reduce(degrees,
                    grb::NoMask(), grb::Plus<RealT>(),
                    grb::PlusMonoid<RealT>(),
                    graph);
        //grb::print_vector(std::cout, degrees, "degrees");

        // Start with all vertices except isolated ones as candidates (NOTE: dense!)
        BoolVector candidates(num_vertices);
        grb::assign(candidates,
                    degrees,
                    grb::NoAccumulate(),
                    true, grb::AllIndices(), grb::REPLACE);

        // Courtesy of Tim Davis: singletons are not candidates.  Add to iset
        grb::assign(independent_set,
                    grb::complement(degrees),
                    grb::NoAccumulate(),
                    true, grb::AllIndices(), grb::REPLACE);

        while (candidates.nvals() > 0)
        {
            //std::cout << "************* ITERATION ************* nnz = "
            //          << candidates.nvals() << std::endl;
            //grb::print_vector(std::cout, candidates, "candidates");

            // assign new random values scaled by inverse degree to
            // all non-zero elements (ensures that any ties that may
            // occur between neighbors will eventually be broken.
            grb::eWiseMult(
                prob,
                grb::NoMask(), grb::NoAccumulate(),
                [&](bool candidate, RealT const &degree)
                { return static_cast<RealT>(
                        0.0001 + distribution(generator)/(1. + 2.*degree)); },
                candidates, degrees);
            //grb::print_vector(std::cout, prob, "prob");

            // find the neighbor of each source node with the max random number
            grb::mxv(neighbor_max,
                     candidates, grb::NoAccumulate(),
                     grb::MaxSecondSemiring<RealT>(),
                     graph, prob, grb::REPLACE);
            //grb::print_vector(std::cout, neighbor_max, "neighbor_max");

            // Select source node if its probability is > neighbor_max
            grb::eWiseAdd(new_members,
                          grb::NoMask(), grb::NoAccumulate(),
                          grb::GreaterThan<RealT>(),
                          prob, neighbor_max);
            // Note the result above IS DENSE, I can add the following apply (or
            // many other operations to make it sparse (just add itself as the
            // mask).
            grb::apply(new_members,
                       new_members, grb::NoAccumulate(),
                       grb::Identity<bool>(),
                       new_members, grb::REPLACE);
            //grb::print_vector(std::cout, new_members, "new_members");

            // Add new members to independent set.
            grb::eWiseAdd(independent_set,
                          grb::NoMask(), //new_members, //
                          grb::NoAccumulate(),
                          grb::LogicalOr<bool>(),
                          independent_set, new_members);
            //grb::print_vector(std::cout, independent_set, "IS");

            // Remove new_members selected for independent set from candidates
            grb::eWiseMult(candidates,
                           grb::complement(new_members),
                           grb::NoAccumulate(),
                           grb::LogicalAnd<bool>(),
                           candidates, candidates, grb::REPLACE);
            //grb::print_vector(std::cout, candidates,
            //                  "candidates (sans new_members)");

            if (candidates.nvals() == 0)
            {
                break;
            }

            // Neighbors of new members can also be removed
            grb::mxv(new_neighbors,
                     candidates, grb::NoAccumulate(),
                     grb::LogicalSemiring<bool>(),
                     graph, new_members);  // REPLACE doesn't seem needed
            //grb::print_vector(std::cout, new_neighbors,
            //                  "new_member neighbors");

            // Zero out candidates of new member neighbors
            grb::eWiseMult(candidates,
                           grb::complement(new_neighbors),
                           grb::NoAccumulate(),
                           grb::LogicalAnd<bool>(),
                           candidates, candidates, grb::REPLACE);
            //grb::print_vector(std::cout, candidates,
            //                  "candidates (sans new_members' neighbors)");
        }

    }

} // algorithms
