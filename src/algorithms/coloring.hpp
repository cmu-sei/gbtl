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

// Adapted from M. Osama, M. Truong, C. Yang, A. Buluc, and J. D. Owens,
// "Graph Coloring on the GPU", in IEEE IPDPS Workshops, 2019.

#pragma once

#include <vector>
#include <random>

#include <graphblas/graphblas.hpp>

//****************************************************************************
namespace algorithms
{
    //************************************************************************
    /**
     * @brief Independent set graph coloring.
     *
     * @param[in]  graph   Binary NxN adjacency matrix of the graph.
     * @param[out] colors  N-vector of color labels (integers)
     *
     * @retval number of unique colors
     *
     */
    template<typename MatrixT, typename ColorT=uint32_t>
    auto coloring(MatrixT             const &graph,
                  grb::Vector<ColorT>       &colors,
                  double                     seed = 0)
    {
        using T = typename MatrixT::ScalarType;
        grb::IndexType N(graph.nrows());

        if (N != graph.ncols() || N != colors.size())
        {
            throw grb::DimensionException();
        }

        std::default_random_engine             generator;
        std::uniform_real_distribution<double> distribution;
        generator.seed(seed);

        grb::Vector<ColorT> weight(N);
        grb::Vector<ColorT> max(N);
        grb::Vector<ColorT> frontier(N);

        // allocate and initialize color array
        //colors.clear();
        grb::assign(colors, grb::NoMask(), grb::NoAccumulate(),
                    static_cast<ColorT>(0), grb::AllIndices());

        // assign random weights to each vertex
        grb::apply(weight, grb::NoMask(), grb::NoAccumulate(),
                   [&](ColorT const &color) {
                       return static_cast<ColorT>(
                           1 + (std::numeric_limits<ColorT>::max() - 1)*
                           distribution(generator));
                   },
                   colors);
        //grb::print_vector(std::cout, weight, "initial weights");

        colors.clear();
        ColorT num_colors(0);

        // loop can be parallelized
        for (ColorT color = 1; color <= N; ++color)
        {
            num_colors = color;
            //std::cout << "================== iter/color = " << color << std::endl;
            // find the max of neighbors
            grb::vxm(max, grb::NoMask(), grb::NoAccumulate(),
                     grb::MaxFirstSemiring<ColorT>(), weight, graph);  // MaxFirst or MaxTimes?
            //grb::print_vector(std::cout, max, "max");

            // find all largest nodes that are uncolored.
            grb::eWiseAdd(frontier, grb::NoMask(), grb::NoAccumulate(),
                          grb::GreaterThan<ColorT>(), weight, max);
            grb::apply(frontier, frontier, grb::NoAccumulate(),
                       grb::Identity<bool>(), frontier, grb::REPLACE);
            //grb::print_vector(std::cout, frontier, "frontier");

            if (frontier.nvals() == 0)
                break;

            // assign new color
            grb::assign(colors, frontier, grb::NoAccumulate(),
                        color, grb::AllIndices());
            //grb::print_vector(std::cout, colors, "colors (with new colors)");

            // get rid of colored nodes in candidate set
            grb::assign(weight, frontier, grb::NoAccumulate(),
                        0, grb::AllIndices());
            //grb::print_vector(std::cout, weight, "weight (sans colored nodes)");
        }

        return num_colors - 1;
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
    template <typename MatrixT, typename ValueT>
    void mis_masked(MatrixT const             &graph,
                    grb::Vector<ValueT> const &exclude_set,
                    grb::Vector<bool>         &independent_set,
                    double                     seed = 0)
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
                    complement(exclude_set), grb::Plus<RealT>(),
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

        // remove exclude_set from independent set
        grb::apply(independent_set, grb::complement(exclude_set),
                   grb::NoAccumulate(), grb::Identity<bool>(),
                   independent_set, grb::REPLACE);
        grb::print_vector(std::cout, independent_set, "***FINAL IS***");
    }

    //************************************************************************
    /**
     * @brief Maximal independent set (mis) graph coloring.
     *
     * @param[in]  graph   Binary adjacency matrix of the graph.
     * @param[out] colors  N-vector of color labels (integers)
     *
     * @retval number of unique colors
     *
     */
    template<typename MatrixT, typename ColorT=uint32_t>
    auto coloring_mis(MatrixT             const &graph,
                      grb::Vector<ColorT>       &colors,
                      double                     seed = 0)
    {
        using T = typename MatrixT::ScalarType;
        grb::IndexType N(graph.nrows());

        if (N != graph.ncols() || N != colors.size())
        {
            throw grb::DimensionException();
        }

        //std::default_random_engine             generator;
        //std::uniform_real_distribution<double> distribution;
        //generator.seed(seed);

        //grb::Vector<ColorT> weight(N);
        grb::Vector<ColorT> max(N);
        grb::Vector<bool> iset(N);

        // allocate and initialize color array
        colors.clear();
        //grb::assign(colors, grb::NoMask(), grb::NoAccumulate(),
        //            static_cast<ColorT>(0), grb::AllIndices());

        // assign random weights to each vertex
        // grb::apply(weight, grb::NoMask(), grb::NoAccumulate(),
        //            [&](ColorT const &color) {
        //                return static_cast<ColorT>(
        //                    1 + (std::numeric_limits<ColorT>::max() - 1)*
        //                    distribution(generator));
        //            },
        //            colors);
        // grb::print_vector(std::cout, weight, "initial weights");

        colors.clear();
        ColorT num_colors(0);

        // loop can be parallelized
        for (ColorT color = 1; color <= N; ++color)
        {
            num_colors = color;
            std::cout << "================== iter/color = " << color << std::endl;
            // find the max of neighbors
            // grb::vxm(max, grb::NoMask(), grb::NoAccumulate(),
            //          grb::MaxFirstSemiring<ColorT>(), weight, graph);  // MaxFirst or MaxTimes?
            // grb::print_vector(std::cout, max, "max");

            // find all largest nodes that are uncolored.
            // grb::eWiseAdd(iset, grb::NoMask(), grb::NoAccumulate(),
            //               grb::GreaterThan<ColorT>(), weight, max);
            // grb::apply(iset, iset, grb::NoAccumulate(),
            //            grb::Identity<bool>(), iset, grb::REPLACE);
            // grb::print_vector(std::cout, iset, "iset");
            mis_masked(graph, colors, iset, seed*color+color);

            if (iset.nvals() == 0)
                break;

            // assign new color
            grb::assign(colors, iset, grb::NoAccumulate(),
                        color, grb::AllIndices());
            grb::print_vector(std::cout, colors, "colors (with new colors)");

            // get rid of colored nodes in candidate set
            //grb::assign(weight, iset, grb::NoAccumulate(),
            //            0, grb::AllIndices());
            //grb::print_vector(std::cout, weight, "weight (sans colored nodes)");
        }

        return num_colors - 1;
    }
}
