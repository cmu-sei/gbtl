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
     * @param[in]  graph   Binary adjacency matrix of the graph.
     * @return    (colors) N-vector of color labels (integers)
     *
     */
    template<typename MatrixT, typename ColorT=uint32_t>
    auto coloring(MatrixT const &graph, double seed = 0)
    {
        using T = typename MatrixT::ScalarType;
        grb::IndexType N(graph.nrows());

        if (N != graph.ncols())
        {
            throw grb::DimensionException();
        }

        std::default_random_engine             generator;
        std::uniform_real_distribution<double> distribution;
        generator.seed(seed);

        grb::Vector<ColorT> colors(N);
        grb::Vector<ColorT> weight(N);
        grb::Vector<ColorT> max(N);
        grb::Vector<ColorT> frontier(N);

        // allocate and initialize color array
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
        grb::print_vector(std::cout, weight, "initial weights");

        colors.clear();
        ColorT num_colors(0);

        // loop can be parallelized
        for (ColorT color = 1; color <= N; ++color)
        {
            num_colors = color;
            std::cout << "================== iter/color = " << color << std::endl;
            // find the max of neighbors
            grb::vxm(max, grb::NoMask(), grb::NoAccumulate(),
                     grb::MaxFirstSemiring<ColorT>(), weight, graph);  // MaxFirst or MaxTimes?
            grb::print_vector(std::cout, max, "max");

            // find all largest nodes that are uncolored.
            grb::eWiseAdd(frontier, grb::NoMask(), grb::NoAccumulate(),
                          grb::GreaterThan<ColorT>(), weight, max);
            grb::apply(frontier, frontier, grb::NoAccumulate(),
                       grb::Identity<bool>(), frontier, grb::REPLACE);
            grb::print_vector(std::cout, frontier, "frontier");

            if (frontier.nvals() == 0)
                break;

            // assign new color
            grb::assign(colors, frontier, grb::NoAccumulate(),
                        color, grb::AllIndices());
            grb::print_vector(std::cout, colors, "colors (with new colors)");

            // get rid of colored nodes in candidate set
            grb::assign(weight, frontier, grb::NoAccumulate(),
                        0, grb::AllIndices());
            grb::print_vector(std::cout, weight, "weight (sans colored nodes)");
        }

        std::cout << "Num colors: " << num_colors << std::endl;
        return colors;
    }
}
