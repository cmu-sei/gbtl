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

#include <functional>
#include <limits>
#include <tuple>

#include <graphblas/graphblas.hpp>

//****************************************************************************
namespace algorithms
{
    //************************************************************************

    /**
     * @brief Compute the lenghts of the single source shortest path(s) of
     *        one specified starting vertex in the given graph.
     *
     * This is an implementation of Algebraic Bellman-Ford from Graph
     * Algorithms in the Language of Linear Algebra (page 47ff).
     * Assumes there are no non-positive weight cycles.
     *
     * @param[in]  graph     The adjacency matrix of the weighted graph for which
     *                       SSSP will be computed.  The implied zero will be
     *                       infinity.
     * @param[in,out] dist   On input, the root vertex of the SSSP is set to zero.
     *                       On output, contains the shortest path distances for
     *                       the specified starting vertex, specified in terms of
     *                       length from the source vertex.
     *
     * @return False, if a negative weight cycle was detected (and therefore
     *         the answer did not converge).
     */
    template<typename MatrixT,
             typename DistVectorT>
    bool sssp(MatrixT const     &graph,
              DistVectorT       &dist)
    {
        using T  = typename MatrixT::ScalarType;  // todo: assert same as DT
        using DT = typename DistVectorT::ScalarType;

        if ((graph.nrows() != dist.size()) ||
            (graph.ncols() != dist.size()))
        {
            throw grb::DimensionException();
        }

        //std::cout << "SSSP " << std::endl;
        //grb::print_vector(std::cout, dist, "dist");
        /// @todo why num_rows-1 iterations? REALLY should be tested for
        /// detection of no change in dist
        for (grb::IndexType k = 0; k < graph.nrows() - 1; ++k)
        {
            grb::vxm(dist,
                     grb::NoMask(), grb::Min<T>(),
                     grb::MinPlusSemiring<T>(),
                     dist, graph);

            //std::cout << "SSSP Iteration " << k << std::endl;
            //grb::print_vector(std::cout, dist, "dist");
        }

        // Perform one more step to determine if any values have changed.
        // If so, that implies a negative cycle.
        DistVectorT dist2(dist);
        grb::vxm(dist2,
                 grb::NoMask(), grb::Min<T>(),
                 grb::MinPlusSemiring<T>(),
                 dist2, graph);
        //grb::print_vector(std::cout, dist2, "Dist2");

        // determine which distances are equal and also detect if any
        // are still changing.
        grb::Vector<bool> equal_flags(graph.nrows());
        bool changed_flag(false);
        grb::eWiseMult(
            equal_flags,
            grb::NoMask(), grb::NoAccumulate(),
            [&changed_flag](DT const &lhs, DT const &rhs)     //equal_test,
            {
                changed_flag |= (lhs != rhs);
                return lhs == rhs;
            },
            dist, dist2);

        //std::cout << "changed_flag = " << changed_flag << std::endl;
        return !changed_flag;
    }

    //****************************************************************************

    /**
     * @brief Compute the lenghts of the single source shortest path(s) of
     *        a batch of specified starting vertex (vertices) in the given graph.
     *
     * This is an implementation of Algebraic Bellman-Ford from Graph
     * Algorithms in the Language of Linear Algebra (page 47ff).
     * Assumes there are no non-positive weight cycles.
     *
     * @param[in]  graph     The adjacency matrix of the graph for which SSSP
     *                       will be computed.  The implied zero will be
     *                       infinity.
     * @param[in,out] dists  Each row contains the lengths of single source
     *                       shortest dist(s) for each the specified starting
     *                       vertices. These roots are specified, in this matrix
     *                       on input, by setting the source vertex to zero in the
     *                       corresponding row.
     *
     * @return False, if a negative weight cycle was detected (and therefore
     *         the answer did not converge).
     */
    template<typename MatrixT,
             typename DistMatrixT>
    bool batch_sssp(MatrixT const     &graph,
                    DistMatrixT       &dists)  // dists are initialized to start
    {
        using T  = typename MatrixT::ScalarType;  // todo: assert same as DT
        using DT = typename DistMatrixT::ScalarType;

        if ((graph.nrows() != dists.ncols()) ||
            (graph.ncols() != dists.ncols()))
        {
            throw grb::DimensionException();
        }

        /// @todo why num_rows-1 iterations?  Should terminate
        /// when there are no changes?
        for (grb::IndexType k = 0; k < graph.nrows()-1; ++k)
        {
            grb::mxm(dists,
                     grb::NoMask(),
                     grb::Min<T>(),
                     grb::MinPlusSemiring<T>(),
                     dists, graph);

            //std::cout << "BATCH SSSP Iteration " << k << std::endl;
            //grb::print_vector(std::cout, dists, "dists");
        }

        // Perform one more step to determine if any values have changed.
        // If so, that implies a negative cycle.
        DistMatrixT dist2(dists);
        grb::mxm(dist2,
                 grb::NoMask(), grb::Min<T>(),
                 grb::MinPlusSemiring<T>(),
                 dist2, graph);
        //grb::print_matrix(std::cout, dist2, "Dist2");

        // determine which distances are equal and also detect if any
        // are still changing.
        grb::Matrix<bool> equal_flags(dists.nrows(), dists.ncols());
        bool changed_flag(false);
        grb::eWiseMult(
            equal_flags,
            grb::NoMask(), grb::NoAccumulate(),
            [&changed_flag](DT const &lhs, DT const &rhs)     //equal_test
            {
                changed_flag |= (lhs != rhs);
                return lhs == rhs;
            },
            dists, dist2);

        //std::cout << "changed_flag = " << changed_flag << std::endl;
        return !changed_flag;
    }

    //************************************************************************

    /**
     * @brief Compute the lengths of the single source shortest path(s) of
     *        one specified starting vertex in the given graph.
     *
     * This is an implementation of a filtered Algebraic Bellman-Ford algorithm.
     * Assumes there are no non-positive weight cycles.
     *
     * @param[in]    graph   The adjacency matrix of the weighted graph for which
     *                       SSSP will be computed.  The implied zero will be
     *                       infinity.
     * @param[in,out] dist   On input, the root vertex of the SSSP is set to zero.
     *                       On output, contains the shortest path distances for
     *                       the specified starting vertex, specified in terms of
     *                       length from the source vertex.
     *
     * @return False, if a negative weight cycle was detected (and therefore
     *         the answer did not converge).
     */
    template<typename MatrixT,
             typename DistVectorT>
    bool filtered_sssp(MatrixT const &graph,
                       DistVectorT   &dist)
    {
        using T = typename DistVectorT::ScalarType;

        if ((graph.nrows() != dist.size()) ||
            (graph.ncols() != dist.size()))
        {
            throw grb::DimensionException();
        }

        grb::Vector<T> new_dist(dist);
        grb::Vector<bool> new_dist_flags(dist.size());

        grb::IndexType iters_left(graph.nrows());
        while (iters_left > 0)
        {
            //std::cout << "============= Filtered BF Iteration =================\n";
            grb::vxm(new_dist,
                     grb::NoMask(), grb::NoAccumulate(),
                     grb::MinPlusSemiring<T>(),
                     new_dist, graph);
            //grb::print_vector(std::cout, new_dist, "new dist");

            // new_dist_flags<new_dist> = new_dist .< dist
            // (works even if new_dist has a stored zero
            grb::eWiseAdd(new_dist_flags,
                          grb::structure(new_dist),
                          grb::NoAccumulate(),
                          grb::LessThan<T>(),
                          new_dist, dist, grb::REPLACE);
            //grb::print_vector(std::cout, new_dist_flags, "new dist flags");

            // clear out any non-contributing paths
            grb::apply(new_dist,
                       new_dist_flags,
                       grb::NoAccumulate(),
                       grb::Identity<T>(), new_dist, grb::REPLACE);
            //grb::print_vector(std::cout, new_dist, "new dist (cleared)");

            if (new_dist.nvals() == 0)
                break;

            grb::eWiseAdd(dist,
                          grb::NoMask(),
                          grb::NoAccumulate(),
                          grb::Min<T>(),
                          new_dist, dist);
            //grb::print_vector(std::cout, dist, "Dist");
            //std::cout << std::endl;

            --iters_left;
        }

        // dist holds return value
        return iters_left > 0;
    }

    //************************************************************************
    //************************************************************************

    //************************************************************************
    template <typename ScalarT>
    struct SelectInRange
    {
        ScalarT const m_low, m_high;

        SelectInRange(ScalarT low, ScalarT high) : m_low(low), m_high(high) {}

        inline bool operator()(ScalarT input)
        {
            if ((m_low <= input) && (input < m_high))
            {
                return true;
            }

            return false;
        }
    };

    //************************************************************************
    template<typename MatrixT>
    void sssp_delta_step(MatrixT const                             &graph,
                         typename MatrixT::ScalarType               delta,
                         grb::IndexType                             src,
                         grb::Vector<typename MatrixT::ScalarType> &paths)
    {
        using T = typename MatrixT::ScalarType;
        //std::cerr << "delta = " << delta << std::endl;
        //grb::print_matrix(std::cerr, graph, "A");

        grb::IndexType n(graph.nrows());  /// @todo assert nrows = ncols
        paths.clear();                    /// @todo assert size = nrows

        grb::Vector<T>    t(n);
        grb::Vector<T>    tmasked(n);
        grb::Vector<T>    tReq(n);

        grb::Vector<bool> tBi(n);
        grb::Vector<bool> tnew(n);
        grb::Vector<bool> tcomp(n);
        grb::Vector<bool> tless(n);
        grb::Vector<bool> s(n);

        // t = infinity, t[src] = 0
        t.setElement(src, 0);

        // AL = A .* (A <= delta)
        MatrixT AL(n, n);
        grb::apply(AL, grb::NoMask(), grb::NoAccumulate(),
                   std::bind(grb::LessEqual<T>(),
                             std::placeholders::_1,
                             static_cast<T>(delta)),
                   graph);
        grb::apply(AL, AL, grb::NoAccumulate(),
                   grb::Identity<T>(), graph, grb::REPLACE);
        //grb::print_matrix(std::cerr, AL, "AL = A(<=delta)");

        // AH = A .* (A > delta)
        MatrixT AH(n, n);
        grb::apply(AH, grb::NoMask(), grb::NoAccumulate(),
                   std::bind(grb::GreaterThan<T>(),
                             std::placeholders::_1,
                             delta),
                   graph);
        grb::apply(AH, AH, grb::NoAccumulate(),
                   grb::Identity<T>(), graph, grb::REPLACE);
        //grb::print_matrix(std::cerr, AH, "AH = A(>delta)");

        // i = 0
        grb::IndexType i(0);

        // t >= i*delta
        grb::apply(tcomp, grb::NoMask(), grb::NoAccumulate(),
                   std::bind(grb::GreaterEqual<T>(),
                             std::placeholders::_1,
                             static_cast<T>(i)*delta),
                   t);
        grb::apply(tcomp, tcomp, grb::NoAccumulate(),
                   grb::Identity<bool>(), tcomp);

        //grb::print_vector(std::cerr, tcomp, "tcomp = t(>=i*delta)");

        // while (t >= i*delta) not empty
        while (tcomp.nvals() > 0)
        {
            //std::cerr << "************************************************\n";
            //std::cerr << "****** Outer loop: i = " << i << std::endl;
            // s = 0 (clear)
            s.clear();

            // tBi = t .* (i*delta <= t < (i+1)*delta)
            SelectInRange<T> in_range((T)i*delta, ((T)(i + 1))*delta);
            grb::apply(tBi, grb::NoMask(), grb::NoAccumulate(),
                       in_range, t);
            grb::apply(tBi, tBi, grb::NoAccumulate(),
                       grb::Identity<T>(), tBi, grb::REPLACE);
            //grb::print_vector(std::cerr, tBi,
            //                        "tBi<tless> = tReq([i*d, (i+1)*d))");

            // tm<tBi> = t
            grb::apply(tmasked, tBi, grb::NoAccumulate(),
                       grb::Identity<T>(), t, grb::REPLACE);
            //grb::print_vector(std::cerr, tmasked, "tm = t<tBi>");

            while (tmasked.nvals() > 0)
            {
                //std::cerr << "******************* inner *********************\n";
                // tReq = AL' (min.+) (t .* tBi)
                grb::vxm(tReq, grb::NoMask(), grb::NoAccumulate(),
                         grb::MinPlusSemiring<T>(), tmasked, AL);
                //grb::print_vector(std::cerr, tReq, "tReq = tm*AL");

                // s = s + tBi
                grb::eWiseAdd(s,
                              grb::NoMask(),
                              grb::NoAccumulate(),
                              grb::LogicalOr<bool>(),
                              s, tBi);
                //grb::print_vector(std::cerr, s, "s += tBi");

                // Don't tBi = 0
                // tBi.clear();

                // Note that the following ("masked less than") only
                // works if the new distances are never zero.
                // TODO: NEED STRUCTURE ONLY MASK here
                // tless<tReq> = tReq .< t
                grb::eWiseAdd(tless,
                              tReq,
                              grb::NoAccumulate(),
                              grb::LessThan<T>(),
                              tReq, t, grb::REPLACE);
                //grb::print_vector(std::cerr, tless, "tless<tReq> = tReq .< t");

                // tBi<tless> = i*delta <= tReq < (i+1)*delta
                grb::apply(tBi,
                           tless,
                           grb::NoAccumulate(),
                           in_range, tReq, grb::REPLACE);
                //grb::apply(tnew, tnew, grb::NoAccumulate(),
                //                 grb::Identity<bool>(), tnew, grb::REPLACE);
                //grb::print_vector(std::cerr, tBi,
                //                        "tBi<tless> = tReq([i*d, (i+1)*d))");

                // t = min(t, tReq)
                grb::eWiseAdd(t,
                              grb::NoMask(),
                              grb::NoAccumulate(),
                              grb::Min<T>(),
                              t, tReq);
                //grb::print_vector(std::cerr, t, "t = min(t, tReq)");

                // tm<tBi> = t
                grb::apply(tmasked, tBi, grb::NoAccumulate(),
                           grb::Identity<T>(), t, grb::REPLACE);
                //grb::print_vector(std::cerr, tmasked, "tm = t<tBi>");
            }
            //std::cerr << "******************** end inner loop *****************\n";

            // (t .* s)
            grb::apply(tmasked, s, grb::NoAccumulate(),
                       grb::Identity<T>(), t, grb::REPLACE);
            //grb::print_vector(std::cerr, tmasked, "tm = t<s>");

            // tReq = AH'(t .* s)
            grb::vxm(tReq,
                     grb::NoMask(),
                     grb::NoAccumulate(),
                     grb::MinPlusSemiring<T>(), tmasked, AH);
            //grb::print_vector(std::cerr, tReq, "tReq = tcomp min.+ AH");

            // t = min(t, tReq)
            grb::eWiseAdd(t,
                          grb::NoMask(),
                          grb::NoAccumulate(),
                          grb::Min<T>(), t, tReq);
            //grb::print_vector(std::cerr, t, "t = min(t,tReq)");

            ++i;

            // t >= i*delta
            grb::apply(tcomp,
                       grb::NoMask(),
                       grb::NoAccumulate(),
                       std::bind(grb::GreaterEqual<T>(),
                                 std::placeholders::_1,
                                 static_cast<T>(i)*delta),
                       t);
            grb::apply(tcomp, tcomp, grb::NoAccumulate(),
                       grb::Identity<bool>(), tcomp, grb::REPLACE);
            //grb::print_vector(std::cerr, tcomp, "tcomp = t(>=i*delta)");
        }

        // result = t
        grb::apply(paths, grb::NoMask(), grb::NoAccumulate(),
                   grb::Identity<T>(), t);
    }
} // algorithms
