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

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <vector>

#include <graphblas/graphblas.hpp>

using namespace graphblas;

//****************************************************************************
namespace algebra
{
    /// Monoid for arithmetic addition
    template <typename T>
    struct ArithmeticAddMonoid
    {
        typedef T ScalarType;
        T identity() const { return static_cast<T>(0); }
        T operator()(T const &a, T const &b) const { return (a + b); }
    };


    /// Monoid for arithmetic multiplication
    template <typename T>
    struct ArithmeticMultiplyMonoid
    {
        typedef T ScalarType;
        T identity() const { return static_cast<T>(1); }
        T operator()(T const &a, T const &b) const { return (a * b); }
    };


    /// Monoid for logical or
    template <typename T>
    struct LogicalOrMonoid
    {
        typedef T ScalarType;
        T identity() const { return static_cast<T>(false); }
        T operator()(T const &a, T const &b) const { return (a || b); }
    };


    /// Monoid for logical and
    template <typename T>
    struct LogicalAndMonoid
    {
        typedef T ScalarType;
        T identity() const { return static_cast<T>(true); }
        T operator()(T const &a, T const &b) const { return (a && b); }
    };


    /// Used to accumulate into a result 'matrix' (overriding assign).
    template<typename ScalarT,
             typename MonoidT = ArithmeticAddMonoid<ScalarT> >
    struct Accum
    {
        ScalarT operator()(ScalarT const &lhs, ScalarT const &rhs) {
            return MonoidT()(lhs, rhs);
        }
    };


    /// Template for all semirings from two monoids
    template <typename AddMonoidT, typename MultiplyMonoidT>
    class Semiring
    {
    public:
        typedef typename MultiplyMonoidT::ScalarType MultiplyType;
        typedef typename AddMonoidT::ScalarType      AddType;
        typedef AddType ScalarType;

        AddType zero() const { return AddMonoidT().identity(); }

        template<typename LhsT, typename RhsT>
        AddType add(LhsT&& a, RhsT&& b) const {
            return AddMonoidT()(std::forward<LhsT>(a), std::forward<RhsT>(b));
        }

        MultiplyType one() const { return MultiplyMonoidT().identity(); }

        template<typename LhsT, typename RhsT>
        MultiplyType mult(LhsT&& a, RhsT&& b) const {
            return MultiplyMonoidT()(std::forward<LhsT>(a), std::forward<RhsT>(b));
        }
    };


    /// Instantiation of arithmetic semiring
    template <typename ScalarT>
    using ArithmeticSemiring = Semiring<ArithmeticAddMonoid<ScalarT>,
                                        ArithmeticMultiplyMonoid<ScalarT>>;

    /// Instantiation of logical semiring
    template <typename ScalarT>
    using LogicalSemiring = Semiring<LogicalOrMonoid<ScalarT>,
                                     LogicalAndMonoid<ScalarT>>;
}

//****************************************************************************
namespace views
{
    /**
     * @brief View of a matrix as if it were structurally complemented.
     *
     * @tparam MatrixT     Implements the 2D matrix concept.
     * @tparam SemiringT   Used to define the behaviour of the negate
     */
    template<typename MatrixT, typename SemiringT>
    class NegateView
    {
    public:
        typedef typename MatrixT::ScalarType ScalarType;

        NegateView(MatrixT const &matrix): m_matrix(matrix) {}
        ~NegateView() {}

        IndexType get_nnz() const
        {
            IndexType rows, cols;
            m_matrix.get_shape(rows, cols);
            return (rows*cols - m_matrix.get_nnz());
        }

        /// Generalized negate/complement/invert?? for illustrative purposes
        ScalarType extractElement(IndexType row, IndexType col) const
        {
            auto value = m_matrix.extractElement(row, col);

            if (value == SemiringT().zero())
                return SemiringT().one();
            else
                return SemiringT().zero();
        }

    private:
        /// Copy assignment not implemented.
        NegateView<MatrixT, SemiringT> &
        operator=(NegateView<MatrixT, SemiringT> const &rhs) = delete;

    private:
        MatrixT const &m_matrix;
    };
}

//****************************************************************************
namespace algorithms
{
    /**
     * @brief Perform a breadth first search (BFS) on the given graph.
     *
     * @param[in]  graph      NxN adjacency matrix of the graph on which to
     *                        perform a BFS (not the transpose).  A value of
     *                        1 indicates an edge (structural zero = 0).
     * @param[in]  wavefront  RxN initial wavefronts to use in the calculation of
     *                        R simultaneous traversals.  A value of 1 in a given
     *                        row indicates a root for the corresponding traversal.
     *                        (structural zero = 0).
     * @param[out] levels     The level (distance in unweighted graphs) from
     *                        the corresponding root of that BFS.  Roots are
     *                        assigned a value of 1. (a value of 0 implies not
     *                        reachable.
     */
    template <typename MatrixT,
              typename WavefrontMatrixT>
    void bfsLevelMasked(MatrixT const     &graph,
                        WavefrontMatrixT   wavefront,  //row vectors, copy made
                        Matrix<IndexType> &levels)
    {
        using T = typename MatrixT::ScalarType;

        /// Assert graph is square/has a compatible shape with wavefront
        IndexType grows, gcols, wrows, wcols;
        graph.get_shape(grows, gcols);
        wavefront.get_shape(wrows, wcols);
        if ((grows != gcols) || (wcols != grows))
        {
            throw DimensionException("grows="+std::to_string(grows)
                    + ", gcols"+std::to_string(gcols)
                    + "\nwcols="+std::to_string(wcols)
                    + ", grows="+std::to_string(grows));
        }

        IndexType depth = 0;
        while (wavefront.get_nnz() > 0)
        {
            // Increment the level (ConstantMatrix is equivalent to a fill)
            ++depth;
           // ConstantMatrix<IndexType> depth_mat(wrows, wcols, depth);

            graphblas::arithmetic_n<
                IndexType,
                algebra::ArithmeticMultiplyMonoid<IndexType> >
                    incr(depth);

            // Apply the level to all newly visited nodes, and
            // accumulate result into the levels vectors.
            apply(wavefront, levels, incr, algebra::Accum<IndexType>());
            //ewisemult(wavefront, depth_mat, levels,
            //          algebra::ArithmeticMultiplyMonoid<IndexType>(),
            //          algebra::Accum<IndexType>());

            // Advance the wavefront and mask out nodes already assigned levels
            mxmMasked(wavefront, graph, wavefront,
                      negate(levels, algebra::ArithmeticSemiring<IndexType>()),
                      algebra::LogicalSemiring<IndexType>());
        }
    }
}

//****************************************************************************
int main(int, char**)
{
    // syntatic sugar
    typedef IndexType ScalarType;

    // Create an adjacency matrix for "Gilbert's" directed graph
    graphblas::IndexType const NUM_NODES(7);
    graphblas::IndexArrayType i = {0, 0, 1, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6};
    graphblas::IndexArrayType j = {1, 3, 4, 6, 5, 0, 2, 6, 5, 2, 2, 3, 4};
    std::vector<ScalarType>   v(i.size(), 1.0);

    Matrix<ScalarType> graph(NUM_NODES, NUM_NODES);
    buildmatrix(graph, i.begin(), j.begin(), v.begin(), i.size());
    print_matrix(std::cout, graph, "Test Graph:");

    // Outputs:
    // Test Graph:: structural zero = 0
    // [[ , 1,  , 1,  ,  ,  ]
    //  [ ,  ,  ,  , 1,  , 1]
    //  [ ,  ,  ,  ,  , 1,  ]
    //  [1,  , 1,  ,  ,  , 1]
    //  [ ,  ,  ,  ,  , 1,  ]
    //  [ ,  , 1,  ,  ,  ,  ]
    //  [ ,  , 1, 1, 1,  ,  ]]


    // Initialize NUM_NODES wavefronts starting from each vertex in the system
    auto roots(identity<Matrix<ScalarType>>(NUM_NODES));
    Matrix<ScalarType> levels(NUM_NODES, NUM_NODES);

    // Perform NUM_NODES traversals simultaneously
    algorithms::bfsLevelMasked(graph, roots, levels);

    // extract the results: get_nnz() method tells us how big
    print_matrix(std::cout, levels, "Levels");

    // Outputs:
    // Levels: structural zero = 0
    // [[1, 2, 3, 2, 3, 4, 3]
    //  [4, 1, 3, 3, 2, 3, 2]
    //  [ ,  , 1,  ,  , 2,  ]
    //  [2, 3, 2, 1, 4, 3, 2]
    //  [ ,  , 3,  , 1, 2,  ]
    //  [ ,  , 2,  ,  , 1,  ]
    //  [3, 4, 2, 2, 2, 3, 1]]

    return 0;
}
