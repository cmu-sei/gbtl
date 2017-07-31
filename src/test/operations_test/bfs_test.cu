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

#include <iostream>

#include <graphblas/graphblas.hpp>
#include <graphblas/utility.hpp>
#include <cusp/graph/breadth_first_search.h>

//#include <algorithms/bfs.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>
#include <cusp/io/dimacs.h>

#define readmm(MTX, FILE_NAME) cusp::io::read_matrix_market_file(MTX, FILE_NAME)
#define readdi(MTX, FILE_NAME) cusp::io::read_dimacs_file(MTX, FILE_NAME)

#define CPV 0


//func here for bfs level (from semiring template):
template <typename T>
__device__ __host__ inline T pick1(T a, T b)
{
    return a;
}
template <typename T>
__device__ __host__ inline
T selectadd(T first, T second, T marker)
{
    return first == marker? marker : ++first;
}

struct pick2{
    template<typename T>
    __device__ __host__ inline T operator()(T a, T b)
    {
        return b;
    }
};

template <typename ScalarT>
class PickAdd1
{
public:
    typedef ScalarT ScalarType;
    typedef ScalarT result_type;

    template<typename LhsT, typename RhsT>
    __host__ __device__ ScalarType add(LhsT&& a, RhsT&& b) const
    { return pick1<ScalarType>(std::forward<LhsT>(a),
                               std::forward<RhsT>(b)); }

    template<typename LhsT, typename RhsT>
    __host__ __device__ ScalarType mult(LhsT&& a, RhsT&& b) const
    { return selectadd<ScalarType>(std::forward<LhsT>(a),
                                std::forward<RhsT>(b), 0); }

    __host__ __device__ ScalarType zero() const
    { return static_cast<ScalarType>(0); }

    __host__ __device__ ScalarType one() const
    { return static_cast<ScalarType>(1); }
};


template <typename T>
struct wf_filter{
    T turn;
    wf_filter(T t):turn(t){}
    __host__ __device__ T operator()(const T& v1, const T& v2) const
    {
        //need to be sequence-agnostic
        return v2 < turn ? 0:v1;
    }
};

struct pl_filter{
    template <typename T>
    __host__ __device__ T operator()(const T& v1, const T& v2) const
    {
        return v1 == 0 ? v2:v1;
    }
};

/**
 * @brief Perform a breadth first search (BFS) on the given graph.
 *
 * @param[in]  graph        The graph to perform a BFS on.  NOT built from
 *                          the transpose of the adjacency matrix.
 *                          (1 indicates edge, structural zero = 0).
 * @param[in]  wavefront    The initial wavefront to use in the calculation.
 *                          (1 indicates root, structural zero = 0).
 * @param[out] levels       The level (distance in unweighted graphs) from
 *                          the corresponding root of that BFS
 */
template <typename MatrixT,
          typename WavefrontMatrixT,
          typename LevelListMatrixT>
void bfs_level(MatrixT const     &graph,
               WavefrontMatrixT   wavefront, //row vector, copy made
               LevelListMatrixT  &levels)
{
    using T = typename MatrixT::ScalarType;

    /// @todo Assert graph is square/have a compatible shape with wavefront?
    // graph.get_shape(rows, cols);

    graphblas::IndexType rows, cols;
    wavefront.get_shape(rows, cols);

    WavefrontMatrixT visited(rows, cols);
    visited = wavefront;

    unsigned int depth = 0;

    graphblas::start_timer();
    while (wavefront.getBackendMatrix().num_entries > 0)
    {
        // Increment the level
        ++depth;
        //graphblas::ConstantMatrix<unsigned int> depth_mat(rows, cols, depth);

        // Apply the level to all newly visited nodes
        //graphblas::ewisemult(wavefront, depth_mat, levels,
        //                     graphblas::math::Times<unsigned int>(),
        //                     graphblas::math::Accum<unsigned int>());

        //std::cout<<"wf bf mxm, dept="<<depth<<std::endl;
        //cusp::print(wavefront.getBackendMatrix());
        graphblas::mxm(wavefront, graph, wavefront,
                       //graphblas::IntLogicalSemiring<unsigned int>());
                       PickAdd1<unsigned int>(),
                       pick2());
        //std::cout<<"wf after mxm"<<std::endl;
        //cusp::print(wavefront.getBackendMatrix());

        //filter:
        //wf_filter<T> wff(depth);
        //graphblas::ewisemult(wavefront, visited, wavefront,
        //                     wff);
        //set difference:
#if CPV==1
        cusp::array1d <T, cusp::device_memory> temp(cols);
#endif
        auto end = thrust::set_difference(
                wavefront.getBackendMatrix().column_indices.begin(),
                wavefront.getBackendMatrix().column_indices.end(),
                visited.getBackendMatrix().column_indices.begin(),
                visited.getBackendMatrix().column_indices.end(),
#if CPV==1
                temp.begin());
        //reset wavefront size:
        auto res = thrust::distance(
                temp.begin(),
                end);

        wavefront.getBackendMatrix().resize(rows,cols,res);
#else
                wavefront.getBackendMatrix().column_indices.begin());

        auto dist = thrust::distance(wavefront.getBackendMatrix().column_indices.begin(), end);
        wavefront.getBackendMatrix().resize(rows,cols,dist);
#endif


        //copy:

#if CPV==1
        thrust::copy(temp.begin(), end ,wavefront.getBackendMatrix().column_indices.begin());
#endif

        //end filter


        //std::cout<<"wf after filter"<<std::endl;
        //cusp::print(wavefront.getBackendMatrix());

        //aggregation:
        //graphblas::ewisemult(visited, wavefront, visited,
        //                     pl_filter());
        //union:
        graphblas::backend::detail::merge(
                wavefront.getBackendMatrix(),
                visited.getBackendMatrix(),
                graphblas::math::Assign<T>());

        //std::cout<<"visited after union, dept="<<depth<<std::endl;
        //cusp::print(visited.getBackendMatrix());



        // Cull previously visited nodes from the wavefront
        // Replace these lines with the negate(levels) mask
        //graphblas::apply(levels, not_visited,
        //                 graphblas::math::IsZero<unsigned int>());
        //graphblas::ewisemult(not_visited, wavefront, wavefront,
        //                     graphblas::math::AndFn<unsigned int>());
    }
    levels = visited;
    graphblas::stop_timer();
    std::cout<<graphblas::get_elapsed_time()<<std::endl;
}

//hard coded graph
#if 0
int main(){
    typedef double T;
    typedef graphblas::Matrix<T, graphblas::DirectedMatrixTag> GrBMatrix;

    graphblas::IndexType const NUM_NODES(9);
    graphblas::IndexType const START_INDEX(5);

    graphblas::VectorIndexType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                   4, 4, 4, 5, 6, 6, 6, 8, 8};
    graphblas::VectorIndexType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                                   2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    GrBMatrix G_tn(NUM_NODES, NUM_NODES, 0);
    graphblas::buildmatrix(G_tn, i.begin(), j.begin(), v.begin(), i.size());

    GrBMatrix root(1, NUM_NODES, 0);

    graphblas::VectorIndexType x = {0};
    graphblas::VectorIndexType y = {START_INDEX};
    std::vector<T> z(x.size(), 1);

    graphblas::buildmatrix(root, x.begin(),y.begin(),z.begin(), x.size());
    //root.setElement(0, START_INDEX, 1);  // multiplicative identity

    GrBMatrix levels(1, NUM_NODES, 0);
    bfs_level(G_tn, root, levels);

    cusp::print(levels.getBackendMatrix());
}
#endif

#if 1
//read graph
int main(int argc, char ** argv){
    typedef double T;
    typedef graphblas::Matrix<T, graphblas::DirectedMatrixTag> GrBMatrix;


    GrBMatrix G_tn;
    //cusp::io::read_matrix_market_file(G_tn.getBackendMatrix(), argv[1]);


    if (atoi(argv[2]) == 1){
        readdi(G_tn.getBackendMatrix(),
                std::string(argv[1]));
    } else {
        readmm(G_tn.getBackendMatrix(),
                std::string(argv[1]));
    }

    graphblas::IndexType NUM_NODES;
    G_tn.get_shape(NUM_NODES, NUM_NODES);

    GrBMatrix root(1, NUM_NODES, 0);

    graphblas::VectorIndexType x = {0};
    graphblas::VectorIndexType y = {0};
    std::vector<T> z(x.size(), 1);

    graphblas::buildmatrix(root, x.begin(),y.begin(),z.begin(), x.size());

    GrBMatrix levels(1, NUM_NODES, 0);
    //cusp::print(G_tn.getBackendMatrix());
    //cusp::print(root.getBackendMatrix());
    bfs_level(G_tn, root, levels);

    //graphblas::start_timer();
    //cusp::graph::breadth_first_search(G_tn.getBackendMatrix(), 1, levels.getBackendMatrix().values);
    //graphblas::stop_timer();
    //std::cout<<graphblas::get_elapsed_time()<<std::endl;

    ///cusp::print(levels.getBackendMatrix());
}
#endif
