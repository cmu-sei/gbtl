cmake_minimum_required(VERSION 2.8)
project(src)

# This seems hokey that we need to include the root as our directory
include_directories(${CMAKE_SOURCE_DIR})

add_definitions(-DGB_USE_SEQUENTIAL)

set(CMAKE_CXX_STANDARD 11)

set(GRAPHBLAS
        graphblas/detail/config.hpp
        graphblas/detail/matrix_tags.hpp
        graphblas/detail/param_unpack.hpp
        graphblas/system/cusp/cusp.hpp
        graphblas/system/cusp/Matrix.hpp
        graphblas/system/cusp/NegateView.hpp
        graphblas/system/cusp/operations.hpp
        graphblas/system/cusp/TransposeView.hpp
        graphblas/system/cusp/types.hpp
        graphblas/system/cusp/utility.hpp
        graphblas/system/cusp/Vector.hpp
        graphblas/system/sequential/BitmapSparseVector.hpp
        graphblas/system/sequential/ColumnExtendedView.hpp
        graphblas/system/sequential/CooMatrix.hpp
        graphblas/system/sequential/CscMatrix.hpp
        graphblas/system/sequential/CsrMatrix.hpp
        graphblas/system/sequential/DiaMatrix.hpp
        graphblas/system/sequential/FunctionView.hpp
        graphblas/system/sequential/LilMatrix.hpp
        graphblas/system/sequential/LilSparseMatrix.hpp
        graphblas/system/sequential/Matrix.hpp
        graphblas/system/sequential/mxv_sparse.hpp
        graphblas/system/sequential/NegateView.hpp
        graphblas/system/sequential/operations.hpp
        graphblas/system/sequential/RowExtendedView.hpp
        graphblas/system/sequential/sequential.hpp
        graphblas/system/sequential/sparse_apply.hpp
        graphblas/system/sequential/sparse_assign.hpp
        graphblas/system/sequential/sparse_extract.hpp
        graphblas/system/sequential/sparse_ewiseadd.hpp
        graphblas/system/sequential/sparse_ewisemult.hpp
        graphblas/system/sequential/sparse_helpers.hpp
        graphblas/system/sequential/sparse_mxm.hpp
        graphblas/system/sequential/sparse_mxv.hpp
        graphblas/system/sequential/sparse_vxm.hpp
        graphblas/system/sequential/TransposeView.hpp
        graphblas/system/sequential/types.hpp
        graphblas/system/sequential/utility.hpp
        graphblas/system/sequential/Vector.hpp
        graphblas/accum.hpp
        graphblas/algebra.hpp
        graphblas/exceptions.hpp
        graphblas/graphblas.cpp
        graphblas/graphblas.hpp
        graphblas/linalg_utils.hpp
        graphblas/Matrix.hpp
        graphblas/matrix_utils.hpp
        graphblas/operations.hpp
        graphblas/types.hpp
        graphblas/utility.hpp
        graphblas/Vector.hpp
        graphblas/View.hpp)


set(ALGORITHMS
        algorithms/algorithms.hpp
        algorithms/bc.hpp
        algorithms/bfs.hpp
        algorithms/cluster.hpp
        algorithms/maxflow.hpp
        algorithms/metrics.hpp
        algorithms/mis.hpp
        algorithms/mst.hpp
        algorithms/page_rank.hpp
        algorithms/sssp.hpp
        algorithms/triangle_count.hpp
#        algorithms/k_truss.hpp
        )


add_executable(test_bitmap_sparse_vector test/sequential/test_bitmap_sparse_vector.cpp ${GRAPHBLAS})
add_executable(test_column_extended_view test/sequential/test_column_extended_view.cpp ${GRAPHBLAS})
add_executable(test_coo_matrix test/sequential/test_coo_matrix.cpp ${GRAPHBLAS})
add_executable(test_csc_matrix test/sequential/test_csc_matrix.cpp ${GRAPHBLAS})
add_executable(test_csr_matrix test/sequential/test_csr_matrix.cpp ${GRAPHBLAS})
add_executable(test_lil_matrix test/sequential/test_lil_matrix.cpp ${GRAPHBLAS})
add_executable(test_lil_sparse_matrix test/sequential/test_lil_sparse_matrix.cpp ${GRAPHBLAS})
add_executable(test_mxm_sparse test/sequential/test_mxm_sparse.cpp ${GRAPHBLAS})
add_executable(test_mxv_sparse test/sequential/test_mxv_sparse.cpp ${GRAPHBLAS})
add_executable(test_row_extended_view test/sequential/test_row_extended_view.cpp ${GRAPHBLAS})

add_executable(test_apply test/test_apply.cpp ${GRAPHBLAS})
add_executable(test_assign test/test_assign.cpp ${GRAPHBLAS})
add_executable(test_bc test/test_bc.cpp ${GRAPHBLAS})
add_executable(test_bfs test/test_bfs.cpp ${GRAPHBLAS})
add_executable(test_buildmatrix test/test_buildmatrix.cpp ${GRAPHBLAS})
add_executable(test_cluster test/test_cluster.cpp ${GRAPHBLAS})
add_executable(test_ewiseapply test/test_ewiseapply.cpp ${GRAPHBLAS})
add_executable(test_extract test/test_extract.cpp ${GRAPHBLAS})
add_executable(test_extracttuples test/test_extracttuples.cpp ${GRAPHBLAS})
add_executable(test_graphblas_algebra test/test_graphblas_algebra.cpp ${GRAPHBLAS})
add_executable(test_homework test/test_homework.cpp ${GRAPHBLAS})
add_executable(test_math test/test_math.cpp ${GRAPHBLAS})
add_executable(test_maxflow test/test_maxflow.cpp ${GRAPHBLAS})
add_executable(test_metrics test/test_metrics.cpp ${GRAPHBLAS})
add_executable(test_mis test/test_mis.cpp ${GRAPHBLAS})
add_executable(test_mst test/test_mst.cpp ${GRAPHBLAS})
add_executable(test_mxm test/test_mxm.cpp ${GRAPHBLAS})
add_executable(test_negate test/test_negate.cpp ${GRAPHBLAS})
add_executable(test_page_rank test/test_page_rank.cpp ${GRAPHBLAS})
add_executable(test_reduce test/test_reduce.cpp ${GRAPHBLAS})
add_executable(test_sparse_algebra test/test_sparse_algebra.cpp ${GRAPHBLAS})
add_executable(test_sparse_apply test/test_sparse_apply.cpp ${GRAPHBLAS})
add_executable(test_sparse_assign test/test_sparse_assign.cpp ${GRAPHBLAS})
add_executable(test_sparse_ewisemult_matrix test/test_sparse_ewisemult_matrix.cpp ${GRAPHBLAS})
add_executable(test_sparse_extract test/test_sparse_extract.cpp ${GRAPHBLAS})
add_executable(test_sparse_mxm test/test_sparse_mxm.cpp ${GRAPHBLAS})
add_executable(test_sparse_mxv test/test_sparse_mxv.cpp ${GRAPHBLAS})
add_executable(test_sparse_vxm test/test_sparse_vxm.cpp ${GRAPHBLAS})
add_executable(test_sparse_reduce test/test_sparse_reduce.cpp ${GRAPHBLAS})
add_executable(test_sssp test/test_sssp.cpp ${GRAPHBLAS})
add_executable(test_transpose test/test_transpose.cpp ${GRAPHBLAS})
add_executable(test_triangle_count test/test_triangle_count.cpp ${GRAPHBLAS})

add_executable(bfs_demo demo/bfs_demo.cpp ${GRAPHBLAS} graphblas/system/sequential/debug.h)
add_executable(bfs_level_demo demo/bfs_level_demo.cpp ${GRAPHBLAS})
add_executable(masked_mxm_homework demo/masked_mxm_homework.cpp ${GRAPHBLAS})
add_executable(mxm_homework demo/mxm_homework.cpp ${GRAPHBLAS})
add_executable(peer_pressure_demo demo/peer_pressure_demo.cpp ${GRAPHBLAS})
add_executable(peer_pressure_karate_demo demo/peer_pressure_karate_demo.cpp ${GRAPHBLAS})
add_executable(sssp_demo demo/sssp_demo.cpp ${GRAPHBLAS})
add_executable(triangle_count_demo demo/triangle_count_demo.cpp ${GRAPHBLAS})
#add_executable(k_truss_demo demo/k_truss_demo.cpp ${GRAPHBLAS})

# TEMPORARY for GraphChallenge data processing.
#add_executable(clean_data demo/clean_data.cpp ${GRAPHBLAS})
#add_executable(snaptsv2csr demo/snaptsv2csr.cpp ${GRAPHBLAS})
#add_executable(tsv2csr demo/tsv2csr.cpp ${GRAPHBLAS})
#add_executable(csr2tsv demo/csr2tsv.cpp ${GRAPHBLAS})

#add_executable(bcc_demo capi/bcc_demo.c capi/bcc.c capi/bcc.h ${GRAPHBLAS} capi/graphblas.cpp capi/TypeAdapter.hpp capi/UnaryAdapter.hpp capi/graphblas_private.h capi/BinaryAdapter.xhpp capi/TypeUnion.hpp capi/SemiringAdapter.hpp capi/FunctionInvocation.cpp capi/FunctionInvocation.h capi/MonoidAdapter.hpp)

#Finished port of
# bfs_demo, test_bfs
# test_bc
