AM_CXXFLAG=-lboost_unit_test_framework

if ENABLE_CUDA

include $(top_srcdir)/mk/cuda.mk

SUBDIRS = #cusp
##$(abs_top_srcdir)
AM_CPPFLAGS = -I$(abs_top_srcdir) -I$(abs_top_srcdir)/cusplibrary  --expt-relaxed-constexpr
AM_LDFLAGS = $(NVCC_LDFLAGS) $(NVCC_LIBS)

noinst_PROGRAMS = \
test_cusp_gpu_math \
test_cusp_gpu_mxm \
test_cusp_gpu_ewiseapply \
test_cusp_gpu_extracttuples \
test_cusp_gpu_apply \
test_cusp_gpu_extract \
test_cusp_gpu_assign \
test_cusp_gpu_reduce \
test_cusp_gpu_buildmatrix \
test_cusp_gpu_transpose \
test_cusp_gpu_negate \
test_cusp_gpu_bfs \
test_cusp_gpu_bc \
test_cusp_gpu_metrics \
test_cusp_gpu_maxflow \
test_cusp_gpu_mst \
test_cusp_gpu_sssp
#test_cusp_gpu_cluster
#test_cusp_gpu_mis

test_cusp_gpu_math_SOURCES = test_math_cuda.cu

test_cusp_gpu_mxm_SOURCES = test_mxm_cuda.cu

test_cusp_gpu_extracttuples_SOURCES = test_extracttuples_cuda.cu

test_cusp_gpu_ewiseapply_SOURCES = test_ewiseapply_cuda.cu

test_cusp_gpu_apply_SOURCES = test_apply_cuda.cu

test_cusp_gpu_extract_SOURCES = test_extract_cuda.cu

test_cusp_gpu_assign_SOURCES = test_assign_cuda.cu

test_cusp_gpu_reduce_SOURCES = test_reduce_cuda.cu

test_cusp_gpu_buildmatrix_SOURCES = test_buildmatrix_cuda.cu

test_cusp_gpu_transpose_SOURCES = test_transpose_cuda.cu

test_cusp_gpu_negate_SOURCES = test_negate_cuda.cu

test_cusp_gpu_bfs_SOURCES = test_bfs_cuda.cu

test_cusp_gpu_bc_SOURCES = test_bc_cuda.cu

test_cusp_gpu_metrics_SOURCES = test_metrics_cuda.cu

test_cusp_gpu_maxflow_SOURCES = test_maxflow_cuda.cu

#test_cusp_gpu_mis_SOURCES = test_mis_cuda.cu

test_cusp_gpu_mst_SOURCES = test_mst_cuda.cu

test_cusp_gpu_sssp_SOURCES = test_sssp_cuda.cu

#test_cusp_gpu_cluster_SOURCES = test_cluster_cuda.cu

else

AM_CPPFLAGS = -I$(abs_top_srcdir) $(BACKEND_CPPFLAGS)

SUBDIRS = sequential

noinst_PROGRAMS = \
test_sequential_math \
test_sequential_mxm \
test_sequential_ewiseapply \
test_sequential_extracttuples \
test_sequential_apply \
test_sequential_extract \
test_sequential_assign \
test_sequential_reduce \
test_sequential_buildmatrix \
test_sequential_transpose \
test_sequential_negate \
test_sequential_bfs \
test_sequential_bc \
test_sequential_metrics \
test_sequential_maxflow \
test_sequential_mis \
test_sequential_mst \
test_sequential_sssp \
test_sequential_cluster \
test_sequential_page_rank


test_sequential_math_SOURCES = test_math.cpp

test_sequential_mxm_SOURCES = test_mxm.cpp

test_sequential_extracttuples_SOURCES = test_extracttuples.cpp

test_sequential_ewiseapply_SOURCES = test_ewiseapply.cpp

test_sequential_apply_SOURCES = test_apply.cpp

test_sequential_extract_SOURCES = test_extract.cpp

test_sequential_assign_SOURCES = test_assign.cpp

test_sequential_reduce_SOURCES = test_reduce.cpp

test_sequential_buildmatrix_SOURCES = test_buildmatrix.cpp

test_sequential_transpose_SOURCES = test_transpose.cpp

test_sequential_negate_SOURCES = test_negate.cpp

test_sequential_bfs_SOURCES = test_bfs.cpp

test_sequential_bc_SOURCES = test_bc.cpp

test_sequential_metrics_SOURCES = test_metrics.cpp

test_sequential_maxflow_SOURCES = test_maxflow.cpp

test_sequential_mis_SOURCES = test_mis.cpp

test_sequential_mst_SOURCES = test_mst.cpp

test_sequential_sssp_SOURCES = test_sssp.cpp

test_sequential_cluster_SOURCES = test_cluster.cpp

test_sequential_page_rank_SOURCES = test_page_rank.cpp

endif
