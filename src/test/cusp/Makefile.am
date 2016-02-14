AM_CXXFLAG=-lboost_unit_test_framework

if ENABLE_CUDA

AM_CPPFLAGS = -I$(abs_top_srcdir) -I$(abs_top_srcdir)/cusplibrary
AM_LDFLAGS = $(NVCC_LDFLAGS) $(NVCC_LIBS)
include $(top_srcdir)/mk/cuda.mk

noinst_PROGRAMS = \
test_cusp_gpu_coo_matrix \
test_cusp_gpu_row_extended_view \
test_cusp_gpu_column_extended_view

test_cusp_gpu_coo_matrix_SOURCES = test_coo_matrix.cu

test_cusp_gpu_row_extended_view_SOURCES = test_row_extended_view.cu

test_cusp_gpu_column_extended_view_SOURCES = test_column_extended_view.cu

else

endif
