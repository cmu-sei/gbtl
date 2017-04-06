AM_CXXFLAG=-lboost_unit_test_framework

if ENABLE_CUDA

else

AM_CPPFLAGS = -I$(abs_top_srcdir) $(BACKEND_CPPFLAGS)

noinst_PROGRAMS = \
test_sequential_lil_matrix \
test_sequential_coo_matrix \
test_sequential_csc_matrix \
test_sequential_csr_matrix \
test_sequential_row_extended_view \
test_sequential_column_extended_view \
test_sequential_lil_sparse_matrix \
test_sequential_bitmap_sparse_vector \
test_sequential_mxm_sparse \
test_sequential_mxv_sparse

test_sequential_lil_matrix_SOURCES = test_lil_matrix.cpp

test_sequential_coo_matrix_SOURCES = test_coo_matrix.cpp

test_sequential_csc_matrix_SOURCES = test_csc_matrix.cpp

test_sequential_csr_matrix_SOURCES = test_csr_matrix.cpp

test_sequential_row_extended_view_SOURCES = test_row_extended_view.cpp

test_sequential_column_extended_view_SOURCES = test_column_extended_view.cpp

test_sequential_lil_sparse_matrix_SOURCES = test_lil_sparse_matrix.cpp

test_sequential_bitmap_sparse_vector_SOURCES = test_bitmap_sparse_vector.cpp

test_sequential_mxm_sparse_SOURCES = test_mxm_sparse.cpp

test_sequential_mxv_sparse_SOURCES = test_mxv_sparse.cpp

endif
