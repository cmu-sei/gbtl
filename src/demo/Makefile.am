# TODO Best way to add the library sources?

AM_CPPFLAGS = -I$(abs_top_srcdir) $(BACKEND_CPPFLAGS)

if ENABLE_CUDA

AM_CPPFLAGS += -I$(abs_top_srcdir)/cusplibrary

include $(top_srcdir)/mk/cuda.mk

noinst_PROGRAMS = bfs_demo bfs_level_demo sssp_demo peer_pressure_demo peer_pressure_karate_demo

bfs_demo_SOURCES = bfs_demo.cpp

bfs_level_demo_SOURCES = bfs_level_demo.cpp

sssp_demo_SOURCES = sssp_demo.cpp

peer_pressure_demo_SOURCES = peer_pressure_demo.cpp

peer_pressure_karate_demo_SOURCES = peer_pressure_karate_demo.cpp

else

noinst_PROGRAMS = bfs_demo bfs_level_demo sssp_demo mxm_homework masked_mxm_homework peer_pressure_demo peer_pressure_karate_demo triangle_count_demo k_truss_demo clean_data snaptsv2csr tsv2csr csr2tsv

bfs_demo_SOURCES = bfs_demo.cpp graphblas.cpp

bfs_level_demo_SOURCES = bfs_level_demo.cpp graphblas.cpp

sssp_demo_SOURCES = sssp_demo.cpp graphblas.cpp

mxm_homework_SOURCES = mxm_homework.cpp graphblas.cpp

masked_mxm_homework_SOURCES = masked_mxm_homework.cpp graphblas.cpp

peer_pressure_demo_SOURCES = peer_pressure_demo.cpp graphblas.cpp

peer_pressure_karate_demo_SOURCES = peer_pressure_karate_demo.cpp graphblas.cpp

triangle_count_demo_SOURCES = triangle_count_demo.cpp graphblas.cpp

k_truss_demo_SOURCES = k_truss_demo.cpp graphblas.cpp

clean_data_SOURCES = clean_data.cpp graphblas.cpp

snaptsv2csr_SOURCES = snaptsv2csr.cpp

tsv2csr_SOURCES = tsv2csr.cpp

csr2tsv_SOURCES = csr2tsv.cpp

AM_CPPFLAGS+=-I$(abs_top_srcdir)
endif
