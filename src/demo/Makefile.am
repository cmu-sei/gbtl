# TODO Best way to add the library sources?

AM_CPPFLAGS = -I$(abs_top_srcdir) $(BACKEND_CPPFLAGS)

if ENABLE_CUDA

AM_CPPFLAGS += -I$(abs_top_srcdir)/cusplibrary

include $(top_srcdir)/mk/cuda.mk

noinst_PROGRAMS = mxm_homework_cuda

mxm_homework_cuda_SOURCES = mxm_homework_cuda.cu

else

noinst_PROGRAMS = bfs_demo bfs_level_demo sssp_demo mxm_homework masked_mxm_homework peer_pressure_demo peer_pressure_karate_demo

bfs_demo_SOURCES = bfs_demo.cpp

bfs_level_demo_SOURCES = bfs_level_demo.cpp

sssp_demo_SOURCES = sssp_demo.cpp

mxm_homework_SOURCES = mxm_homework.cpp
masked_mxm_homework_SOURCES = masked_mxm_homework.cpp

peer_pressure_demo_SOURCES = peer_pressure_demo.cpp

peer_pressure_karate_demo_SOURCES = peer_pressure_karate_demo.cpp

AM_CPPFLAGS+=-I$(abs_top_srcdir)/src
endif
