#
# Copyright (c) 2017 Carnegie Mellon University.
# All Rights Reserved.
#
# THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
# MELLON UNIVERSITY EXPRESSLY DISCLAIMS TO THE FULLEST EXTENT PERMITTED BY
# LAW ALL EXPRESS, IMPLIED, AND STATUTORY WARRANTIES, INCLUDING, WITHOUT
# LIMITATION, THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
#
# This Program is distributed under a BSD license.  Please see LICENSE file or
# permission@sei.cmu.edu for more information.  DM-0002659
#

cmake_minimum_required(VERSION 2.8)

project(GRAPHBLAS_TEMPLATE_LIBRARY_2)

# For debugging
message("CMake SRC: ${CMAKE_SOURCE_DIR}")
message("Project Name: ${PROJECT_NAME}")
message("Bin Directory: ${CMAKE_BINARY_DIR}")
message("Project Name: ${CMAKE_SOURCE_DIR}")
message("Project Bin Directory: ${PROJECT_BINARY_DIR}")

# Default platform is sequential
if (NOT PLATFORM)
  set(PLATFORM sequential)
endif()

message("Configuring platform: ${PLATFORM}")

if (EXISTS ${CMAKE_SOURCE_DIR}/graphblas/system/${PLATFORM})
    file(COPY ${CMAKE_SOURCE_DIR}/graphblas/system/${PLATFORM}/backend_include.hpp DESTINATION ${CMAKE_SOURCE_DIR}/graphblas)
else()
    message(FATAL_ERROR "Backend directory does not exist: ${PLATFORM}")
endif()

unset($PLATFORM)

# https://stackoverflow.com/questions/14306642/adding-multiple-executables-in-cmake

# This seems hokey that we need to include the root as our directory
include_directories(${CMAKE_SOURCE_DIR})

# Compiler flags
set(CMAKE_CXX_STANDARD 11)

# Build a list of all the graphblas headers.
file(GLOB GRAPHBLAS_HEADERS graphblas/*.hpp)

# Build tests and demos into a separate bin directory so it doesn't get mixed in
# with settings and config files. Note the library dependency is not in here.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

### Make basic tests
file( GLOB TEST_SOURCES LIST_DIRECTORIES false ${CMAKE_SOURCE_DIR}/test/*.cpp )
foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" "" testname ${justname} )
    message("Adding: ${testname}")
    add_executable( ${testname} ${testsourcefile} ${GRAPHBLAS_HEADERS})
endforeach( testsourcefile ${TEST_SOURCES} )

### Make extra sequential tests
file( GLOB TEST_SOURCES LIST_DIRECTORIES false ${CMAKE_SOURCE_DIR}/test/sequential/*.cpp )
foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" "" testname ${justname} )
    message("Adding: ${testname}_sequential ")
    add_executable( ${testname}_sequential ${testsourcefile} ${GRAPHBLAS_HEADERS})
endforeach( testsourcefile ${TEST_SOURCES} )

## Make demos
file( GLOB TEST_SOURCES LIST_DIRECTORIES false ${CMAKE_SOURCE_DIR}/demo/*.cpp )
foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" "" testname ${justname} )
    message("Adding: ${testname}")
    add_executable( ${testname} ${testsourcefile} ${GRAPHBLAS_HEADERS})
endforeach( testsourcefile ${TEST_SOURCES} )

# TODO: Make capi tests
#add_executable(bcc_demo capi/bcc_demo.c capi/bcc.c capi/bcc.h ${GRAPHBLAS} capi/graphblas.cpp capi/TypeAdapter.hpp capi/UnaryAdapter.hpp capi/graphblas_private.h capi/BinaryAdapter.xhpp capi/TypeUnion.hpp capi/SemiringAdapter.hpp capi/FunctionInvocation.cpp capi/FunctionInvocation.h capi/MonoidAdapter.hpp)
