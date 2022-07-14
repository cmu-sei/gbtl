# GraphBLAS Template Library (GBTL), Version 3.0
#
# Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
# Authors.
#
# THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
# THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
# UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
# DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
# EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
# DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
# ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
# OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
# DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
# RIGHTS.
#
# Released under a BSD-style license, please see LICENSE file or contact
# permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release
# and unlimited distribution.  Please see Copyright notice for non-US
# Government use and distribution.
#
# DM20-0442

cmake_minimum_required(VERSION 2.8)

project(GRAPHBLAS_TEMPLATE_LIBRARY_3)

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

if (EXISTS ${CMAKE_SOURCE_DIR}/graphblas/platforms/${PLATFORM})
    set(PLATFORM_SOURCE_DIR ${CMAKE_SOURCE_DIR}/graphblas/platforms/${PLATFORM})
else()
    unset(PLATFORM CACHE)
    message(FATAL_ERROR "Specified platform directory does not exist.")
endif()

message("Configured platform: ${PLATFORM}")

# https://stackoverflow.com/questions/14306642/adding-multiple-executables-in-cmake

# This seems hokey that we need to include the root as our directory
include_directories(${CMAKE_SOURCE_DIR} ${PLATFORM_SOURCE_DIR})

# Compiler flags
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

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

### Make extra PLATFORM-specific tests
file( GLOB TEST_SOURCES LIST_DIRECTORIES false ${PLATFORM_SOURCE_DIR}/test/*.cpp )
foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" "" testname ${justname} )
    message("Adding: ${testname}_${PLATFORM} ")
    add_executable( ${testname}_${PLATFORM} ${testsourcefile} ${GRAPHBLAS_HEADERS})
endforeach( testsourcefile ${TEST_SOURCES} )

## Make demos
file( GLOB TEST_SOURCES LIST_DIRECTORIES false ${CMAKE_SOURCE_DIR}/demo/*.cpp )
foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" "" testname ${justname} )
    message("Adding: ${testname}")
    add_executable( ${testname} ${testsourcefile} ${GRAPHBLAS_HEADERS})
endforeach( testsourcefile ${TEST_SOURCES} )
