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

cmake_minimum_required(VERSION 3.17)

# -----------------------------------------------------------------------------
# Extend the module path so we can find our custom modules
# -----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake)
#list(APPEND CMAKE_MODULE_PATH /opt/intel/onapi/tbb/latest/modulefiles)

project(GRAPHBLAS_TEMPLATE_LIBRARY VERSION 3.0.1 DESCRIPTION "NWGraph platform")

# For debugging
message("Project Name:     ${PROJECT_NAME}")
message("Project Version:  ${PROJECT_VERSION}")
message("Source Directory: ${CMAKE_SOURCE_DIR}")
message("Bin Directory:    ${CMAKE_BINARY_DIR}")
message("Module path:      ${CMAKE_MODULE_PATH}")

# Default platform is sequential
if (NOT PLATFORM)
    set(PLATFORM sequential)
endif()

#------------------------------------------------------------------------------

message("Configuring platform: ${PLATFORM}")

if (EXISTS ${CMAKE_SOURCE_DIR}/graphblas/platforms/${PLATFORM})
    set(PLATFORM_SOURCE_DIR ${CMAKE_SOURCE_DIR}/graphblas/platforms/${PLATFORM})
else()
    unset(PLATFORM CACHE)
    message(FATAL_ERROR "Specified platform directory does not exist.")
endif()

list(APPEND EXTRA_LIBS "")
list(APPEND EXTRA_LIB_DIRS "")

if (PLATFORM STREQUAL "nwgraph")
  # TODO find_package(nwgraph) HARDCODED FOR NOW
  set(NWGRAPH_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/../../NWgr/include" )
  
  set(ENV{TBB_ROOT} "/opt/intel/onapi/tbb/latest/")
  find_package(TBB REQUIRED)
  if (TBB_FOUND)
    message("TBB FOUND")
    message("TBB Version:  ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR}.${TBB_INTERFACE_VERSION}")
    message("TBB_INCLUDE_DIRS: ${TBB_INCLUDE_DIRS}")
    message("TBB_LIBRARIES ${TBB_LIBRARIES}")
    #include_directories( ${TBB_INCLUDE_DIRS} )
    #list(APPEND EXTRA_LIBS "tbb")
    #list(APPEND EXTRA_LIB_DIRS "")
  else()
    unset(CACHE)
    message(FATAL_ERROR "TBB package not found.")
  endif()
endif()

message("Configured platform: ${PLATFORM}")

#------------------------------------------------------------------------------

# https://stackoverflow.com/questions/14306642/adding-multiple-executables-in-cmake

#include_directories( ${CMAKE_SOURCE_DIR} ${PLATFORM_SOURCE_DIR} )

# Compiler flags
if (PLATFORM STREQUAL "nwgraph")
  set(CMAKE_CXX_STANDARD 20)
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/intel/oneapi/tbb/2021.1.1/include -I${CMAKE_SOURCE_DIR}/../../NWgr/include -L/opt/intel/oneapi/tbb/2021.1.1/lib/intel64/gcc4.8 -ltbb")
  #message("nwgraph platform require C++20: ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_STANDARD 17)
endif()
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
    target_link_libraries( ${testname} ${TBB_LIBRARIES} )
    target_include_directories(
      ${testname}
      PUBLIC ${CMAKE_SOURCE_DIR}
      PRIVATE ${PLATFORM_SOURCE_DIR}
      )
    if (PLATFORM STREQUAL "nwgraph")
      target_include_directories(
        ${testname}
        PUBLIC ${TBB_INCLUDE_DIRS}
        PUBLIC ${NWGRAPH_INCLUDE_DIRECTORIES}
        )
    endif()
endforeach( testsourcefile ${TEST_SOURCES} )

### Make extra PLATFORM-specific tests
file( GLOB TEST_SOURCES LIST_DIRECTORIES false ${PLATFORM_SOURCE_DIR}/test/*.cpp )
foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" "" testname ${justname} )
    message("Adding: ${testname}_${PLATFORM} ")
    add_executable( ${testname}_${PLATFORM} ${testsourcefile} ${GRAPHBLAS_HEADERS})
    target_link_libraries( ${testname} ${TBB_LIBRARIES} )
    target_include_directories(
      ${testname}
      PUBLIC ${CMAKE_SOURCE_DIR}
      PRIVATE ${PLATFORM_SOURCE_DIR}
      )
    if (PLATFORM STREQUAL "nwgraph")
      target_include_directories(
        ${testname}
        PUBLIC ${TBB_INCLUDE_DIRS}
        PUBLIC ${NWGRAPH_INCLUDE_DIRECTORIES}
        )
    endif()
endforeach( testsourcefile ${TEST_SOURCES} )

## Make demos
file( GLOB TEST_SOURCES LIST_DIRECTORIES false ${CMAKE_SOURCE_DIR}/demo/*.cpp )
foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" "" testname ${justname} )
    message("Adding: ${testname}")
    add_executable( ${testname} ${testsourcefile} ${GRAPHBLAS_HEADERS})
    target_link_libraries( ${testname} ${TBB_LIBRARIES} )
    target_include_directories(
      ${testname}
      PUBLIC ${CMAKE_SOURCE_DIR}
      PRIVATE ${PLATFORM_SOURCE_DIR}
      )
    if (PLATFORM STREQUAL "nwgraph")
      target_include_directories(
        ${testname}
        PUBLIC ${TBB_INCLUDE_DIRS}
        PUBLIC ${NWGRAPH_INCLUDE_DIRECTORIES}
        )
    endif()
endforeach( testsourcefile ${TEST_SOURCES} )
