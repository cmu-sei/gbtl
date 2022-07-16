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
# Disable building in same directory as CMakeLists.txt
# -----------------------------------------------------------------------------
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)


# -----------------------------------------------------------------------------
# Extend the module path so we can find our custom modules
# -----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
#list(APPEND CMAKE_MODULE_PATH /opt/intel/onapi/tbb/latest/modulefiles)

project(GRAPHBLAS_TEMPLATE_LIBRARY VERSION 3.0.1 DESCRIPTION "NWGraph platform")

#-------------------------------------------------------------------------------
# determine build type
#-------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)

# For development only, not for end-users:
# set ( CMAKE_BUILD_TYPE Debug )

# Could also be: RelWithDebInfo

if ( NOT CMAKE_BUILD_TYPE )
    set ( CMAKE_BUILD_TYPE Release )
endif ( )

# For debugging
message(STATUS "CMake SRC: ${CMAKE_SOURCE_DIR}")
message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Bin Directory: ${CMAKE_BINARY_DIR}")
message(STATUS "Project Name: ${CMAKE_SOURCE_DIR}")
message(STATUS "Project Bin Directory: ${PROJECT_BINARY_DIR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Default platform is sequential
if (NOT PLATFORM)
    set(PLATFORM sequential)
endif()


#------------------------------------------------------------------------------


option(GBTL_USE_TBBMALLOC "Link to tbbmalloc" OFF)
option(TBB_TEST "" OFF)


#------------------------------------------------------------------------------

message(STATUS "Configuring platform: ${PLATFORM}")

if (EXISTS ${CMAKE_SOURCE_DIR}/src/graphblas/platforms/${PLATFORM})
    set(PLATFORM_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/graphblas/platforms/${PLATFORM})
else()
    unset(PLATFORM CACHE)
    message(FATAL_ERROR "Specified platform directory does not exist.")
endif()

list(APPEND EXTRA_LIBS "")
list(APPEND EXTRA_LIB_DIRS "")

if (PLATFORM STREQUAL "nwgraph")
  set(CMAKE_CXX_STANDARD 20)

  add_library(nwgraph INTERFACE)
  target_include_directories(nwgraph INTERFACE ${CMAKE_SOURCE_DIR}/external/NWGraph/include)

  # Use nwgraph submodule
  # set(NWGRAPH_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/external/NWGraph/include" )

  option(TBB_TEST "" OFF)

  message("++++ ${GBTL_USE_TBBMALLOC}")


  if (GBTL_USE_TBBMALLOC AND APPLE AND ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "arm64")
    message(WARNING "TBB malloc will probably not work on Apple arm64.
    You should probably disable it with -DGBTL_USE_TBBMALLOC=OFF (which is the default)")
  endif()


  if (NOT GBTL_USE_TBBMALLOC)
    option(TBBMALLOC_BUILD "Enable tbbmalloc build" OFF)
  else()
    option(TBBMALLOC_BUILD "Enable tbbmalloc build" OFF)
  endif()


  # Use tbb submodule
  add_subdirectory(external/oneTBB)

  #find_package(NWGRAPH REQUIRED)
  # Use fetch content rather than as a submodule
  # FetchContent_Declare(
  #   nwgraph
  #   GIT_REPOSITORY https://github.com/pnnl/NWGraph.git
  #   GIT_TAG master
  #   message(STATUS "Fetching nwgraph")
  # FetchContent_MakeAvailable(nwgraph)


  # Use fetch content rather than as a submodule
  # FetchContent_Declare(
  #   oneTBB
  #   GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
  #   GIT_TAG onetbb_2021)
  #   message(STATUS "Fetching TBB")
  # FetchContent_MakeAvailable(oneTBB)


  # find_package(TBB REQUIRED)
  # if (TBB_FOUND)
  #   message(STATUS "TBB FOUND")
  #  message(STATUS "TBB Version:  ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR}.${TBB_INTERFACE_VERSION}")
  #  message(STATUS "TBB_INCLUDE_DIRS: ${TBB_INCLUDE_DIRS}")
  #  message(STATUS "TBB_LIBRARIES ${TBB_LIBRARIES}")
  #  #include_directories( ${TBB_INCLUDE_DIRS} )
  #  #list(APPEND EXTRA_LIBS "tbb")
  #  #list(APPEND EXTRA_LIB_DIRS "")
  # else()
  #   unset(CACHE)
  #  message(FATAL_ERROR "TBB package not found.")
  # endif()
endif()


#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

message(STATUS "Configured platform: ${PLATFORM}")

#------------------------------------------------------------------------------
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})


#------------------------------------------------------------------------------

add_subdirectory(src)
