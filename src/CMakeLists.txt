cmake_minimum_required(VERSION 3.21.1)

project(GRAPHBLAS_TEMPLATE_LIBRARY_3)

# For debugging
message("CMake SRC: ${CMAKE_SOURCE_DIR}")
message("Project Name: ${PROJECT_NAME}")
message("Bin Directory: ${CMAKE_BINARY_DIR}")
message("Project Name: ${CMAKE_SOURCE_DIR}")
message("Project Bin Directory: ${PROJECT_BINARY_DIR}")

# Default platform is sequential
if (NOT PLATFORM)
    set(PLATFORM metall)
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


if (("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))
   if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.1)
      message(FATAL_ERROR "GCC version must be at least 8.1")
   endif ()
endif ()

set (Boost_NO_BOOST_CMAKE ON)
find_package(Boost 1.64)


if (Boost_FOUND)
# Just a sanity check
    if (NOT Boost_INCLUDE_DIRS)
        message(FATAL_ERROR "Something is wrong: Boost_INCLUDE_DIRS is empty although Boost_FOUND is true.")
    endif()
else ()
    message(FATAL_ERROR "Can not find Boost C++ Libraries")
endif ()
include_directories(${Boost_INCLUDE_DIRS})

if (METALL_ROOT)
	find_path(METALL_INCLUDE_DIR metall/metall.hpp PATHS ${METALL_ROOT}/include REQUIRED)
else ()
	find_path(METALL_INCLUDE_DIR metall/metall.hpp REQUIRED)
endif ()

include_directories(${METALL_INCLUDE_DIR})

message("Metall Path: ${METALL_INCLUDE_DIR}")



set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread")

# Build a list of all the graphblas headers.
file(GLOB GRAPHBLAS_HEADERS graphblas/*.hpp)

# Build tests and demos into a separate bin directory so it doesn't get mixed in
# with settings and config files. Note the library dependency is not in here.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)



# Make demos
file( GLOB TEST_SOURCES LIST_DIRECTORIES false ${CMAKE_SOURCE_DIR}/demo/metall_demo/*.cpp )
foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" "" testname ${justname} )
    message("Adding: ${testname}")
    add_executable( ${testname} ${testsourcefile} ${GRAPHBLAS_HEADERS})
endforeach( testsourcefile ${TEST_SOURCES} )

configure_file(${CMAKE_SOURCE_DIR}/demo/metall_demo/facebook_combined.txt ${CMAKE_BINARY_DIR}/bin/facebook_combined.txt COPYONLY)

