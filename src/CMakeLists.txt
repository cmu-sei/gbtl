cmake_minimum_required(VERSION 2.8)

project(SPIRALGRAPH)

# https://stackoverflow.com/questions/14306642/adding-multiple-executables-in-cmake

# For debugging
#message("CMake SRC: ${CMAKE_SOURCE_DIR}")
#message("Project Name: ${PROJECT_NAME}")
#message("Bin Directory: ${CMAKE_BINARY_DIR}")
#message("Project Name: ${CMAKE_SOURCE_DIR}")
#message("PProject Bin Directory: ${PROJECT_BINARY_DIR}")

# This seems hokey that we need to include the root as our directory
include_directories(${CMAKE_SOURCE_DIR})

# Compiler flags
set(CMAKE_CXX_STANDARD 11)

# First we build a "library" that we include with each test.
set(GRAPHBLAS_SOURCES graphblas/graphblas.cpp)
file(GLOB GRAPHBLAS_HEADERS graphblas/*.hpp)
add_library(GraphBLASLib ${GRAPHBLAS_SOURCES} ${GRAPHBLAS_HEADERS})

# Build tests and demos into a separate bin directory so it doesn't get mixed in
# with settings and config files. Note the library dependency is not in here.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

### Make basic tests
file( GLOB TEST_SOURCES LIST_DIRECTORIES false ${CMAKE_SOURCE_DIR}/test/*.cpp )
foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" "" testname ${justname} )
    message("Adding: ${testname}")
    add_executable( ${testname} ${testsourcefile} )
    target_link_libraries( ${testname} GraphBLASLib )
endforeach( testsourcefile ${TEST_SOURCES} )

### Make extra sequential tests
file( GLOB TEST_SOURCES LIST_DIRECTORIES false ${CMAKE_SOURCE_DIR}/test/sequential/*.cpp )
foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" "" testname ${justname} )
    message("Adding: ${testname}_sequential ")
    add_executable( ${testname}_sequential ${testsourcefile} )
    target_link_libraries( ${testname}_sequential GraphBLASLib )
endforeach( testsourcefile ${TEST_SOURCES} )

## Make demos
file( GLOB TEST_SOURCES LIST_DIRECTORIES false ${CMAKE_SOURCE_DIR}/demo/*.cpp )
foreach( testsourcefile ${TEST_SOURCES} )
    get_filename_component(justname ${testsourcefile} NAME)
    string( REPLACE ".cpp" "" testname ${justname} )
    message("Adding: ${testname}")
    add_executable( ${testname} ${testsourcefile} )
    target_link_libraries( ${testname} GraphBLASLib )
endforeach( testsourcefile ${TEST_SOURCES} )

# TODO: Make capi tests
#add_executable(bcc_demo capi/bcc_demo.c capi/bcc.c capi/bcc.h ${GRAPHBLAS} capi/graphblas.cpp capi/TypeAdapter.hpp capi/UnaryAdapter.hpp capi/graphblas_private.h capi/BinaryAdapter.xhpp capi/TypeUnion.hpp capi/SemiringAdapter.hpp capi/FunctionInvocation.cpp capi/FunctionInvocation.h capi/MonoidAdapter.hpp)
