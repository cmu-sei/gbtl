

# Build a list of all the graphblas headers.
file(GLOB GRAPHBLAS_HEADERS graphblas/*.hpp)

# Build tests and demos into a separate bin directory so it doesn't get mixed in
# with settings and config files. Note the library dependency is not in here.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(graphblas INTERFACE)
target_include_directories(graphblas INTERFACE ${CMAKE_SOURCE_DIR}/src)

add_subdirectory(test)
add_subdirectory(demo)
