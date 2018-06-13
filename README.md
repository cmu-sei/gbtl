# GraphBLAS Template Library (GBTL)

This is Version 2.0 of the C++ implementation that is mathematically
equivalent to the GraphBLAS C API.  Unlike the first version (which
predates the GraphBLAS C API Specification), this only contains the
backend written for a single CPU (support for GPUs that was in the
first version is currently not available but can be accessed using the
git tag: 'Version1').

The implementation of this backend is currently focused on correctness
over performance.  It also contains implementations of many common
graph algorithms using the C++ API.

The first version was the result of a collaboration between Carnegie
Mellon University's Software Engineering Institute (an FFRDC) and
Indiana University's Center for Research in Extreme Scale Technologies
(CREST).  Version 2.0 borrows heavily from the first one.

## Prerequisites

A detailed study of which C++ compilers are require has not been carried
out.  Anecdotally however, g++ 4.8.5 is not capable of compiling the
examples and tests, while g++ 6.3.0 works.

## Building

This project is designed to use cmake to build and use an "out of
source" style build to make it easy to clean up. We build in to a
"build" directory in the top-level directory by following these steps:

$ mkdir build
$ cd build
$ cmake [-DPLATFORM=<backend>] ../src
$ make

The PLATFORM argument to cmake specifies which backend source code should
configured for the build and the value must correspond to a subdirectory
in "gbtl/src/graphblas/system/" and that subdirectory must have a
"backend_include.hpp" file.  If this argument is omitted it defaults to
configuring the "sequential" backend.

Using "make -i -j8" tries to build every test (ignoring all erros) and
uses all eight the CPU's cores to speed up the build (use a number
appropriate for your system).

There is a convenience script to do this from scratch called
rebuild.sh that also removes all the old content from a previous use
of clean_build.sh.

For CLion support in the cmake project settings "Build, Execution,
Deployment > CMake > Generation path:" set it to "../build" to use the
same make files as that created by the clean build process so that
there aren't two different build trees.

There is also a script called "clean_cmake.sh" that removes all the
files, if cmake is run in the src directory or in the root (gbtl)
directory.


## Installation

The current library is set up as a header only library.  To install
this library, copy the graphblas directory (and its subfolders) to a
location in your include path.

## Documentation

Documentation can be generated using the Doxygen documentation system.  To
generate documentation run doxygen from the src directory:

$ cd src
$ doxygen

All documentation is built in the 'docs' subdirectory.

## Project Goals

* Complete, mathematically equivalent implementation of the GraphBLAS C API
specification in C++ (specification http://graphblas.org).
* Graph algorithm library supporting commonly used graph algorithms.
* Graph input/output library (TBD).
