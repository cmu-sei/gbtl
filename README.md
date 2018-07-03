# GraphBLAS Template Library (GBTL), v. 2.0

## Project Goals

* Complete, mathematically equivalent implementation of the GraphBLAS C API
specification in C++ (specification http://graphblas.org).
* Graph algorithm library containing commonly used graph algorithms
implemented with the GraphBLAS primitive operations.

This is Version 2.0 of the C++ implementation that is mathematically
equivalent to the GraphBLAS C API.  Unlike the first version (which
predates the GraphBLAS C API Specification), this only contains the
'sequential' backend (in the platforms directory) written for a single
CPU.  Support for GPUs that was in version 1.0 is currently not
available but can be accessed using the git tag: 'Version1').

The implementation of the sequential backend is currently focused on
correctness over performance.  The project also contains implementations of
many common graph algorithms using the C++ API:

* Breadth-first traversal (aka BFS)
  * level BFS
  * parent list BFS
  * batched BFS
* Single-source shortest path (SSSP)
  * Floyd-Warshall
  * Delta stepping
* All-pairs shortest path (APSP)
* Centrality measures
  * Vertex Betweenness Centrality (batch variant too)
  * Edge Betweenness Centrality
  * Closeness centrality
* Clustering
  * peer pressure clustering
  * Markov clustering
* Triangle counting (many variants)
* K-truss enumeration
  * incidence matrix variant
  * adjacency matrix variant
* PageRank
* Maximal Independent Set (MIS)
* Minimum Spanning Tree (MST)
* Maxflow
  * Ford-Fulkerson
* Metrics
  * degree, in and out
  * graph distance
  * radius, diameter
  * vertex eccentricity

## Prerequisites

A detailed study of which C++ compilers are required has not been carried
out.  Anecdotally however, g++ 4.8.5 is not capable of compiling the
examples and tests, while g++ 6.3.0 and 7.3.0 both work.

## Building

This project is designed to use cmake to build and use an "out of
source" style build to make it easy to clean up. We build in to a
"build" directory in the top-level directory by following these steps:

$ mkdir build
$ cd build
$ cmake [-DPLATFORM=<backend>] ../src
$ make

The PLATFORM argument to cmake specifies which platform-specific source
code (also referred to as the backend) should be configured for the build
and the value must correspond to a subdirectory in
"gbtl/src/graphblas/platforms/" and that subdirectory must have a
"backend_include.hpp" file.  If this argument is omitted it defaults to
configuring the "sequential" platform.

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

The current library is set up as a header only library.  To install this
library, copy the graphblas directory, its subdirectories and the
specific platform subdirectory (sans the platform's test directories) to
a location in your include path.

## Documentation

Documentation can be generated using the Doxygen documentation system.  To
generate documentation run doxygen from the src directory:

$ cd src
$ doxygen

All documentation is built in the 'docs' subdirectory.

## Acknowledgments and Disclaimers

This material is based upon work funded and supported by the United
States Department of Defense under Contract No. FA8702-15-D-0002 with
Carnegie Mellon University for the operation of the Software
Engineering Institute, a federally funded research and development
center and by the United States Department of Energy under Contract
DE-AC05-76RL01830 with Battelle Memorial Institute for the Operation
of the Pacific Northwest National Laboratory.

THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN
AGENCY OF THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES
GOVERNMENT NOR THE UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED
STATES DEPARTMENT OF DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR
BATTELLE, NOR ANY OF THEIR EMPLOYEES, NOR ANY JURISDICTION OR
ORGANIZATION THAT HAS COOPERATED IN THE DEVELOPMENT OF THESE
MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY
LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR
USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR
PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE
PRIVATELY OWNED RIGHTS.

Reference herein to any specific commercial product, process, or
service by trade name, trademark, manufacturer, or otherwise does not
necessarily constitute or imply its endorsement, recommendation, or
favoring by the United States Government or any agency thereof, or
Carnegie Mellon University, or Battelle Memorial Institute. The views
and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

Please see “AUTHORS” file for a list of known contributors.

This release is an update of:

1. GraphBLAS Template Library (GBTL)
(https://github.com/cmu-sei/gbtl/blob/master/LICENSE) Copyright 2015
Carnegie Mellon University and The Trustees of Indiana. DM17-0037,
DM-0002659
