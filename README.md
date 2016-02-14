# GraphBLAS Template Library (GBTL)

This package provides an implementation of the current GraphBLAS API in C++,
as well as implementations of common graph algorithms built on top of this
GraphBLAS implementation.  Two backends are provided: a GPU backend built on
cusp/thrust libraries (called cusp_gpu), and a simple CPU backend (called
sequential).

This is the result of a collaboration between Carnegie Mellon University's
Software Engineering Institute (an FFRDC) and Indiana University's Center for
Research in Extreme Scale Technologies (CREST).

## Prerequisites

If it is desired to build using the cusp backend, CUSP must also be added
to this project.  To do so, download cusp from http://cusplibrary.github.io/.
Click on "Get Cusp", the rightmost menu item, to download the library.  Extract
the downloaded archive, then copy the contents in the extracted directory to a
directory called cusplibrary located at the top of where you have put this
project's source.

NOTE: There is a problem with v0.5.1 of the CUSP library and one must get a
snapshot of the development branch instead

## Building

Currently, there are two different GraphBLAS backends implemented: A CPU
backend, which is named sequential, and a GPU backend, which is implemented on
top of CUSP, called cusp_gpu.

Since the code is heavily templated, it is a header only library.  However,
to compile the demo programs and unit tests first cd to the src directory.

$ cd src

### Autotools configuration

Configure the initial build system using autotools.  To do
this, run:

$ ./autogen.sh

### Configuring for Backend Selection

Only one "backend" can be built at a time.  We currently support a sequential
CPU build and one for use with the CUSP library on NVIDIA GPUs.  This is
controlled with the following command:

$ ./configure [--with-backend={cusp_gpu|sequential}]

The default backend is cusp_gpu.  If you wish explicitly specify to build
using the cusp_gpu backend, supply the argument --with-backend=cusp_gpu
to the configure script.  To build the sequential backend, supply the argument
--with-backend=sequential

### Building

$ make

## Installation

The current library is set up as a header only library.  To install this
library, copy the graphblas directory to a location in your include path
(probably /usr/include or /usr/local/include).  If you wish to use the cusp
backend, make sure to place the cusp library in your include path as well.

## Documentation

Documentation can be generated using the Doxygen documentation system.  To
generate documentation, Configure

## Project Goals

* Full GraphBLAS specification implementation (specification in process at
http://graphblas.org).
* Graph input/output library.
* Graph algorithm library supporting commonly used graph algorithms.
