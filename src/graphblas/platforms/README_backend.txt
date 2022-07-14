Each subdirectory in this (system) directory, contains the backend code for
one platform.  The name of the subdirectory is the name of the platform used
when configuring the build.  Note that only one backend platform can be
configured and compiled at a time.

Within each platform directory, a backend_include.hpp file should be created.
It should be based on the template in this directory
(backend_include_template.hpp).  It should specify which include files within
the backend directory contain the declarations for various classes and
functions needed by the frontend code.  Comments in the template header file
provide more information about what to specify:

 
1. Global search and replace GB_BACKEND_NAME with the name of the platform
directory.

2. Use GB_INCLUDE_BACKEND_ALL to specify a single include file that contains
the include directives for all of the platform's header files

3. Use GB_INCLUDE_BACKEND_MATRIX to specify the include file that defines
the backend::Matrix type (in param_unpack.hpp in sequential).

4. Use GB_INCLUDE_BACKEND_VECTOR to specify the include file that defines
the backend::Vector type (in param_unpack.hpp in sequential).

5. Use GB_INCLUDE_BACKEND_OPERATIONS to specify the include file(s) that
defines the platform's operations functions.

6. Implement code (e.g., param_unpack.hpp) to map template tags from
src/graphblas/detail/matrix_tags.hpp to backend data types for matrices
and vectors.
