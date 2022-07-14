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

3. Use GB_INCLUDE_BACKEND_MATRIX to specify the include file that defines the
platform's Matrix base class

4. Use GB_INCLUDE_BACKEND_VECTOR to specify the include file that defines the
platform's Vector base class

5. Use GB_INCLUDE_BACKEND_UTILITY to specify the include file that defines
the pretty_print and pretty_print_matrix functions.

6. Use GB_INCLUDE_BACKEND_TRANSPOSE_VIEW to specify the include file that
defines the platform's TransposeView class

7. Use GB_INCLUDE_BACKEND_COMPLEMENT_VIEW to specify the include file that defines the platform's ComplementView class

8. Use GB_INCLUDE_BACKEND_OPERATIONS to specify the include file(s) that
defines the platform's operations functions.
