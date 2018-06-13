Each subdirectory in this (system) directory, contains the backend code for
one platform.  The name of the subdirectory is the name of the platform used
when configuring the build.  Note that only one backend can be configured and
compiled at a time.

Each backend should copy backend_include_temp.hpp into the backend directory,
rename it "backend_include.hpp", and modify the contents to point to a set of
header files within the backend directory:
  <backend name>.hpp  - that includes all of the other backend header files
  Matrix.hpp
  Vector.hpp
  utility.hpp
  TransposeView.hpp
  ComplementView.hpp
  operations.hpp

This backend_include.hpp file is copied to the frontend during cmake build
configuration.
