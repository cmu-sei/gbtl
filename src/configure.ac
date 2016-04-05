AC_INIT([graphblas], [0.0], [email@site.ext])
AM_INIT_AUTOMAKE([-Wall -Werror])
AC_CONFIG_MACRO_DIRS([m4])
AC_PROG_CXX
AC_PROG_CC
AX_CXX_COMPILE_STDCXX_11([noext], [mandatory])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([
    Makefile
    demo/Makefile
    test/Makefile
    test/sequential/Makefile
])
#find out what version we are running
ARCH=`uname -m`
if [[ $ARCH == "x86_64" ]];
then
  SUFFIX="64"
else
  SUFFIX=""
fi

# Setup NVCC paths
# ------------------------------------------------------------------------------
AC_ARG_WITH([nvcc],
   [AS_HELP_STRING([--with-nvcc=PATH], [prefix where nvcc is installed @<:@default=/usr/local/nvcc@:>@])])

if test -n "$with_nvcc"
then
   NVCC_CXXFLAGS="-I$with_nvcc/include -Xcompiler -Wall"
   NVCC_LIBS="-lcudadevrt -rdc=true"
   NVCC_LDFLAGS="-L$with_nvcc/lib$SUFFIX"
   NVCC="$with_nvcc/bin/nvcc"
   LINK="$with_nvcc/bin/nvcc -o \$@"
else
   NVCC_CXXFLAGS="-I/usr/local/cuda/include --gpu-architecture=sm_35"
   NVCC_LIBS="-lcudart"
   NVCC_LDFLAGS="-L/usr/local/cuda/lib$SUFFIX"
   NVCC="nvcc"
   LINK="nvcc -o \$@"
fi
AC_SUBST(NVCC_CXXFLAGS)
AC_SUBST(NVCC_LIBS)
AC_SUBST(NVCC_LDFLAGS)
AC_SUBST(NVCC)
AC_SUBST(LINK)

AC_ARG_WITH([backend],
   [AS_HELP_STRING([--with-backend=BACKEND], [sequential or cusp_gpu @<:@default=cusp_gpu@:>@])],
                   [],
                   [with_backend=cusp_gpu])

AS_CASE([$with_backend],
      [cusp_gpu],
                 [BACKEND_CPPFLAGS="-Xcompiler -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DGB_USE_CUSP_GPU"
                  USING_CUDA="true"],
      [sequential],
                 [BACKEND_CPPFLAGS="-DGB_USE_SEQUENTIAL"
                 USING_CUDA="false"],
      [AC_MSG_ERROR([No backend specified.])]
)

AC_SUBST(BACKEND_CPPFLAGS)
AC_SUBST(USING_CUDA)

AM_CONDITIONAL([ENABLE_CUDA], [test x$USING_CUDA = "xtrue"])


# Setup doxygen.
# http://chris-miceli.blogspot.com/2011/01/integrating-doxygen-with-autotools.html
AC_CHECK_PROGS([DOXYGEN], [doxygen])
if test -z "$DOXYGEN";
   then AC_MSG_WARN([Doxygen not found - continuing without Doxygen support])
fi

AC_SUBST(DOXYGEN)
AM_CONDITIONAL([HAVE_DOXYGEN], [test -n "$DOXYGEN"])

AC_OUTPUT
