# Instructions to build nwgraph platform

## Requirements

- g++-11  Need to install and default to g++-11 compiler (required by Intel OneAPI TBB library)
- Intel OneAPI TBB version 2021.1.1(?) found at (https://github.com/oneapi-src/oneTBB/releases)
  - Later versions have not yet been tested

Ubuntu APT instructions can be found at: https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/apt.html#apt

For the lastest release:

`sudo apt install intel-oneapi-tbb`

For the specific release currently tested:

`sudo apt-get install intel-oneapi-tbb-2021.1.1`

Installed in /opt/intel/oneapi/tbb/2021.1.1/

export LD_LIBRARY_PATH=/opt/intel/oneapi/tbb/2021.1.1/lib/intel64/gcc4.8

/usr/bin/c++  -I/opt/intel/oneapi/tbb/2021.1.1/include -I/home/smcmillan/github/gbtl/src/../../NWgr/include -L/opt/intel/oneapi/tbb/2021.1.1/lib/intel64/gcc4.8 CMakeFiles/nwgraph_demo.dir/demo/nwgraph_demo.cpp.o -o bin/nwgraph_demo -ltbb
