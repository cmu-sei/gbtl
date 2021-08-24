# Metallizing GraphBlas Template Library

The goal of this project is to prototype a persistent memory allocator for GBTL container.

Metall is a memory allocator for persistent memory.
More Info at : https://github.com/LLNL/metall


## Getting Started

Demo programs are `src/demo/metall_demo/`

Currently, we have a working demo of GBTL graph construction, Breadth First Search, Triangle Counting and Page Rank


### Requirements:

Preferred versions 
`spack-0.15+, cmake@3.21.1+, gcc-9+, boost-1.75+, metall-0.15+`


        module load gcc/8.1.0 [Optional]
        source spack/share/spack/setup-env.sh
        
        spack install cmake@3.21.1
        spack install gcc@9.3.0
        spack install boost@1.77.0
        spack install metall@0.15
        
        spack load cmake@3.21.1
        spack load gcc@9.3.0
        spack load boost@1.77.0
        spack load metall@0.15


### To Compile and Run


Optional, Just make sure `echo $METALL_ROOT` and `echo $BOOST_ROOT` is not empty after spack installations. 

Create a build dir 

        mkdir build
        cd build
        cmake ../src/
        make
        cd bin  
                
        executable inputfile metall_datastore_location        

inputfile is needed only at the metall construction
metall_datastore_location can be any location such as optane nvm or nvme SSD

        ./m_metall_construction facebook_combined.txt /dev/shm/d1
        ./m_metall_algo1_bfs /dev/shm/d1
        ./m_metall_algo2_tc /dev/shm/d1
        ./m_metall_algo3_pr /dev/shm/d1


### To compile without CMake or Makefile just for reference

        g++ 
            -std=gnu++1z 
            -I./src/graphblas/detail 
            -I./src 
            -I./src/graphblas/platforms/metall
            -I. ~/metall-0.15/include/ 
            -I. ~/spack/boost-1.77.0/
            -pthread 
            ./src/demo/metall-demo/m_metall_construction.cpp   -o  m_metall_construction.exe   -lstdc++fs   -O3 