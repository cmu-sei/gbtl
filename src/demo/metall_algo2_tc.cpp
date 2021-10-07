#include <iostream>
#include <fstream>
#include <chrono>
#include "Timer.hpp"

#include <graphblas/graphblas.hpp>
#include <algorithms/triangle_count.hpp>

#include <metall/metall.hpp>
#include <metall/utility/fallback_allocator_adaptor.hpp>

//****************************************************************************
int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cerr << "ERROR: Metall datastore path is not given." << std::endl;
    exit(1);
  }

    Timer<std::chrono::steady_clock, std::chrono::milliseconds> my_timer;
    using T = int32_t;

    using allocator_t = metall::utility::fallback_allocator_adaptor<metall::manager::allocator_type<char>>;
    using Metall_MatType = grb::Matrix<T, allocator_t>;


    //================= Triangle Counting in Metall Scope =========================

    {
        my_timer.start();
        metall::manager manager(metall::open_read_only, argv[1]);
        Metall_MatType *A = manager.find<Metall_MatType>("gbtl_vov_matrix").first;
        my_timer.stop();
        std::cout << "TC re-attach time: \t\t" << my_timer.elapsed()
                   << " milli seconds." << std::endl;

        my_timer.start();
        T count(0);
        count = algorithms::triangle_count_masked_noT(*A);
        my_timer.stop();
        std::cout << "TC Algorithm time: \t\t"  << my_timer.elapsed() 
		  << " milli seconds. " 	<< std::endl
		  << "Num Triangles: \t\t" 	<< count  	<< std::endl;

    }
    return 0;
}

