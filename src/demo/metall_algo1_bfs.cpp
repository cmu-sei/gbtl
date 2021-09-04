#include <iostream>
#include <fstream>
#include <chrono>
#include "Timer.hpp"

#include <graphblas/graphblas.hpp>
#include <algorithms/bfs.hpp>

#include <metall/metall.hpp>
#include <metall/utility/fallback_allocator_adaptor.hpp>

//****************************************************************************
int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cerr << "ERROR: Metall datastore path is not given." << std::endl;
    exit(1);
  }
    using T = int32_t;
    // Read the numnodes

    Timer<std::chrono::steady_clock, std::chrono::milliseconds> my_timer;

    using allocator_t = metall::utility::fallback_allocator_adaptor<metall::manager::allocator_type<char>>;
    using Metall_MatType = grb::Matrix<T, allocator_t>;



    //================= single BFS in Metall Scope ================================
    {
        my_timer.start();
        metall::manager manager(metall::open_read_only, argv[1]);
        Metall_MatType *A = manager.find<Metall_MatType>("gbtl_vov_matrix").first;
        my_timer.stop();
        std::cout << "BFS re-attach time: \t\t" << my_timer.elapsed()
                   << " milli seconds." << std::endl;

        my_timer.start();
        grb::IndexType NUM_NODES = *(manager.find<grb::IndexType>("NUM_NODES").first);
        T iA_front = *(manager.find<T>("iA.front()").first);
        T jA_front = *(manager.find<T>("jA.front()").first);

        grb::Vector<T> parent_list(NUM_NODES);
        grb::Vector<T> root(NUM_NODES);
        root.setElement(iA_front, jA_front);
        algorithms::bfs(*A, root, parent_list);
        my_timer.stop();
        std::cout << "BFS Algorithm time: \t\t" << my_timer.elapsed()
                  << " milli seconds." << std::endl;

    }

    return 0;
}

