#include <iostream>
#include <fstream>
#include <chrono>
#include "Timer.hpp"

#include <graphblas/graphblas.hpp>

#include <metall/metall.hpp>
#include <metall/utility/fallback_allocator_adaptor.hpp>

//****************************************************************************
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "ERROR: too few arguments. arg1: input file. arg2: datastore location." << std::endl;
        exit(1);
    }
    // Read the edgelist and create the tuple arrays
    std::string pathname(argv[1]);

    Timer<std::chrono::steady_clock, std::chrono::milliseconds> my_timer;

    grb::IndexArrayType iL, iU, iA;
    grb::IndexArrayType jL, jU, jA;
    uint64_t num_rows = 0;
    uint64_t max_id = 0;
    uint64_t src, dst;

    my_timer.start();
    {
        std::ifstream infile(pathname);
        while (infile)
        {
            infile >> src >> dst;
            //std::cout << "Read: " << src << ", " << dst << std::endl;
            if (src > max_id) max_id = src;
            if (dst > max_id) max_id = dst;

            if (src < dst)
            {
                iA.push_back(src);
                jA.push_back(dst);

                iU.push_back(src);
                jU.push_back(dst);
            }
            else if (dst < src)
            {
                iA.push_back(src);
                jA.push_back(dst);

                iL.push_back(src);
                jL.push_back(dst);
            }
            // else ignore self loops

            ++num_rows;
        }
    }

    std::cout << "Read " << num_rows << " rows." << std::endl;
    std::cout << "#Nodes = " << (max_id + 1) << std::endl;

    // sort the
    using DegIdx = std::tuple<grb::IndexType,grb::IndexType>;
    std::vector<DegIdx> degrees(max_id + 1);
    for (grb::IndexType idx = 0; idx <= max_id; ++idx)
    {
        degrees[idx] = {0UL, idx};
    }

    {
        std::ifstream infile(pathname);
        while (infile)
        {
            infile >> src >> dst;
            if (src != dst)
            {
                std::get<0>(degrees[src]) += 1;
            }
        }
    }

    std::sort(degrees.begin(), degrees.end(),
              [](DegIdx a, DegIdx b) { return std::get<0>(b) < std::get<0>(a); });

    //relabel

    for (auto &idx : iL) { idx = std::get<1>(degrees[idx]); }
    for (auto &idx : jL) { idx = std::get<1>(degrees[idx]); }


    using T = int32_t;
    using allocator_t = metall::utility::fallback_allocator_adaptor<metall::manager::allocator_type<char>>;
    using Metall_MatType = grb::Matrix<T, allocator_t>;

    grb::IndexType NUM_NODES(max_id + 1);
    std::vector<T> v(iA.size(), 1);

    std::cout <<  "NUM_NODES :"  <<  NUM_NODES  << std::endl;
    std::cout <<  "iA.front() :" <<  iA.front() << std::endl;
    std::cout <<  "jA.front() :" <<  jA.front() << std::endl;


    //================= Graph Construction in Metall Scope ========================

    {
        metall::manager manager(metall::create_only, argv[2]);
        Metall_MatType *A = manager.construct<Metall_MatType>("gbtl_vov_matrix")
                        ( NUM_NODES, NUM_NODES, manager.get_allocator());
        A->build(iA.begin(), jA.begin(), v.begin(), iA.size());

        manager.construct<grb::IndexType>("NUM_NODES")(NUM_NODES);
        manager.construct<T>("iA.front()")(iA.front());
        manager.construct<T>("jA.front()")(jA.front());
    }
    my_timer.stop();
    std::cout << "Graph Construction time: \t" << my_timer.elapsed()
                << " milli seconds." << std::endl;

    return 0;
}
