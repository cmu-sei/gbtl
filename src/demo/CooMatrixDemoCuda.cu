#include <graphblas/graphblas.hpp>

// COO is in graphblas, so need this unless we write graphblas::COO.
using namespace graphblas;

int main()
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                             {7, 0, 0, 0},
                                             {0, 0, 9, 4},
                                             {2, 5, 0, 3},
                                             {2, 0, 0, 1},
                                             {0, 0, 0, 0},
                                             {0, 1, 0, 2}};

    // Build some sparse matrices.
    graphblas::Matrix<double, DirectedMatrixTag> m1(mat);
    graphblas::Matrix<double, DirectedMatrixTag> m3(m1);
    // std::cout << "Here" << std::endl;
    // std::cout << m1 << std::endl;

    // Should be:
    // [[6, 0, 0, 4]
    //  [7, 0, 0, 0]
    //  [0, 0, 9, 4]
    //  [2, 5, 0, 3]
    //  [2, 0, 0, 1]
    //  [0, 0, 0, 0]
    //  [0, 1, 0, 2]]
    // A: 6 4 7 9 4 2 5 3 2 1 1 2
    // IA: 0 2 3 5 8 10 10 12
    // JA: 0 3 0 2 3 0 1 3 0 3 1 3

    m1.set_value_at(0, 0, 8);
    // std::cout << m1 << std::endl;
    // Should be:
    // [[8, 0, 0, 4]
    //  [7, 0, 0, 0]
    //  [0, 0, 9, 4]
    //  [2, 5, 0, 3]
    //  [2, 0, 0, 1]
    //  [0, 0, 0, 0]
    //  [0, 1, 0, 2]]
    // A: 8 4 7 9 4 2 5 3 2 1 1 2
    // IA: 0 2 3 5 8 10 10 12
    // JA: 0 3 0 2 3 0 1 3 0 3 1 3

    m1.set_value_at(0, 1, 8);
    // std::cout << m1 << std::endl;

    // Should be:
    // [[6, 0, 0, 4]
    //  [7, 0, 0, 0]
    //  [0, 0, 9, 4]
    //  [2, 5, 0, 3]
    //  [2, 0, 0, 1]
    //  [0, 0, 0, 0]
    //  [0, 1, 0, 2]]
    // A: 6 4 7 9 4 2 5 3 2 1 1 2
    // IA: 0 2 3 5 8 10 10 12
    // JA: 0 3 0 2 3 0 1 3 0 3 1 3

    // std::cout << m3 << std::endl;

    // std::cout << ((m1 == m3) ? "true" : "false") << std::endl;
    //Should be:
    // false (since deep copy).

    std::vector<std::vector<double> > debug = {{0, 1, 2, 3},
                                               {4, 0, 6, 7},
                                               {8, 9, 1, 0}};
    graphblas::Matrix<double, DirectedMatrixTag> m4(debug);
    //std::cout << m4 << std::endl;
    m4.set_value_at(1, 1, 1);
    //std::cout << m4 << std::endl;
    m4.set_value_at(2, 2, 0);
    std::cout << m4.get_value_at(2, 2) << std::endl;

    return 0;
}
