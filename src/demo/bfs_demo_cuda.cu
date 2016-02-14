#include <iostream>
#include <vector>

#include <graphblas/graphblas.hpp>
#include <algorithms/bfs.hpp>

int main()
{
    // TODO Assignment from Initalizer list.
    // graphblas::CsrMatrix<double> G_tn({{0, 0, 0, 1, 0, 0, 0, 0, 0},
    //                                     {0, 0, 0, 1, 0, 0, 1, 0, 0},
    //                                     {0, 0, 0, 0, 1, 1, 1, 0, 1},
    //                                     {1, 1, 0, 0, 1, 0, 1, 0, 0},
    //                                     {0, 0, 1, 1, 0, 0, 0, 0, 1},
    //                                     {0, 0, 1, 0, 0, 0, 0, 0, 0},
    //                                     {0, 1, 1, 1, 0, 0, 0, 0, 0},
    //                                     {0, 0, 0, 0, 0, 0, 0, 0, 0},
    //                                     {0, 0, 1, 0, 1, 0, 0, 0, 0}});

    // graphblas::CsrMatrix<double> identity
    //     = linalg::identity<graphblas::CsrMatrix<double> >(9);

    // // TODO Conversions between matrix classes.
    // graphblas::CsrMatrix<double> G_tn_res(9, 9);
    // algorithms::bfs
    //     <graphblas::CsrMatrix<double>, graphblas::CsrMatrix<double>,
    //      graphblas::CsrMatrix<double> >

    //hardcoded r/c
    graphblas::Matrix<double, graphblas::DirectedMatrixTag> G_tn(9,9);
    graphblas::Matrix<double, graphblas::DirectedMatrixTag> G_tn_res(9,9);
    std::vector<std::vector <double> > matrixData =
        {{0, 0, 0, 1, 0, 0, 0, 0, 0},
         {0, 0, 0, 1, 0, 0, 1, 0, 0},
         {0, 0, 0, 0, 1, 1, 1, 0, 1},
         {1, 1, 0, 0, 1, 0, 1, 0, 0},
         {0, 0, 1, 1, 0, 0, 0, 0, 1},
         {0, 0, 1, 0, 0, 0, 0, 0, 0},
         {0, 1, 1, 1, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 1, 0, 1, 0, 0, 0, 0}};

    graphblas::buildmatrix(G_tn, matrixData.begin(),
                           matrixData.size(), graphblas::math::Plus<double>());

    auto identity = graphblas::identity<
        graphblas::Matrix<double, graphblas::DirectedMatrixTag> >(9);

    algorithms::bfs(G_tn, identity, G_tn_res);

    //std::cout << G_tn_res << std::endl;
    return 0;
}
