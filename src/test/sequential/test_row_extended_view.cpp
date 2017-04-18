/*
 * Copyright notice.
 */

#include <iostream>

#include <graphblas/graphblas.hpp>

using namespace graphblas;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE row_extended_view_suite

#include <boost/test/included/unit_test.hpp>


BOOST_AUTO_TEST_SUITE(row_extended_view_suite)

//****************************************************************************
// LIL basic constructor
BOOST_AUTO_TEST_CASE(rev_test_construction_vector)
{
    // Build a matrix that is a column vector
    std::vector<std::vector<double> > col_vector = {{6}, {1}, {-1}, {4}};
    graphblas::Matrix<double,
                      graphblas::DirectedMatrixTag > column_matrix(col_vector);

    graphblas::IndexType num_cols = 3;
    graphblas::RowExtendedView<graphblas::Matrix<
        double, graphblas::DirectedMatrixTag > >
        m1(0,
           column_matrix,
           num_cols);

    graphblas::IndexType M, N;
    m1.get_shape(M, N);
    BOOST_CHECK_EQUAL(col_vector.size(), M);
    BOOST_CHECK_EQUAL(num_cols, N);
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1.extractElement(i, j), col_vector[i][0]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(rev_test_construction_column_view)
{
/*
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                             {7, 0, 0, 0},
                                             {0, 0, 9, 4},
                                             {2, 5, 0, 3},
                                             {2, 0, 0, 1},
                                             {0, 0, 0, 0},
                                             {0, 1, 0, 2}};

    // Create a column view of a matrix (4th col).
    graphblas::IndexType orig_col_index = 3;
    graphblas::Matrix<double, graphblas::DirectedMatrixTag > matrix(mat);
    graphblas::ColumnView<graphblas::Matrix<
        double, graphblas::DirectedMatrixTag > >
        col_matrix(orig_col_index, matrix);

    // Replicate that column 3 times
    graphblas::IndexType num_cols = 3;
    graphblas::RowExtendedView<
        graphblas::ColumnView<
            graphblas::Matrix<double, graphblas::DirectedMatrixTag > > >
        rev_mat(0, col_matrix, num_cols);


    graphblas::IndexType M, N;
    //matrix.get_shape(M, N);
    //std::cerr << "matrix: " << M << "x" << N << std::endl;
    //col_matrix.get_shape(M, N);
    //std::cerr << "colmat: " << M << "x" << N << std::endl;

    rev_mat.get_shape(M, N);
    //std::cerr << "revmat: " << M << "x" << N << std::endl;
    BOOST_CHECK_EQUAL(M, mat.size());
    BOOST_CHECK_EQUAL(N, num_cols);
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(rev_mat.extractElement(i, j),
                              mat[i][orig_col_index]);
        }
    }
*/
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(rev_test_construction_matrix_col_index)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                             {7, 0, 0, 0},
                                             {0, 0, 9, 4},
                                             {2, 5, 0, 3},
                                             {2, 0, 0, 1},
                                             {0, 0, 0, 0},
                                             {0, 1, 0, 2}};

    // Create a column view of a matrix (4th col).
    graphblas::IndexType orig_col_index = 3;
    graphblas::Matrix<double, graphblas::DirectedMatrixTag > matrix(mat);

    // Replicate the 4th column 3 times
    graphblas::IndexType num_cols = 3;
    graphblas::RowExtendedView<
        graphblas::Matrix<double, graphblas::DirectedMatrixTag > >
        rev_mat(orig_col_index,
                matrix,
                num_cols);

    graphblas::IndexType M, N;
    rev_mat.get_shape(M, N);

    BOOST_CHECK_EQUAL(M, mat.size());
    BOOST_CHECK_EQUAL(N, num_cols);
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(rev_mat.extractElement(i, j),
                              mat[i][orig_col_index]);
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(rev_test_modify_element)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                             {7, 0, 0, 0},
                                             {0, 0, 9, 4},
                                             {2, 5, 0, 3},
                                             {2, 0, 0, 1},
                                             {0, 0, 0, 0},
                                             {0, 1, 0, 2}};

    // Create a column view of a matrix (4th col).
    graphblas::IndexType orig_col_index = 3;
    graphblas::Matrix<double, graphblas::DirectedMatrixTag > matrix(mat);

    // Replicate the 4th column 3 times
    graphblas::IndexType num_cols = 3;
    graphblas::RowExtendedView<
        graphblas::Matrix<double, graphblas::DirectedMatrixTag > >
        rev_mat(orig_col_index,
                matrix,
                num_cols);

    graphblas::IndexType M, N;
    rev_mat.get_shape(M, N);

    BOOST_CHECK_EQUAL(M, mat.size());
    BOOST_CHECK_EQUAL(N, num_cols);

    // modify the second value
    double new_value = 55.0;
    rev_mat.setElement(1, 0, new_value);
    mat[1][orig_col_index] = new_value;

    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(rev_mat.extractElement(i, j),
                              mat[i][orig_col_index]);
        }
    }

    matrix.get_shape(M, N);
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(matrix.extractElement(i, j), mat[i][j]);
        }
    }
}



BOOST_AUTO_TEST_SUITE_END()
