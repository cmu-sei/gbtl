#include <iostream>

#include <graphblas/graphblas.hpp>

using namespace graphblas;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE coo_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(coo_suite)

BOOST_AUTO_TEST_CASE(coo_constructor)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                          {7, 0, 0, 0},
                                          {0, 0, 9, 4},
                                          {2, 5, 0, 3},
                                          {2, 0, 0, 1},
                                          {0, 0, 0, 0},
                                          {0, 1, 0, 2}};

    graphblas::COO<double> m1(mat);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m1.get_shape(num_rows, num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1[i][j], mat[i][j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(coo_assignment_empty_location)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                          {7, 0, 0, 0},
                                          {0, 0, 9, 4},
                                          {2, 5, 0, 3},
                                          {2, 0, 0, 1},
                                          {0, 0, 0, 0},
                                          {0, 1, 0, 2}};

    graphblas::COO<double> m1(mat);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m1.get_shape(num_rows, num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);

    mat[0][1] = 8;
    m1[0][1] = 8;
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1[i][j], mat[i][j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(coo_assignment_previous_value)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                          {7, 0, 0, 0},
                                          {0, 0, 9, 4},
                                          {2, 5, 0, 3},
                                          {2, 0, 0, 1},
                                          {0, 0, 0, 0},
                                          {0, 1, 0, 2}};

    graphblas::COO<double> m1(mat);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m1.get_shape(num_rows, num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);

    mat[0][0] = 8;
    m1[0][0] = 8;
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1[i][j], mat[i][j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(coo_remove_value)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                          {7, 0, 0, 0},
                                          {0, 0, 9, 4},
                                          {2, 5, 0, 3},
                                          {2, 0, 0, 1},
                                          {0, 0, 0, 0},
                                          {0, 1, 0, 2}};

    graphblas::COO<double> m1(mat);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m1.get_shape(num_rows, num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);

    mat[0][2] = 0;
    m1[0][2] = 0;
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1[i][j], mat[i][j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(coo_test_copy_assignment)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                             {7, 0, 0, 0},
                                             {0, 0, 9, 4},
                                             {2, 5, 0, 3},
                                             {2, 0, 0, 1},
                                             {0, 0, 0, 0},
                                             {0, 1, 0, 2}};

    graphblas::COO<double> m1(mat);
    graphblas::IndexType M, N;
    m1.get_shape(M, N);
    graphblas::COO<double> m2(M, N);

    m2 = m1;

    BOOST_CHECK_EQUAL(m1, m2);
}

BOOST_AUTO_TEST_CASE(coo_test_get_rows)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                             {7, 0, 0, 0},
                                             {0, 0, 9, 4},
                                             {2, 5, 0, 3},
                                             {2, 0, 0, 1},
                                             {0, 0, 0, 0},
                                             {0, 1, 0, 2}};

    graphblas::COO<double> m1(mat);
    graphblas::IndexType M, N;
    m1.get_shape(M, N);

    graphblas::IndexType row_index = 2;
    auto mat_row2a = m1.get_row(row_index);
    auto mat_row2b = m1[row_index];

    for (graphblas::IndexType j = 0; j < N; ++j)
    {
        BOOST_CHECK_EQUAL(mat_row2a[j], m1.get_value_at(row_index, j));
        BOOST_CHECK_EQUAL(mat_row2b[j], m1.get_value_at(row_index, j));
    }
}

// @todo add this to a TransposeView test suite (if the class survives).
BOOST_AUTO_TEST_CASE(coo_test_transpose_view)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                             {7, 0, 0, 0},
                                             {0, 0, 9, 4},
                                             {2, 5, 0, 3},
                                             {2, 0, 0, 1},
                                             {0, 0, 0, 0},
                                             {0, 1, 0, 2}};

    graphblas::COO<double> m1(mat);
    graphblas::IndexType M, N;
    m1.get_shape(M, N);

    graphblas::TransposeView<graphblas::COO<double> > mT(m1);

    graphblas::IndexType m, n;
    mT.get_shape(m, n);

    BOOST_CHECK_EQUAL(m, N);
    BOOST_CHECK_EQUAL(M, n);

    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1.get_value_at(i, j), mT.get_value_at(j, i));
        }
    }
}

BOOST_AUTO_TEST_CASE(coo_double_assignment)
{
    std::vector<std::vector<double> > mat = {{6, 0, 0, 4},
                                          {7, 0, 0, 0},
                                          {0, 0, 9, 4},
                                          {2, 5, 0, 3},
                                          {2, 0, 0, 1},
                                          {0, 0, 0, 0},
                                          {0, 1, 0, 2}};

    graphblas::COO<double> m1(mat);

    graphblas::IndexType M = mat.size();
    graphblas::IndexType N = mat[0].size();
    graphblas::IndexType num_rows, num_cols;
    m1.get_shape(num_rows, num_cols);
    BOOST_CHECK_EQUAL(num_rows, M);
    BOOST_CHECK_EQUAL(num_cols, N);

    mat[1][1] = 1;
    mat[2][2] = 0;
    m1[1][1] = 1;
    m1[2][2] = 0;
    for (graphblas::IndexType i = 0; i < M; i++)
    {
        for (graphblas::IndexType j = 0; j < N; j++)
        {
            BOOST_CHECK_EQUAL(m1[i][j], mat[i][j]);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
