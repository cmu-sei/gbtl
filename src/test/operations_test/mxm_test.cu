#include <cstdlib>
#include <iostream>
#include <algorithm>

#include <graphblas/graphblas.hpp>
#include <vector>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE ewise

#include <boost/test/included/unit_test.hpp>
#include <cusp/print.h>

BOOST_AUTO_TEST_SUITE(mxm)
//mxm == gemm

/**
 * >>> a
 array([[   0,    0,    0,    0],
         [4213,    0,    0,    0],
         [   0,  234,  242,    0],
         [   0, 1123,    0, 3342]])
 >>> b
     array([[0, 0, 0, 0],
             [2, 0, 0, 0],
             [0, 2, 2, 0],
             [0, 2, 0, 2]])
 >>> a*b
 matrix([[   0,    0,    0,    0],
         [   0,    0,    0,    0],
         [ 468,  484,  484,    0],
         [2246, 6684,    0, 6684]])

     */

BOOST_AUTO_TEST_CASE(mxm)
{
    using namespace graphblas;
    Matrix<int> a(4,4);
    Matrix<int> b(4,4);
    Matrix<int> c(4,4);
    //test on iterators:
    std::vector<IndexType> i = { 1, 2 , 2, 3, 3};
    std::vector<IndexType> j = { 0, 1 , 2, 1, 3};
    std::vector<int> v = { 4213, 234, 242, 1123, 3342};
    std::vector<int> v_b = { 2, 2, 2, 2, 2};
    graphblas::IndexType count = 5;
    graphblas::buildmatrix(a, i.begin(), j.begin(), v.begin(), count );
    graphblas::buildmatrix(b, i.begin(), j.begin(), v_b.begin(), count );

    //c<-a*b
    graphblas::mxm(a,b,c);
    //extracttuples:
    VectorIndexType rows(6), cols(6);
    std::vector<int> vals(6);
    extracttuples(c, rows, cols, vals);

    std::vector<IndexType> i_res = {2,2,2,3,3,3};
    std::vector<IndexType> j_res = { 0, 1 , 2,0, 1, 3};
    std::vector<int> v_res = { 468,484,484,2246,6684,6684};


    BOOST_CHECK_EQUAL_COLLECTIONS(rows.begin(), rows.end(), i_res.begin(), i_res.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(cols.begin(), cols.end(), j_res.begin(), j_res.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(vals.begin(), vals.end(), v_res.begin(), v_res.end());

}

BOOST_AUTO_TEST_SUITE_END()
