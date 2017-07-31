#define GB_USE_CUSP

#include <cstdlib>
#include <iostream>

#include <graphblas/graphblas.hpp>
//#include <cusp/print.h>
#include <vector>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE assign

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(assign)

BOOST_AUTO_TEST_CASE(assign_to_empty_matrix)
{
    using namespace graphblas;
    Matrix<int> a(4,4);
    //test on iterators:
    std::vector<int> i = { 1, 2 , 2, 3, 3};
    std::vector<int> j = { 0, 1 , 2, 1, 3};
    std::vector<int> v = { 4213, 234 , 242, 1123, 3342};
    graphblas::IndexType count = 5;
    graphblas::buildmatrix(a, i.begin(), j.begin(), v.begin(), count );
    //output:
    Matrix<int> cm(6,6);
    //4x4
    std::vector<int> ia = {4, 3, 2, 0};
    std::vector<int> ja = {1, 4, 3, 5};
    //call assign (matrix version)
    graphblas::assign(a,ia.begin(), ja.begin(), cm);
    //should only be five values:
    VectorIndexType r(5), c(5);
    std::vector<int> vals(5);
    extracttuples(cm, r.begin(), c.begin(), vals.begin());
    int r_answers[5] ={0,0,2,2,3};
    int c_answers[5] ={4,5,4,3,1};
    int v_answers[5] ={1123,3342,234,242,4213};
    BOOST_CHECK_EQUAL_COLLECTIONS(r.begin(), r.end(), r_answers, r_answers+5);
    BOOST_CHECK_EQUAL_COLLECTIONS(c.begin(), c.end(), c_answers, c_answers+5);
    BOOST_CHECK_EQUAL_COLLECTIONS(vals.begin(), vals.end(), v_answers,
            v_answers+5);
}

BOOST_AUTO_TEST_SUITE_END()
