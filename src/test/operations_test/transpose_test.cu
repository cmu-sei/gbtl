#define GB_USE_CUSP

#include <cstdlib>
#include <iostream>
#include <algorithm>

#include <graphblas/graphblas.hpp>
//#include <cusp/print.h>
#include <vector>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE transpose

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(transpose)

//this test SHOULD work once the linking issues are figured out
BOOST_AUTO_TEST_CASE(transpose)
{
    using namespace graphblas;
    Matrix<int> stuff(4,4);
    Matrix<int> transposed_stuff(4,4);
    //test on iterators:
    std::vector<IndexType> i = { 1, 2 , 2, 3, 3};
    std::vector<IndexType> j = { 0, 1 , 2, 1, 3};
    std::vector<int> v = { 4213, 234, 242, 1123, 3342};
    graphblas::IndexType count = 5;
    graphblas::buildmatrix(stuff, i.begin(), j.begin(), v.begin(), count );
    graphblas::transpose(stuff, transposed_stuff);

    //extracttuples:
    VectorIndexType r(5), c(5);
    std::vector<int> vals(5);
    //&c or just c?
    extracttuples(transposed_stuff, r, c, vals);

    std::sort(j.begin(), j.end());
    std::vector<IndexType> i_t = { 1, 2 , 3, 2, 3};
    std::vector<int> v_t = { 4213, 234, 1123, 242,3342};

    BOOST_CHECK_EQUAL_COLLECTIONS(r.begin(), r.end(), j.begin(), j.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(c.begin(), c.end(), i_t.begin(), i_t.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(vals.begin(), vals.end(), v_t.begin(), v_t.end());
}

BOOST_AUTO_TEST_SUITE_END()
