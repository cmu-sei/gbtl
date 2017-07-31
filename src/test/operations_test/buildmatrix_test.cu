#define GB_USE_CUSP

#include <cstdlib>
#include <iostream>
#include <algorithm>

#include <graphblas/graphblas.hpp>
//#include <cusp/print.h>
#include <vector>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE buildmatrix

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(buildmatrix)

//this test SHOULD work once the linking issues are figured out
BOOST_AUTO_TEST_CASE(build_matrix_using_iterators)
{
    using namespace graphblas;
    Matrix<int> stuff(4,4);
    //test on iterators:
    std::vector<IndexType> i = { 1, 2 , 2, 3, 3};
    std::vector<IndexType> j = { 0, 1 , 2, 1, 3};
    std::vector<int> v = { 4213, 234 , 242, 1123, 3342};
    graphblas::IndexType count = 5;


    //test:

    //typedef math::Assign<int> AccumT;

    graphblas::buildmatrix(stuff, i.begin(), j.begin(), v.begin(), count);
            //AccumT() );
    //extracttuples:
    VectorIndexType r(5), c(5);
    std::vector<int> vals(5);
    //&c or just c?
    extracttuples(stuff, r.begin(), c.begin(), vals.begin());

    BOOST_CHECK_EQUAL_COLLECTIONS(r.begin(), r.end(), i.begin(), i.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(c.begin(), c.end(), j.begin(), j.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(vals.begin(), vals.end(), v.begin(), v.end());
}

BOOST_AUTO_TEST_SUITE_END()
