#include <cstdlib>
#include <iostream>
#include <algorithm>

#include <graphblas/graphblas.hpp>
#include <vector>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE apply

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(apply)

//this test SHOULD work once the linking issues are figured out
BOOST_AUTO_TEST_CASE(apply)
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


    //TODO: need to actively toggle std::negate or thrust::negate
    thrust::negate<int> op;
    graphblas::apply(stuff, transposed_stuff, op);

    //extracttuples:
    VectorIndexType r(5), c(5);
    std::vector<int> vals(5);
    extracttuples(transposed_stuff, r.begin(), c.begin(), vals.begin());

    std::vector<int> v_true;
    for(graphblas::IndexType idx=0; idx<5;idx++){
        v_true.push_back(-1*v[idx]);
    }
    BOOST_CHECK_EQUAL_COLLECTIONS(r.begin(), r.end(), i.begin(), i.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(c.begin(), c.end(), j.begin(), j.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(vals.begin(), vals.end(), v_true.begin(),
            v_true.end());
}

BOOST_AUTO_TEST_SUITE_END()
