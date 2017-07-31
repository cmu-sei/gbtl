#include <cstdlib>
#include <iostream>
#include <algorithm>

#include <graphblas/graphblas.hpp>
#include <vector>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE ewise

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(ewise)

template <typename T>
struct multiply_operator{
    typedef T result_type;
    T operator() (T &a, T &b)
    {
        return a * b;
    }
};

template <typename T>
struct plus_operator{
    typedef T result_type;
    T operator() (T &a, T &b)
    {
        return a + b;
    }
};

BOOST_AUTO_TEST_CASE(ewisemult)
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
    graphblas::ewisemult(a,b,c);
    //extracttuples:
    VectorIndexType rows(5), cols(5);
    std::vector<int> vals(5);
    extracttuples(c, rows, cols, vals);

    for(auto it = v.begin(); it<v.end(); ++it) {
        *it = *it * 2;
    }

    BOOST_CHECK_EQUAL_COLLECTIONS(rows.begin(), rows.end(), i.begin(), i.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(cols.begin(), cols.end(), j.begin(), j.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(vals.begin(), vals.end(), v.begin(), v.end());

}

BOOST_AUTO_TEST_CASE(ewiseadd)
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

    //c<-a+b
    graphblas::ewiseadd(a,b,c);
    //extracttuples:
    VectorIndexType rows(5), cols(5);
    std::vector<int> vals(5);
    extracttuples(c, rows, cols, vals);

    for(auto it = v.begin(); it<v.end(); ++it) {
        *it = *it + 2;
    }

    BOOST_CHECK_EQUAL_COLLECTIONS(rows.begin(), rows.end(), i.begin(), i.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(cols.begin(), cols.end(), j.begin(), j.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(vals.begin(), vals.end(), v.begin(), v.end());
}

BOOST_AUTO_TEST_SUITE_END()
