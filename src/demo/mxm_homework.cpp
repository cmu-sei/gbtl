#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <vector>

#include <graphblas/graphblas.hpp>

int main(int, char**)
{
    // syntatic sugar
    typedef double ScalarType;
    graphblas::IndexType const NUM_ROWS = 3;
    graphblas::IndexType const NUM_COLS = 3;

    // Note: size of dimensions require at ccnstruction
    graphblas::Matrix<ScalarType, graphblas::DirectedMatrixTag> a(NUM_ROWS,
                                                                  NUM_COLS);
    graphblas::Matrix<ScalarType, graphblas::DirectedMatrixTag> b(NUM_ROWS,
                                                                  NUM_COLS);
    graphblas::Matrix<ScalarType, graphblas::DirectedMatrixTag> c(NUM_ROWS,
                                                                  NUM_COLS);

    // initialize matrices
    graphblas::IndexArrayType i = {0,  1,  2};
    graphblas::IndexArrayType j = {0,  1,  2};
    std::vector<ScalarType>   v = {1., 1., 1.};

    graphblas::buildmatrix(a, i.begin(), j.begin(), v.begin(), i.size());
    graphblas::buildmatrix(b, i.begin(), j.begin(), v.begin(), i.size());

    std::cout << "A = " << std::endl;
    graphblas::pretty_print_matrix(std::cout, a);
    std::cout << "B = " << std::endl;
    graphblas::pretty_print_matrix(std::cout, b);

    // matrix multiply (default parameter values used for some)
    graphblas::mxm(a, b, c);

    std::cout << "A * B = " << std::endl;
    graphblas::pretty_print_matrix(std::cout, c);

    // extract the results: get_nnz() method tells us how big
    graphblas::IndexType nnz = c.get_nnz();
    graphblas::IndexArrayType rows(nnz), cols(nnz);
    std::vector<ScalarType> vals(nnz);

    graphblas::extracttuples(c, rows, cols, vals);

    graphblas::IndexArrayType i_res = {0,  1,  2};
    graphblas::IndexArrayType j_res = {0,  1,  2};
    std::vector<ScalarType>   v_res = {1., 1., 1.};

    bool success = true;
    for (graphblas::IndexType ix = 0; ix < vals.size(); ++ix)
    {
        // Note: no semantics defined for extractTuples regarding the
        // order of returned values, so using an O(N^2) approach
        // without sorting:
        bool found = false;
        for (graphblas::IndexType iy = 0; iy < v_res.size(); ++iy)
        {
            if ((i_res[iy] == rows[ix]) && (j_res[iy] == cols[ix]))
            {
                std::cout << "Found result: result, ans index = "
                          << ix << ", " << iy << ": " << vals[ix]
                          << " ?= " << v_res[iy] << std::endl;
                found = true;
                if (v_res[iy] != vals[ix])
                {
                    success = false;
                }
                break;
            }
        }
        if (!found)
        {
            success = false;
        }
    }

    return (success ? 0 : 1);
}
