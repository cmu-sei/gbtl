#pragma once

#include <thrust/device_ptr.h>
#include "../header.hpp"
#include "../config.hpp"

namespace graphblas{
namespace backend{
    //raiterator: host iterator
    template<typename MatrixT,
             typename RAIteratorI,
             typename RAIteratorJ,
             typename RAIteratorV>
    inline void buildmatrix(MatrixT     &m,
                            RAIteratorI  i,
                            RAIteratorJ  j,
                            RAIteratorV  v,
                            IndexType  n)
    {
        m.resize(m.num_rows, m.num_cols, n);

        //TODO: decide what to do with duplicates
        //merge by key, maybe?
        utility::htod_matrix_member_async_memcpy(
                thrust::raw_pointer_cast(m.row_indices),
                thrust::raw_pointer_cast(m.column_indices),
                thrust::raw_pointer_cast(m.values),
                &(*i), &(*j), &(*v),
                n);
        //sorting:
        temp.sort_by_row_and_column();
    }
} //backend
} // graphblas
