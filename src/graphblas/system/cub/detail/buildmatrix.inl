#pragma once

#include <thrust/device_ptr.h>
#include "../header.hpp"
#include "../config.hpp"
#include "../utility.hpp"

namespace graphblas{
namespace backend{
    //raiterator: host iterator
    template<typename MatrixT,
             typename RAIteratorI,
             typename RAIteratorJ,
             typename RAIteratorV,
             typename AccumT>
    inline void buildmatrix(MatrixT     &m,
                            RAIteratorI  i,
                            RAIteratorJ  j,
                            RAIteratorV  v,
                            IndexType  n,
                            AccumT accum)
    {
        m.resize(m.num_rows, m.num_cols, n);

        utility::htod_matrix_member_async_memcpy(
                thrust::raw_pointer_cast(m.row_indices),
                thrust::raw_pointer_cast(m.column_indices),
                thrust::raw_pointer_cast(m.values),
                &(*i), &(*j), &(*v),
                n);
        //sorting:
        m.sort_by_row_and_column();
    }

    template<typename MatrixT,
             typename AccumT>
    inline void buildmatrix(MatrixT              &m,
                            IndexArrayType const &i,
                            IndexArrayType const &j,
                            std::vector<typename MatrixT::ScalarType> const &v,
                            AccumT accum)
    {
        /// @todo Add dimension checks
        backend::buildmatrix(m, i.begin(), j.begin(), v.begin(), i.size(), accum);
    }
} //backend
} // graphblas
