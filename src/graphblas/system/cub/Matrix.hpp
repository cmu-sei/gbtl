#pragma once

#include <cstddef>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/iterator/iterator_facade.h>
#include <cub/cub.cuh>
#include <graphblas/header.hpp>
#include <graphblas/config.hpp>
#include <graphblas/utility.hpp>
#include <utility>

namespace graphblas
{
namespace backend
{
    //TODO: support ``constant'' tag as matrix using iterators in future.
    template<typename ScalarT, typename... TagsT>
    class Matrix
    {
    private:
        IndexType * row;
        IndexType * col;
        ScalarT * val;
        IndexType allocated_size;

        inline IndexType calc_alloc_size(IndexType requested)
        {
            return requested < 1024 ?
                    1024
                    :
                    (IndexType) requested*1.1;
        }
    public:

        typedef IndexType index_type;

        thrust::device_ptr<IndexType> row_indices;
        thrust::device_ptr<IndexType> column_indices;
        thrust::device_ptr<ScalarT> values;
        IndexType num_rows;
        IndexType num_cols;
        IndexType num_entries;

        Matrix() = delete;

        Matrix(const IndexType rows, const IndexType columns) :
            num_rows(rows),
            num_cols(columns),
            num_entries(0),
            allocated_size(0)
        {}

        //num vals:
        template<typename RowIndexType,
                 typename ColumnIndexType,
                 typename ValueType >
        Matrix(const RowIndexType rows,
               const ColumnIndexType columns,
               const ValueType num_vals) :
            num_rows(rows),
            num_cols(columns),
            num_entries(num_vals)
        {
            this->allocated_size = this->calc_alloc_size(num_entries);
            cudaMalloc((void**)&row, sizeof(IndexType)*allocated_size);
            cudaMalloc((void**)&col, sizeof(IndexType)*allocated_size);
            cudaMalloc((void**)&val, sizeof(ScalarT)*allocated_size);

            //continuous chunk:
            //cudaMalloc((void**)&row,
            //        2*sizeof(IndexType)*num_entries
            //        + sizeof(ScalarT)*num_entries);
            //col = (IndexType*)((void*)row)+sizeof(IndexType)*num_entries;
            //val = (ScalarT*)((void*)col)+sizeof(IndexType)*num_entries;

            //or associative?

            row_indices = thrust::device_pointer_cast(row);
            column_indices = thrust::device_pointer_cast(col);
            values = thrust::device_pointer_cast(val);
        }

        //copy
        template<typename MatrixT>
        Matrix(const MatrixT &matrix) :
            Matrix(matrix.num_rows,
                   matrix.num_cols,
                   matrix.num_entries)
        {
            utility::dtod_matrix_member_async_memcpy(
                    row, col, val,
                    matrix.row, matrix.col, matrix.val,
                    matrix.num_entries);
        }

        ~Matrix()
        {
            cudaFree(row);
            cudaFree(col);
            cudaFree(val);
        }

        //similar to cusp interface
        inline void resize(
                IndexType num_rows,
                IndexType num_cols,
                IndexType num_entries,
                bool keep_data=false)
        {
            this->num_rows = num_rows;
            this->num_cols = num_cols;

            if (num_entries <= this->allocated_size) {
                this->num_entries = num_entries;
                return;
            }

            //reallocate:
            if (!keep_data) {
                if (row) cudaFree(row);
                if (col) cudaFree(col);
                if (val) cudaFree(val);

                this->allocated_size = this->calc_alloc_size(num_entries);

                cudaMalloc((void**)&row, sizeof(IndexType)*allocated_size);
                cudaMalloc((void**)&col, sizeof(IndexType)*allocated_size);
                cudaMalloc((void**)&val, sizeof(ScalarT)*allocated_size);

                this->num_entries = num_entries;

                return;
            } else {
                IndexType *t1 = row, *t2 = col;
                ScalarT *t3 = val;

                this->allocated_size = this->calc_alloc_size(num_entries);

                cudaMalloc((void**)&row, sizeof(IndexType)*allocated_size);
                cudaMalloc((void**)&col, sizeof(IndexType)*allocated_size);
                cudaMalloc((void**)&val, sizeof(ScalarT)*allocated_size);

                if(t1 && t2 && t3) {
                    utility::dtod_matrix_member_async_memcpy(
                            row, col, val,
                            t1, t2, t3,
                            this->num_entries);
                }

                if (t1) cudaFree(t1);
                if (t2) cudaFree(t2);
                if (t3) cudaFree(t3);

                this->num_entries = num_entries;
                return;
            }
        }

        inline void sort_by_row_and_column()
        {
            //sort by key
            thrust::sort_by_key(column_indices, column_indices+num_entries, row_indices);
            thrust::stable_sort_by_key(row_indices, row_indices+num_entries, column_indices);
        }

#if 0
        inline void sort_by_row() {
            //sort by row

            IndexType *d_key_alt_buf;
            IndexType *d_value_alt_buf;
            IndexType *d_value_buf;

            cudaMalloc(&d_key_alt_buf, num_entries*sizeof(IndexType));
            cudaMalloc(&d_value_alt_buf, num_entries*sizeof(IndexType));
            cudaMalloc(&d_value_buf, num_entries*sizeof(IndexType));
            //fill d value:
            IndexType threads, blocks;

            utility::get_threads_blocks(num_entries, &blocks, &threads, 1024);
            utility::sequence<<<blocks, threads>>>(d_value_buf, num_entries);

            cub::DoubleBuffer<IndexType> d_keys(row, d_key_alt_buf);
            cub::DoubleBuffer<IndexType> d_values(d_value_buf, d_value_alt_buf);

            // Determine temporary device storage requirements
            void     *d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;

            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_entries);
            // Allocate temporary storage
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cudaDeviceSynchronize();
            // Run sorting operation
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_entries);

            cudaDeviceSynchronize();

            //TODO: gather
            //
            //should just use thrust::gather
            utility::gather<<<blocks, threads>>>(
                    d_value_buf,
                    col,
                    val,
                    d_key_alt_buf,
                    reinterpret_cast<ScalarT*>(d_value_alt_buf),
                    num_entries);

            std::swap(col, d_key_alt_buf);
            //may not be the best practice, but since value_alt_buf is 64bit
            //data, we **can** (not that we should) take spillover if casted to 32bit
            ScalarT * tmp = val;
            val = reinterpret_cast<ScalarT*>(d_value_alt_buf);
            d_value_alt_buf = reinterpret_cast<IndexType*>(tmp);


            cudaDeviceSynchronize();

            cudaFree(d_key_alt_buf);
            cudaFree((IndexType*)d_value_alt_buf);

        }
#endif
    };
} // backend
} // graphblas
