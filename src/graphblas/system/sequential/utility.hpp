/*
 * Copyright (c) 2015 Carnegie Mellon University and The Trustees of Indiana
 * University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY AND THE TRUSTEES OF INDIANA UNIVERSITY EXPRESSLY DISCLAIM
 * TO THE FULLEST EXTENT PERMITTED BY LAW ALL EXPRESS, IMPLIED, AND STATUTORY
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */

#pragma once

#ifndef GB_SEQUENTIAL_UTILITY_HPP
#define GB_SEQUENTIAL_UTILITY_HPP

#include <limits>

namespace GraphBLAS
{
    namespace backend
    {
        template <typename ScalarT>
        void pretty_print(std::ostream &ostr, std::vector<std::tuple<IndexType, ScalarT> > const &vec)
        {
            IndexType index;
            ScalarT   value;

            IndexType curr_idx = 0;

            ostr << "[";

            auto vec_it = vec.begin();
            while (vec_it != vec.end())
            {
                std::tie(index, value) = *vec_it;
                while (curr_idx < index)
                {
                    ostr << ((curr_idx == 0) ? " " : ",  " );
                    ++curr_idx;
                }

                // Put out the value at index
                ostr << ((curr_idx == 0) ? "" : ", " );
                ostr << value;

                ++vec_it;
                ++curr_idx;
            }

            // TODO: Go to N.
            ostr << "]";
        }

        /**
         *  @brief Output the matrix in array form.  Mainly for debugging
         *         small matrices.
         *
         *  @param[in] ostr  The output stream to send the contents
         *  @param[in] mat   The matrix to output
         *
         */
        // @deprecated - use stream print-info/stream inserter
        template <typename MatrixT >
        void pretty_print_matrix(std::ostream &ostr, MatrixT const &mat)
        {
            typedef typename MatrixT::ScalarType ScalarT;
            typedef std::vector<std::tuple<GraphBLAS::IndexType, ScalarT> > RowType;

            IndexType num_rows = mat.nrows();
            IndexType num_cols = mat.ncols();

            for (IndexType row_idx = 0; row_idx < num_rows; ++row_idx)
            {
                ostr << ((row_idx == 0) ? "[[" : " [");

                RowType const &row(mat.getRow(row_idx));
                IndexType curr_idx = 0;

                if (row.empty())
                {
                    while (curr_idx < num_cols)
                    {
                        ostr << ((curr_idx == 0) ? " " : ",  " );
                        ++curr_idx;
                    }
                }
                else
                {
                    // Now walk the columns.  A sparse iter would be handy here...
                    IndexType col_idx;
                    ScalarT cell_val;

                    auto row_it = row.begin();
                    while (row_it != row.end())
                    {
                        std::tie(col_idx, cell_val) = *row_it;
                        while (curr_idx < col_idx)
                        {
                            ostr << ((curr_idx == 0) ? " " : ",  " );
                            ++curr_idx;
                        }

                        if (curr_idx != 0)
                            ostr << ", ";
                        ostr << cell_val;

                        ++row_it;
                        ++curr_idx;
                    }

                    // Fill in the rest to the end
                    while (curr_idx < num_cols)
                    {
                        ostr << ",  ";
                        ++curr_idx;
                    }
                }
                ostr << ((row_idx == num_rows - 1 ) ? "]]\n" : "]\n");
            }
        }

        template<typename ScalarT>
        std::ostream &operator<<(std::ostream &os, std::vector<std::tuple<IndexType, ScalarT> > &vec)
        {
            pretty_print(os, vec);
            return os;
        }

    } //backend
}

#endif // GB_SEQUENTIAL_UTILITY_HPP
