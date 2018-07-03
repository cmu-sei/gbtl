/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
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
