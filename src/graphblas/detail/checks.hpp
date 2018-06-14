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

#ifndef GB_CHECKS_HPP
#define GB_CHECKS_HPP

#define GB_INCLUDE_BACKEND_MATRIX 1
#define GB_INCLUDE_BACKEND_VECTOR 1
#include <backend_include.hpp>
#include "logging.h"

namespace GraphBLAS
{


    // Helper method to make a nicely formatted messages
    std::string make_message(const std::string &msg, const std::string &msg2, IndexType dim1, IndexType dim2)
    {
        std::ostringstream ss;
        ss << msg << ", " << msg2 << ", (" << dim1 << " != " << dim2 << ")";
        return ss.str();
    }

    // Helper method to make a nicely formatted messages
    std::string make_message(const std::string &msg, IndexType dim1, IndexType dim2)
    {
        std::ostringstream ss;
        ss << msg << ", (" << dim1 << " != " << dim2 << ")";
        return ss.str();
    }

    // This is the basis for everything
    inline void check_val_equals(IndexType val1, IndexType val2,
                                 const std::string &msg1,
                                 const std::string &msg2)
    {
        if (val1 != val2)
            throw DimensionException(make_message(msg1, msg2, val1, val2));
    };

    inline void check_index_val_lt(IndexType val1, IndexType val2,
                                   const std::string &msg1,
                                   const std::string &msg2)
    {
        if (val1 >= val2)
        {
            std::ostringstream ss;
            ss << msg1 << ", " << msg2 << ", (" << val1 << " >= " << val2 << ")";
            throw InvalidIndexException(ss.str());
        }
    };

    inline void check_index_val_leq(IndexType val1, IndexType val2,
                                    const std::string &msg1,
                                    const std::string &msg2)
    {
        if (val1 > val2)
        {
            std::ostringstream ss;
            ss << msg1 << ", " << msg2 << ", (" << val1 << " > " << val2 << ")";
            throw DimensionException(ss.str());
        }
    };

    // ================================================

    template <typename M1, typename M2>
    void check_nrows_nrows(M1 m1, M2 m2, const std::string &msg)
    {
        check_val_equals(m1.nrows(), m2.nrows(), "nrows != nrows", msg);
    }

    template <typename M1>
    void check_nrows_nrows(M1 m1, NoMask mask, const std::string &msg)
    {
        // No op
    }

    // ================================================

    template <typename M1, typename M2>
    void check_ncols_ncols(M1 m1, M2 m2, const std::string &msg)
    {
        check_val_equals(m1.ncols(), m2.ncols(), "ncols != ncols", msg);
    }

    template <typename M1>
    void check_ncols_ncols(M1 m1, NoMask mask, const std::string &msg)
    {
        // No op
    }

    // ================================================

    template <typename M1, typename M2>
    void check_ncols_nrows(M1 m1, M2 m2, const std::string &msg)
    {
        check_val_equals(m1.ncols(), m2.nrows(), "ncols != nrows", msg);
    };

    template <typename M>
    void check_ncols_nrows(M m, NoMask mask, const std::string &msg)
    {
        // No op
    };

    template <typename M>
    void check_ncols_nrows(NoMask mask, M m, const std::string &msg)
    {
        // No op
    };

    // =========================================================================

    template <typename M, typename S>
    void check_nrows_nindices(const M &m, const S &s, const std::string &msg)
    {
        check_val_equals(m.nrows(), s.size(), "nrows != nindices", msg);
    };

    template <typename M>
    void check_nrows_nindices(const M &m, const AllIndices &s, const std::string &msg)
    {
    };

    template <typename S>
    void check_nrows_nindices(const NoMask &m, const S &s, const std::string &msg)
    {
    };

    // =========================================================================

    template <typename M, typename S>
    void check_ncols_nindices(const M &m, const S &s, const std::string &msg)
    {
        check_val_equals(m.ncols(), s.size(), "ncols != nindices", msg);
    };

    template <typename M>
    void check_ncols_nindices(const M &m, const AllIndices &s, const std::string &msg)
    {
    };

    template <typename S>
    void check_ncols_nindices(const NoMask &m, const S &s, const std::string &msg)
    {
    };

    // =========================================================================

    template <typename V1, typename V2>
    void check_size_size(const V1 &v1, const V2 &v2, const std::string &msg)
    {
        check_val_equals(v1.size(), v2.size(), "size != size", msg);
    };

    template <typename V>
    void check_size_size(const V &v, const NoMask &mask, const std::string &msg)
    {
        // No op
    };

    template <typename V>
    void check_size_size(const NoMask &mask, const V &v, const std::string &msg)
    {
        // No op
    };

    // =========================================================================

    template <typename V, typename M>
    void check_size_nrows(const V &v, const M &A, const std::string &msg)
    {
        check_val_equals(v.size(), A.nrows(), "size != nrows", msg);
    };

    // Note: NoMask support for assign(col)
    template <typename M>
    void check_size_nrows(const NoMask &mask, const M &A, const std::string &msg)
    {
        // No op
    };

    // =========================================================================

    template <typename V, typename M>
    void check_size_ncols(const V &v, const M &A, const std::string &msg)
    {
        check_val_equals(v.size(), A.ncols(), "size != nrows", msg);
    };

    // Note: NoMask support for assign(row)
    template <typename M>
    void check_size_ncols(const NoMask &mask, const M &A, const std::string &msg)
    {
        // No op
    };

    // Note: no NoMask support

    // ================================================

    template <typename V, typename S>
    void check_size_nindices(const V &v, const S &s, const std::string &msg)
    {
        check_val_equals(v.size(), s.size(), "size != nindices", msg);
    };

    template <typename V>
    void check_size_nindices(const V &v,
                             const GraphBLAS::AllIndices &s,
                             const std::string &msg)
    {
    };

    // ================================================

    template <typename M>
    void check_index_within_ncols(IndexType val, const M &m,
                                  const std::string &msg)
    {
        check_index_val_lt(val, m.ncols(), "value < ncols", msg);
    }

    // ================================================

    template <typename M>
    void check_index_within_nrows(IndexType val, const M &m,
                                  const std::string &msg)
    {
        check_index_val_lt(val, m.nrows(), "value < nrows", msg);
    }

    // ================================================

    template <typename S, typename V>
    void check_nindices_within_size(const S &seq, const V &vec,
                                    const std::string &msg)
    {
        check_index_val_leq(seq.size(), vec.size(), "seq.size < vec.size", msg);
    }

    template <typename V>
    void check_nindices_within_size(const GraphBLAS::AllIndices &seq,
                                    const V &vec,
                                    const std::string &msg)
    {
    }

    // ================================================

    template <typename S, typename M>
    void check_nindices_within_ncols(const S &seq, const M &mat,
                                     const std::string &msg)
    {
        check_index_val_leq(seq.size(), mat.ncols(), "seq.size < mat.ncols", msg);
    }

    template <typename M>
    void check_nindices_within_ncols(const GraphBLAS::AllIndices &seq,
                                     const M &mat,
                                     const std::string &msg)
    {
    }

    // ================================================

    template <typename S, typename M>
    void check_nindices_within_nrows(const S &seq,
                                     const M &mat,
                                     const std::string &msg)
    {
        check_index_val_leq(seq.size(), mat.nrows(), "seq.size < mat.nrows", msg);
    }

    template <typename M>
    void check_nindices_within_nrows(const GraphBLAS::AllIndices &seq,
                                     const M &mat,
                                     const std::string &msg)
    {
    }

} // end namespace GraphBLAS

#endif //GB_CHECKS_HPP
