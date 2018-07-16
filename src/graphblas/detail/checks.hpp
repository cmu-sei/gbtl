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

#ifndef GB_CHECKS_HPP
#define GB_CHECKS_HPP

#include <graphblas/types.hpp>

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

    // ***********************************************************************
    // ***********************************************************************

    // This is the basis for everything
    inline Info check_val_equals(IndexType val1, IndexType val2,
                                 const std::string &msg1,
                                 const std::string &msg2)
    {
        if (val1 != val2)
        {
            //throw DimensionException(make_message(msg1, msg2, val1, val2));
            return DIMENSION_MISMATCH;
        }
        return SUCCESS;
    };

    inline Info check_index_val_lt(IndexType val1, IndexType val2,
                                   const std::string &msg1,
                                   const std::string &msg2)
    {
        if (val1 >= val2)
        {
            std::ostringstream ss;
            ss << msg1 << ", " << msg2 << ", (" << val1 << " >= " << val2 << ")";
            //throw InvalidIndexException(ss.str());
            return DIMENSION_MISMATCH;
        }
        return SUCCESS;
    };

    inline Info check_index_val_leq(IndexType val1, IndexType val2,
                                    const std::string &msg1,
                                    const std::string &msg2)
    {
        if (val1 > val2)
        {
            std::ostringstream ss;
            ss << msg1 << ", " << msg2 << ", (" << val1 << " > " << val2 << ")";
            //throw DimensionException(ss.str());
            return DIMENSION_MISMATCH;
        }
        return SUCCESS;
    };

    // ================================================

    template <typename M1, typename M2>
    Info check_nrows_nrows(M1 m1, M2 m2, const std::string &msg)
    {
        return check_val_equals(m1.nrows(), m2.nrows(), "nrows != nrows", msg);
    }

    template <typename M1>
    Info check_nrows_nrows(M1 m1, NoMask mask, const std::string &msg)
    {
        // No op
        return SUCCESS;
    }

    // ================================================

    template <typename M1, typename M2>
    Info check_ncols_ncols(M1 m1, M2 m2, const std::string &msg)
    {
        return check_val_equals(m1.ncols(), m2.ncols(), "ncols != ncols", msg);
    }

    template <typename M1>
    Info check_ncols_ncols(M1 m1, NoMask mask, const std::string &msg)
    {
        // No op
        return SUCCESS;
    }

    // ================================================

    template <typename M1, typename M2>
    Info check_ncols_nrows(M1 m1, M2 m2, const std::string &msg)
    {
        return check_val_equals(m1.ncols(), m2.nrows(), "ncols != nrows", msg);
    };

    template <typename M>
    Info check_ncols_nrows(M m, NoMask mask, const std::string &msg)
    {
        // No op
        return SUCCESS;
    };

    template <typename M>
    Info check_ncols_nrows(NoMask mask, M m, const std::string &msg)
    {
        // No op
        return SUCCESS;
    };

    // =========================================================================

    template <typename M, typename S>
    Info check_nrows_nindices(const M &m, const S &s, const std::string &msg)
    {
        return check_val_equals(m.nrows(), s.size(), "nrows != nindices", msg);
    };

    template <typename M>
    Info check_nrows_nindices(const M &m, const AllIndices &s, const std::string &msg)
    {
        return SUCCESS;
    };

    template <typename S>
    Info check_nrows_nindices(const NoMask &m, const S &s, const std::string &msg)
    {
        return SUCCESS;
    };

    // =========================================================================

    template <typename M, typename S>
    Info check_ncols_nindices(const M &m, const S &s, const std::string &msg)
    {
        return check_val_equals(m.ncols(), s.size(), "ncols != nindices", msg);
    };

    template <typename M>
    Info check_ncols_nindices(const M &m, const AllIndices &s, const std::string &msg)
    {
        return SUCCESS;
    };

    template <typename S>
    Info check_ncols_nindices(const NoMask &m, const S &s, const std::string &msg)
    {
        return SUCCESS;
    };

    // =========================================================================

    template <typename V1, typename V2>
    Info check_size_size(const V1 &v1, const V2 &v2, const std::string &msg)
    {
        return check_val_equals(v1.size(), v2.size(), "size != size", msg);
    };

    template <typename V>
    Info check_size_size(const V &v, const NoMask &mask, const std::string &msg)
    {
        // No op
        return SUCCESS;
    };

    template <typename V>
    Info check_size_size(const NoMask &mask, const V &v, const std::string &msg)
    {
        // No op
        return SUCCESS;
    };

    // =========================================================================

    template <typename V, typename M>
    Info check_size_nrows(const V &v, const M &A, const std::string &msg)
    {
        return check_val_equals(v.size(), A.nrows(), "size != nrows", msg);
    };

    // Note: NoMask support for assign(col)
    template <typename M>
    Info check_size_nrows(const NoMask &mask, const M &A, const std::string &msg)
    {
        // No op
        return SUCCESS;
    };

    // =========================================================================

    template <typename V, typename M>
    Info check_size_ncols(const V &v, const M &A, const std::string &msg)
    {
        return check_val_equals(v.size(), A.ncols(), "size != nrows", msg);
    };

    // Note: NoMask support for assign(row)
    template <typename M>
    Info check_size_ncols(const NoMask &mask, const M &A, const std::string &msg)
    {
        // No op
        return SUCCESS;
    };

    // Note: no NoMask support

    // ================================================

    template <typename V, typename S>
    Info check_size_nindices(const V &v, const S &s, const std::string &msg)
    {
        return check_val_equals(v.size(), s.size(), "size != nindices", msg);
    };

    template <typename V>
    Info check_size_nindices(const V &v,
                             const GraphBLAS::AllIndices &s,
                             const std::string &msg)
    {
        return SUCCESS;
    };

    // ================================================

    template <typename M>
    Info check_index_within_ncols(IndexType val, const M &m,
                                  const std::string &msg)
    {
        return check_index_val_lt(val, m.ncols(), "value < ncols", msg);
    }

    // ================================================

    template <typename M>
    Info check_index_within_nrows(IndexType val, const M &m,
                                  const std::string &msg)
    {
        return check_index_val_lt(val, m.nrows(), "value < nrows", msg);
    }

    // ================================================

    template <typename S, typename V>
    Info check_nindices_within_size(const S &seq, const V &vec,
                                    const std::string &msg)
    {
        return check_index_val_leq(seq.size(), vec.size(), "seq.size < vec.size", msg);
    }

    template <typename V>
    Info check_nindices_within_size(const GraphBLAS::AllIndices &seq,
                                    const V &vec,
                                    const std::string &msg)
    {
        return SUCCESS;
    }

    // ================================================

    template <typename S, typename M>
    Info check_nindices_within_ncols(const S &seq, const M &mat,
                                     const std::string &msg)
    {
        return check_index_val_leq(seq.size(), mat.ncols(), "seq.size < mat.ncols", msg);
    }

    template <typename M>
    Info check_nindices_within_ncols(const GraphBLAS::AllIndices &seq,
                                     const M &mat,
                                     const std::string &msg)
    {
        return SUCCESS;
    }

    // ================================================

    template <typename S, typename M>
    Info check_nindices_within_nrows(const S &seq,
                                     const M &mat,
                                     const std::string &msg)
    {
        return check_index_val_leq(seq.size(), mat.nrows(), "seq.size < mat.nrows", msg);
    }

    template <typename M>
    Info check_nindices_within_nrows(const GraphBLAS::AllIndices &seq,
                                     const M &mat,
                                     const std::string &msg)
    {
        return SUCCESS;
    }

} // end namespace GraphBLAS

#endif //GB_CHECKS_HPP
