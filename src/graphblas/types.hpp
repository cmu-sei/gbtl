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

/**
 * @brief Some basic typedefs.
 */

#ifndef GB_TYPES_HPP
#define GB_TYPES_HPP

#include <cstdint>
#include <exception>
#include <vector>
#include <iostream>

namespace GraphBLAS
{
    typedef uint64_t IndexType;
    typedef std::vector<IndexType> IndexArrayType;

    //**************************************************************************
    // Error codes to replace some of the exceptions
    enum Info {
        SUCCESS,
        INVALID_VALUE,
        INVALID_INDEX,
        DOMAIN_MISMATCH,
        DIMENSION_MISMATCH,
        OUTPUT_NOT_EMPTY,
        NO_VALUE,
        INDEX_OUT_OF_BOUNDS,
        OUT_OF_MEMORY
    };


    //**************************************************************************
    struct NoAccumulate
    {
        // It doesn't really matter what the type is, it never gets executed.
        typedef bool result_type;
        inline bool operator()(bool lhs, bool rhs) { return true; }
    };


    //**************************************************************************

    // This is the "Matrix" class for this example
    struct matrix_tag {};

    // This is the "Vector" class for this example
    struct vector_tag {};


} // namespace GraphBLAS

#define CHECK_STATUS(x)     do { Info status(x); if (status != SUCCESS) return status; } while(0)

namespace std
{

// @TODO; It seems that unit tests can't find this!
    inline std::ostream &
    operator<<(std::ostream &os, const std::vector<long unsigned int> vec)
    {
        bool first = true;
        for (auto it = vec.begin(); it != vec.end(); ++it)
        {
            os << (first ? "" : ",") << *it;
            first = false;
        }
        return os;
    }

} // namespace std

#endif // GB_TYPES_HPP
