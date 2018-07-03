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
// matrix tags should be strictly internal
#include "matrix_tags.hpp"

#define GB_INCLUDE_BACKEND_MATRIX 1
#define GB_INCLUDE_BACKEND_VECTOR 1
#include <backend_include.hpp>

//#define __GB_SYSTEM_MATRIX_HEADER <graphblas/system/__GB_SYSTEM_ROOT/Matrix.hpp>
//#include __GB_SYSTEM_MATRIX_HEADER
//#undef __GB_SYSTEM_MATRIX_HEADER
//
//#define __GB_SYSTEM_VECTOR_HEADER <graphblas/system/__GB_SYSTEM_ROOT/Vector.hpp>
//#include __GB_SYSTEM_VECTOR_HEADER
//#undef __GB_SYSTEM_VECTOR_HEADER

//this file contains the variadic template parameters unpacking utility.

//****************************************************************************
//****************************************************************************

namespace GraphBLAS
{
    namespace detail
    {

        // Substitute template to decide if a tag goes into a given slot
        template<typename TagCategory, typename Tag>
        struct substitute {
            using type = TagCategory;
        };


        template<>
        struct substitute<detail::SparsenessCategoryTag, DenseTag> {
            using type = DenseTag;
        };

        template<>
        struct substitute<detail::SparsenessCategoryTag, SparseTag> {
            using type = SparseTag;
        };

        template<>
        struct substitute<detail::DirectednessCategoryTag, UndirectedMatrixTag> {
            using type = UndirectedMatrixTag;
        };

        template<>
        struct substitute<detail::DirectednessCategoryTag, DirectedMatrixTag> {
            using type = DirectedMatrixTag;
        };

        template<>
        struct substitute<detail::DirectednessCategoryTag, detail::NullTag> {
            //default values
            using type = DirectedMatrixTag; // default directedness
        };

        template<>
        struct substitute<detail::SparsenessCategoryTag, detail::NullTag> {
            using type = SparseTag; // default sparseness
        };


        // hidden part in the frontend (detail namespace somewhere) to unroll
        // template parameter pack

        struct matrix_generator {
            // recursive call: shaves off one of the tags and puts it in the right
            // place (no error checking yet)
            template<typename ScalarT, typename Sparseness, typename Directedness,
                typename InputTag, typename... Tags>
            struct result {
                using type = typename result<ScalarT,
                      typename detail::substitute<Sparseness, InputTag >::type,
                      typename detail::substitute<Directedness, InputTag >::type,
                      Tags... >::type;
            };

            //null tag shortcut:
            template<typename ScalarT, typename Sparseness, typename Directedness>
            struct result<ScalarT, Sparseness, Directedness, detail::NullTag, detail::NullTag>
            {
                using type = typename backend::Matrix<ScalarT,
                      typename detail::substitute<Sparseness, detail::NullTag >::type,
                      typename detail::substitute<Directedness, detail::NullTag >::type >;
            };

            // base case returns the matrix from the backend
            template<typename ScalarT, typename Sparseness, typename Directedness, typename InputTag>
            struct result<ScalarT, Sparseness, Directedness, InputTag>
            {
                using type = typename backend::Matrix<ScalarT,
                      typename detail::substitute<Sparseness, InputTag >::type,
                      typename detail::substitute<Directedness, InputTag >::type > ;
            };
        };

        /// @todo remove directedness from the vector generator
        struct vector_generator {
            // recursive call: shaves off one of the tags and puts it in the right
            // place (no error checking yet)
            template<typename ScalarT, typename Sparseness,
                typename InputTag, typename... Tags>
            struct result {
                using type = typename result<ScalarT,
                      typename detail::substitute<Sparseness, InputTag >::type,
                      Tags... >::type;
            };

            //null tag shortcut:
            template<typename ScalarT, typename Sparseness>
            struct result<ScalarT, Sparseness, detail::NullTag>
            {
                using type = typename backend::Vector<ScalarT,
                      typename detail::substitute<Sparseness, detail::NullTag >::type >;
            };

            // base case returns the vector from the backend
            template<typename ScalarT, typename Sparseness, typename InputTag>
            struct result<ScalarT, Sparseness, InputTag>
            {
                using type = typename backend::Vector<ScalarT,
                      typename detail::substitute<Sparseness, InputTag >::type > ;
            };
        };

    }//end detail
}
