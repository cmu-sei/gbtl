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
//matrix tags should be strictly internal
#include "matrix_tags.hpp"

// Include matrix definitions from the appropriate backend.
#define __GB_SYSTEM_MATRIX_HEADER <graphblas/system/__GB_SYSTEM_ROOT/Matrix.hpp>
#include __GB_SYSTEM_MATRIX_HEADER
#undef __GB_SYSTEM_MATRIX_HEADER

//this file contains the variadic template parameters unpacking utility.

namespace graphblas
{
    namespace detail
    {

        // Substitute template to decide if a tag goes into a given slot
        template<typename TagCategory, typename Tag>
        struct substitute {
            using type = TagCategory;
        };


        template<>
        struct substitute<detail::SparsenessCategoryTag, DenseMatrixTag> {
            using type = DenseMatrixTag;
        };

        template<>
        struct substitute<detail::SparsenessCategoryTag, SparseMatrixTag> {
            using type = SparseMatrixTag;
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
            using type = DirectedMatrixTag;
        };

        template<>
        struct substitute<detail::SparsenessCategoryTag, detail::NullTag> {
            using type = SparseMatrixTag;
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
    }//end detail
}

