/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors.
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
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DM20-0442
 */

#pragma once
// Tags are API-defined (i.e., frontend)
#include <graphblas/detail/matrix_tags.hpp>

#include <graphblas/platforms/optimized_sequential/LilSparseMatrix.hpp>
#include <graphblas/platforms/optimized_sequential/BitmapSparseVector.hpp>

//this file contains the variadic template parameters unpacking utility.

//****************************************************************************
//****************************************************************************

namespace grb
{
    namespace backend
    {

        // Substitute template to decide if a tag goes into a given slot
        template<typename TagCategory, typename Tag>
        struct substitute {
            using type = TagCategory;
        };


        template<>
        struct substitute<grb::detail::SparsenessCategoryTag, grb::DenseTag> {
            using type = grb::DenseTag;
        };

        template<>
        struct substitute<grb::detail::SparsenessCategoryTag, grb::SparseTag> {
            using type = grb::SparseTag;
        };

        template<>
        struct substitute<grb::detail::SparsenessCategoryTag, grb::detail::NullTag> {
            using type = grb::SparseTag; // default sparseness
        };


        template<>
        struct substitute<grb::detail::DirectednessCategoryTag, grb::UndirectedMatrixTag> {
            using type = grb::UndirectedMatrixTag;
        };

        template<>
        struct substitute<grb::detail::DirectednessCategoryTag, grb::DirectedMatrixTag> {
            using type = grb::DirectedMatrixTag;
        };

        template<>
        struct substitute<grb::detail::DirectednessCategoryTag, grb::detail::NullTag> {
            //default values
            using type = grb::DirectedMatrixTag; // default directedness
        };


        // hidden part in the frontend (detail namespace somewhere) to unroll
        // template parameter pack

        struct matrix_generator {
            // recursive call: shaves off one of the tags and puts it in the right
            // place (no error checking yet)
            template<typename ScalarT, typename Sparseness, typename Directedness,
                typename InputTag, typename... TagsT>
            struct result {
                using type = typename result<
                    ScalarT,
                    typename substitute<Sparseness, InputTag >::type,
                    typename substitute<Directedness, InputTag >::type,
                    TagsT...>::type;
            };

            //null tag shortcut:
            template<typename ScalarT, typename Sparseness, typename Directedness>
            struct result<ScalarT, Sparseness, Directedness, grb::detail::NullTag, grb::detail::NullTag>
            {
                using type = LilSparseMatrix<ScalarT>;
            };

            // base case returns the matrix from the backend
            template<typename ScalarT, typename Sparseness, typename Directedness, typename InputTag>
            struct result<ScalarT, Sparseness, Directedness, InputTag>
            {
                using type = LilSparseMatrix<ScalarT>;
            };
        };

        // helper to replace backend Matrix class
        template<typename ScalarT, typename... TagsT>
        using Matrix = typename matrix_generator::result<
            ScalarT,
            detail::SparsenessCategoryTag,
            detail::DirectednessCategoryTag,
            TagsT...,
            detail::NullTag,
            detail::NullTag>::type;

        //********************************************************************
        struct vector_generator {
            // recursive call: shaves off one of the tags and puts it in the right
            // place (no error checking yet)
            template<typename ScalarT, typename Sparseness,
                typename InputTag, typename... Tags>
            struct result {
                using type = typename result<
                    ScalarT,
                    typename substitute<Sparseness, InputTag>::type,
                    Tags... >::type;
            };

            //null tag shortcut:
            template<typename ScalarT, typename Sparseness>
            struct result<ScalarT, Sparseness, grb::detail::NullTag>
            {
                using type = BitmapSparseVector<ScalarT>;
            };

            // base case returns the vector from the backend
            template<typename ScalarT, typename Sparseness, typename InputTag>
            struct result<ScalarT, Sparseness, InputTag>
            {
                using type = BitmapSparseVector<ScalarT>;
            };
        };

        // helper to replace backend Vector class
        template<typename ScalarT, typename... TagsT>
        using Vector = typename vector_generator::result<
            ScalarT,
            detail::SparsenessCategoryTag,
            TagsT... ,
            detail::NullTag>::type;

    } // namespace backend
} // namespace grb
