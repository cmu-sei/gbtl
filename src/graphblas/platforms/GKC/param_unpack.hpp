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
// Should matrix tags be implementation-defined (backend) or API-defined (frontend)?
#include <graphblas/detail/matrix_tags.hpp>

#include <graphblas/platforms/GKC/LilSparseMatrix.hpp>
#include <graphblas/platforms/GKC/BitmapSparseVector.hpp>

#include <graphblas/platforms/GKC/GKCMatrix.hpp>
#include <graphblas/platforms/GKC/GKCSparseVector.hpp>

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


        //--

        template<>
        struct substitute<grb::detail::ImplementationCategoryTag, grb::GKCTag> {
            using type = grb::GKCTag;
        };

        template<>
        struct substitute<grb::detail::ImplementationCategoryTag, grb::OrigTag> {
            using type = grb::OrigTag;
        };

        template<>
        struct substitute<grb::detail::ImplementationCategoryTag, grb::detail::NullTag> {
            using type = grb::OrigTag; // default implementation
        };

        //--

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

        //--

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


        //--

        // hidden part in the frontend (detail namespace somewhere) to unroll
        // template parameter pack

        struct matrix_generator {
            // recursive call: shaves off one of the tags and puts it in the right
            // place (no error checking yet)
            template<typename ScalarT,
                     typename Implementation, typename Sparseness, typename Directedness,
                     typename InputTag, typename... TagsT>
            struct result {
                using type = typename result<
                    ScalarT,
                    typename substitute<Implementation, InputTag >::type,
                    typename substitute<Sparseness, InputTag >::type,
                    typename substitute<Directedness, InputTag >::type,
                    TagsT...>::type;
            };

            // special case returns the backend GKC matrix type
            template<typename ScalarT,
                     typename Implementation, typename Sparseness, typename Directedness,
                     typename... TagsT>
            struct result<ScalarT,
                          Implementation, Sparseness, Directedness,
                          grb::GKCTag, TagsT...>
            {
                using type = GKCMatrix<ScalarT>;
            };

            // null tag shortcut:
            template<typename ScalarT,
                     typename Implementation, typename Sparseness, typename Directedness>
            struct result<ScalarT,
                          Implementation, Sparseness, Directedness,
                          grb::detail::NullTag, grb::detail::NullTag, grb::detail::NullTag>
            {
                using type = LilSparseMatrix<ScalarT>;
            };

            // base case returns the matrix from the backend
            template<typename ScalarT,
                     typename Implementation, typename Sparseness, typename Directedness,
                     typename InputTag>
            struct result<ScalarT, Implementation, Sparseness, Directedness, InputTag>
            {
                using type = LilSparseMatrix<ScalarT>;
            };
        };

        // helper to replace backend Matrix class
        template<typename ScalarT, typename... TagsT>
        using Matrix = typename matrix_generator::result<
            ScalarT,
            detail::ImplementationCategoryTag,
            detail::SparsenessCategoryTag,
            detail::DirectednessCategoryTag,
            TagsT...,
            detail::NullTag,
            detail::NullTag,
            detail::NullTag>::type;

        //********************************************************************
        //********************************************************************
        struct vector_generator {
            // recursive call: shaves off one of the tags and puts it in the right
            // place (no error checking yet)
            template<typename ScalarT,
                     typename Implementation, typename Sparseness,
                     typename InputTag, typename... Tags>
            struct result {
                using type = typename result<
                    ScalarT,
                    typename substitute<Implementation, InputTag >::type,
                    typename substitute<Sparseness, InputTag>::type,
                    Tags... >::type;
            };

            // special case returns the backend GKC vector type
            template<typename ScalarT,
                     typename Implementation, typename Sparseness,
                     typename... TagsT>
            struct result<ScalarT,
                          Implementation, Sparseness,
                          grb::GKCTag, TagsT...>
            {
                using type = GKCSparseVector<ScalarT>;
            };

            //null tag shortcut:
            template<typename ScalarT,
                     typename Implementation, typename Sparseness>
            struct result<ScalarT, Implementation, Sparseness,
                          grb::detail::NullTag, grb::detail::NullTag>
            {
                using type = BitmapSparseVector<ScalarT>;
            };

            // base case returns the vector from the backend
            template<typename ScalarT,
                     typename Implementation, typename Sparseness,
                     typename InputTag>
            struct result<ScalarT, Implementation,  Sparseness, InputTag>
            {
                using type = BitmapSparseVector<ScalarT>;
            };
        };

        // helper to replace backend Vector class
        template<typename ScalarT, typename... TagsT>
        using Vector = typename vector_generator::result<
            ScalarT,
            detail::ImplementationCategoryTag,
            detail::SparsenessCategoryTag,
            TagsT... ,
            detail::NullTag,
            detail::NullTag>::type;

    } // namespace backend
} // namespace grb
