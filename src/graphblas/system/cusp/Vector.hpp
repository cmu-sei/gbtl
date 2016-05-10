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

#include <cstddef>
#include <iostream>

#include <graphblas/detail/config.hpp>
#include <cusp/array1d.h>

namespace graphblas
{
    namespace backend
    {
        //**********************************************************************
        //ignoring all tags here, array1d only.
        template<typename ScalarT, typename... TagsT>
        class Vector : public cusp::array1d <ScalarT, cusp::device_memory>
        {
        private:
            typedef typename cusp::array1d <ScalarT, cusp::device_memory>
                ParentVectorT;
        public:
            typedef ScalarT ScalarType;

            Vector(): ParentVectorT(){};

            //use parent copy constructor:
            //use parent constructor with num vals:
            template<typename OtherVectorT>
            Vector(OtherVectorT const &vec)
                : ParentVectorT(vec) {}

            template<typename SizeT>
            Vector(SizeT const &count, ScalarT const &value)
                : ParentVectorT(count, value) {}
        };
    }
}
