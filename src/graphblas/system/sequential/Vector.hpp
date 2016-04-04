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
#include <vector>

namespace graphblas
{
    namespace backend
    {
        //ignoring all tags here, coo matrices only.
        template<typename ScalarT, typename... TagsT>
        class Vector : public std::vector <ScalarT>
        {
        private:
            typedef typename std::vector <ScalarT> ParentVectorT;
        public:
            typedef ScalarT ScalarType;
            Vector(): ParentVectorT(){};

            //use parent copy constructor:
            //use parent constructor with num vals:
            template<typename T>
            Vector(const T &v)
                : ParentVectorT(v) {}

            template<typename T1, typename T2>
            Vector(const T1 &v1, const T2 & v2)
                : ParentVectorT(v1, v2) {}
        };
    }
}
