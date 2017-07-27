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
//****************************************************************************

namespace GraphBLAS
{
    // The default matrix is sparse and directed, and the default vector is sparse,
    // so we need tags that modify that
    struct DirectedMatrixTag {};
    struct UndirectedMatrixTag {};
    struct DenseTag {};
    struct SparseTag {};

    namespace detail
    {
        // add category tags in the detail namespace
        struct SparsenessCategoryTag {};
        struct DirectednessCategoryTag {};
        struct NullTag {};
    } //end detail
}//end GraphBLAS

//****************************************************************************
