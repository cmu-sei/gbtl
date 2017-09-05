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

#ifndef GRAPHBLAS_SRC_BINARYADAPTER_HPP
#define GRAPHBLAS_BINARYADAPTER_HPP

#include "graphblas.h"
#include "graphblas_private.h"
#include "TypeAdapter.hpp"

class BinaryAdapter
{
public:
    BinaryAdapter(GrB_BinaryOp op)
    : m_binary_op(op)
            {
            }

    typedef TypeAdapter result_type;
    TypeAdapter operator()(TypeAdapter input1, TypeAdapter input2)
    {
        // @TODO: Is this always true that we take the second?
        if (m_binary_op == GrB_NULL)
            return input2;

        return binary_call(m_binary_op->m_output,
                           m_binary_op->m_input1,
                           m_binary_op->m_input2,
                           m_binary_op->m_binary_fp,
                           input1,
                           input2);
    }

private:
    const GrB_BinaryOp m_binary_op;
};




#endif //GRAPHBLAS_BINARYADAPTER_HPP
