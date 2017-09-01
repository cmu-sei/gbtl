//
// Created by aomellinger on 5/19/17.
//

#ifndef SRC_BINARYADAPTER_HPP
#define SRC_BINARYADAPTER_HPP

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


#endif //SRC_BINARYADAPTER_HPP
