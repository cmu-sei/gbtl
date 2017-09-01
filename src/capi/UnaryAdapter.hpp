//
// Created by aomellinger on 5/17/17.
//

#ifndef SRC_UNARYADAPTER_H
#define SRC_UNARYADAPTER_H

#include "graphblas.h"
#include "graphblas_private.h"
#include "TypeAdapter.hpp"
#include "FunctionInvocation.h"

/**
 * Adapts the GraphBLAS unarfy function point to the template TypeAdapter
 * we pass into the tempkate C++ backend.
 */
class UnaryAdapter
{
public:
    UnaryAdapter(GrB_UnaryOp op)
    : m_unary_op(op)
    {
    }

    typedef TypeAdapter result_type;
    TypeAdapter operator()(TypeAdapter input)
    {
        return unary_call(m_unary_op->m_output,
                          m_unary_op->m_input,
                          m_unary_op->m_unary_fp,
                          input);
    }

private:
    const GrB_UnaryOp m_unary_op;
};

#endif //SRC_UNARYADAPTER_H
