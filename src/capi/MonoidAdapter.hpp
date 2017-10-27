//
// Created by aomellinger on 5/23/17.
//

#ifndef SRC_MONOIDADAPTER_HPP
#define SRC_MONOIDADAPTER_HPP

#include "graphblas.h"
#include "graphblas_private.h"
#include "TypeAdapter.hpp"
#include "FunctionInvocation.h"

class MonoidAdapter
{
public:
    MonoidAdapter(GrB_Monoid op)
            : m_monoid(op)
    {
    }

    typedef TypeAdapter ScalarType;
    typedef TypeAdapter result_type;

    TypeAdapter identity()
    {
        return TypeAdapter(m_monoid->m_identity_type, m_monoid->m_identity);
    }

    TypeAdapter operator()(TypeAdapter input1, TypeAdapter input2)
    {
        return binary_call(m_monoid->m_binary_op->m_binary_fp,
                           m_monoid->m_binary_op->m_output_type,
                           m_monoid->m_binary_op->m_input1_type,
                           m_monoid->m_binary_op->m_input2_type,
                           input1,
                           input2);
    }

private:
    const GrB_Monoid m_monoid;
};

#endif //SRC_MONOIDADAPTER_HPP
