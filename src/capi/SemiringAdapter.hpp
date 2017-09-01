//
// Created by aomellinger on 5/23/17.
//

#ifndef SRC_SEMIRINGADAPTER_HPP
#define SRC_SEMIRINGADAPTER_HPP

#include "graphblas.h"
#include "graphblas_private.h"
#include "TypeAdapter.hpp"
#include "FunctionInvocation.h"

class SemiringAdapter
{
public:
    SemiringAdapter(GrB_Semiring semiring)
    : m_semiring(semiring)
    {
    }

    typedef TypeAdapter result_type;

    TypeAdapter add(TypeAdapter a, TypeAdapter b) const
    {
        return binary_call(m_semiring->m_monoid->m_binary_op->m_output,
                           m_semiring->m_monoid->m_binary_op->m_input1,
                           m_semiring->m_monoid->m_binary_op->m_input2,
                           m_semiring->m_monoid->m_binary_op->m_binary_fp,
                           a,
                           b);
    }

    TypeAdapter mult(TypeAdapter a, TypeAdapter b) const
    {
        return binary_call(m_semiring->m_binary_op->m_output,
                           m_semiring->m_binary_op->m_input1,
                           m_semiring->m_binary_op->m_input2,
                           m_semiring->m_binary_op->m_binary_fp,
                           a,
                           b);
    }

    TypeAdapter zero() const
    {
        return TypeAdapter(m_semiring->m_monoid->m_identity_type,
                           m_semiring->m_monoid->m_identity);
    }

private:
    const GrB_Semiring m_semiring;
};


#endif //SRC_SEMIRINGADAPTER_HPP
