//
// Created by aomellinger on 5/19/17.
//

#include "graphblas.h"
#include "TypeUnion.hpp"
#include "TypeAdapter.hpp"

#include <graphblas/graphblas.hpp>

#ifndef SRC_GRAPHBLAS_PRIVATE_H
#define SRC_GRAPHBLAS_PRIVATE_H

// Contains a bunch of different constants for internal use.

// Opaque struct to support 4.2.1.2
struct GrB_UnaryOp_Struct
{
    GrB_UnaryOp_Struct(GrB_Type d1, GrB_Type d2, void *unary_func)
            : m_input(d1), m_output(d2), m_unary_fp(unary_func)
    {
    }

    const GrB_Type          m_input;
    const GrB_Type          m_output;
    void * const            m_unary_fp;
};

// Opaque struct to support 4.2.1.3
struct GrB_BinaryOp_Struct
{
    GrB_BinaryOp_Struct(GrB_Type d1, GrB_Type d2, GrB_Type d3, void *binary_func)
            : m_input1(d1), m_input2(d2), m_output(d3), m_binary_fp(binary_func)
    {
    }

    const GrB_Type          m_input1;
    const GrB_Type          m_input2;
    const GrB_Type          m_output;
    void * const            m_binary_fp;
};

// Opaque to support 4.2.1.4
struct GrB_Monoid_Struct
{
    GrB_Monoid_Struct(GrB_BinaryOp binary_op, GrB_INT32 val)
            : m_identity_type(GrB_INT32_Type), m_binary_op(binary_op)
    {
        m_identity._int32 = val;
    }

    GrB_Monoid_Struct(GrB_BinaryOp binary_op, GrB_FP32 val)
            : m_identity_type(GrB_FP32_Type), m_binary_op(binary_op)
    {
        m_identity._fp32 = val;
    }

    const GrB_Type         m_identity_type;
    const GrB_BinaryOp     m_binary_op;

    GrB_TypeUnion      m_identity;
};

// Opaque struct to support 4.2.1.5
struct GrB_Semiring_Struct
{
    GrB_Semiring_Struct(GrB_Monoid add_op,
                        GrB_BinaryOp mul_op )
    : m_monoid(add_op), m_binary_op(mul_op)
    {
    }

    const GrB_Monoid    m_monoid;           // Mult
    const GrB_BinaryOp  m_binary_op;        // Add
};

// Opaque struct to support 4.2.2.1
struct GrB_Vector_Struct
{
    GrB_Vector_Struct(GrB_Type type, GrB_Index nrows)
            : m_type(type)
    {
        m_vec = new GraphBLAS::Vector<TypeAdapter>(nrows);
    }

    ~GrB_Vector_Struct()
    {
        delete m_vec;
    }

    const GrB_Type m_type;
    GraphBLAS::Vector<TypeAdapter> *m_vec;
};

// Opaque struct to support 4.2.3.1
struct GrB_Matrix_Struct
{
    GrB_Matrix_Struct(GrB_Type type, GrB_Index nrows, GrB_Index ncols )
            : m_type(type)
    {
        m_matrix = new GraphBLAS::Matrix<TypeAdapter>(nrows, ncols);
    }

    ~GrB_Matrix_Struct()
    {
        delete m_matrix;
    }

    // Describes our dynamic type
    const GrB_Type m_type;
    GraphBLAS::Matrix<TypeAdapter> *m_matrix;
};

// Opaque struct to support descriptors 3.7, 4.2.4.1 & 4.2.4.2
struct GrB_Descriptor_Struct
{
    void set(GrB_Field field, GrB_Value value)
    {
        m_field |= 0x1 << (field * 8 + value);
    }

    GrB_Value get(GrB_Field field)
    {
        return (GrB_Value) (0xFF & (m_field >> (field * 8)));
    }

    bool isSet(GrB_Field field, GrB_Value value)
    {
        return get(field) & (0x1 << value);
    }

    int32_t  m_field;
};

#endif //SRC_GRAPHBLAS_PRIVATE_H
