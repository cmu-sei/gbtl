/*
 * Copyright (c) 2017 Carnegie Mellon University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY EXPRESSLY DISCLAIMS TO THE FULLEST EXTENT PERMITTED BY
 * LAW ALL EXPRESS, IMPLIED, AND STATUTORY WARRANTIES, INCLUDING, WITHOUT
 * LIMITATION, THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */

#include "graphblas.h"
#include "graphblas_private.h"

#include <graphblas/graphblas.hpp>
#include <capi/detail/index_support.hpp>

#include "TypeAdapter.hpp"
#include "UnaryAdapter.hpp"
#include "BinaryAdapter.hpp"
#include "SemiringAdapter.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// I want these to be const *, but I get const violations when I do this
//const void *GrB_NULL = (void *) &("GrB_NULL");
void *GrB_NULL = (void *) &("GrB_NULL");


//=============================================================================

// Mask has 3 options: Null, regular, complement (SCMP)
// Accum has 2 options: Null, regular
// A has 2 options:  plain, transpose
// B has 2 options:  plain, transpose

// Total = 3 x 2 x 2 x 2 = 24

// MASK_ACCUM_IN0_IN1
enum BinaryInputSelectorEnum {
    NULL_NULL_REG_REG,
    NULL_NULL_REG_TRANS,
    NULL_NULL_TRANS_REG,
    NULL_NULL_TRANS_TRANS,
    NULL_REG_REG_REG,
    NULL_REG_REG_TRANS,
    NULL_REG_TRANS_REG,
    NULL_REG_TRANS_TRANS,
    REG_NULL_REG_REG,
    REG_NULL_REG_TRANS,
    REG_NULL_TRANS_REG,
    REG_NULL_TRANS_TRANS,
    REG_REG_REG_REG,
    REG_REG_REG_TRANS,
    REG_REG_TRANS_REG,
    REG_REG_TRANS_TRANS,
    SCMP_NULL_REG_REG,
    SCMP_NULL_REG_TRANS,
    SCMP_NULL_TRANS_REG,
    SCMP_NULL_TRANS_TRANS,
    SCMP_REG_REG_REG,
    SCMP_REG_REG_TRANS,
    SCMP_REG_TRANS_REG,
    SCMP_REG_TRANS_TRANS
};


BinaryInputSelectorEnum makeBinaryInputSelector(GrB_Matrix C,
                                                const GrB_Matrix Mask,
                                                const GrB_BinaryOp accum,
                                                const GrB_Descriptor desc)
{

}



//=============================================================================
// 4.2.1.2
GrB_Info GrB_UnaryOp_new(GrB_UnaryOp *unary_op,
                         GrB_Type d1,
                         GrB_Type d2,
                         void *unary_func)
{
    *unary_op = new GrB_UnaryOp_Struct(d1, d2, unary_func);
    return GrB_SUCCESS;
}

//=============================================================================
// 4.2.1.3
GrB_Info GrB_BinaryOp_new(GrB_BinaryOp *binary_op,
                          GrB_Type d1,
                          GrB_Type d2,
                          GrB_Type d3,
                          void *binary_func)
{
    *binary_op = new GrB_BinaryOp_Struct(d1, d2, d3, binary_func);
    return GrB_SUCCESS;
}

//=============================================================================
// 4.2.1.4
GrB_Info GrB_Monoid_INT32_new(GrB_Monoid *monoid,
                              GrB_BinaryOp binary_op,
                              GrB_INT32 identity)
{
    // Type check!
    *monoid = new GrB_Monoid_Struct(binary_op, identity);
    return GrB_SUCCESS;
}

GrB_Info GrB_Monoid_FP32_new(GrB_Monoid *monoid,
                             GrB_BinaryOp binary_op,
                             GrB_FP32 identity)
{
    // Type check!
    *monoid = new GrB_Monoid_Struct(binary_op, identity);
    return GrB_SUCCESS;
}

//=============================================================================
// 4.2.1.5
GrB_Info GrB_Semiring_new(GrB_Semiring *semiring,
                          GrB_Monoid add_op,
                          GrB_BinaryOp mul_op)
{
    *semiring = new GrB_Semiring_Struct(add_op, mul_op);
    return GrB_SUCCESS;
}


//=============================================================================
// 4.2.2.1
GrB_Info GrB_Vector_new(GrB_Vector *v,
                        GrB_Type d,
                        GrB_Index nsize)
{
    *v = new GrB_Vector_Struct(d, nsize);
    return GrB_SUCCESS;
}

//=============================================================================
// 4.2.2.1

GrB_Info GrB_Vector_extractElement_FP32(GrB_FP32 *val,
                                        const GrB_Vector u,
                                        GrB_Index index)
{
    *val = (GrB_FP32) u->m_vec->extractElement(index);
    return GrB_SUCCESS;
}

//=============================================================================

// 4.2.3.1
GrB_Info GrB_Matrix_new(GrB_Matrix *A,
                        GrB_Type d,
                        GrB_Index nrows,
                        GrB_Index ncols)
{
    *A = new GrB_Matrix_Struct(d, nrows, ncols);
    return GrB_SUCCESS;
}

//=============================================================================
// 4.2.3.4

GrB_Info GrB_Matrix_nrows(GrB_Index *nrows,
                          const GrB_Matrix A)
{
    *nrows = A->m_matrix->nrows();
    return GrB_SUCCESS;
}

//=============================================================================
// 4.2.3.5

GrB_Info GrB_Matrix_ncols(GrB_Index *ncols,
                          const GrB_Matrix A)
{
    *ncols = A->m_matrix->ncols();
    return GrB_SUCCESS;
}

//=============================================================================
// 4.2.3.6

GrB_Info GrB_Matrix_nvals(GrB_Index *nvals,
                          const GrB_Matrix A)
{
    *nvals = A->m_matrix->nvals();
    return GrB_SUCCESS;
}


//=============================================================================

// 4.2.3.7
GrB_Info GrB_Matrix_build_INT32(GrB_Matrix C,
                                const GrB_Index *row_indices,
                                const GrB_Index *col_indices,
                                const GrB_INT32 *values,
                                GrB_Index nvals,
                                const GrB_BinaryOp dup)
{
    if (C->m_type != GrB_INT32_Type)
        return GrB_TYPE_MISMATCH;

    // Since it is a const, can we use a wrapper?
    GraphBLAS::IndexArrayType row_vec(row_indices, &(row_indices[nvals]));
    GraphBLAS::IndexArrayType col_vec(col_indices, &(col_indices[nvals]));

    // Convert the type to the type adapter
    std::vector<TypeAdapter> values_vec;
    for (int i = 0; i < nvals; ++i)
        values_vec.push_back(TypeAdapter(values[i]));

    C->m_matrix->build(row_vec, col_vec, values_vec);

    return GrB_SUCCESS;
}


GrB_Info GrB_Matrix_build_FP64(GrB_Matrix C,
                               const GrB_Index *row_indices,
                               const GrB_Index *col_indices,
                               const GrB_FP64 *values,
                               GrB_Index nvals,
                               const GrB_BinaryOp dup)
{
    if (C->m_type != GrB_FP64_Type)
        return GrB_TYPE_MISMATCH;

    // Since it is a const, can we use a wrapper?
    GraphBLAS::IndexArrayType row_vec(row_indices, &(row_indices[nvals]));
    GraphBLAS::IndexArrayType col_vec(col_indices, &(col_indices[nvals]));

    // Convert the type to the type adapter
    std::vector<TypeAdapter> values_vec;
    for (int i = 0; i < nvals; ++i)
        values_vec.push_back(TypeAdapter(values[i]));

    C->m_matrix->build(row_vec, col_vec, values_vec);

    return GrB_SUCCESS;
}

//=============================================================================
// 4.2.4 Descriptors

// 4.2.4.1
GrB_Info GrB_Descriptor_new(GrB_Descriptor *desc)
{
    *desc = new GrB_Descriptor_Struct;
    return GrB_SUCCESS;
}

// 4.2.4.2
GrB_Info GrB_Descriptor_set(GrB_Descriptor desc,
                            GrB_Field field,
                            GrB_Value val)
{
    desc->set(field, val);
    return GrB_SUCCESS;
}

//=============================================================================

// 4.2.5
GrB_Info GrB_UnaryOp_free(GrB_UnaryOp *obj)
{
    delete *obj;
    *obj = NULL;
    return GrB_SUCCESS;
}

GrB_Info GrB_BinaryOp_free(GrB_BinaryOp *obj)
{
    delete *obj;
    *obj = NULL;
    return GrB_SUCCESS;
}

GrB_Info GrB_Monoid_free(GrB_Monoid *obj)
{
    delete *obj;
    *obj = NULL;
    return GrB_SUCCESS;
}

GrB_Info GrB_Semiring_free(GrB_Semiring *obj)
{
    delete *obj;
    *obj = NULL;
    return GrB_SUCCESS;
}

GrB_Info GrB_Vector_free(GrB_Vector *obj)
{
    delete *obj;
    *obj = NULL;
    return GrB_SUCCESS;
}

GrB_Info GrB_Matrix_free(GrB_Matrix *obj)
{
    delete *obj;
    *obj = NULL;
    return GrB_SUCCESS;
}

GrB_Info GrB_Descriptor_free(GrB_Descriptor *obj)
{
    delete *obj;
    *obj = NULL;
    return GrB_SUCCESS;
}

//=============================================================================

// 4.3.1
GrB_Info GrB_mxm(GrB_Matrix C,
                 const GrB_Matrix Mask,
                 const GrB_BinaryOp accum,
                 const GrB_Semiring op,
                 const GrB_Matrix A,
                 const GrB_Matrix B,
                 const GrB_Descriptor desc)
{
    bool replace = false;
    if (desc != GrB_NULL)
        replace = desc->isSet(GrB_OUTP, GrB_REPLACE);

    if (Mask == GrB_NULL)
    {
        if (accum == GrB_NULL)
        {
            if (desc != GrB_NULL &&
                desc->isSet(GrB_INP0, GrB_TRAN) &&
                desc->isSet(GrB_INP1, GrB_TRAN))
            {
                GraphBLAS::mxm(*C->m_matrix,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               SemiringAdapter(op),
                               GraphBLAS::transpose(*A->m_matrix),
                               GraphBLAS::transpose(*B->m_matrix),
                               replace);

                return GrB_SUCCESS;
            }
            else if (desc != GrB_NULL && desc->isSet(GrB_INP0, GrB_TRAN))
            {
                return GrB_PANIC;
            }
            else if (desc != GrB_NULL && desc->isSet(GrB_INP1, GrB_TRAN))
            {
                return GrB_PANIC;
            }
            else
            {
                GraphBLAS::mxm(*C->m_matrix,
                               GraphBLAS::NoMask(),
                               GraphBLAS::NoAccumulate(),
                               SemiringAdapter(op),
                               *A->m_matrix,
                               *B->m_matrix,
                               replace);

                return GrB_SUCCESS;
            }
        }
        else
        {
            // @TODO: Build out all four
            return GrB_PANIC;
        }
    }
    else if (desc != GrB_NULL && desc->isSet(GrB_MASK, GrB_SCMP))
    {
        if (accum == GrB_NULL)
        {
            if (desc->isSet(GrB_INP0, GrB_TRAN) &&
                desc->isSet(GrB_INP1, GrB_TRAN))
            {
                return GrB_SUCCESS;
            }
            else if (desc->isSet(GrB_INP0, GrB_TRAN))
            {
                return GrB_PANIC;
            }
            else if (desc->isSet(GrB_INP1, GrB_TRAN))
            {
                return GrB_PANIC;
            }
            else
            {
                GraphBLAS::mxm(*C->m_matrix,
                               GraphBLAS::complement(*Mask->m_matrix),
                               GraphBLAS::NoAccumulate(),
                               SemiringAdapter(op),
                               *A->m_matrix,
                               *B->m_matrix,
                               replace);
                return GrB_SUCCESS;
            }
        }
        else
        {
            // @TODO: Build out all four
            return GrB_PANIC;
        }

        // @todo: Add all four transpose options times op options
    }
    else
    {
        if (accum == GrB_NULL)
        {
            if (desc != GrB_NULL && desc->isSet(GrB_INP0, GrB_TRAN) &&
                desc->isSet(GrB_INP1, GrB_TRAN))
            {
                return GrB_PANIC;
            }
            else if (desc != GrB_NULL && desc->isSet(GrB_INP0, GrB_TRAN))
            {
                return GrB_PANIC;
            }
            else if (desc != GrB_NULL && desc->isSet(GrB_INP1, GrB_TRAN))
            {
                GraphBLAS::mxm(*C->m_matrix,
                               *Mask->m_matrix,
                               GraphBLAS::NoAccumulate(),
                               SemiringAdapter(op),
                               *A->m_matrix,
                               GraphBLAS::transpose(*B->m_matrix),
                               replace);
                return GrB_SUCCESS;
            }
            else
            {
                GraphBLAS::mxm(*C->m_matrix,
                               *Mask->m_matrix,
                               GraphBLAS::NoAccumulate(),
                               SemiringAdapter(op),
                               *A->m_matrix,
                               *B->m_matrix,
                               replace);
                return GrB_SUCCESS;
            }
        }
        else
        {
            // @TODO: Build out all four
            return GrB_PANIC;
        }
    }


//    if (desc != GrB_NULL && desc->isSet(GrB_INP0, GrB_TRAN) && desc->isSet(GrB_INP1, GrB_TRAN))
//    {
//        return GrB_PANIC;
//    }
//    else if (desc != GrB_NULL && desc->isSet(GrB_INP0, GrB_TRAN))
//    {
//        return GrB_PANIC;
//    }
//    else if (desc != GrB_NULL && desc->isSet(GrB_INP1, GrB_TRAN))
//    {
//        return GrB_PANIC;
//    }
//    else
//    {
//        return GrB_SUCCESS;
//    }


    return GrB_PANIC;
}


GrB_Info GrB_mxm_v2(GrB_Matrix C,
                 const GrB_Matrix Mask,
                 const GrB_BinaryOp accum,
                 const GrB_Semiring op,
                 const GrB_Matrix A,
                 const GrB_Matrix B,
                 const GrB_Descriptor desc)
{
    int foo = 0;
    switch(foo)
    {
        case NULL_NULL_REG_REG:
            break;
        case NULL_NULL_REG_TRANS:
            break;
        case NULL_NULL_TRANS_REG:
            break;
        case NULL_NULL_TRANS_TRANS:
            break;
        case NULL_REG_REG_REG:
            break;
        case NULL_REG_REG_TRANS:
            break;
        case NULL_REG_TRANS_REG:
            break;
        case NULL_REG_TRANS_TRANS:
            break;
        case REG_NULL_REG_REG:
            break;
        case REG_NULL_REG_TRANS:
            break;
        case REG_NULL_TRANS_REG:
            break;
        case REG_NULL_TRANS_TRANS:
            break;
        case REG_REG_REG_REG:
            break;
        case REG_REG_REG_TRANS:
            break;
        case REG_REG_TRANS_REG:
            break;
        case REG_REG_TRANS_TRANS:
            break;
        case SCMP_NULL_REG_REG:
            break;
        case SCMP_NULL_REG_TRANS:
            break;
        case SCMP_NULL_TRANS_REG:
            break;
        case SCMP_NULL_TRANS_TRANS:
            break;
        case SCMP_REG_REG_REG:
            break;
        case SCMP_REG_REG_TRANS:
            break;
        case SCMP_REG_TRANS_REG:
            break;
        case SCMP_REG_TRANS_TRANS:
            break;
        default:
            break;
    }
}

//=============================================================================

// 4.3.4.2

GrB_Info GrB_Martrix_eWiseMult_BinaryOp(GrB_Matrix C,
                                        const GrB_Matrix Mask,
                                        const GrB_BinaryOp accum,
                                        const GrB_BinaryOp op,
                                        const GrB_Matrix A,
                                        const GrB_Matrix B,
                                        const GrB_Descriptor desc)
{
    bool replace = false;
    if (desc != GrB_NULL)
        replace = desc->isSet(GrB_OUTP, GrB_REPLACE);

    // Add debug check for transpose and scmp

    if (Mask == GrB_NULL)
    {
        if (accum == GrB_NULL)
        {
            GraphBLAS::eWiseMult(*C->m_matrix,
                                 GraphBLAS::NoMask(),
                                 GraphBLAS::NoAccumulate(),
                                 BinaryAdapter(op),
                                 *A->m_matrix,
                                 *B->m_matrix,
                                 replace);
        }
        else
        {
            GraphBLAS::eWiseMult(*C->m_matrix,
                                 GraphBLAS::NoMask(),
                                 BinaryAdapter(accum),
                                 BinaryAdapter(op),
                                 *A->m_matrix,
                                 *B->m_matrix,
                                 replace);
        }
    }
    else
    {
        if (accum == GrB_NULL)
        {
            GraphBLAS::eWiseMult(*C->m_matrix,
                                 *Mask->m_matrix,
                                 GraphBLAS::NoAccumulate(),
                                 BinaryAdapter(op),
                                 *A->m_matrix,
                                 *B->m_matrix,
                                 replace);
        }
        else
        {
            GraphBLAS::eWiseMult(*C->m_matrix,
                                 *Mask->m_matrix,
                                 BinaryAdapter(accum),
                                 BinaryAdapter(op),
                                 *A->m_matrix,
                                 *B->m_matrix,
                                 replace);
        }
    }

    return GrB_SUCCESS;
}

//=============================================================================

// 4.3.5.2
GrB_Info GrB_Matrix_eWiseAdd_BinaryOp(GrB_Matrix C,
                                      const GrB_Matrix Mask,
                                      const GrB_BinaryOp accum,
                                      const GrB_BinaryOp op,
                                      const GrB_Matrix A,
                                      const GrB_Matrix B,
                                      const GrB_Descriptor desc)
{
    bool replace = false;
    if (desc != GrB_NULL)
        replace = desc->isSet(GrB_OUTP, GrB_REPLACE);

    if (Mask != GrB_NULL)
        return GrB_PANIC;

    if (accum != GrB_NULL)
        return GrB_PANIC;

    GraphBLAS::eWiseAdd(*C->m_matrix,
                        GraphBLAS::NoMask(),
                        GraphBLAS::NoAccumulate(),
                        BinaryAdapter(op),
                        *A->m_matrix,
                        *B->m_matrix,
                        replace);
    return GrB_SUCCESS;
}


//=============================================================================

// 4.3.6.2
GrB_Info GrB_extract(GrB_Matrix C,
                     const GrB_Matrix Mask,
                     const GrB_BinaryOp accum,
                     const GrB_Matrix A,
                     const GrB_Index *row_indices,
                     const GrB_Index nrows,
                     const GrB_Index *col_indices,
                     const GrB_Index ncols,
                     const GrB_Descriptor desc)
{
    bool replace = false;
    if (desc != GrB_NULL)
        replace = desc->isSet(GrB_OUTP, GrB_REPLACE);

    if (accum != GrB_NULL)
        return GrB_PANIC;

    if ((void *) Mask == GrB_NULL)
    {
        GraphBLAS::extract(*C->m_matrix,
                           GraphBLAS::NoMask(),
                           GraphBLAS::NoAccumulate(),
                           *A->m_matrix,
                           GraphBLAS::IndexProxy(row_indices, nrows),
                           GraphBLAS::IndexProxy(col_indices, ncols),
                           replace);
    }
    else
    {
        GraphBLAS::extract(*C->m_matrix,
                           *Mask->m_matrix,
                           GraphBLAS::NoAccumulate(),
                           *A->m_matrix,
                           GraphBLAS::IndexProxy(row_indices, nrows),
                           GraphBLAS::IndexProxy(col_indices, ncols),
                           replace);
    }

//    inline void extract(CMatrixT             &C,
//                        MaskT          const &Mask,
//                        AccumT                accum,
//                        AMatrixT       const &A,
//                        IndexArrayType const &row_indices,
//                        IndexArrayType const &col_indices,
//                        bool                  replace_flag = false)


    return GrB_SUCCESS;
}

//=============================================================================
// 4.3.7.6 - GrB_assign

GrB_Info GrB_Vector_assign_FP32(GrB_Vector w,
                                const GrB_Vector mask,
                                const GrB_BinaryOp accum,
                                const GrB_FP32 val,
                                const GrB_Index *row_indicies,
                                const GrB_Index nindicies,
                                const GrB_Descriptor desc)
{
    bool replace = false;
    if (desc != GrB_NULL)
        replace = desc->isSet(GrB_OUTP, GrB_REPLACE);

    if (mask != GrB_NULL)
        return GrB_PANIC;

    if (accum == GrB_NULL)
    {
        GraphBLAS::assign(*w->m_vec,
                           GraphBLAS::NoMask(),
                           GraphBLAS::NoAccumulate(),
                           val,
                           GraphBLAS::IndexProxy(row_indicies, nindicies),
                           replace);
    }
    else
    {
        GraphBLAS::assign(*w->m_vec,
                           GraphBLAS::NoMask(),
                           BinaryAdapter(accum),
                           val,
                           GraphBLAS::IndexProxy(row_indicies, nindicies),
                           replace);
    }
    return GrB_SUCCESS;
}


// Matrix FP32 VARIANT
GrB_Info GrB_Matrix_assign_FP32(GrB_Matrix C,
                                const GrB_Matrix Mask,
                                const GrB_BinaryOp accum,
                                const GrB_FP32 val,
                                const GrB_Index *row_indicies,
                                const GrB_Index nrows,
                                const GrB_Index *col_indicies,
                                const GrB_Index ncols,
                                const GrB_Descriptor desc)
{
    bool replace = false;
    if (desc != GrB_NULL)
        replace = desc->isSet(GrB_OUTP, GrB_REPLACE);

    if (Mask != GrB_NULL)
        return GrB_PANIC;

    if (accum != GrB_NULL)
        return GrB_PANIC;

    GraphBLAS::assign(*C->m_matrix,
                       GraphBLAS::NoMask(),
                       GraphBLAS::NoAccumulate(),
                       val,
                       GraphBLAS::IndexProxy(row_indicies, nrows),
                       GraphBLAS::IndexProxy(col_indicies, ncols),
                       replace);

    return GrB_SUCCESS;
}

//=============================================================================
// 4.3.8.2
GrB_Info GrB_apply(GrB_Matrix C,
                   const GrB_Matrix Mask,
                   const GrB_BinaryOp accum,
                   const GrB_UnaryOp op,
                   const GrB_Matrix A,
                   const GrB_Descriptor desc)
{
    bool replace = false;
    if (desc != GrB_NULL)
        replace = desc->isSet(GrB_OUTP, GrB_REPLACE);

    // Ignore mask and accum for now
    GraphBLAS::apply(*C->m_matrix,
                     GraphBLAS::NoMask(),
                     GraphBLAS::NoAccumulate(),
                     UnaryAdapter(op),
                     *A->m_matrix,
                     replace);

    return GrB_SUCCESS;
}

//=============================================================================
// 4.3.9.1 - GrB_reduce

GrB_Info GrB_Vector_reduce_BinaryOp(GrB_Vector w,
                                    const GrB_Vector mask,
                                    const GrB_BinaryOp accum,
                                    const GrB_BinaryOp op,
                                    const GrB_Matrix A,
                                    const GrB_Descriptor desc)
{
    bool replace = false;
    if (desc != GrB_NULL)
        replace = desc->isSet(GrB_OUTP, GrB_REPLACE);

    if (accum != GrB_NULL)
    {
        if (desc != GrB_NULL && desc->isSet(GrB_INP0, GrB_TRAN))
        {
            GraphBLAS::reduce(*w->m_vec,
                              GraphBLAS::NoMask(),
                              BinaryAdapter(accum),
                              BinaryAdapter(op),
                              GraphBLAS::transpose(*A->m_matrix),
                              replace);
            return GrB_SUCCESS;
        }
        else
        {
            return GrB_PANIC;
        }
    }
    else
    {
        return GrB_PANIC;
    }


    return GrB_PANIC;
}


//=============================================================================
// ####  ##### ####  #   #  ####
// #   # #     #   # #   # #
// #   # ####  ####  #   # #  ##
// #   # #     #   # #   # #   #
// ####  ##### ####   ###   ###
//=============================================================================

void capi_print_matrix(GrB_Matrix matrix, char *label)
{
    GraphBLAS::print_matrix(std::cout, *(matrix->m_matrix), label);
    std::cout << std::endl;
}

void capi_print_vector(GrB_Vector vector, char *label)
{
    std::cout << label << " " << *vector->m_vec << std::endl;
}

#ifdef __cplusplus
}
#endif
