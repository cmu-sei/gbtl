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
#include <memory>

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

// Helper function used to setup the mask. Move to a utility file.
GraphBLAS::Matrix<TypeAdapter> *prepareMask(const GrB_Matrix C,
                                            const GrB_Matrix Mask,
                                            const GrB_Descriptor desc)
{
    // FullMask.  Return a constant mask.
    if ((void *) Mask == GrB_NULL)
    {
        return new GraphBLAS::Matrix<TypeAdapter>(
                C->m_matrix->nrows(), C->m_matrix->ncols(), 1);
    }

    // They gave us a mask AND they set to structural complement
    if (desc != GrB_NULL && desc->isSet(GrB_MASK, GrB_SCMP))
    {
        GraphBLAS::Matrix<TypeAdapter> * new_matrix =
            new GraphBLAS::Matrix<TypeAdapter>(C->m_matrix->nrows(), C->m_matrix->ncols());
        new_matrix->build_complement_from(*Mask->m_matrix);
        return new_matrix;
    }

    return Mask->m_matrix;
}

void cleanupMask(const GrB_Matrix Mask,
                 const GrB_Descriptor desc,
                 GraphBLAS::Matrix<TypeAdapter> *usedMask)
{
    // If either case from above, delete the mask
    if ((void *) Mask == GrB_NULL ||
        (desc != GrB_NULL && desc->isSet(GrB_MASK, GrB_SCMP)))
    {
        delete usedMask;
    }
}

//=============================================================================
// 4.2.1.2
GrB_Info GrB_UnaryOp_new(GrB_UnaryOp   *unary_op,
                         GrB_UnaryFunc  unary_func,
                         GrB_Type       d_out,
                         GrB_Type       d_in)
{
    *unary_op = new GrB_UnaryOp_Struct(unary_func, d_out, d_in);
    return GrB_SUCCESS;
}

//=============================================================================
// 4.2.1.3
GrB_Info GrB_BinaryOp_new(GrB_BinaryOp   *binary_op,
                          GrB_BinaryFunc  binary_func,
                          GrB_Type        d_out,
                          GrB_Type        d_in1,
                          GrB_Type        d_in2)
{
    *binary_op = new GrB_BinaryOp_Struct(binary_func, d_out, d_in1, d_in2);
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

    GraphBLAS::Matrix<TypeAdapter> *mask_matrix = prepareMask(C, Mask, desc);

    if (desc != GrB_NULL &&
        desc->isSet(GrB_INP0, GrB_TRAN) &&
        desc->isSet(GrB_INP1, GrB_TRAN))
    {
        GraphBLAS::mxm(*C->m_matrix,
                       *mask_matrix,
                       BinaryAdapter(accum),
                       SemiringAdapter(op),
                       GraphBLAS::transpose(*A->m_matrix),
                       GraphBLAS::transpose(*B->m_matrix),
                       replace);
        return GrB_SUCCESS;
    }
    else if (desc != GrB_NULL && desc->isSet(GrB_INP0, GrB_TRAN))
    {
        GraphBLAS::mxm(*C->m_matrix,
                       *mask_matrix,
                       BinaryAdapter(accum),
                       SemiringAdapter(op),
                       GraphBLAS::transpose(*A->m_matrix),
                       *B->m_matrix,
                       replace);
        return GrB_SUCCESS;
    }
    else if (desc != GrB_NULL && desc->isSet(GrB_INP1, GrB_TRAN))
    {
        GraphBLAS::mxm(*C->m_matrix,
                       *mask_matrix,
                       BinaryAdapter(accum),
                       SemiringAdapter(op),
                       *A->m_matrix,
                       GraphBLAS::transpose(*B->m_matrix),
                       replace);
        return GrB_SUCCESS;
    }
    else
    {
        GraphBLAS::mxm(*C->m_matrix,
                       *mask_matrix,
                       BinaryAdapter(accum),
                       SemiringAdapter(op),
                       *A->m_matrix,
                       *B->m_matrix,
                       replace);
        return GrB_SUCCESS;
    }


//
//    if (Mask == GrB_NULL)
//    {
//        if (accum == GrB_NULL)
//        {
//            if (desc != GrB_NULL &&
//                desc->isSet(GrB_INP0, GrB_TRAN) &&
//                desc->isSet(GrB_INP1, GrB_TRAN))
//            {
//                GraphBLAS::mxm(*C->m_matrix,
//                               GraphBLAS::NoMask(),
//                               GraphBLAS::NoAccumulate(),
//                               SemiringAdapter(op),
//                               GraphBLAS::transpose(*A->m_matrix),
//                               GraphBLAS::transpose(*B->m_matrix),
//                               replace);
//
//                return GrB_SUCCESS;
//            }
//            else if (desc != GrB_NULL && desc->isSet(GrB_INP0, GrB_TRAN))
//            {
//                return GrB_PANIC;
//            }
//            else if (desc != GrB_NULL && desc->isSet(GrB_INP1, GrB_TRAN))
//            {
//                return GrB_PANIC;
//            }
//            else
//            {
//                GraphBLAS::mxm(*C->m_matrix,
//                               GraphBLAS::NoMask(),
//                               GraphBLAS::NoAccumulate(),
//                               SemiringAdapter(op),
//                               *A->m_matrix,
//                               *B->m_matrix,
//                               replace);
//
//                return GrB_SUCCESS;
//            }
//        }
//        else
//        {
//            // @TODO: Build out all four
//            return GrB_PANIC;
//        }
//    }
//    else if (desc != GrB_NULL && desc->isSet(GrB_MASK, GrB_SCMP))
//    {
//        if (accum == GrB_NULL)
//        {
//            if (desc->isSet(GrB_INP0, GrB_TRAN) &&
//                desc->isSet(GrB_INP1, GrB_TRAN))
//            {
//                return GrB_SUCCESS;
//            }
//            else if (desc->isSet(GrB_INP0, GrB_TRAN))
//            {
//                return GrB_PANIC;
//            }
//            else if (desc->isSet(GrB_INP1, GrB_TRAN))
//            {
//                return GrB_PANIC;
//            }
//            else
//            {
//                GraphBLAS::mxm(*C->m_matrix,
//                               GraphBLAS::complement(*Mask->m_matrix),
//                               GraphBLAS::NoAccumulate(),
//                               SemiringAdapter(op),
//                               *A->m_matrix,
//                               *B->m_matrix,
//                               replace);
//                return GrB_SUCCESS;
//            }
//        }
//        else
//        {
//            // @TODO: Build out all four
//            return GrB_PANIC;
//        }
//
//        // @todo: Add all four transpose options times op options
//    }
//    else
//    {
//        if (accum == GrB_NULL)
//        {
//            if (desc != GrB_NULL && desc->isSet(GrB_INP0, GrB_TRAN) &&
//                desc->isSet(GrB_INP1, GrB_TRAN))
//            {
//                return GrB_PANIC;
//            }
//            else if (desc != GrB_NULL && desc->isSet(GrB_INP0, GrB_TRAN))
//            {
//                return GrB_PANIC;
//            }
//            else if (desc != GrB_NULL && desc->isSet(GrB_INP1, GrB_TRAN))
//            {
//                GraphBLAS::mxm(*C->m_matrix,
//                               *Mask->m_matrix,
//                               GraphBLAS::NoAccumulate(),
//                               SemiringAdapter(op),
//                               *A->m_matrix,
//                               GraphBLAS::transpose(*B->m_matrix),
//                               replace);
//                return GrB_SUCCESS;
//            }
//            else
//            {
//                GraphBLAS::mxm(*C->m_matrix,
//                               *Mask->m_matrix,
//                               GraphBLAS::NoAccumulate(),
//                               SemiringAdapter(op),
//                               *A->m_matrix,
//                               *B->m_matrix,
//                               replace);
//                return GrB_SUCCESS;
//            }
//        }
//        else
//        {
//            // @TODO: Build out all four
//            return GrB_PANIC;
//        }
//    }
}

//=============================================================================

// 4.3.4.2 - Semiring variant
//// @TODO:
//GrB_Info GrB_Martrix_eWiseMult_Semiring(GrB_Matrix C,
//                                        const GrB_Matrix Mask,
//                                        const GrB_BinaryOp accum,
//                                        const GrB_BinaryOp op,
//                                        const GrB_Matrix A,
//                                        const GrB_Matrix B,
//                                        const GrB_Descriptor desc)
//{
//
//}


// 4.3.4.2 - BinaryOp Variant
GrB_Info GrB_Martrix_eWiseMult_BinaryOp(GrB_Matrix C,
                                        const GrB_Matrix Mask,
                                        const GrB_BinaryOp accum,
                                        const GrB_BinaryOp op,
                                        const GrB_Matrix A,
                                        const GrB_Matrix B,
                                        const GrB_Descriptor desc)
{
    if ((void *) op == GrB_NULL)
    {
        printf("!!GrB_Martrix_eWiseMult_BinaryOp: operator can't be GrB_NULL!");
        return GrB_PANIC;
    }

    bool replace = false;
    if (desc != GrB_NULL)
        replace = desc->isSet(GrB_OUTP, GrB_REPLACE);

    // @TODO: Implement transpose
    if (desc != GrB_NULL && desc->isSet(GrB_INP0, GrB_TRAN))
        return GrB_PANIC;

    // @TODO: Implement transpose
    if (desc != GrB_NULL && desc->isSet(GrB_INP1, GrB_TRAN))
        return GrB_PANIC;

    GraphBLAS::Matrix<TypeAdapter> *mask_matrix = prepareMask(C, Mask, desc);

    // The real call
    GraphBLAS::eWiseMult(*C->m_matrix,
                         *mask_matrix,
                         BinaryAdapter(accum),
                         BinaryAdapter(op),
                         *A->m_matrix,
                         *B->m_matrix,
                         replace);

    cleanupMask(Mask, desc, mask_matrix);

    return GrB_SUCCESS;
}

// 4.3.4.2 - Monoid Variant
// @TODO:
//GrB_Info GrB_Martrix_eWiseMult_Monoid(GrB_Matrix C,
//                                        const GrB_Matrix Mask,
//                                        const GrB_BinaryOp accum,
//                                        const GrB_Monoid op,
//                                        const GrB_Matrix A,
//                                        const GrB_Matrix B,
//                                        const GrB_Descriptor desc)
//{
//
//}

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

    // @TODO: Implement transpose
    if (desc != GrB_NULL && desc->isSet(GrB_INP0, GrB_TRAN))
        return GrB_PANIC;

    // @TODO: Implement transpose
    if (desc != GrB_NULL && desc->isSet(GrB_INP1, GrB_TRAN))
        return GrB_PANIC;

    GraphBLAS::Matrix<TypeAdapter> *mask_matrix = prepareMask(C, Mask, desc);

    GraphBLAS::eWiseAdd(*C->m_matrix,
                        *mask_matrix,
                        BinaryAdapter(accum),
                        BinaryAdapter(op),
                        *A->m_matrix,
                        *B->m_matrix,
                        replace);

    cleanupMask(Mask, desc, mask_matrix);

    return GrB_SUCCESS;
}


//=============================================================================

// 4.3.6.2 - IMPL_STATUS: ALPHA
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

    // @TODO: Implement transpose
    if (desc != GrB_NULL && desc->isSet(GrB_INP0, GrB_TRAN))
        return GrB_PANIC;

    GraphBLAS::Matrix<TypeAdapter> *mask_matrix = prepareMask(C, Mask, desc);

    GraphBLAS::extract(*C->m_matrix,
                       *mask_matrix,
                       BinaryAdapter(accum),
                       *A->m_matrix,
                       GraphBLAS::IndexProxy(row_indices, nrows),
                       GraphBLAS::IndexProxy(col_indices, ncols),
                       replace);

    cleanupMask(Mask, desc, mask_matrix);

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

    GraphBLAS::assign(*w->m_vec,
                       GraphBLAS::NoMask(),
                       BinaryAdapter(accum),
                       val,
                       GraphBLAS::IndexProxy(row_indicies, nindicies),
                       replace);

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
// 4.3.8.2 - IMPL_STATUS: ALPHA
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

    // @TODO: Implement transpose
    if (desc != GrB_NULL && desc->isSet(GrB_INP0, GrB_TRAN))
        return GrB_PANIC;

    // Setup mask
    GraphBLAS::Matrix<TypeAdapter> *mask_matrix = prepareMask(C, Mask, desc);

    // Ignore mask and accum for now
    GraphBLAS::apply(*C->m_matrix,
                     *mask_matrix,
                     BinaryAdapter(accum),
                     UnaryAdapter(op),
                     *A->m_matrix,
                     replace);

    cleanupMask(Mask, desc, mask_matrix);

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

    if (mask != GrB_NULL)
        return GrB_PANIC;

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
