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

#include <stdio.h>
#include <stdlib.h>

#include "graphblas.h"

#define check(x) \
    if ((status = x ) != GrB_SUCCESS ) \
    {   \
        printf("Operation failed: %s\n", #x); \
        return status;    \
    }


void bool_identity_fn(void *retVal, const void *val)
{
    *(GrB_BOOL *) retVal = *(GrB_BOOL *) val;
}

void int32_plus_fn(void *retVal, const void *input1, const void *input2)
{
    *(GrB_INT32 *) retVal = *(GrB_INT32 *) input1 + *(GrB_INT32 *) input2;
}

void fp32_plus_fn(void *retVal, const void *input1, const void *input2)
{
    *(GrB_FP32 *) retVal = *(GrB_FP32 *) input1 + *(GrB_FP32 *) input2;
}

void int32_times_fn(void *retVal, const void *input1, const void *input2)
{
    *(GrB_INT32 *) retVal = *(GrB_INT32 *) input1 * *(GrB_INT32 *) input2;
}

void fp32_times_fn(void *retVal, const void *input1, const void *input2)
{
    *(GrB_FP32 *) retVal = *(GrB_FP32 *) input1 * *(GrB_FP32 *) input2;
}

void fp32_mult_inv_fn(void *retVal, const void *val)
{
    *(GrB_FP32 *) retVal = (GrB_FP32) 1.0 / *(GrB_FP32 *) val;
    //printf("val: %f retVal: %f\n", *(GrB_FP32*)val, *(GrB_FP32*)retVal);
}


GrB_Info vertex_betweenness_centrality(float *result,
                                       GrB_Matrix A,
                                       GrB_Index *s,
                                       GrB_Index num_indicies)
{
    GrB_Info status = GrB_SUCCESS;

    printf("Computing betweenness centrality\n");

    capi_print_matrix(A, "Graph");

    GrB_Index nsver = num_indicies;
    if (nsver == 0)
    {
        return GrB_DIMENSION_MISMATCH;
    }

    printf("batch side (p): %lu\n", nsver);

    GrB_Index m;
    check(GrB_Matrix_nrows(&m, A));

    GrB_Index n;
    check(GrB_Matrix_ncols(&n, A));

    if (m != n)
    {
        return GrB_DIMENSION_MISMATCH;
    }

    printf("Num nodes (n): %lu\n", n);

    // ======

    // The current frontier for all BFS's (from all roots)
    // It is initialized to the out neighbors of the specified roots
    GrB_Matrix Frontier;
    check(GrB_Matrix_new(&Frontier, GrB_INT32_Type, nsver, n));
    check(GrB_extract(Frontier,           // C
                      GrB_NULL,           // Mask
                      GrB_NULL,           // accum
                      A,                  // A
                      s,                  // row_indicies
                      nsver,              // num_row_indicies
                      GrB_ALL,            // col_indicies
                      n,                  // num_col_indicies
                      GrB_NULL));         // desc
    capi_print_matrix(Frontier, "initial frontier");

    // NumSP holds number of shortest paths to a vertex from a given root
    // NumSP is initialized with the each starting root in 's':
    // NumSP[i,s[i]] = 1 where 0 <= i < nsver; implied zero elsewhere
    GrB_Matrix NumSP;

    GrB_Index *row_indicies = (GrB_Index *) malloc(sizeof(GrB_Index) * nsver);
    for (GrB_Index i = 0; i < nsver; ++i)
        row_indicies[i] = i;

    GrB_INT32 *one_values= (GrB_INT32 *) malloc(sizeof(GrB_INT32) * nsver);
    for (GrB_Index i = 0; i < nsver; ++i)
        one_values[i] = 1;

    check(GrB_Matrix_new(&NumSP, GrB_INT32_Type, nsver, n));
    check(GrB_Matrix_build_INT32(NumSP,                // C
                                 row_indicies,         // row_indices
                                 s,                    // col_indicies
                                 one_values,           // values
                                 nsver,                // nvals
                                 GrB_NULL));           // dup
    capi_print_matrix(NumSP, "initial NumSP");

    free(row_indicies);
    free(one_values);

    // ==================== BFS phase ====================
    // Placeholders for GraphBLAS operators

    printf("======= START BFS phase ======\n");

    // Hack.  We have a hard stop right now of 10
    GrB_Matrix Sigmas[10];
    int32_t d = 0;

    // For testing purpose we only allow 10 iterations so it doesn't
    // get into an infinite loop

    //=============

    GrB_UnaryOp bool_identity;
    check(GrB_UnaryOp_new(&bool_identity, GrB_BOOL_Type, GrB_BOOL_Type,
                          (void *) &bool_identity_fn));

    GrB_BinaryOp int32_plus;
    check(GrB_BinaryOp_new(&int32_plus, GrB_INT32_Type, GrB_INT32_Type,
                           GrB_INT32_Type, (void *) &int32_plus_fn))

    GrB_BinaryOp fp32_plus;
    check(GrB_BinaryOp_new(&fp32_plus, GrB_FP32_Type, GrB_FP32_Type,
                           GrB_FP32_Type, (void *) &fp32_plus_fn))

    GrB_Monoid int32_plus_monoid;
    check(GrB_Monoid_INT32_new(&int32_plus_monoid, int32_plus, 0));

    GrB_Monoid fp32_plus_monoid;
    check(GrB_Monoid_FP32_new(&fp32_plus_monoid, fp32_plus, 0));

    GrB_BinaryOp int32_times;
    check(GrB_BinaryOp_new(&int32_times, GrB_INT32_Type, GrB_INT32_Type,
                           GrB_INT32_Type, (void *) &int32_times_fn))

    GrB_BinaryOp fp32_times;
    check(GrB_BinaryOp_new(&fp32_times, GrB_FP32_Type, GrB_FP32_Type,
                           GrB_FP32_Type, (void *) &fp32_times_fn))

    GrB_UnaryOp fp32_mult_inv;
    check(GrB_UnaryOp_new(&fp32_mult_inv, GrB_FP32_Type, GrB_FP32_Type,
                          (void *) &fp32_mult_inv_fn));

    //=============

    GrB_Descriptor desc_replace;
    GrB_Descriptor_new(&desc_replace);
    GrB_Descriptor_set(desc_replace, GrB_OUTP, GrB_REPLACE);

    GrB_Descriptor desc_CompMask_OutReplace;
    GrB_Descriptor_new(&desc_CompMask_OutReplace);
    GrB_Descriptor_set(desc_CompMask_OutReplace, GrB_MASK, GrB_SCMP);
    GrB_Descriptor_set(desc_CompMask_OutReplace, GrB_OUTP, GrB_REPLACE);

    GrB_Descriptor desc_In0Tans_OutReplace;
    GrB_Descriptor_new(&desc_In0Tans_OutReplace);
    GrB_Descriptor_set(desc_In0Tans_OutReplace, GrB_INP0, GrB_TRAN);
    GrB_Descriptor_set(desc_In0Tans_OutReplace, GrB_OUTP, GrB_REPLACE);

    GrB_Descriptor desc_In1Tans_OutReplace;
    GrB_Descriptor_new(&desc_In1Tans_OutReplace);
    GrB_Descriptor_set(desc_In1Tans_OutReplace, GrB_INP1, GrB_TRAN);
    GrB_Descriptor_set(desc_In1Tans_OutReplace, GrB_OUTP, GrB_REPLACE);

    //=============

    GrB_Index nvals;
    check(GrB_Matrix_nvals(&nvals, Frontier));

    GrB_Semiring int32_arithmeticSemiring;
    GrB_Semiring_new(&int32_arithmeticSemiring, int32_plus_monoid, int32_times);

    GrB_Semiring fp32_arithmeticSemiring;
    GrB_Semiring_new(&fp32_arithmeticSemiring, fp32_plus_monoid, fp32_times);

    //===================================================================

    while (nvals > 0 && d < 10)
    {
        printf("------- BFS iteration %d --------\n", d);

        check(GrB_Matrix_new(&Sigmas[d], GrB_BOOL_Type, nsver, n));

        // Sigma[d] = (bool)
        check(GrB_apply(Sigmas[d],      // C
                        GrB_NULL,       // Mask
                        GrB_NULL,       // accum
                        bool_identity,  // op
                        Frontier,       // A
                        GrB_NULL));     // desc
        capi_print_matrix(Sigmas[d], "Sigma[d] = (bool)Frontier");

        // P = F + P
        // NOTE:  In the original code in bc.hpp it says the int is a monoid
        // but it's type is binary op, so I am just using that.
        check(GrB_Matrix_eWiseAdd_BinaryOp(NumSP,
                                           GrB_NULL,
                                           GrB_NULL,
                                           int32_plus,
                                           NumSP,
                                           Frontier,
                                           GrB_NULL));
        capi_print_matrix(NumSP, "NumSP");

        // F<!P> = F +.* A
        check(GrB_mxm(Frontier,                                  // C
                      NumSP,                                     // M
                      GrB_NULL,                                  // accum
                      int32_arithmeticSemiring,                  // op
                      Frontier,                                  // A
                      A,                                         // B
                      desc_CompMask_OutReplace));                // replace
        capi_print_matrix(Frontier, "New frontier");

        ++d;
        check(GrB_Matrix_nvals(&nvals, Frontier));
    }
    printf("======= END BFS phase =======\n");
    GrB_Descriptor_free(&desc_CompMask_OutReplace);

    // ================== backprop phase ==================

    GrB_Matrix NspInv;
    check(GrB_Matrix_new(&NspInv, GrB_FP32_Type, nsver, n));
    check(GrB_apply(NspInv,               // C
                    GrB_NULL,             // M
                    GrB_NULL,             // accum
                    fp32_mult_inv,        // op
                    NumSP,                // A
                    GrB_NULL));           // desc
    capi_print_matrix(NspInv, "(1 ./ P)");

    GrB_Matrix BCu;
    check(GrB_Matrix_new(&BCu, GrB_FP32_Type, nsver, n));
    check(GrB_Matrix_assign_FP32(BCu,                 // C
                                 GrB_NULL,            // Mask
                                 GrB_NULL,            // accum
                                 1.0f,                // val
                                 GrB_ALL,             // row_indicies
                                 nsver,               // n row indicies
                                 GrB_ALL,             // col_indicies
                                 n,                   // n col indicies
                                 GrB_NULL));          // desc
    capi_print_matrix(BCu, "U");

    printf("======= BEGIN BACKPROP phase =======\n");

    GrB_Matrix W;
    check(GrB_Matrix_new(&W, GrB_FP32_Type, nsver, n));

    for (int32_t i = d - 1; i > 0; --i)
    {
        printf("------- BACKPROP iteration %d --------\n", i);

        // W<Sigma[i]> = (1 ./ P) .* U
        check(GrB_Martrix_eWiseMult_BinaryOp(W,                     // C,
                                             Sigmas[i],             // Mask,
                                             GrB_NULL,              // accum,
                                             fp32_times,            // op,
                                             NspInv,                // A,
                                             BCu,                   // B,
                                             desc_replace));        // desc
        capi_print_matrix(W, "W<Sigma[i]> = (1 ./ P) .* U");

        // W<Sigma[i-1]> = A +.* W
        check(GrB_mxm(W,                            // C
                      Sigmas[i - 1],                // Mask
                      GrB_NULL,                     // accum
                      fp32_arithmeticSemiring,      // op
                      W,                            // A
                      A,                            // B
                      desc_In1Tans_OutReplace));    // desc
        capi_print_matrix(W, "W<Sigma[i-1]> = A +.* W");

        // U += W .* P
        check(GrB_Martrix_eWiseMult_BinaryOp(BCu,           // C
                                             GrB_NULL,      // Mask
                                             fp32_plus,     // accum
                                             fp32_times,    // op
                                             W,             // A
                                             NumSP,         // B
                                             GrB_NULL));    // desc
        capi_print_matrix(BCu, "U += W .* P");

        --d;
    }
    printf("======= END BACKPROP phase =======\n");
    capi_print_matrix(BCu, "BC Updates");

    GrB_Vector tmpResult;
    check(GrB_Vector_new(&tmpResult, GrB_FP32_Type, n));
    printf("nsver %d, nsver * -1.0: %f, n: %dn", (int) nsver, nsver * -1.0f, (int)n);
    check(GrB_Vector_assign_FP32(tmpResult,              // w
                                 GrB_NULL,               // mask
                                 GrB_NULL,               // accum
                                 nsver * -1.0f,          // val
                                 GrB_ALL,                // row_undicies
                                 n,                      // nindicies
                                 GrB_NULL));             // desc
    capi_print_vector(tmpResult, "prepped result:");

    check(GrB_Vector_reduce_BinaryOp(tmpResult,                  // w
                                     GrB_NULL,                   // mask
                                     fp32_plus,                  // accum
                                     fp32_plus,                  // op
                                     BCu,                        // A
                                     desc_In0Tans_OutReplace));  //desc

    // Fill in result
    printf("RESULT: ");
    bool first = true;
    for (GrB_Index i = 0; i < n; ++i)
    {
        GrB_Vector_extractElement_FP32(&result[i], tmpResult, i);
        printf("%s%f", first ? "" : ",", result[i]);
        first = false;
    }
    printf("\n");

    return GrB_SUCCESS;
}