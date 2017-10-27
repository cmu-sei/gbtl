//
// Created by aomellinger on 5/15/17.
//

#ifndef GRAPHBLAS_H
#define GRAPHBLAS_H

// This is a total hack.  We need to break this out to different parts.

#include <sys/types.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GrB_SUCCESS,
    GrB_DIMENSION_MISMATCH,
    GrB_INVALID_VALUE,
    GrB_PANIC,
    GrB_TYPE_MISMATCH
} GrB_Info;

// I want these to be const *, but I get const violations when I do this
//extern const void *GrB_NULL;
extern void *GrB_NULL;

// What is the right type to use that matches the C++ one?
//typedef u_int64_t GrB_Index;
typedef unsigned long GrB_Index;

#define GrB_ALL ((GrB_Index *)NULL)

typedef bool GrB_BOOL;
// GrB_INT8 int8
// GrB_UINT8 uint8
// GrB_INT16 int16
// GrB_UINT16 uint16
typedef int32_t GrB_INT32;
// GrB_UINT32 uint32
// GrB_INT64 int64
// GrB_UINT64 uint64
typedef float GrB_FP32;
typedef double GrB_FP64;

// How are we supposed to do user defined types this way??
typedef enum
{
    // @TODO HACK!! - Unset isn't a valid public type.  We use it internally
    // to mark a type that hasn't been set yet.  When we move to UDT's we'll
    // need to handle this in a different way anyway.
    GrB_Unset_Type         = 0,
    GrB_BOOL_Type,
    GrB_INT32_Type,
    GrB_FP32_Type,
    GrB_FP64_Type
} GrB_Type;


// The types are all opaque
typedef struct GrB_UnaryOp_Struct *GrB_UnaryOp;
typedef struct GrB_BinaryOp_Struct *GrB_BinaryOp;
typedef struct GrB_Monoid_Struct *GrB_Monoid;
typedef struct GrB_Semiring_Struct *GrB_Semiring;
typedef struct GrB_Matrix_Struct *GrB_Matrix;
typedef struct GrB_Vector_Struct *GrB_Vector;
typedef struct GrB_Descriptor_Struct *GrB_Descriptor;


// Descriptor parts 3.7
typedef enum
{
    GrB_OUTP,           // Output
    GrB_INP0,           // First input
    GrB_INP1,           // Second input
    GrB_MASK            // Mask input
} GrB_Field;

typedef enum
{
    GrB_SCMP,           // Structural complement
    GrB_TRAN,           // Transpose
    GrB_REPLACE         // Clear the output before assigning computed values.
} GrB_Value;

// @TODO: Look to see if the spec passes these into the new functions
typedef void (* GrB_UnaryFunc)(void *, const void *);
typedef void (* GrB_BinaryFunc)(void *, const void *, const void *);


#define GrB_UNARY_ADD ((GrBUnaryOp) 1000);


// 4.2.1.1
// Huh??
//GrB_Info GrB_Type_new(GrB_Type *utype,
//<type> ctype);

//=============================================================================
// 4.2.1.2
// d_in = arg1, d_out = result
GrB_Info GrB_UnaryOp_new(GrB_UnaryOp   *unary_op,
                         GrB_UnaryFunc  unary_func,
                         GrB_Type       d_out,
                         GrB_Type       d_in);

//=============================================================================
// 4.2.1.3
GrB_Info GrB_BinaryOp_new(GrB_BinaryOp   *binary_op,
                          GrB_BinaryFunc  binary_func,
                          GrB_Type        d_out,
                          GrB_Type        d_in1,
                          GrB_Type        d_in2);

//=============================================================================
// 4.2.1.4
// NONPOLYMORPHIC INTERFACES
GrB_Info GrB_Monoid_INT32_new(GrB_Monoid *monoid,
                              GrB_BinaryOp binary_op,
                              GrB_INT32 identity);

GrB_Info GrB_Monoid_FP32_new(GrB_Monoid *monoid,
                             GrB_BinaryOp binary_op,
                             GrB_FP32 identity);

//=============================================================================
// 4.2.1.5
GrB_Info GrB_Semiring_new(GrB_Semiring *semiring,
                          GrB_Monoid add_op,
                          GrB_BinaryOp mul_op);

//=============================================================================
// 4.2.2.1
GrB_Info GrB_Vector_new(GrB_Vector *v,
                        GrB_Type d,
                        GrB_Index nsize);


//=============================================================================
// 4.2.2.1
GrB_Info GrB_Vector_extractElement_FP32(GrB_FP32 *val,
                                        const GrB_Vector u,
                                        GrB_Index index);

//=============================================================================
// 4.2.3.1
GrB_Info GrB_Matrix_new(GrB_Matrix *A,
                        GrB_Type d,
                        GrB_Index nrows,
                        GrB_Index ncols);

//=============================================================================
// 4.2.3.4

GrB_Info GrB_Matrix_nrows(GrB_Index *nrows,
                          const GrB_Matrix A);

//=============================================================================
// 4.2.3.5

GrB_Info GrB_Matrix_ncols(GrB_Index *ncols,
                          const GrB_Matrix A);


//=============================================================================
// 4.2.3.6

GrB_Info GrB_Matrix_nvals(GrB_Index *nvals,
                         const GrB_Matrix A);

//=============================================================================
// 4.2.3.7
GrB_Info GrB_Matrix_build_FP64(GrB_Matrix C,
                          const GrB_Index *row_indices,
                          const GrB_Index *col_indices,
                          const GrB_FP64 *values,
                          GrB_Index nvals,
                          const GrB_BinaryOp dup);

GrB_Info GrB_Matrix_build_INT32(GrB_Matrix C,
                               const GrB_Index *row_indices,
                               const GrB_Index *col_indices,
                               const GrB_INT32 *values,
                               GrB_Index nvals,
                               const GrB_BinaryOp dup);

//=============================================================================
// 4.2.4 Descriptors

// 4.2.4.1
GrB_Info GrB_Descriptor_new(GrB_Descriptor *desc);

// 4.2.4.2
GrB_Info GrB_Descriptor_set(GrB_Descriptor desc,
                            GrB_Field field,
                            GrB_Value val);


//=============================================================================
// 4.2.5

//GrB_Info GrB_free(GrB_Object *obj);
GrB_Info GrB_UnaryOp_free(GrB_UnaryOp *obj);
GrB_Info GrB_BinaryOp_free(GrB_BinaryOp *obj);
GrB_Info GrB_Monoid_free(GrB_Monoid *obj);
GrB_Info GrB_Semiring_free(GrB_Semiring *obj);
//GrB_Info GrB_Vector_free(GrB_Vector *obj);
GrB_Info GrB_Matrix_free(GrB_Matrix *obj);
GrB_Info GrB_Descriptor_free(GrB_Descriptor *obj);

//=============================================================================
// 4.3.1
GrB_Info GrB_mxm(GrB_Matrix C,
                 const GrB_Matrix Mask,
                 const GrB_BinaryOp accum,
                 const GrB_Semiring op,
                 const GrB_Matrix A,
                 const GrB_Matrix B,
                 const GrB_Descriptor desc);


//=============================================================================
// 4.3.4.2

// Binary op variant
GrB_Info GrB_Martrix_eWiseMult_BinaryOp(GrB_Matrix               C,
                                        const GrB_Matrix         Mask,
                                        const GrB_BinaryOp       accum,
                                        const GrB_BinaryOp       op,
                                        const GrB_Matrix         A,
                                        const GrB_Matrix         B,
                                        const GrB_Descriptor     desc);


//=============================================================================
// 4.3.5.2
GrB_Info GrB_Matrix_eWiseAdd_BinaryOp(GrB_Matrix C,
                      const GrB_Matrix Mask,
                      const GrB_BinaryOp accum,
                      const GrB_BinaryOp op,
                      const GrB_Matrix A,
                      const GrB_Matrix B,
                      const GrB_Descriptor desc);

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
                     const GrB_Descriptor desc);

//=============================================================================
// 4.3.7.5 - GrB_assign

GrB_Info GrB_Vector_assign_FP32(GrB_Vector w,
                                const GrB_Vector mask,
                                const GrB_BinaryOp accum,
                                const GrB_FP32 val,
                                const GrB_Index *row_indicies,
                                const GrB_Index nindicies,
                                const GrB_Descriptor desc);


//=============================================================================
// 4.3.7.6 - GrB_assign

// Matrix FP32 VARIANT
GrB_Info GrB_Matrix_assign_FP32(GrB_Matrix C,
                               const GrB_Matrix Mask,
                               const GrB_BinaryOp accum,
                               const GrB_FP32 val,
                               const GrB_Index *row_indicies,
                               const GrB_Index nrows,
                               const GrB_Index *col_indicies,
                               const GrB_Index ncols,
                               const GrB_Descriptor desc);

//=============================================================================
// 4.3.8.2
GrB_Info GrB_apply(GrB_Matrix C,
                   const GrB_Matrix Massk,
                   const GrB_BinaryOp accum,
                   const GrB_UnaryOp op,
                   const GrB_Matrix A,
                   const GrB_Descriptor desc);


//=============================================================================
// 4.3.9.1 - GrB_reduce

GrB_Info GrB_Vector_reduce_BinaryOp(GrB_Vector w,
                                    const GrB_Vector mask,
                                    const GrB_BinaryOp accum,
                                    const GrB_BinaryOp op,
                                    const GrB_Matrix A,
                                    const GrB_Descriptor desc);


//=============================================================================
// ####  ##### ####  #   #  ####
// #   # #     #   # #   # #
// #   # ####  ####  #   # #  ##
// #   # #     #   # #   # #   #
// ####  ##### ####   ###   ###
//=============================================================================

void capi_print_matrix(GrB_Matrix matrix, char *label);
void capi_print_vector(GrB_Vector vector, char *label);

#ifdef __cplusplus
}
#endif

#endif //SRC_GRAPHBLAS_H_H
