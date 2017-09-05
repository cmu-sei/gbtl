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

#include "FunctionInvocation.h"

//=============================================================================

TypeAdapter unary_call(GrB_Type returnType,
                       GrB_Type input1Type,
                       void *fn,
                       TypeAdapter input)
{
    // The function pointer is of type void(*)(void *, const void*, const void *);
    GrB_UnaryFunc func = (GrB_UnaryFunc)fn;

    TypeAdapter retVal = TypeAdapter(returnType);

    // The stored value (e.g. in input1) may NOT be of the type that
    // the function calls for (e.g. input1Type) so we always convert.

    func(retVal.getUnion(),
         input.convert(input1Type).getUnion());

    //std::cout << "unary_call input1 " << input << ", retVal: " << retVal << ", retType: " << returnType << std::endl;

    return retVal;
}

//=============================================================================

TypeAdapter binary_call(GrB_Type returnType,
                        GrB_Type input1Type,
                        GrB_Type input2Type,
                        void *fn,
                        TypeAdapter input1,
                        TypeAdapter input2)
{
    // The function pointer is of type void(*)(void *, const void*, const void *);
    GrB_BinaryFunc func = (GrB_BinaryFunc)fn;

    TypeAdapter retVal = TypeAdapter(returnType);

    // The stored value (e.g. in input1) may NOT be of the type that
    // the function calls for (e.g. input1Type) so we always convert.
    func(retVal.getUnion(),
         input1.convert(input1Type).getUnion(),
         input2.convert(input2Type).getUnion());
    return retVal;
}

TypeAdapter binary_call_choose_second(GrB_Type returnType,
                        GrB_Type input1Type,
                        GrB_Type input2Type,
                        void *fn,
                        TypeAdapter input1,
                        TypeAdapter input2)
{
    // We just return the second value
    return input2;
}



