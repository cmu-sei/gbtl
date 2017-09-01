//
// Created by aomellinger on 5/23/17.
//

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
