//
// Created by aomellinger on 5/23/17.
//

#ifndef SRC_FUNCTIONSUPPORT_H
#define SRC_FUNCTIONSUPPORT_H

#include "graphblas.h"
#include "TypeAdapter.hpp"

TypeAdapter unary_call(GrB_UnaryFunc unary_func,
                       GrB_Type      returnType,
                       GrB_Type      input1Type,
                       TypeAdapter   input);

TypeAdapter binary_call(GrB_BinaryFunc binary_func,
                        GrB_Type       returnType,
                        GrB_Type       input1Type,
                        GrB_Type       input2Type,
                        TypeAdapter    input1,
                        TypeAdapter    input2);

#endif //SRC_FUNCTIONSUPPORT_H
