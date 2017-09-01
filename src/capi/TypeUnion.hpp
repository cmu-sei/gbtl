//
// Created by aomellinger on 5/22/17.
//

#ifndef SRC_TYPEUNION_HPP
#define SRC_TYPEUNION_HPP

#include "graphblas.h"

union GrB_TypeUnion {
    GrB_BOOL _bool;
    GrB_INT32 _int32;
    GrB_FP32 _fp32;
    GrB_FP64 _fp64;
};

#endif //SRC_TYPEUNION_HPP
