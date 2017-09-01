//
// Created by aomellinger on 5/15/17.
//

#ifndef SRC_BC_H
#define SRC_BC_H

#include "graphblas.h"

/**
 * @param result        Pre-sized area (ncols) in which to place values.
 * @param matrix        The input matrix.
 * @param s             The set of source vertex indices from which to compute
 *                        BC contributions
 * @param num_indicies  The number of source verticies provided.
 * @return GrB_Info result.
 */
GrB_Info vertex_betweenness_centrality(float *result,
                                  GrB_Matrix matrix,
                                  GrB_Index *s,
                                  GrB_Index num_indicies);

#endif //SRC_BC_H
