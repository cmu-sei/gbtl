//
// Created by aomellinger on 5/15/17.
//

#include <stdio.h>
#include "bcc.h"

int main(int argc, char **argv)
{
    int ncols = 8;
    GrB_Info info;

    GrB_Matrix betweenness;
    if ((info = GrB_Matrix_new(&betweenness, GrB_FP64_Type, 8, 8)) != GrB_SUCCESS)
    {
        printf("Failed to construct matrix! %d", info);
        return -1;
    }

    GrB_Index row_indicies[] = {0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6};
    GrB_Index col_indicies[] = {1, 2, 3, 2, 4, 4, 2, 4, 5, 6, 7, 7};
    GrB_FP64 values[]        = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    if ((info = GrB_Matrix_build_FP64(betweenness, row_indicies, col_indicies, values, 12, NULL)) != GrB_SUCCESS)
    {
        printf("Failed to build matrix! %d", info);
        return -1;
    }

    // ============

    float result[ncols];
    GrB_Index seed_set[] = {0};

    if ((info = vertex_betweenness_centrality((float *)&result, betweenness, seed_set, 1)) != GrB_SUCCESS)
    {
        printf("Failed to execute vertex betweenness centrality: %d", info);
        return -1;
    }

    // ==== Print output
    printf("Result: %f", result[0]);
    for (int i = 1; i < ncols; ++i)
        printf(", %f", result[i]);
    printf("\n");

    printf("Done\n");
    return 0;
}

