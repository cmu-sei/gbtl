//
// Created by aomellinger on 5/15/17.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bcc.h"

#define EPSILON 0.001

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


    // Check the values
    float correct_result[] = { 0.000000, 1.333333, 1.333333, 1.333333, 3.000000, 0.500000, 0.500000, 0.000000 };
    bool found_error = false;
    for (int i = 0; i < ncols; ++i)
    {
        if (fabs(result[i] - correct_result[i]) > EPSILON)
        {
            printf("ERROR at entry %d.  Expected %f, found %f\n", i,
                   correct_result[i], result[i]);
            found_error = true;
        }
    }

    if (found_error)
        printf(">>>\n>>>>>>>>>>>>>>>>>>> FAILURE :( :( :( >>>>>>>>>>>>>>>>>>>>>>>\n>>>\n");
    else
        printf(">>>\n>>>>>>>>>>>>>>>>>>> SUCCESS :) :) :) >>>>>>>>>>>>>>>>>>>>>>>\n>>>\n");
    return 0;
}

