#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <graphblas/graphblas.hpp>

using namespace grb;
// call in loop:

using ITYPE  = typename grb::IndexType;
using VTYPE  = int32_t;
// Pull kernel for non-unit-weight-based pagerank
inline void pr_dot_weights_kernel(
    ITYPE *IA, ITYPE *JA, VTYPE *VA, ITYPE N,
    // Algo data
    VTYPE *scores, VTYPE *out_scores)
{
    // #pragma omp for
    for (uint32_t i = 0; i < N; i++)
    {
        ITYPE vidx = i;
        ITYPE st = *(IA + vidx);
        ITYPE nd = *(IA + vidx + 1);

        VTYPE tmp = 0;
        bool set = false;

        for (uint32_t j = st; j < nd; j++)
        {
            ITYPE neighbor = JA[j];
            // if (scores[neighbor] == 0) continue;
            // std::cout << neighbor << ", ";
            set = true;
            tmp += (VA[j] * scores[neighbor]);
        }
        if (set)
        {
            out_scores[i] = tmp;
            // std::cout << std::endl;
        }
    }
    return;
}

// Pull kernel for non-unit-weight-based pagerank
inline void pr_dot_weights_kernel_bmask(
    ITYPE *IA, ITYPE *JA, VTYPE *VA, ITYPE N,
    // Algo data
    VTYPE *scores, VTYPE *out_scores,
    std::vector<bool>& b1, std::vector<bool>& b2)
{
    // #pragma omp for
    for (uint32_t i = 0; i < N; i++)
    {
        ITYPE vidx = i;
        ITYPE st = *(IA + vidx);
        ITYPE nd = *(IA + vidx + 1);

        VTYPE tmp = 0;
        bool set = false;

        for (uint32_t j = st; j < nd; j++)
        {
            if (!b1[JA[j]]) continue;
            ITYPE neighbor = JA[j];
            // std::cout << neighbor << ", ";
            tmp += (VA[j] * scores[neighbor]);
            set = true;
        }
        if (set)
        {
            out_scores[i] = tmp;
            b2[i] = true;
            // std::cout << std::endl;
        }
    }
    return;
}

// Stripped down pull bfs kernel with weights...so it's sssp
inline void dot_kernel_bfs(
    ITYPE *IA, ITYPE *JA, VTYPE *VA, ITYPE N,
    // Algo data
    VTYPE *parents_in, VTYPE *parents_out,
    ITYPE * front_in, ITYPE * front_out,
    ITYPE front_sz, ITYPE & front_out_sz)
{
    // #pragma omp for
    front_out_sz = 0;
    for (uint32_t i = 0; i < N; i++)
    {
        ITYPE vidx = i;
        ITYPE st = *(IA + vidx);
        ITYPE nd = *(IA + vidx + 1);
        ITYPE fidx = 0;

        VTYPE tmp = 0;
        VTYPE j = st;
        bool set = false;

        // Dot sparse parents with sparse row
        while (j < nd && fidx < front_sz)
        {
            if (front_in[fidx] == JA[j])
            {
                ITYPE neighbor = JA[j];
                // std::cout << neighbor << ", ";
                tmp += (VA[j] * parents_in[neighbor]);
                set = true;
                j++;
                fidx++;
            }
            else if (front_in[fidx] < JA[j])
            {
                fidx++;
            }
            else
            {
                j++;
            }
        }
        if (set){
            parents_out[i] = tmp;
            front_out[front_out_sz++] = i;
            // std::cout << std::endl;
        }
    }
    return;
}


/*
inline void axpy_kernel_bfs(
    VTYPE *IA, VTYPE *JA, VTYPE *VA, VTYPE N,
    // Algo data
    F_TYPE *parents_in, F_TYPE *parents_out,
    VTYPE * front_in, VTYPE * front_out,
    VTYPE front_sz, VTYPE & front_out_sz)
{
    // Assume parents_out is clear
	for (VTYPE idx = 0; idx < front_sz; idx++){
		VTYPE vidx = front_in[idx];
		VTYPE * st = JA + IA[vidx];
		VTYPE * nd = JA + IA[vidx+1];
		uint32_t * val_ptr = VA + IA[vidx];
		
		uint32_t x_val = parents_in[vidx];

		for (VTYPE * e_ptr = st; e_ptr < nd; e_ptr++, val_ptr++){
			uint32_t A_val = *val_ptr;
			// Mult op
            uint32_t tmp_val = A_val * x_val;
            // Reduce (no accum) op
            if (parents_out[*e_ptr]==0)
            {
                front_out[front_out_sz++] = *e_ptr;
            }
            parents_out[*e_ptr] += tmp_val;
		}
	}
}
*/