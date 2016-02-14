#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <iostream>
int main(){
    int A_keys[6] = {1, 3, 5, 7, 9, 11};
    int A_vals[6] = {0, 0, 0, 0, 0, 0};
    int B_keys[13] = {1, 1, 2, 3, 5, 8, 13,0,0,0,0,0,0};
    int B_vals[13] = {1, 1, 1, 1, 1, 1, 1,0,0,0,0,0,0};
    int keys_result[13];
    int vals_result[13];
    thrust::pair<int*,int*> end =
        thrust::merge_by_key(thrust::host,
                A_keys, A_keys + 6,
                B_keys, B_keys + 7,
                A_vals,
                B_vals,
                B_keys,
                B_vals);
    for(int i=0;i<13;i++){
        //std::cout<<keys_result[i]<<" "<<vals_result[i]<<std::endl;
        std::cout<<B_vals[i]<<" "<<B_keys[i]<<std::endl;
    }
    return 0;

}
