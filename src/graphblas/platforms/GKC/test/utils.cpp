#include "utils.h"

unsigned long long rdtsc()
{
 unsigned a, d;

 __asm__ volatile("rdtsc" : "=a" (a), "=d" (d));

 return ((unsigned long long)a) | (((unsigned long long)d) << 32);
}

int cmp_u32(const void* p1, const void* p2){
  uint32_t v1 = *(uint32_t*)p1;
  uint32_t v2 = *(uint32_t*)p2;
  if (v1 < v2) return -1;
  if (v1 > v2) return 1;
  return 0;
}

#define BROOM_SIZE 1024*1024*30 // 30 MB
auto broom = std::vector<char>(BROOM_SIZE);
void clean_caches(){
#pragma omp parallel
  {
    for (uint32_t i = 0; i < BROOM_SIZE; i++)
    {
      broom[i] = broom[i] + 2;
    }
  }
}

char* truncate_fname(char* input_fname){
	char *output = (char*) malloc(256);
	uint32_t MAX_LEN = 255;
	uint32_t idx=0; 
	while (idx < 255 && input_fname[idx] != '\0') idx++;
	MAX_LEN = (idx-1 >= 0) ? idx-1 : 0;	
	uint32_t s_idx = 0, e_idx=MAX_LEN;
	// find start (find last / or take beginning):
	idx = 0;
	while (idx < MAX_LEN-1)
	{
		if (input_fname[idx] == '/') s_idx = idx+1;
		idx++;	
	}
	// Find end (find first '.' from end or end):
	idx = MAX_LEN;
	while (idx > 0) 
	{
		if (input_fname[idx] == '.') break;
		idx--;
	}
	e_idx = idx;
	strncpy(output, input_fname+s_idx, e_idx - s_idx);
 output[e_idx - s_idx] = '\0';
	return output;
}

