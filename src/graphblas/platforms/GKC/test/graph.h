/*This file contains utitlity functions to operate on
adjaceny matrices and frontier vectors*/
#ifndef GRAPH_HEADER
#define GRAPH_HEADER
#include <stdlib.h>
#include <stdbool.h>
#include <memory.h>
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <map>
#include <cassert>
#include <cmath>
#include "utils.h"
#include <sys/stat.h>
#include <limits>
#include <atomic>

/* 
// for aligned_alloc on OSX:
#ifdef __APPLE__
#include <boost/align/aligned_alloc.hpp>
using boost::alignment::aligned_alloc;
#endif
*/

uint64_t read_galois_size(const char * filename, 
		uint64_t& nodes, uint64_t& edges, uint64_t& edge_size, bool& weighted);
uint64_t read_galois_data(const char * filename, 
		uint32_t * IA, uint32_t * JA, uint32_t * VA);

#ifndef VTYPE
#define VTYPE uint32_t
#endif

template <typename T>
class VertexDatum{
	public:
		VTYPE idx;
		T val;
	VertexDatum(VTYPE index, T value){
		idx = index;
		val = value;
	}
	bool operator<(VertexDatum<T> const & a){
		return val < a.val;
	}
};

template <typename T>
void print_top_k(T * values, size_t num_elems, size_t k){
	k = MIN(k, num_elems);
	printf("Printing top %lu elements\n", k);
	// Copy values to class containing original index and value:
	auto vdata = std::vector<VertexDatum<T> >();
	for (uint32_t i = 0; i < num_elems; i++){
		vdata.push_back( VertexDatum<T>(i, values[i]) );
	}
	// Sort the array of class objects
	// (reverse iterators to get descending order):
	std::sort(vdata.rbegin(), vdata.rend());
	// Print the top K values
	for (uint32_t i = 0; i < k; i++){
		printf("PR %d %0.20lf\n", vdata[i].idx, vdata[i].val);
	}
	// Free the array (implicit)
}

// Read non-binary (text) file. Reads first element, which is the number
// of subsequent elements. Subsequent elements are read into array.
// The input is expected to be comma-separated, and a whole graph would
// be divided between two or three text files. One for IA (neighborhood offsets),
// one for JA (neighbors/edges), and one optional one for edge weights.
// Return value is number of elements successfully read into array.
uint64_t read_txt(const char * filename, uint32_t * array,
		char delimiter=',');

// Get size of text IA. JA, or VA file by reading and returning
// The first value (before delimiter).
uint64_t tell_size_txt(const char* filename);

// returns number of lines in <filename>
uint32_t tell_size(const char * filename);

// need not really return anything
// reads 0-indexed file
uint32_t read_binary( const char * filename, uint32_t *array);

// Read buffered. Faster!
uint32_t read_binary_buffers( const char * filename, uint32_t *array);

// Read buffered. Faster! 
// Reads up to n_elems input elements, even if the file may be larger.
// Returns the number actually read.
uint32_t read_binary_buffers_n( const char * filename, uint32_t *array, size_t n_elems);

 //create adjacency matrix
uint32_t init( char ** argv, uint32_t ** IA, uint32_t ** JA);

void print_v( uint32_t * v, uint32_t length );

void print_v_f(float * v, uint32_t length );

void init_vector(uint32_t * v, uint32_t length,  uint32_t val);

void init_vector_f(float * v, uint32_t length,  float val);

template<typename T>
size_t nunique_vector(T* v, size_t length){
	std::map<T, uint64_t> counters;
	//std::unordered_map<T, uint64_t> counters;
	for(size_t i = 0; i< length; i++){
    counters[v[i]]++;
  }
	return counters.size();
}


uint32_t nz_v(uint32_t *v, uint32_t val, uint32_t length);

// convert to CSC
bool csr_to_csc(uint32_t *IAr, uint32_t * JAr, uint32_t ** IAc, uint32_t ** JAc, uint32_t length);
bool csr_to_csc_parallel(uint32_t *IAr, uint32_t * JAr, uint32_t ** IAc, uint32_t ** JAc, uint32_t length);

// Full symmetric matrix to lower tri:
void csr_to_lower(uint32_t * IAf, uint32_t * JAf, 
  uint32_t * IAl, uint32_t * JAl, uint32_t N);

void sort_neighborhoods(uint32_t * IA, uint32_t * JA, uint32_t N);

// Convert IA and JA to center-offset IA and JA, such that the values in 
// IA and JA can be signed integers and the centered pointers point into 
// the middle of each array.
// NOTE: IA_cent and JA_cent are allocated within this function! Don't pre-allocate!
void csr_to_center_csr(uint32_t * IA, uint32_t * JA, 
  int32_t ** IA_cent, int32_t ** JA_cent, uint32_t N);

// Test if matrix is symmetric, using provided transpose. 
bool is_symmetric(uint32_t * IA, uint32_t * JA, uint32_t * IAc,	uint32_t * JAc, uint32_t N);

#endif
