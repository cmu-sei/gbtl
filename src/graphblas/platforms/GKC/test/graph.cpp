/*Implementation of Graph utitlty functions as
described in "graph.h"*/
#include "graph.h"


uint64_t read_galois_size(const char * filename, 
		uint64_t& nodes, uint64_t& edges, uint64_t& edge_size, bool& weighted){
	struct stat stat_buf;
	int rc = stat(filename, &stat_buf);
	if (rc != 0 || stat_buf.st_size < 64 * 4) return 0;

	printf("File size is %lu bytes\n", stat_buf.st_size);
	
	std::ifstream f_obj(filename, std::ios::in | std::ios::binary );
	if (!f_obj.is_open()) {
		printf("Error: could not open galois graph file.\n");
		return 0;
	}

	uint64_t buf_head[4];
	uint64_t version_code;

	f_obj.read((char*)buf_head, 4*sizeof(uint64_t));
	f_obj.close();

	version_code = buf_head[0];
	edge_size = buf_head[1];
	nodes = buf_head[2];
	edges = buf_head[3];

	weighted=((stat_buf.st_size - nodes * sizeof(uint64_t) - edges * edge_size) >= edges * edge_size);

	printf("Version code: %d\n", version_code);
	printf("Edge size: %lu\t Nodes: %lu\t Edges: %lu\n", 
			edge_size, nodes, edges);
	printf("Galois graph is weighted: %s\n", weighted ? "yes" : "no");
	
	//TODO: replace with VTYPE
	uint64_t vertex_limit = std::numeric_limits<uint32_t>::max();
	if (vertex_limit < edges) return 1;
	if (!(version_code == 1 && edge_size == 4) && !(version_code == 2 && edge_size == 8)){
		// Either the edges should be 4 bytes with v code 1, or 8 with v code 2.
		return 1;
	}
	return 0;
}

uint64_t read_galois_data(const char * filename, 
		uint32_t * IA, uint32_t * JA, uint32_t * VA){
	struct stat stat_buf;
	int rc = stat(filename, &stat_buf);
	if (rc != 0 || stat_buf.st_size < 64 * 4) return 0;

	std::ifstream f_obj(filename, std::ios::in | std::ios::binary );
	if (!f_obj.is_open()) {
		printf("Error: could not open galois graph file.\n");
		return 0;
	}

	uint64_t buf_head[4];
	uint64_t version_code, nodes, edges, edge_size;

	f_obj.read((char*)buf_head, 4*sizeof(uint64_t));

	version_code = buf_head[0];
	edge_size = buf_head[1];
	nodes = buf_head[2];
	edges = buf_head[3];

	bool weighted=((stat_buf.st_size - nodes * sizeof(uint64_t) - edges * edge_size) > edges);

	const int read_buf_sz=100000;
	uint64_t long_buf[read_buf_sz];
	uint32_t int_buf[read_buf_sz];
	char char_buf[read_buf_sz];

	// Read data in chunks, copy to corresponding arrays for nodes, edges, weights:
	IA[0] = 0;
	uint64_t array_idx = 1;
	uint64_t elems_to_read = nodes;
	while (elems_to_read > 0){
		uint64_t chunk = MIN(read_buf_sz, elems_to_read);
		f_obj.read((char*)long_buf, chunk * sizeof(uint64_t));
		for (uint64_t idx = 0; idx < chunk; idx++, array_idx++){
			IA[array_idx] = (uint32_t)long_buf[idx];
		}
		elems_to_read -= chunk;
	}

	char* edge_buf;
	array_idx = 0;
	elems_to_read = edges;
	if (version_code == 1){
		edge_buf = (char*)int_buf;
		while (elems_to_read > 0){
			uint64_t chunk = MIN(4096, elems_to_read);
			f_obj.read(edge_buf, chunk * edge_size);
			for (uint64_t idx = 0; idx < chunk; idx++, array_idx++){
				JA[array_idx] = (uint32_t)((uint32_t*)edge_buf)[idx];
			}
			elems_to_read -= chunk;
		}
	} else if (version_code == 2){
		edge_buf = (char*)long_buf;
		while (elems_to_read > 0){
			uint64_t chunk = MIN(4096, elems_to_read);
			f_obj.read(edge_buf, chunk * edge_size);
			for (uint64_t idx = 0; idx < chunk; idx++, array_idx++){
				JA[array_idx] = (uint32_t)((uint64_t*)edge_buf)[idx];
			}
			elems_to_read -= chunk;
		}
	}

	// Not sure why galois does this. Probably for alignment?
	if (edges % 2){
		char c;
		for (uint32_t idx = 0; idx < edge_size; idx++){
			f_obj.get(c);
		}
	}
	printf("Remaining bytes: %lu\n", stat_buf.st_size - f_obj.tellg());
	printf("Remaining ints: %lu\n", (stat_buf.st_size - f_obj.tellg()) / 4);
	// Now read the edge weights
	if (weighted && VA != NULL){
		// TODO: are they actually chars, or is galois doing something odd?
		array_idx = 0;
		elems_to_read = edges;
		while (elems_to_read > 0){
			uint64_t chunk = MIN(4096, elems_to_read);
			f_obj.read((char*)int_buf, chunk * sizeof(uint32_t));
			for (uint64_t idx = 0; idx < chunk; idx++, array_idx++){
				VA[array_idx] = (uint32_t)int_buf[idx];
			}
			elems_to_read -= chunk;
		}
	}
	f_obj.close();
	return 0;
}


// Read non-binary (text) file. Reads first element, which is the number
// of subsequent elements. Subsequent elements are read into array.
// The input is expected to be comma-separated, and a whole graph would
// be divided between two or three text files. One for IA (neighborhood offsets),
// one for JA (neighbors/edges), and one optional one for edge weights.
// Return value is number of elements successfully read into array.
uint64_t read_txt(const char * filename, uint32_t * array,
		char delimiter)
{
	std::ifstream f_obj(filename, std::ios::in);
	uint64_t n, count = 0;
	uint32_t vid;
	std::string sep;
	if (!f_obj.is_open()) {
		std::cerr << "ERROR opening txt file!\n";    
		f_obj.close();
    exit(EXIT_FAILURE);
	}
	f_obj >> n;
	//std::cout << "Preparing to read " << n << " elements\n";
	while ( !f_obj.eof() && count < n )
	{
		// Advance to next comma or other delimiter char
		std::getline(f_obj, sep, delimiter);	
		// Get the next vertex ID or integer edge weight
		f_obj >> vid;
		//std::cout << "Read value: " << vid << std::endl;
		if (f_obj.good()){
			array[count++] = vid;
		}
		else {
			std::cerr << "ERROR reading data from txt file!\n";
			f_obj.close();
			exit(EXIT_FAILURE);
		}
	}
	f_obj.close();
	return count;
}

// Get size of text IA. JA, or VA file by reading and returning
// The first value (before delimiter).
uint64_t tell_size_txt(const char* filename)
{
	std::ifstream f_obj(filename, std::ios::in);
	uint64_t n;
	if (!f_obj.is_open()) {
		std::cerr << "ERROR opening txt file!\n";    
		f_obj.close();
    exit(EXIT_FAILURE);
	}
	f_obj >> n;
	if (!f_obj.good()) {
    std::cerr << "ERROR reading size from txt file!\n";
		f_obj.close();
    exit(EXIT_FAILURE);
	}
	f_obj.close();
	return n;
}

// Get size of binary IA, JA, or VA file by reading first
// 4 bytes representing a uint32_t value, and returning it.
uint32_t tell_size(const char * filename) {
  FILE *fptr= fopen(filename,"rb");
  uint32_t n;
  if ( fread(&n, sizeof(n), 1, fptr) != 1) {
    std::cerr << "ERROR reading size from txt file!\n";
		fclose(fptr);
    exit(EXIT_FAILURE);
  }
  fclose(fptr);
  return n;
}
uint32_t read_binary( const char * filename, uint32_t *array) {
  FILE *fptr= fopen(filename,"rb");
  fseek(fptr,0,SEEK_END);
  uint64_t end = ftell(fptr);
  fseek(fptr,0,SEEK_SET);
  printf("File has %lu entries.\n", end/4);
  uint64_t counter =0;
  uint32_t n;

  if ( fread(&n, sizeof(n), 1, fptr) != 1) {
    fprintf(stderr, "ERROR reading size from bin file!\n");
    exit(EXIT_FAILURE);
  }
  printf("First 4 bytes indicate %u entries.\n", n);
  counter+=sizeof(uint32_t);
  fseek(fptr,counter,SEEK_SET);
  uint32_t el;
  while(counter<end) {
    if ( fread(&el,sizeof(el),1,fptr) != 1){
      fprintf(stderr, "ERROR reading data from bin file!\n");
      exit(EXIT_FAILURE);
    }
    counter+=sizeof(el);
    fseek(fptr,counter,SEEK_SET);
    array[counter/4-2]=el;
  }

  fclose(fptr);
  return n;
}

uint32_t read_binary_buffers( const char * filename, uint32_t *array) {
  FILE *fptr= fopen(filename,"rb");
  uint32_t n;
  // reads first element from file (number of elements to follow)
  size_t s = fread(&n, sizeof(uint32_t), 1, fptr);
  printf(" %u elements to read \n", n);

	// after reading the number of elements, the file stream points to the
	// second element in the file.
	// read the rest of the elements into the array.
	uint32_t stride = 500000000;
	uint32_t index = 0;
	uint32_t loop_iters = n/stride + (n % stride > 0);
	printf(" %u reads\n ", loop_iters);
	for(uint32_t i = 0; i < loop_iters ; i++){
		uint32_t read_amt = MIN(stride, n-index);
		s = fread(array+index, sizeof(uint32_t), read_amt, fptr);
		index += read_amt;
		printf("%lu bytes read\n", (uint64_t)s*sizeof(uint32_t));
	}

  fclose(fptr);
  return n;
}

// Reads up to n_elems input elements, even if the file may be larger.
// Returns the number actually read.
uint32_t read_binary_buffers_n( const char * filename, uint32_t *array, size_t n_elems){
  FILE *fptr= fopen(filename,"rb");
  uint32_t n;
  // reads first element from file (number of elements to follow)
  size_t s = fread(&n, sizeof(uint32_t), 1, fptr);
	n = MIN(n_elems, n);
  printf(" %u elements to read \n", n);

  uint32_t stride = 500000000;
  uint32_t index = 0;
  uint32_t loop_iters = n/stride + (n % stride > 0);
  printf(" %u reads\n ", loop_iters);
  for(uint32_t i = 0; i < loop_iters ; i++){
    uint32_t read_amt = MIN(stride, n-index);
    s = fread(array+index, sizeof(uint32_t), read_amt, fptr);
    index += read_amt;
    printf("%lu bytes read\n", (uint64_t)s*sizeof(uint32_t));
  }

  fclose(fptr);
  return n;
}

uint32_t init( char ** argv, uint32_t ** IA, uint32_t ** JA){

  uint32_t NUM_VERTICES=tell_size(argv[1])-1;
  uint32_t NUM_EDGES=tell_size(argv[2]);

  // Allocate contiguous sections of memory for each thread in the test:

  // for non-triangularized data:
  char *I_J;

  // Compute rounded up number of entries for data
  // (such that each segment will be cache-aligned):
  uint64_t I_bytes = (NUM_VERTICES+1) * sizeof(uint32_t);
  I_bytes = ((I_bytes >> 6) << 6)  + 64 * ((I_bytes & 0x3F) != 0);
  uint64_t J_bytes = NUM_EDGES * sizeof(uint32_t);
  J_bytes  = ((J_bytes >> 6) << 6) + 64 * ((J_bytes & 0x3F) != 0);
  // Num bytes for original data:
  uint64_t bytes_aligned = I_bytes + J_bytes;

  // Allocate the contiguous segments of memory:
  I_J = (char*)aligned_alloc(64, bytes_aligned);
  // Set pointers:
  *IA = (uint32_t*) (I_J + 0);
  *JA = (uint32_t*) (I_J + I_bytes);

  read_binary_buffers(argv[1],*IA);
  read_binary_buffers(argv[2],*JA);

  return NUM_VERTICES;
}


void print_v( uint32_t * v, uint32_t length ){
  for(uint32_t i = 0; i< length /*&& v[i]<length*/ ; i++){
    printf(" %d ", v[i]);
  }
  printf("\n" );
}

void print_v_f(float * v, uint32_t length ){
  for(uint32_t i = 0; i< length /*&& v[i]<length*/ ; i++){
    printf(" %f \n", v[i]);
  }
  printf("\n" );
}

void init_vector(uint32_t * v, uint32_t length,  uint32_t val){
  for(uint32_t i = 0; i< length; i++){
    v[i]= val;
  }
}
void init_vector_f(float * v, uint32_t length,  float val){
  for(uint32_t i = 0; i< length; i++){
    v[i]= val;
  }
}

uint32_t nz_v(uint32_t *v, uint32_t val, uint32_t length){
  uint32_t nz = 0;
  for(uint32_t i = 0; i < length; i++ ){
    if(v[i] == val)
      nz++;
  }
  return nz;
}

bool csr_to_csc(uint32_t *IAr, uint32_t * JAr, uint32_t ** IAc, uint32_t ** JAc, uint32_t length){
  // Memory alloc:
  if ((*IAc)==NULL || (*JAc)==NULL){
		printf("one or both null\n");
    if (*IAc!=NULL || *JAc!=NULL) return false; // Either both or none should be allocated.
    // Allocate cache-aligned data
    printf("Allocating memory for transpose.\n");
		uint32_t edges = IAr[length];
    unsigned long long I_bytes = (length+1) * sizeof(uint32_t);
    I_bytes = ((I_bytes >> 6) << 6)  + 64 * ((I_bytes & 0x3F) != 0);
    unsigned long long J_bytes = edges * sizeof(uint32_t);
    J_bytes  = ((J_bytes >> 6) << 6) + 64 * ((J_bytes & 0x3F) != 0);

    *IAc = (uint32_t*)aligned_alloc(64, I_bytes);
    *JAc = (uint32_t*)aligned_alloc(64, J_bytes);
    if (!*IAc || !*JAc) return false;
  }
  printf("Transposing.\n");
  // Transpose:
  std::vector<std::vector<uint32_t>> remap(length);
  for (uint32_t idx = 0; idx < length; idx++){
    uint32_t st = IAr[idx];
    uint32_t nd = IAr[idx+1];
    for (uint32_t edx = st; edx < nd; edx++) {
      uint32_t jdx = JAr[edx];
      remap[jdx].push_back(idx);
    }  
  }
  // Sort and copy (note swapped jdx and idx):
  *(IAc)[0] = 0;
  for (uint32_t jdx = 0; jdx < length; jdx++){
    auto st = remap[jdx].begin();
    auto nd = remap[jdx].end();
    std::sort(st, nd);
    // Refresh start and end iterators (is that even necessary?)
    st = remap[jdx].begin();
    nd = remap[jdx].end();
    // Cumulative sum:
    (*IAc)[jdx+1] = (*IAc)[jdx] + remap[jdx].size();
    // Copy to CSC:
    for (uint32_t edx = 0; edx < remap[jdx].size(); edx++) {
      (*JAc)[(*IAc)[jdx] + edx] = remap[jdx][edx];
    }
  }
  return true;
}

bool csr_to_csc_parallel(uint32_t *IAr, uint32_t * JAr, uint32_t ** IAc, uint32_t ** JAc, uint32_t length){
 // Memory alloc:
  uint32_t edges = IAr[length];
  unsigned long long I_bytes = (length+1) * sizeof(uint32_t);
  I_bytes = ((I_bytes >> 6) << 6)  + 64 * ((I_bytes & 0x3F) != 0);
  unsigned long long J_bytes = edges * sizeof(uint32_t);
  J_bytes  = ((J_bytes >> 6) << 6) + 64 * ((J_bytes & 0x3F) != 0);
 if ((*IAc)==NULL || (*JAc)==NULL){
  printf("one or both null\n");
  if ((*IAc) || (*JAc)) return false; // Either both or none should be allocated.
  // Allocate cache-aligned data
  printf("Allocating memory for transpose.\n");

  *IAc = (uint32_t*)aligned_alloc(64, I_bytes);
  *JAc = (uint32_t*)aligned_alloc(64, J_bytes);
  if (!(*IAc) || !(*JAc)) return false;
 }
 uint32_t * degrees = (uint32_t *)calloc(length, sizeof(uint32_t));
 uint32_t * IAc_main = *IAc;
 if (!degrees) return false;
 
 printf("Transposing.\n");
 // First get size of incoming neighborhoods:
#pragma omp parallel for schedule(dynamic, 64)
 for (uint32_t edx = 0; edx < edges; edx++) {
  uint32_t jdx = JAr[edx];
#pragma omp atomic
  degrees[jdx]++;
 }

 uint32_t num_elems = 1024*1024 / sizeof(uint32_t); // About the size of L2 cache on SKX
	uint32_t num_blocks= (length+num_elems-1) / num_elems;

	// Coarse parallel sum
	IAc_main[0] = 0;
#pragma omp parallel for
	for (uint32_t bidx = 0; bidx < num_blocks; bidx++){
		uint32_t b_st = bidx * num_elems;
		uint32_t b_nd = MIN(length, (bidx+1) * num_elems);
		uint32_t t_sum = 0;
		for (uint32_t idx = b_st; idx < b_nd; idx++){
			// TODO: SIMD reduction
			t_sum += degrees[idx];
			IAc_main[idx+1] = t_sum;
		}
	}
	// At this point each b_nd location of IAc_main
	// has a prefix sum for a block
	
	// Coarse prefix summation of sums at each b_nd:
	for (uint32_t bidx = 1; bidx < num_blocks; bidx++){
		uint32_t b_st = bidx * num_elems;
		uint32_t b_nd = MIN(length, (bidx+1) * num_elems);
		IAc_main[b_nd] += IAc_main[b_st];
	}

	// Now each block has a local sum, except that the entry at the 
	// block start (1 + last block end) has a global prefix sum value

	// Parallel local summation
#pragma omp parallel for
	for (uint32_t bidx = 1; bidx < num_blocks; bidx++){
		uint32_t b_st = bidx * num_elems;
		uint32_t b_nd = MIN(length, (bidx+1) * num_elems);
		uint32_t t_sum = IAc_main[b_st];
		// TODO: SIMD
		for (uint32_t idx = b_st+1; idx < b_nd; idx++){
			IAc_main[idx] += t_sum;
		}
	}

 // IAc_shadow can be used as a temporary for size of unwritten neighborhoods:
 uint32_t * IAc_shadow = (uint32_t*)aligned_alloc(64, I_bytes);
#pragma omp parallel for
 for (uint32_t idx = 0; idx < length; idx++){
  IAc_shadow[idx] = 0;
 }

 // Then copy source over (expensive :( )
#pragma omp parallel for schedule(dynamic, 64)
 for (uint32_t idx = 0; idx < length; idx++){
  uint32_t st = IAr[idx];
  uint32_t nd = IAr[idx+1];
  for (uint32_t edx = st; edx < nd; edx++) {
   uint32_t jdx = JAr[edx];
   uint32_t nbhd_st = IAc_main[jdx];
   uint32_t curr_nbhd_sz = __sync_fetch_and_add(&(IAc_shadow[jdx]), 1);
   (*JAc)[nbhd_st + curr_nbhd_sz] = idx;
  }  
 }
 // Sort new neighborhoods:
#pragma omp parallel for schedule(dynamic, 64)
 for (uint32_t idx = 0; idx < length; idx++){
  uint32_t j_st = IAc_main[idx];
  uint32_t len = IAc_main[idx+1] - IAc_main[idx];
  qsort(*JAc + j_st, len, sizeof(uint32_t), cmp_u32);
 }

 free(degrees);
 free(IAc_shadow);
 return true;
}

// Convert symmetric, full matrix from CSR/CSC to lower triangular CSR.
void csr_to_lower(uint32_t * IAf, uint32_t * JAf, 
  uint32_t * IAl, uint32_t * JAl, uint32_t N){
  uint32_t Ml = 0;
  IAl[0] = 0;
  for (uint32_t idx = 0; idx < N; idx++){
    uint32_t st = IAf[idx];
    uint32_t nd = IAf[idx+1];
    //uint32_t old_M = Ml;
    for (auto j = st; j != nd; j++){
      if ( JAf[j] < idx) {
        JAl[Ml] = JAf[j];
        Ml++; 
      }
      else {break;}
    }
    //if ((long)Ml-(long)old_M != 0){
    //printf("%u: Filled half neighborhood with %u of %u edges.\n",
    //  idx, Ml-old_M, nd-st);
    //}
    IAl[idx+1] = Ml;
  }
  printf("Got lower triangular (%u of %u edges)\n", IAl[N], IAf[N]);
}


// Inputs: IA array, JA array, and N=number of vertices. (IA is size N+1)
void sort_neighborhoods(uint32_t * IA, uint32_t * JA, uint32_t N){
  for (uint32_t idx = 0; idx < N; idx++){
    uint32_t st = IA[idx];
    uint32_t nd = IA[idx+1]; 
    qsort(JA+st, nd-st, sizeof(uint32_t), cmp_u32);
  }
}

// See detailed description of this function in header.
void csr_to_center_csr(uint32_t * IA, uint32_t * JA, 
  int32_t ** IA_cent, int32_t ** JA_cent, uint32_t N){
  
  uint32_t M = IA[N]; // Retrieve num edges
  
  // IA_cent and JA_cent should not already be allocated.
  assert(!(*IA_cent) && !(*JA_cent));

  // Now allocate
  int32_t * IA_int = (int32_t *)malloc(sizeof(int32_t)*(N+1));
  int32_t * JA_int = (int32_t *)malloc(sizeof(int32_t)*M);
  assert(IA_int!=NULL && JA_int!=NULL);

  // Set pointers to centered offsets
  *IA_cent = IA_int + (N+1) / 2;
  *JA_cent = JA_int + M / 2;

  // Copy IA and JA into offset arrays, modifying values as needed.
  for (uint32_t idx = 0; idx < N+1; idx++){
    IA_int[idx] = (int32_t)((int64_t)IA[idx] - (int64_t)(M/2));
  }

  for (uint32_t idx = 0; idx < M; idx++){
    JA_int[idx] = (int32_t)((int64_t)JA[idx] - (int64_t)((N+1)/2));
  }
}

// Test if matrix is symmetric, using provided transpose. 
bool is_symmetric(
		uint32_t * IA, 
		uint32_t * JA, 
		uint32_t * IAc,
		uint32_t * JAc,
		uint32_t N){
	std::atomic<bool> success(true);
		// Iterate over each out-neighborhood, searching for the 
		// corresponding entry in the in-neighborhood
#pragma omp parallel for 
	for (uint64_t i = 0; i < N; i++){
		if (!success) continue;
		if (IA[i] != IAc[i])
			success = false;
	}
#pragma omp parallel for 
	for (uint64_t i = 0; i < IA[N]; i++){
		if (!success) continue;
		if (JA[i] != JAc[i])
			success = false;
	}
	return (bool)success;
}
