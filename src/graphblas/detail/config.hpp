/*
 * Copyright (c) 2015 Carnegie Mellon University and The Trustees of Indiana
 * University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY AND THE TRUSTEES OF INDIANA UNIVERSITY EXPRESSLY DISCLAIM
 * TO THE FULLEST EXTENT PERMITTED BY LAW ALL EXPRESS, IMPLIED, AND STATUTORY
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */
#pragma once

/**
*	@todo We need to differentiate between cusp cpu and cusp gpu
*/	
#if defined(GB_USE_CUSP_GPU)
#    define __GB_SYSTEM_ROOT cusp
#elif defined(GB_USE_SEQUENTIAL)
#    define __GB_SYSTEM_ROOT sequential
// ....
#else
#    error GraphBLAS library type unspecified at compile time!
#endif

#if !defined(__CUDACC__)
#ifndef __host__
#define __host__
#endif // __host__

#ifndef __device__
#define __device__
#endif // __device__

#endif // __CUDACC__
