#pragma once

/**
   This file gives default configuration parameters. Any changes
   should be givein in config-user.h.
 */
#include "config-user.h"

/**
   CONFIG_USE_FAT_BITMAP - whether to use one word per entry in the
   queues rather than one bit per entry.

   In theory one word per entry should trade memory bandwidth for
   fewer atomics, potentially improving performance. In practice, this
   option tends to slow things down, so we leave it off by default.
 */
#ifndef CONFIG_USE_FAT_BITMAP
#define CONFIG_USE_FAT_BITMAP 0
#endif

/**
   CONFIG_WARP_REAP - specifies whether to use warp-level cooperation
   to process vertices with a small number of adjacent vertices.

   This dramatically improves performance, so we turn it on by default.
*/
#ifndef CONFIG_WARP_REAP
#define CONFIG_WARP_REAP 1
#endif

/**
   CONFIG_SORT_EDGES - enables edge sorting.

   This leads to a little longer runtime due to the time to sort the
   edges, but we get higher TEPS due to better memory locality.
*/
#ifndef CONFIG_SORT_EDGES
#define CONFIG_SORT_EDGES 1
#endif

/**
   CONFIG_GPU_DRIVER - run the iteration loop on the GPU.

   This theoretically improves performance by decrease CPU-GPU round
   trips. In practice, it doesn't.
*/
#ifndef CONFIG_GPU_DRIVER
#define CONFIG_GPU_DRIVER 0
#endif

/**
   CONFIG_UNROLL_CHILD_KERNEL - process multiple vertices from child
   kernel thread.

   This could theoretically give us better latency hiding, but really
   it just slows us down.
*/
#ifndef CONFIG_UNROLL_CHILD_KERNEL
#define CONFIG_UNROLL_CHILD_KERNEL 0
#endif
