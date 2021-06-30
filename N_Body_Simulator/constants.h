#pragma once

#define MAX_CACHING_NODES_NUM
#define x64 1
#define x86 2

#if (PLATFORM == x64)
	#define MAX_CACHING_NODES_NUM 16000000
#elif (PLATFORM == x86)
	#define MAX_CACHING_NODES_NUM 8000000
#else
	#define MAX_CACHING_NODES_NUM 100000
#endif

#define MAX_CACHING_NODES_NUM 100000

#define CALCULATION_TYPE float

#define CACHE_OVERFLOW_EXCEPT exception("Cache overflowed. Try to increase its size.")

# define PI 3.1415926535897932384626433

#undef x64
#undef x86