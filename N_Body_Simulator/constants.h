#pragma once

#define MAX_CACHING_NODES_NUM
#define x64 1
#define x86 2

#if (PLATFORM == x64)
	#define MAX_CACHING_NODES_NUM 16000000
	#define SIZE_TYPE unsigned long long
#elif (PLATFORM == x86)
	#define MAX_CACHING_NODES_NUM 8000000
	#define SIZE_TYPE unsigned int
#else
	#define MAX_CACHING_NODES_NUM 100000
	#define SIZE_TYPE unsigned int
#endif

#define SIZE_TYPE unsigned int
#define MAX_CACHING_NODES_NUM 100000

#define CALCULATION_TYPE double

#undef x64
#undef x86