#pragma once

#define MAX_CACHING_NODES_NUM
#define x64 1
#define x86 2

#if (PLATFORM == x64)
	#define MAX_CACHING_NODES_NUM 70000000
#elif (PLATFORM == x86)
	#define MAX_CACHING_NODES_NUM 8000000
#else
	#define MAX_CACHING_NODES_NUM 100000
#endif

#undef x64
#undef x86

using namespace std;

static const unsigned int W = 800;
static const unsigned int H = 800;
static unsigned int N = 5000;
static const double G = 100000;
static const double K = 1000 * G;
static double DeltaT = 0.00001;
static double MAX_START_SPEED = 30;
static const double min_width = 10;
static const double theta = 0.3;
static const double scroll_speed = 10;
static const double min_accel = 0.05;
static const double max_accel = 1000000000;
static const double max_dist = 40000. * 40000.;
static const bool useBH = true;
static bool record = true;
static int max_cache = 1073741824;
static int caching_nodes_num = MAX_CACHING_NODES_NUM;

static const int configN = 5;