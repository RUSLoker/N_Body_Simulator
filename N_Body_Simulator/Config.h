#pragma once
#include "constants.h"

using namespace std;

class Config
{
public:
	unsigned int W = 800;
	unsigned int H = 800;
	unsigned int N = 5000;
	double G = 100000;
	double K = 1000 * G;
	double DeltaT = 0.00001;
	double MAX_START_SPEED = 30;
	double min_width = 10;
	double theta = 0.3;
	double scroll_speed = 10;
	double min_accel = 0.05;
	double max_accel = 1000000000;
	double max_dist = 40000. * 40000.;
	bool useBH = true;
	bool record = true;
	unsigned long long max_cache = 1073741824;
	unsigned long long caching_nodes_num = MAX_CACHING_NODES_NUM;

	static const int configN = 5;

	void readConfig(char* path);
	void readConfig(const char* path) {
		readConfig((char*)path);
	}
};

