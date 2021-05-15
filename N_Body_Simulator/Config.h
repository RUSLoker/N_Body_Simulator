#pragma once
#include "constants.h"
#include <string>
#include <windows.h>

using namespace std;

class Config
{
public:
	unsigned int W = 800;
	unsigned int H = 800;
	unsigned int N = 5000;
	CALCULATION_TYPE G = 100000;
	CALCULATION_TYPE K = 1000 * G;
	CALCULATION_TYPE DeltaT = 0.00001;
	CALCULATION_TYPE MAX_START_SPEED = 30;
	CALCULATION_TYPE theta = 0.3;
	double scroll_speed = 10;
	CALCULATION_TYPE min_accel = 0.05;
	CALCULATION_TYPE max_accel = 1000000000;
	CALCULATION_TYPE max_dist = 40000. * 40000.;
	bool useBH = false;
	bool record = false;
	size_t max_cache = 1073741824;
	string capture_dir = "captures";
	string record_dir = "records";
	string capture_path = capture_dir + "\\capture.cptr";
	string record_path = record_dir + "\\record.rcd";
	bool read_capture = true;
	bool useCUDA = true;

	static const int configN = 6;

	void readConfig(char* path);
	void readConfig(const char* path) {
		readConfig((char*)path);
	}

#pragma optimize("", off)
	Config() {
		CreateDirectory(capture_dir.c_str(), NULL);
		CreateDirectory(record_dir.c_str(), NULL);
		capture_path = capture_dir + "\\capture.cptr";
		record_path = record_dir + "\\record.rcd";
	}
};

