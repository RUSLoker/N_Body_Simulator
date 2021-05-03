#pragma once
#include <vector>
#include <fstream>
#include <string>
#include "constants.h"

using namespace std;

class BH_tree
{
public:
	double* body_coords;
	double body_mass = -1;
	double center[2];
	double node_mass;
	double node_width;
	BH_tree* children[4];
	bool hasNodes = false;

	BH_tree(double x, double y, double width);
	BH_tree();

	void add(double* coords, double mass);

	void setNew(double x, double y, double width);

	void clear();

	vector<BH_tree*> getNodes();

	double* calcAccel(double* coords);

	~BH_tree();

	unsigned int depth() {
		return node_depth;
	}

private:
	unsigned int node_depth = 1;

	void calcAccel(double* coords, double* holder);
};

static void readConfig() {
	ifstream cfg("config.cfg");
	unsigned int cfg_s;
	cfg.seekg(0, cfg._Seekend);
	cfg_s = cfg.tellg();
	cfg.seekg(0, cfg._Seekbeg);
	vector<bool> config_readed(configN, false);
	while (cfg) {
		string s;
		cfg >> s;
		if (s == "MAX_START_SPEED:") {
			cfg >> MAX_START_SPEED;
			config_readed[0] = true;
		}
		else if (s == "RECORD:") {
			cfg >> record_default;
			config_readed[1] = true;
		}
		else if (s == "N:") {
			cfg >> N;
			config_readed[2] = true;
		}
	}
	cfg.close();
	ofstream cfgo("config.cfg", ios::app | ios::out);
	if (!config_readed[0]) {
		cfgo << "MAX_START_SPEED: " << MAX_START_SPEED << endl;
	}
	if (!config_readed[1]) {
		cfgo << "RECORD: " << record_default << endl;
	}
	if (!config_readed[2]) {
		cfgo << "N: " << N << endl;
	}
}