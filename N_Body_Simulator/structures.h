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
	BH_tree* children;
	bool hasNodes = false;

	void add(double* coords, double mass);

	void setNew(double x, double y, double width);

	void clear();

	vector<BH_tree*> getNodes();

	double* calcAccel(double* coords);

	static BH_tree* newTree();

	~BH_tree();

	unsigned int depth() {
		return node_depth;
	}

	int totalNodeCount() {
		return ((int)*next_caching - (int)node_cache) / sizeof(BH_tree);
	}	
	
	int activeNodeCount() {
		return *active_node_count;
	}

private:
	unsigned int node_depth = 1;
	BH_tree* node_cache;
	BH_tree** next_caching;
	int* active_node_count;

	BH_tree() {};

	void newNode(BH_tree* cache, BH_tree** next, int* node_counter);

	void calcAccel(double* coords, double* holder);
};

static void readConfig() {
	fstream cfg;
	cfg.open("config.cfg", ios::in);
	bool exist = cfg.good();
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
			cfg >> record;
			config_readed[1] = true;
		}
		else if (s == "N:") {
			cfg >> N;
			config_readed[2] = true;
		}
		else if (s == "DeltaT:") {
			cfg >> DeltaT;
			config_readed[3] = true;
		}
	}
	cfg.close();

	cfg.open("config.cfg", ios::in);
	cfg.seekg(-1, cfg._Seekend);
	char last_ch = 0;
	cfg.read(&last_ch, sizeof(last_ch));
	cfg.close();

	cfg.open("config.cfg", ios::out | ios::app);
	if (last_ch != '\n' && cfg_s > 0 && exist) 
		cfg << endl;
	if (!config_readed[0]) {
		cfg << "MAX_START_SPEED: " << MAX_START_SPEED << endl;
	}
	if (!config_readed[1]) {
		cfg << "RECORD: " << record << endl;
	}
	if (!config_readed[2]) {
		cfg << "N: " << N << endl;
	}
	if (!config_readed[3]) {
		cfg << "DeltaT: " << DeltaT << endl;
	}
}