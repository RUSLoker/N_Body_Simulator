#pragma once
#include "constants.h"
#include "Config.h"

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

	BH_tree** getNodes();

	double* calcAccel(double* coords);

	static BH_tree* newTree(Config config);

	~BH_tree();

	unsigned int depth() {
		return node_depth;
	}

	SIZE_TYPE totalNodeCount() {
		return ((SIZE_TYPE)*next_caching - (SIZE_TYPE)node_cache) / (SIZE_TYPE)sizeof(BH_tree);
	}

	SIZE_TYPE activeNodeCount() {
		return *active_node_count;
	}

private:
	unsigned int node_depth = 1;
	BH_tree* node_cache;
	BH_tree** next_caching;
	SIZE_TYPE* active_node_count;
	Config* tree_config;

	BH_tree() {};

	void newNode(BH_tree* cache, BH_tree** next, SIZE_TYPE* node_counter, Config* config);

	void getNodes(BH_tree*** next);

	void calcAccel(double* coords, double* holder);
};

