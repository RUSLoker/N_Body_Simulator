#pragma once
#include "constants.h"
#include "Config.h"

using namespace std;


template <typename T>

class BH_tree
{
public:
	T* body_coords;
	T body_mass = -1;
	T center[2];
	T center_of_mass[2];
	T node_mass;
	T node_width;
	BH_tree<T>* children;
	bool hasNodes = false;

	void add(T* coords, T mass);

	void setNew(T x, T y, T width);

	void clear();

	BH_tree<T>* getNodes();

	T* calcAccel(T* coords);

	void calcAccel(T* coords, T* holder);

	static BH_tree<T>* newTree(Config config);

	~BH_tree();

	unsigned int depth() {
		return node_depth;
	}

	size_t totalNodeCount() {
		return ((size_t)*next_caching - (size_t)node_cache) / (size_t)sizeof(BH_tree<T>);
	}

	size_t activeNodeCount() {
		return *active_node_count;
	}

private:
	unsigned int node_depth = 1;
	BH_tree<T>* node_cache;
	BH_tree<T>** next_caching;
	size_t* active_node_count;
	Config* tree_config;

	BH_tree() {};

	void newNode(BH_tree<T>* cache, BH_tree<T>** next, size_t* node_counter, Config* config);
};