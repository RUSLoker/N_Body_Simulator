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
	T node_mass;
	T node_width;
	BH_tree<T>* children;
	bool hasNodes = false;

	void add(T* coords, T mass);

	void setNew(T x, T y, T width);

	void clear();

	BH_tree<T>* getNodes();

	T* calcAccel(T* coords);

	static BH_tree<T>* newTree(Config config);

	~BH_tree();

	unsigned int depth() {
		return node_depth;
	}

	SIZE_TYPE totalNodeCount() {
		return ((SIZE_TYPE)*next_caching - (SIZE_TYPE)node_cache) / (SIZE_TYPE)sizeof(BH_tree<T>);
	}

	SIZE_TYPE activeNodeCount() {
		return *active_node_count;
	}

private:
	unsigned int node_depth = 1;
	BH_tree<T>* node_cache;
	BH_tree<T>** next_caching;
	SIZE_TYPE* active_node_count;
	Config* tree_config;

	BH_tree() {};

	void newNode(BH_tree<T>* cache, BH_tree<T>** next, SIZE_TYPE* node_counter, Config* config);

	void calcAccel(T* coords, T* holder);
};