#pragma once
#include "constants.h"
#include "Config.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

	__host__ __device__ void add(T* coords, T mass) {
		unsigned int curd = 0;
		node_mass += mass;
		if (body_mass > 0 && !hasNodes) {
			if (children == 0) {
				children = *next_caching;

				for (BH_tree<T>* i = children; i < children + 4; i++) {
					i->newNode(node_cache, next_caching, active_node_count, tree_config);
				}
			}
			children[0].setNew(center[0] - node_width / 4, center[1] - node_width / 4, node_width / 2);
			children[1].setNew(center[0] + node_width / 4, center[1] - node_width / 4, node_width / 2);
			children[2].setNew(center[0] - node_width / 4, center[1] + node_width / 4, node_width / 2);
			children[3].setNew(center[0] + node_width / 4, center[1] + node_width / 4, node_width / 2);
			hasNodes = true;
		}
		if (hasNodes) {
			if (coords[0] < center[0] && coords[1] < center[1]) {
				children[0].add(coords, mass);
				curd = children[0].node_depth;
				if (curd > node_depth) node_depth = curd;
			}
			else if (coords[0] > center[0] && coords[1] < center[1]) {
				children[1].add(coords, mass);
				curd = children[1].node_depth;
				if (curd > node_depth) node_depth = curd;
			}
			else if (coords[0] < center[0] && coords[1] > center[1]) {
				children[2].add(coords, mass);
				curd = children[2].node_depth;
				if (curd > node_depth) node_depth = curd;
			}
			else if (coords[0] > center[0] && coords[1] > center[1]) {
				children[3].add(coords, mass);
				curd = children[3].node_depth;
				if (curd > node_depth) node_depth = curd;
			}
			if (body_mass > 0) {
				if (body_coords[0] < center[0] && body_coords[1] < center[1]) {
					children[0].add(body_coords, body_mass);
					curd = children[0].node_depth;
					if (curd > node_depth) node_depth = curd;
				}
				else if (body_coords[0] > center[0] && body_coords[1] < center[1]) {
					children[1].add(body_coords, body_mass);
					curd = children[1].node_depth;
					if (curd > node_depth) node_depth = curd;
				}
				else if (body_coords[0] < center[0] && body_coords[1] > center[1]) {
					children[2].add(body_coords, body_mass);
					curd = children[2].node_depth;
					if (curd > node_depth) node_depth = curd;
				}
				else if (body_coords[0] > center[0] && body_coords[1] > center[1]) {
					children[3].add(body_coords, body_mass);
					curd = children[3].node_depth;
					if (curd > node_depth) node_depth = curd;
				}
				body_mass = -1;
			}
		}
		else {
			body_mass = mass;
			body_coords = coords;
		}
		if (curd + 1 > node_depth) node_depth = curd + 1;
	}

	__host__ __device__ void setNew(T x, T y, T width) {
		center[0] = x;
		center[1] = y;
		node_width = width;
		(*active_node_count)++;
	}

	__host__ __device__ void clear() {
		hasNodes = false;
		body_mass = -1;
		node_mass = 0;
		node_depth = 1;
		children = 0;
		*next_caching = this + 1;
		(*active_node_count) = 0;
	}

	BH_tree<T>* getNodes();

	__host__ __device__ T* calcAccel(T* coords) {
		T* a = new T[2]{ 0, 0 };
		this->calcAccel(coords, a);
		return a;
	}

	__host__ __device__ void calcAccel(T* coords, T* holder) {
		if (node_mass <= 0) return;
		T cr[] = { center[0] - coords[0], center[1] - coords[1] };
		T cdist = sqrt(cr[0] * cr[0] + cr[1] * cr[1]);
		if (cdist < 0.000001) cdist = 0.000001;
		if (body_mass < 0. && node_width / cdist <= tree_config->theta) {
			T t1 = node_mass / pow(cdist, 3) * tree_config->G;
			T t2 = node_mass / pow(cdist + 0.6, 14) * tree_config->K;
			if (abs(t1 - t2) < tree_config->max_accel) {
				holder[0] += t1 * cr[0];
				holder[1] += t1 * cr[1];
				holder[0] -= t2 * cr[0];
				holder[1] -= t2 * cr[1];
			}
			return;
		}
		else if (body_mass > 0.) {
			if (coords == body_coords) return;
			T r[] = { body_coords[0] - coords[0], body_coords[1] - coords[1] };
			T mr = sqrt(r[0] * r[0] + r[1] * r[1]);
			if (mr < 0.000001) mr = 0.000001;
			T t1 = body_mass / pow(mr, 3) * tree_config->G;
			T t2 = body_mass / pow(mr + 0.6, 14) * tree_config->K;
			if (abs(t1 - t2) < tree_config->max_accel) {
				holder[0] += t1 * r[0];
				holder[1] += t1 * r[1];
				holder[0] -= t2 * r[0];
				holder[1] -= t2 * r[1];
			}
			return;
		}
		else if (hasNodes) {
			for (BH_tree<T>* i = children; i < children + 4; i++) {
				if (i->node_mass > 0) {
					i->calcAccel(coords, holder);
				}
			}
		}
	}

	static BH_tree<T>* newTree(Config config);	
	
	static BH_tree<T>* newTreeCUDA(Config config);

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

	__host__ __device__ void newNode(BH_tree<T>* cache, BH_tree<T>** next, SIZE_TYPE* node_counter, Config* config) {
		hasNodes = false;
		body_mass = -1;
		node_mass = 0;
		node_depth = 1;
		children = 0;
		node_cache = cache;
		next_caching = next;
		*next_caching = *next_caching + 1;
		active_node_count = node_counter;
		tree_config = config;
	}
};