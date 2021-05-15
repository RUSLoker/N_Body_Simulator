#include "BH_tree.h"
#include "constants.h"
#include <cmath>

using namespace std;

template <typename T>

BH_tree<T>* BH_tree<T>::newTree(Config config) {
	BH_tree<T>* cache = (BH_tree<T>*)malloc(config.max_cache);
	BH_tree<T>** next = new BH_tree<T> * ();
	*next = cache;
	size_t* counter = new size_t(0);
	cache[0].tree_config = (Config*)malloc(sizeof(Config));
	memcpy(cache[0].tree_config, &config, sizeof(Config));
	cache[0].newNode(cache, next, counter, cache[0].tree_config);
	cache[0].setNew(0, 0, 100000);
	return cache;
}

template <typename T>

void BH_tree<T>::newNode(BH_tree<T>* cache, BH_tree<T>** next, size_t* node_counter, Config* config) {
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

template <typename T>

void BH_tree<T>::add(T* coords, T mass) {
	unsigned int curd = 0;
	center_of_mass[0] = (center_of_mass[0] * node_mass + coords[0] * mass) / (node_mass + mass);
	center_of_mass[1] = (center_of_mass[1] * node_mass + coords[1] * mass) / (node_mass + mass);
	node_mass += mass;
	if (body_mass > 0 && !hasNodes) {
		if (children == 0) {
			if ((char*)(*next_caching + 5) > (char*)node_cache + tree_config->max_cache) {
				throw CACHE_OVERFLOW_EXCEPT;
			}
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

template <typename T>

BH_tree<T>* BH_tree<T>::getNodes() {
	return node_cache;
}

template <typename T>

BH_tree<T>::~BH_tree() {
	if (this == node_cache) {
		delete next_caching;
		delete active_node_count;
		free(node_cache);
	}
}

template <typename T>

void BH_tree<T>::setNew(T x, T y, T width) {
	center[0] = x;
	center[1] = y;
	center_of_mass[0] = 0;
	center_of_mass[1] = 0;
	node_mass = 0;
	node_width = width;
	(*active_node_count)++;
}

template <typename T>

void BH_tree<T>::clear() {
	hasNodes = false;
	body_mass = -1;
	node_mass = 0;
	node_depth = 1;
	children = 0;
	*next_caching = this + 1;
	(*active_node_count) = 0;
}

template <typename T>

T* BH_tree<T>::calcAccel(T* coords) {
	T* a = new T[2]{ 0, 0 };
	this->calcAccel(coords, a);
	return a;
}

template <typename T>

void BH_tree<T>::calcAccel(T* coords, T* holder) {
	if (node_mass <= 0) return;
	T cr[] = { center_of_mass[0] - coords[0], center_of_mass[1] - coords[1] };
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