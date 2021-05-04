#include "structures.h"
#include "constants.h"

using namespace std;

BH_tree* BH_tree::newTree() {
	BH_tree* cache = new BH_tree[caching_nodes_num];
	BH_tree** next = new BH_tree * ();
	*next = cache;
	int* counter = new int(0);
	(*next)->newNode(cache, next, counter);
	cache[0].setNew(0, 0, 100000);
	return cache;
}

void BH_tree::newNode(BH_tree* cache, BH_tree** next, int* node_counter) {
	hasNodes = false;
	body_mass = -1;
	node_mass = 0;
	node_depth = 1;
	children = 0;
	node_cache = cache;
	next_caching = next;
	*next_caching = *next_caching + 1;
	active_node_count = node_counter;
}

void BH_tree::add(double* coords, double mass) {
	int curd = 0;
	node_mass += mass;
	if (body_mass > 0 && !hasNodes) {
		if (children == 0) {
			children = *next_caching;
			for (BH_tree* i = children; i < children + 4; i++) {
				i->newNode(node_cache, next_caching, active_node_count);
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

BH_tree** BH_tree::getNodes() {
	BH_tree** nodes = new BH_tree*[*active_node_count];
	BH_tree** next = nodes;
	this->getNodes(&next);
	return nodes;
}

void BH_tree::getNodes(BH_tree*** next) {
	**next = this;
	(*next)++;
	if (hasNodes) {
		for (BH_tree* i = children; i < children + 4; i++) {
			i->getNodes(next);
		}
	}
}

BH_tree::~BH_tree() {
	if (this == node_cache) {
		delete next_caching;
		delete[] node_cache;
	}
}

void BH_tree::setNew(double x, double y, double width) {
	center[0] = x;
	center[1] = y;
	node_width = width;
	(*active_node_count)++;
}

void BH_tree::clear() {
	BH_tree** nodes;
	nodes = getNodes();
	for (BH_tree** i = nodes; i < nodes + (*active_node_count); i++) {
		(*i)->hasNodes = false;
		(*i)->body_mass = -1;
		(*i)->node_mass = 0;
		(*i)->node_depth = 1;
	}
	(*active_node_count) = 0;
	delete[] nodes;
}

double* BH_tree::calcAccel(double* coords) {
	double* a = new double[2]{ 0, 0 };
	this->calcAccel(coords, a);
	return a;
}

void BH_tree::calcAccel(double* coords, double* holder) {
	if (node_mass <= 0) return;
	double cr[] = { center[0] - coords[0], center[1] - coords[1] };
	double cdist = sqrt(cr[0] * cr[0] + cr[1] * cr[1]);
	if (cdist < 0.000001) cdist = 0.000001;
	if (body_mass < 0. && node_width / cdist <= theta) {
		double t1 = node_mass / pow(cdist, 3) * G;
		double t2 = node_mass / pow(cdist, 14) * K;
		if (abs(t1 - t2) < max_accel) {
			holder[0] += t1 * cr[0];
			holder[1] += t1 * cr[1];
			holder[0] -= t2 * cr[0];
			holder[1] -= t2 * cr[1];
		}
		return;
	}
	else if (body_mass > 0.) {
		if (coords == body_coords) return;
		double r[] = { body_coords[0] - coords[0], body_coords[1] - coords[1] };
		double mr = sqrt(r[0] * r[0] + r[1] * r[1]);
		if (mr < 0.000001) mr = 0.000001;
		double t1 = body_mass / pow(mr, 3) * G;
		double t2 = body_mass / pow(mr, 14) * K;
		if (abs(t1 - t2) < max_accel) {
			holder[0] += t1 * r[0];
			holder[1] += t1 * r[1];
			holder[0] -= t2 * r[0];
			holder[1] -= t2 * r[1];
		}
		return;
	}
	else if (hasNodes) {
		for (BH_tree* i = children; i < children + 4; i++) {
			if (i->node_mass > 0) {
				i->calcAccel(coords, holder);
			}
		}
	}
}