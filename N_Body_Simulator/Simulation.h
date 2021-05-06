#pragma once
#include "Config.h"
#include "BH_tree.h"
#include "constants.h"

using namespace std;

template <typename T>

class Simulation
{
public:

	T* points;
	T* vels;
	T* masses;
	bool* skip;
	double ups = 0;
	volatile bool alive = false;
	SIZE_TYPE totalTreeNodes = 0;
	SIZE_TYPE activeTreeNodes = 0;
	int treeDepth = 0;

	Simulation(Config config);

	void run();

	void stop() {
		work = false;
	}

private:

	Config config;
	BH_tree<T>* tree;
	bool cptr_loaded;
	volatile bool work = false;

	Simulation() {};

	void calculateForces();
};
