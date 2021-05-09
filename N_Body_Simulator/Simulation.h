#pragma once
#include "Config.h"
#include "BH_tree.cuh"
#include "constants.h"
#include <chrono>
#include <ctime>

using namespace std;

template <typename T>

class Simulation
{
public:

	T* points;
	T* vels;
	T* masses;
	bool* skip;
	double UPS = 0;
	double totalUpdates = 0;
	volatile bool alive = false;
	SIZE_TYPE totalTreeNodes = 0;
	SIZE_TYPE activeTreeNodes = 0;
	int treeDepth = 0;

	Simulation(Config config);

	void run();

	void stop() {
		work = false;
	}

	double evaluationTime() {
		auto now = chrono::system_clock::now();
		chrono::duration<double> elapsed_seconds = now - startTime;
		return elapsed_seconds.count();
	}

	double meanUPS() {
		return totalUpdates / evaluationTime();
	}

private:

	Config config;
	Config* config_d;
	BH_tree<T>* tree;
	BH_tree<T>* tree_d;
	T* points_d;
	T* vels_d;
	T* masses_d;
	bool* skip_d;
	bool cptr_loaded;
	volatile bool work = false;
	chrono::system_clock::time_point startTime;

	Simulation() {};

	void calculateForces();

	void makeTree();
};
