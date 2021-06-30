#pragma once
#include "Config.h"
#include "BH_tree.h"
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
	size_t totalTreeNodes = 0;
	size_t activeTreeNodes = 0;
	int treeDepth = 0;
	exception except;

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

	void getPoints(T* dst);

	~Simulation() {
		free(points);
		delete[] skip;
		tree->~BH_tree();
	}

private:
	Config config;
	Config* config_d;
	BH_tree<T>* tree;
	T* points_d;
	T* pointsTMP_d;
	T* vels_d;
	T* velsTMP_d;
	T* masses_d;
	bool cptr_loaded;
	volatile bool work = false;
	chrono::system_clock::time_point startTime;

	Simulation() {};

	void calculateForces();
};
