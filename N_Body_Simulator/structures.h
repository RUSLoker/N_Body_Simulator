#pragma once
#include <vector>
#include <fstream>

using namespace std;

class BH_tree
{
public:
	double* body_coords;
	double body_mass = -1;
	double center[2];
	double node_mass;
	double node_width;
	BH_tree* children[4];
	bool hasNodes = false;

	BH_tree(double x, double y, double width);
	BH_tree();

	void add(double* coords, double mass);

	void setNew(double x, double y, double width);

	void clear();

	vector<BH_tree*> getNodes();

	double* calcAccel(double* coords);

	~BH_tree();

	unsigned int depth() {
		return node_depth;
	}

private:
	unsigned int node_depth = 1;

	void calcAccel(double* coords, double* holder);
};

template <typename T>

std::ofstream& operator<<(std::ofstream& out, const T& data) {
	out.write((char*)&data, sizeof(data));
	return out;
}

template <typename T>

std::ifstream& operator>>(std::ifstream& in, T& data) {
	in.read((char*)&data, sizeof(data));
	return in;
}