#include "BH_tree.cuh"
#include "constants.h"
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

template <typename T>

BH_tree<T>* BH_tree<T>::newTree(Config config) {
	BH_tree<T>* cache = (BH_tree<T>*)malloc(config.max_cache);
	BH_tree<T>** next = new BH_tree<T> *();
	*next = cache;
	SIZE_TYPE* counter = new SIZE_TYPE(0);
	cache[0].tree_config = new Config();
	*cache[0].tree_config = config;
	cache[0].newNode(cache, next, counter, cache[0].tree_config);
	
	return cache;
}

template <typename T>

BH_tree<T>* BH_tree<T>::newTreeCUDA(Config config) {
	BH_tree<T>* cache;
	cudaMalloc(&cache, config.max_cache);
	BH_tree<T>** next = new BH_tree<T> *();
	*next = cache;
	SIZE_TYPE* counter;
	cudaMalloc(&counter, sizeof(SIZE_TYPE));

	BH_tree<T> first;
	cudaMalloc(&first.tree_config, sizeof(Config));
	first.newNode(cache, next, counter, first.tree_config);
	
	cudaMemcpy(cache, &first, sizeof(BH_tree<T>), cudaMemcpyHostToDevice);
	return cache;
}

template <typename T>

BH_tree<T>* BH_tree<T>::getNodes() {
	return node_cache;
}

template <typename T>

BH_tree<T>::~BH_tree() {
	if (this == node_cache) {
		delete next_caching;
		delete[] node_cache;
	}
}