#include <cmath>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaFunctions.cuh"

#define points(i, j) points[i*2 + j]
#define vels(i, j) vels[i*2 + j]

__global__ void calculateForcesDEVICE(
    CALCULATION_TYPE* points,
    CALCULATION_TYPE* vels,
    CALCULATION_TYPE* masses,
    bool* skip,
    BH_tree<CALCULATION_TYPE>* tree,
    Config* config
) {
    int i = threadIdx.x + 1024 * blockIdx.x;
    if ( i < config->N && !skip[i]) {
        CALCULATION_TYPE* ca;
        ca = tree->calcAccel(points + i * 2);
        vels(i, 0) += ca[0] * config->DeltaT;
        vels(i, 1) += ca[1] * config->DeltaT;
        if (ca[0] * ca[0] + ca[1] * ca[1] < config->min_accel
            && points(i, 0) * points(i, 0) + points(i, 1) * points(i, 1) > config->max_dist) {
            skip[i] = true;
        }
        delete[] ca;
    }
}

void calculateForcesCUDA(
    CALCULATION_TYPE* points,
    CALCULATION_TYPE* vels,
    CALCULATION_TYPE* masses,
    bool* skip,
    BH_tree<CALCULATION_TYPE>* tree,
    Config* config,
    unsigned int N
) {
    dim3 blockSize;
    dim3 gridSize;
    int threadNum;
    threadNum = 1024;
    blockSize = dim3(threadNum, 1, 1);
    gridSize = dim3(N / threadNum + 1, 1, 1);
    calculateForcesDEVICE<<<gridSize, blockSize >>> (points, vels, masses, skip, tree, config);
}

__global__ void makeTreeDEVICE(
    CALCULATION_TYPE* points,
    CALCULATION_TYPE* vels,
    CALCULATION_TYPE* masses,
    bool* skip,
    BH_tree<CALCULATION_TYPE>* tree,
    Config* config
) {
    CALCULATION_TYPE maxD = 0;
    for (int i = 0; i < config->N; i++) {
        points(i, 0) += vels(i, 0) * config->DeltaT;
        points(i, 1) += vels(i, 1) * config->DeltaT;
        if (skip[i]) continue;
        maxD = abs(points(i, 0)) > maxD ? abs(points(i, 0)) : maxD;
        maxD = abs(points(i, 1)) > maxD ? abs(points(i, 1)) : maxD;
    }
    maxD *= 2;
    maxD += 100;
    tree->clear();
    tree->setNew(0, 0, maxD);
    //for (int i = 0; i < config->N; i++) {
    //    if (skip[i]) continue;
    //    tree->add(points + i * 2, masses[i]);
    //}
    //treeDepth = tree->depth();
    //totalTreeNodes = tree->totalNodeCount();
    //activeTreeNodes = tree->activeNodeCount();
}

void makeTreeCUDA(
    CALCULATION_TYPE* points,
    CALCULATION_TYPE* vels,
    CALCULATION_TYPE* masses,
    bool* skip,
    BH_tree<CALCULATION_TYPE>* tree,
    Config* config
) {
    dim3 blockSize;
    dim3 gridSize;
    blockSize = dim3(1, 1, 1);
    gridSize = dim3(1, 1, 1);
    makeTreeDEVICE <<<1, 1 >>> (points, vels, masses, skip, tree, config);
}