#include <cmath>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaFunctions.cuh"

#define points(i, j) points[i*2 + j]
#define pointsTMP(i, j) pointsTMP[i*2 + j]
#define vels(i, j) vels[i*2 + j]

__global__ void calculateForcesDEVICE(
    CALCULATION_TYPE* points,
    CALCULATION_TYPE* pointsTMP,
    CALCULATION_TYPE* vels,
    CALCULATION_TYPE* masses,
    Config* config,
    int threadsPerBlock
) {
    int i = threadIdx.x + threadsPerBlock * blockIdx.x;
    if (i < config->N) {
            CALCULATION_TYPE ca[] = { 0, 0 };
            for (int j = 0; j < config->N; j++) {
                if (i == j) continue;
                CALCULATION_TYPE r[] = { points(j, 0) - points(i, 0), points(j, 1) - points(i, 1) };
                CALCULATION_TYPE mr = sqrt(r[0] * r[0] + r[1] * r[1]);
                if (mr < 0.000001) mr = 0.000001;
                CALCULATION_TYPE t1 = masses[j] / pow(mr, 3) * config->G;
                CALCULATION_TYPE t2 = masses[j] / pow(mr + 0.6, 14) * config->K;
                if (abs(t1 - t2) < config->max_accel) {
                    ca[0] += t1 * r[0];
                    ca[1] += t1 * r[1];
                    ca[0] -= t2 * r[0];
                    ca[1] -= t2 * r[1];
                }
            }
            vels(i, 0) += ca[0] * config->DeltaT;
            vels(i, 1) += ca[1] * config->DeltaT;
            pointsTMP(i, 0) = points(i, 0) + vels(i, 0) * config->DeltaT;
            pointsTMP(i, 1) = points(i, 1) + vels(i, 1) * config->DeltaT;
    }
}

void calculateForcesCUDA(
    CALCULATION_TYPE* points,
    CALCULATION_TYPE* pointsTMP,
    CALCULATION_TYPE* vels,
    CALCULATION_TYPE* masses,
    Config* config,
    unsigned int N
) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    int mp = props.multiProcessorCount;
    dim3 blockSize;
    dim3 gridSize;
    int threadNum;
    threadNum = N / mp + 1;
    blockSize = dim3(threadNum, 1, 1);
    gridSize = dim3(mp, 1, 1);
    calculateForcesDEVICE <<<gridSize, blockSize >>> (points, pointsTMP, vels, masses, config, threadNum);
}