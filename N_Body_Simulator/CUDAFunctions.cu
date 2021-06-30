#include <cmath>
#include "Config.h"
#include "constants.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaFunctions.cuh"

#define points(i, j) points[i*2 + j]
#define pointsTMP(i, j) pointsTMP[i*2 + j]
#define vels(i, j) vels[i*2 + j]
#define velsTMP(i, j) velsTMP[i*2 + j]

template <typename T> 
__device__ int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template <typename T>

__global__ void calculateForcesDEVICE(
    T* points,
    T* pointsTMP,
    T* vels,
    T* velsTMP,
    T* masses,
    Config* config,
    int threadsPerBlock
) {
    int i = threadIdx.x + threadsPerBlock * blockIdx.x;
    if (i < config->N) {
        T ca[] = { 0, 0 };
        for (int j = 0; j < config->N; j++) {
            if (i == j) continue;
            T r[] = { points(j, 0) - points(i, 0), points(j, 1) - points(i, 1) };
            T mr = sqrt(r[0] * r[0] + r[1] * r[1]);
            T t1 = masses[j] / pow(mr + 1.0f, 3) * config->G;
            ca[0] += t1 * r[0];
            ca[1] += t1 * r[1];
        }
        velsTMP(i, 0) = vels(i, 0) + ca[0] * config->DeltaT;
        velsTMP(i, 1) = vels(i, 1) + ca[1] * config->DeltaT;
        pointsTMP(i, 0) = points(i, 0) + velsTMP(i, 0) * config->DeltaT;
        pointsTMP(i, 1) = points(i, 1) + velsTMP(i, 1) * config->DeltaT;
    }
}

void calculateForcesCUDA(
    CALCULATION_TYPE* points,
    CALCULATION_TYPE* pointsTMP,
    CALCULATION_TYPE* vels,
    CALCULATION_TYPE* velsTMP,
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
    threadNum = 128;
    blockSize = dim3(threadNum, 1, 1);
    gridSize = dim3(N / threadNum + 1, 1, 1);
    calculateForcesDEVICE<CALCULATION_TYPE> <<<gridSize, blockSize >>> (points, pointsTMP, vels, velsTMP, masses, config, threadNum);
}