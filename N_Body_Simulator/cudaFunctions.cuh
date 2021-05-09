#pragma once
#include "Config.h"
#include "BH_tree.cuh"
#include "constants.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void calculateForcesDEVICE(
    CALCULATION_TYPE* points,
    CALCULATION_TYPE* vels,
    CALCULATION_TYPE* masses,
    bool* skip,
    BH_tree<CALCULATION_TYPE>* tree,
    Config* config
);

void calculateForcesCUDA(
    CALCULATION_TYPE* points,
    CALCULATION_TYPE* vels,
    CALCULATION_TYPE* masses,
    bool* skip,
    BH_tree<CALCULATION_TYPE>* tree,
    Config* config,
    unsigned int N
);

__global__ void makeTreeDEVICE(
    CALCULATION_TYPE* points,
    CALCULATION_TYPE* vels,
    CALCULATION_TYPE* masses,
    bool* skip,
    BH_tree<CALCULATION_TYPE>* tree,
    Config* config
);

void makeTreeCUDA(
    CALCULATION_TYPE* points,
    CALCULATION_TYPE* vels,
    CALCULATION_TYPE* masses,
    bool* skip,
    BH_tree<CALCULATION_TYPE>* tree,
    Config* config
);