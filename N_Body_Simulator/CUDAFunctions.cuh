#pragma once
#include "Config.h"
#include "constants.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void calculateForcesDEVICE(
    CALCULATION_TYPE* points,
    CALCULATION_TYPE* pointsTMP,
    CALCULATION_TYPE* vels,
    CALCULATION_TYPE* masses,
    Config* config
);

void calculateForcesCUDA(
    CALCULATION_TYPE* points,
    CALCULATION_TYPE* pointsTMP,
    CALCULATION_TYPE* vels,
    CALCULATION_TYPE* masses,
    Config* config,
    unsigned int N
);
