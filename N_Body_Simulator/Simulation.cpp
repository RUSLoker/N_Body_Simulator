#include "Simulation.h"
#include <cmath>
#include <chrono>
#include <ctime>
#include "omp.h"
#include "cudaFunctions.cuh"
#include <fstream>

using namespace std;

template <typename T>

Simulation<T>::Simulation(Config config) {
    this->config = config;
    T* propsArr = (T*)malloc(config.N * (size_t)5 * sizeof(T));
    points = propsArr;
    vels = propsArr + config.N * 2;
    masses = propsArr + config.N * 4;
    skip = new bool[config.N];
    tree = BH_tree<T>::newTree(config);

#define points(i, j) points[i*2 + j]
#define vels(i, j) vels[i*2 + j]

    cptr_loaded = false;
    if (config.read_capture) {
        ifstream cptr(config.capture_path, ios::binary | ios::in);
        unsigned int cptr_s;
        cptr.seekg(0, cptr._Seekend);
        cptr_s = cptr.tellg();
        cptr.seekg(0, cptr._Seekbeg);
        if (cptr_s == sizeof(T) * config.N * 5) {
            cptr.read((char*)points, sizeof(T) * config.N * 2);
            cptr.read((char*)vels, sizeof(T) * config.N * 2);
            cptr.read((char*)masses, sizeof(T) * config.N);
            cptr_loaded = true;
        }
        cptr.close();
    }
    if(!cptr_loaded) {
        for (int i = 0; i < config.N; i++) {
            T dst = pow(((T)rand() / RAND_MAX), 10) * config.W * 0.5;
            T angl = ((T)rand() / RAND_MAX) * 2 * PI;
            points(i, 0) = dst * cos(angl);
            points(i, 1) = dst * sin(angl);

            int sgn;
            if (rand() > RAND_MAX * 0.) {
                sgn = 1;
            }
            else {
                sgn = -1;
            }
            vels(i, 0) = sgn * (points(i, 1) * config.MAX_START_SPEED - points(i, 0));
            vels(i, 1) = sgn * (-points(i, 0) * config.MAX_START_SPEED - points(i, 1));
            masses[i] = 100;
            skip[i] = false;
        }
    }
    for (int i = 0; i < config.N; i++) {
        skip[i] = false;
    }

    if (config.useCUDA) {
        cudaMalloc(&config_d, sizeof(Config));
        cudaMemcpy(config_d, &config, sizeof(Config), cudaMemcpyHostToDevice);
        T* propsArr_d;
        cudaMalloc(&propsArr_d, sizeof(T) * config.N * 9);
        pointsTMP_d = propsArr_d;
        points_d = propsArr_d + config.N * 2;
        vels_d = propsArr_d + config.N * 4;
        velsTMP_d = propsArr_d + config.N * 6;
        masses_d = propsArr_d + config.N * 8;
        cudaMemcpy(points_d, points, sizeof(T) * config.N * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(vels_d, vels, sizeof(T) * config.N * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(masses_d, masses, sizeof(T) * config.N, cudaMemcpyHostToDevice);
    }

#undef points
#undef vels
}

#define points(i, j) points[i*2 + j]
#define vels(i, j) vels[i*2 + j]

template <typename T>

void Simulation<T>::calculateForces() {
    if (config.useCUDA) {
        calculateForcesCUDA(points_d, pointsTMP_d, vels_d, velsTMP_d, masses_d, config_d, config.N);
        T* tmp = pointsTMP_d;
        pointsTMP_d = points_d;
        points_d = tmp; 
        tmp = velsTMP_d;
        velsTMP_d = vels_d;
        vels_d = tmp;
    }
    else if (!config.useBH) {
#pragma omp parallel for
        for (int i = 0; i < config.N; i++) {
            T ca[] = { 0, 0 };
            for (int j = 0; j < config.N; j++) {
                if (i == j) continue;
                T r[] = { points(j, 0) - points(i, 0), points(j, 1) - points(i, 1) };
                T mr = sqrt(r[0] * r[0] + r[1] * r[1]);
                T t1 = masses[j] / pow(mr + 1.0f, 3) * config.G;
                ca[0] += t1 * r[0];
                ca[1] += t1 * r[1];
            }
            vels(i, 0) += ca[0] * config.DeltaT;
            vels(i, 1) += ca[1] * config.DeltaT;
        }
    }
    else {
#pragma omp parallel for
        for (int i = 0; i < config.N; i++) {
            if (!skip[i]) {
                T* ca;
                ca = tree->calcAccel(points + i * 2);
                vels(i, 0) += ca[0] * config.DeltaT;
                vels(i, 1) += ca[1] * config.DeltaT;
                if (ca[0] * ca[0] + ca[1] * ca[1] < config.min_accel
                    && points(i, 0) * points(i, 0) + points(i, 1) * points(i, 1) > config.max_dist) {
                    skip[i] = true;
                }
                delete[] ca;
            }
        }
    }

}

template <typename T>

void Simulation<T>::run() {
    auto start = std::chrono::system_clock::now();
    startTime = start;
    work = true;
    alive = true;
    double updates = 0;
    ofstream rec;
    cudaStream_t memReading; //Stream for reading device memory when CUDA is used
    cudaStreamCreate(&memReading);
    if (config.record)
        if (cptr_loaded) {
            rec.open(config.record_path, ios::binary | ios::out | ios_base::app);
        }
        else {
            rec.open(config.record_path, ios::binary | ios::out);
            rec.write((char*)&config.N, sizeof(config.N));
            rec.write((char*)&config.DeltaT, sizeof(config.DeltaT));
            unsigned int size = sizeof(T) * config.N * 2;
            rec.write((char*)&size, sizeof(size));
        }
    while (work) {
        if (config.useCUDA) {
            cudaMemcpyAsync(points, points_d, sizeof(T) * config.N * 2, cudaMemcpyDeviceToHost, memReading);
            cudaDeviceSynchronize();
        }
        if (config.useBH && !config.useCUDA) {
            T maxD = 0;
            for (int i = 0; i < config.N; i++) {
                if (skip[i]) continue;
                maxD = abs(points(i, 0)) > maxD ? abs(points(i, 0)) : maxD;
                maxD = abs(points(i, 1)) > maxD ? abs(points(i, 1)) : maxD;
            }
            maxD *= 2;
            maxD += 100;
            tree->clear();
            tree->setNew(0, 0, maxD);
            for (int i = 0; i < config.N; i++) {
                if (skip[i]) continue;
                try {
                    tree->add(points + i * 2, masses[i]);
                }
                catch (exception e) {
                    except = e;
                    goto END_RUN;
                }
            }
            treeDepth = tree->depth();
            totalTreeNodes = tree->totalNodeCount();
            activeTreeNodes = tree->activeNodeCount();
        }
        calculateForces();
        if (!config.useCUDA)  {
            for (int i = 0; i < config.N; i++) {
                points(i, 0) += vels(i, 0) * config.DeltaT;
                points(i, 1) += vels(i, 1) * config.DeltaT;
            }
        }
        if (config.record) {
            rec.write((char*)points, sizeof(T) * config.N * 2);
        }
        auto now = chrono::system_clock::now();
        chrono::duration<double> elapsed_seconds = now - start;
        double elaps = elapsed_seconds.count();
        updates++;
        totalUpdates++;
        if (elaps >= 1) {
            UPS = updates / elaps;
            start = now;
            updates = 0;
        }
    }
    END_RUN:
    rec.close();
    ofstream cptr(config.capture_path, ios::binary | ios::out);
    cptr.write((char*)points, sizeof(T) * config.N * 2);
    cptr.write((char*)vels, sizeof(T) * config.N * 2);
    cptr.write((char*)masses, sizeof(T) * config.N);
    cptr.close();
    alive = false;
}

template <typename T>

void Simulation<T>::getPoints(T* dst) {
    memcpy(dst, points, sizeof(T) * config.N * 2);
}

#undef points
#undef vels