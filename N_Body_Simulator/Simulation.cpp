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
    T* propsArr = (T*)malloc(sizeof(T) * config.N * 5);
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
            points(i, 0) = ((T)rand() / RAND_MAX) * config.W - 0.5 * config.W;
            points(i, 1) = ((T)rand() / RAND_MAX) * config.H - 0.5 * config.H;
            //vels[i][0] = ((double)rand() / RAND_MAX) * 2 * MAX_START_SPEED - MAX_START_SPEED;
            //vels[i][1] = ((double)rand() / RAND_MAX) * 2 * MAX_START_SPEED - MAX_START_SPEED;
            // 
            //vels[i][0] = points[i][1] * MAX_START_SPEED / sqrt(pow(points[i][1], 2) + pow(points[i][0], 2));
            //vels[i][1] = -points[i][0] * MAX_START_SPEED / sqrt(pow(points[i][1], 2) + pow(points[i][0], 2));

            vels(i, 0) = points(i, 1) * config.MAX_START_SPEED;
            vels(i, 1) = -points(i, 0) * config.MAX_START_SPEED;
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
        cudaMalloc(&propsArr_d, sizeof(T) * config.N * 5);
        points_d = propsArr_d;
        vels_d = propsArr_d + config.N * 2;
        masses_d = propsArr_d + config.N * 4;
        cudaMalloc(&skip_d, sizeof(bool) * config.N);
        tree_d = BH_tree<T>::newTreeCUDA(config);
        cudaMemcpy(propsArr_d, propsArr, sizeof(T) * config.N * 5, cudaMemcpyHostToDevice);
        cudaMemcpy(skip_d, skip, sizeof(bool) * config.N, cudaMemcpyHostToDevice);
    }

#undef points
#undef vels
}

#define points(i, j) points[i*2 + j]
#define vels(i, j) vels[i*2 + j]

template <typename T>

void Simulation<T>::calculateForces() {
    if (config.useCUDA) {
        calculateForcesCUDA(points_d, vels_d, masses_d, skip_d, tree_d, config_d, config.N);
    }
    else if (!config.useBH) {
#pragma omp parallel for
        for (int i = 0; i < config.N; i++) {
            T ca[] = { 0, 0 };
            for (int j = 0; j < config.N; j++) {
                if (i == j) continue;
                T r[] = { points(j, 0) - points(i, 0), points(j, 1) - points(i, 1) };
                T mr = sqrt(r[0] * r[0] + r[1] * r[1]);
                if (mr < 0.000001) mr = 0.000001;
                T t1 = masses[j] / pow(mr, 3) * config.G;
                T t2 = masses[j] / pow(mr + 0.6, 14) * config.K;
                if (abs(t1 - t2) < config.max_accel) {
                    ca[0] += t1 * r[0];
                    ca[1] += t1 * r[1];
                    ca[0] -= t2 * r[0];
                    ca[1] -= t2 * r[1];
                }
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
    if (config.record) {
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
        rec.write((char*)points, sizeof(T) * config.N * 2);
    }
    while (work) {
        if (config.useBH) {
            makeTree();
        }
        calculateForces();
        if (!config.useBH) {
            for (int i = 0; i < config.N; i++) {
                points(i, 0) += vels(i, 0) * config.DeltaT;
                points(i, 1) += vels(i, 1) * config.DeltaT;
            }
        }
        if (config.useCUDA) {
            cudaMemcpy(points, points_d, sizeof(T) * config.N * 2, cudaMemcpyDeviceToHost);
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
    rec.close();
    ofstream cptr(config.capture_path, ios::binary | ios::out);
    cptr.write((char*)points, sizeof(T) * config.N * 2);
    cptr.write((char*)vels, sizeof(T) * config.N * 2);
    cptr.write((char*)masses, sizeof(T) * config.N);
    cptr.close();
    alive = false;
}

template <typename T>

void Simulation<T>::makeTree() {
    if (config.useCUDA) {
        makeTreeCUDA(points_d, vels_d, masses_d, skip_d, tree_d, config_d);
        BH_tree<T>* dt = (BH_tree<T>*)malloc(sizeof(BH_tree<T>));
        cudaMemcpy(dt, tree_d, sizeof(BH_tree<T>), cudaMemcpyDeviceToHost);
        bool hn = dt->hasNodes;
    }
    else {
        T maxD = 0;
        for (int i = 0; i < config.N; i++) {
            points(i, 0) += vels(i, 0) * config.DeltaT;
            points(i, 1) += vels(i, 1) * config.DeltaT;
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
            tree->add(points + i * 2, masses[i]);
        }
        treeDepth = tree->depth();
        totalTreeNodes = tree->totalNodeCount();
        activeTreeNodes = tree->activeNodeCount();
    } 
}

#undef points
#undef vels