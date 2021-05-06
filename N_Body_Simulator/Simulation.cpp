#include "Simulation.h"
#include <cmath>
#include <chrono>
#include <ctime>
#include "omp.h"
#include <fstream>

using namespace std;

template <typename T>

Simulation<T>::Simulation(Config config) {
    this->config = config;
    points = new T[config.N * 2];
    vels = new T[config.N * 2];
    masses = new T[config.N];
    skip = new bool[config.N];
    tree = BH_tree<T>::newTree(config);

    ifstream cptr("capture.cptr", ios::binary | ios::in);

#define points(i, j) points[i*2 + j]
#define vels(i, j) vels[i*2 + j]

    cptr_loaded = false;
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
    else {
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

    cptr.close();

#undef points
#undef vels
}

#define points(i, j) points[i*2 + j]
#define vels(i, j) vels[i*2 + j]

template <typename T>

void Simulation<T>::calculateForces() {
    if (!config.useBH) {
#pragma omp parallel for
        for (int i = 0; i < config.N; i++) {
            T ca[] = { 0, 0 };
            for (int j = 0; j < config.N; j++) {
                if (i == j) continue;
                T r[] = { points(j, 0) - points(i, 0), points(j, 1) - points(i, 1) };
                T mr = sqrt(r[0] * r[0] + r[1] * r[1]);
                if (mr < 0.000001) mr = 0.000001;
                T t1 = masses[j] / pow(mr, 3) * config.G;
                T t2 = masses[j] / pow(mr, 14) * config.K;
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
    work = true;
    alive = true;
    double updates = 0;
    ofstream rec;
    if (config.record)
        if (cptr_loaded) {
            rec.open("record.rcd", ios::binary | ios::out | ios_base::app);
        }
        else {
            rec.open("record.rcd", ios::binary | ios::out);
            rec.write((char*)&config.N, sizeof(config.N));
            rec.write((char*)&config.DeltaT, sizeof(config.DeltaT));
            unsigned int size = sizeof(T) * config.N * 2;
            rec.write((char*)&size, sizeof(size));
        }
    while (work) {
        if (config.useBH) {
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
                tree->add(points + i * 2, masses[i]);
            }
            treeDepth = tree->depth();
            totalTreeNodes = tree->totalNodeCount();
            activeTreeNodes = tree->activeNodeCount();
        }
        calculateForces();
        for (int i = 0; i < config.N; i++) {
            points(i, 0) += vels(i, 0) * config.DeltaT;
            points(i, 1) += vels(i, 1) * config.DeltaT;
        }
        if (config.record) {
            rec.write((char*)points, sizeof(T) * config.N * 2);
        }
        auto now = chrono::system_clock::now();
        chrono::duration<double> elapsed_seconds = now - start;
        double elaps = elapsed_seconds.count();
        if (elaps < 1) {
            updates++;
        }
        else {
            updates++;
            ups = updates / elaps;
            start = now;
            updates = 0;
        }
    }
    rec.close();
    ofstream cptr("capture.cptr", ios::binary | ios::out);
    cptr.write((char*)points, sizeof(T) * config.N * 2);
    cptr.write((char*)vels, sizeof(T) * config.N * 2);
    cptr.write((char*)masses, sizeof(T) * config.N);
    cptr.close();
    alive = false;
}

#undef points
#undef vels