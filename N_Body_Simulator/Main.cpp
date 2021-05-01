#include <SFML/Graphics.hpp>
#include <chrono>
#include <ctime>
#include <thread>
#include <fstream>
#include "structures.h"
#include "omp.h"
#include "constants.h"

using namespace sf;
using namespace std;

bool drawBH = false;
double* points = new double[N * 2];
double* vels = new double[N * 2];
double* masses = new double[N];
bool* skip = new bool[N];
Uint8 pixels[W * H * 4];
bool run = true;
double fps = 0, ups = 0;
double scale = 1;
bool record = record_default;
BH_tree tree;
int comp = 0;
int depth = 0;
bool useBH = useBH_default;
bool centrilize = false;
volatile bool ended = false;
volatile bool cptr_loaded = false;


#define points(i, j) points[i*2 + j]
#define vels(i, j) vels[i*2 + j]

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


void calculateForces() {
    if (!useBH) {
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            double ca[] = { 0, 0 };
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                double r[] = { points(j, 0) - points(i, 0), points(j, 1) - points(i, 1) };
                double mr = sqrt(r[0] * r[0] + r[1] * r[1]);
                if (mr < 0.000001) mr = 0.000001;
                double t1 = masses[j] / pow(mr, 3) * G;
                double t2 = masses[j] / pow(mr, 14) * K;
                ca[0] += t1 * r[0];
                ca[1] += t1 * r[1];
                ca[0] -= t2 * r[0];
                ca[1] -= t2 * r[1];
            }
            vels(i, 0) += ca[0] * DeltaT;
            vels(i, 1) += ca[1] * DeltaT;
        }
    }
    else {
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            if (!skip[i]) {
                double* ca;
                ca = tree.calcAccel(points + i * 2);
                vels(i, 0) += ca[0] * DeltaT;
                vels(i, 1) += ca[1] * DeltaT;
                if (ca[0] * ca[0] + ca[1] * ca[1] < min_accel && points(i, 0) * points(i, 0) + points(i, 1) * points(i, 1) > max_dist) {
                    skip[i] = true;
                }
                delete[] ca;
            }
        }
    }
}

void compute() {
    auto start = std::chrono::system_clock::now();
    double updates = 0;
    ofstream rec;
    if(record)
        if (cptr_loaded) {
            rec.open("record.rcd", ios::binary | ios::out | ios_base::app);
        }
        else {
            rec.open("record.rcd", ios::binary | ios::out);
            rec << N << DeltaT << sizeof(points);
        }
    while (run) {
        if (useBH) {
            double maxD = 0;
            for (int i = 0; i < N; i++) {
                if (skip[i]) continue;
                maxD = abs(points(i, 0)) > maxD ? abs(points(i, 0)) : maxD;
                maxD = abs(points(i, 1)) > maxD ? abs(points(i, 1)) : maxD;
            }
            maxD *= 2;
            maxD += 100;
            tree.clear();
            tree.setNew(0, 0, maxD);
            for (int i = 0; i < N; i++) {
                if (skip[i]) continue;
                tree.add(points + i * 2, masses[i]);
            }
            depth = tree.depth();
        }
        calculateForces();
        for (int i = 0; i < N; i++) {
            points(i, 0) += vels(i, 0) * DeltaT;
            points(i, 1) += vels(i, 1) * DeltaT;
        }
        if (record) {
            rec << points;
        }
        auto now = chrono::system_clock::now();
        chrono::duration<double> elapsed_seconds = now - start;
        double elaps = elapsed_seconds.count();
        if (elaps < 1) {
            updates++;
        }
        else {
            ups = updates / elaps;
            start = now;
            updates = 0;
        }
    }
    rec.close();
    ofstream cptr("capture.cptr", ios::binary | ios::out);
    cptr << points << vels << masses;
    ended = true;
}

int main() {
    ContextSettings settings;
    settings.antialiasingLevel = 8;
    RenderWindow window(VideoMode(W, H), "simulation", Style::Default, settings);
    //window.setFramerateLimit(60);

    auto start = std::chrono::system_clock::now();

    srand(start.time_since_epoch().count());

    readConfig();

    ifstream cptr("capture.cptr", ios::binary | ios::in);
    
    unsigned int cptr_s;
    cptr.seekg(0, cptr._Seekend);
    cptr_s = cptr.tellg();
    cptr.seekg(0, cptr._Seekbeg);
    if (cptr_s == sizeof(points) + sizeof(vels) + sizeof(masses)) {
        cptr >> points >> vels >> masses;
        cptr_loaded = true;
    }
    else {
        for (int i = 0; i < N; i++) {
            points(i, 0) = ((double)rand() / RAND_MAX) * W - 0.5 * W;
            points(i, 1) = ((double)rand() / RAND_MAX) * H - 0.5 * H;
            //vels[i][0] = ((double)rand() / RAND_MAX) * 2 * MAX_START_SPEED - MAX_START_SPEED;
            //vels[i][1] = ((double)rand() / RAND_MAX) * 2 * MAX_START_SPEED - MAX_START_SPEED;
            // 
            //vels[i][0] = points[i][1] * MAX_START_SPEED / sqrt(pow(points[i][1], 2) + pow(points[i][0], 2));
            //vels[i][1] = -points[i][0] * MAX_START_SPEED / sqrt(pow(points[i][1], 2) + pow(points[i][0], 2));

            vels(i, 0) = points(i, 1) * MAX_START_SPEED;
            vels(i, 1) = -points(i, 0) * MAX_START_SPEED;
            masses[i] = 100;
            skip[i] = false;
        }
    }

    cptr.close();

    thread th(compute);
    th.detach();

    double frames = 0;

    double posX = 0, posY = 0;

    BH_tree* vtree = new BH_tree();

    double* point_b = new double[N * 2];

#define point_b(i, j) points[i*2 + j]

    while (window.isOpen())
    {
        Event event;
        while (window.pollEvent(event))
        {
            int tics;
            switch (event.type) {
            case Event::Closed:
                run = false;
                while (!ended);
                window.close();
                break;
            case Event::MouseWheelMoved:
                tics = event.mouseWheel.delta;
                scale += tics * 0.1 * scale;
                if (scale <= 0) scale = 0.000000001;
                break;
            case Event::KeyPressed:
                if (event.key.code == Keyboard::W) {
                    posY -= scroll_speed / scale;
                }
                if (event.key.code == Keyboard::S) {
                    posY += scroll_speed / scale;
                }
                if (event.key.code == Keyboard::A) {
                    posX -= scroll_speed / scale;
                }
                if (event.key.code == Keyboard::D) {
                    posX += scroll_speed / scale;
                }
                if (event.key.code == Keyboard::B) {
                    drawBH = !drawBH;
                }
                if (event.key.code == Keyboard::C) {
                    centrilize = !centrilize;
                }
            }
        }

        Texture tex;
        tex.create(W, H);
        Sprite sp(tex);

        for (int i = 0; i < N; i++) {
            point_b(i, 0) = points(i, 0);
            point_b(i, 1) = points(i, 1);
        }

        if (drawBH || centrilize) {
            double maxD = 0;
            for (int i = 0; i < N; i++) {
                if (skip[i]) continue;
                maxD = abs(point_b(i, 0)) > maxD ? abs(point_b(i, 0)) : maxD;
                maxD = abs(point_b(i, 1)) > maxD ? abs(point_b(i, 1)) : maxD;
            }
            maxD *= 2;
            maxD += 100;

            vtree->clear();
            vtree->setNew(0, 0, maxD);
            for (int i = 0; i < N; i++) {
                if (skip[i]) continue;
                vtree->add(point_b + i * 2, masses[i]);
            }
        }

        if (centrilize) {
            double center[2]{ 0, 0 };

            //for (int i = 0; i < N; i++) {
            //    center[0] += point_b[i][0];
            //    center[1] += point_b[i][1];
            //}
            //center[0] /= N;
            //center[1] /= N;

            BH_tree* maxMtree = vtree;
            int maxdpt = vtree->depth();
            for (int i = 0; i < maxdpt; i++) {
                int maxM = 0;
                if (maxMtree->hasNodes) {
                    BH_tree** start = begin(maxMtree->children);
                    for (BH_tree** i = start; i < start + 4; i++) {
                        if ((*i)->node_mass > maxM) {
                            maxM = (*i)->node_mass;
                            maxMtree = (*i);
                        }
                    }
                }
                else break;
            }

            center[0] = maxMtree->body_coords[0];
            center[1] = maxMtree->body_coords[1];

            posX = center[0];
            posY = center[1];
        }

        for (int i = 0; i < W * H * 4; i++) pixels[i] = 0;

        for (int i = 0; i < N; i++) {
            int x = (point_b(i, 0) - posX) * scale + 0.5 * W;
            int y = (point_b(i, 1) - posY) * scale + 0.5 * H;
            if (x >= 0 && x < W && y >= 0 && y < H) {
                int p = 4 * (y * W + x);
                pixels[p] = 255;
                pixels[p + 1] = 255;
                pixels[p + 2] = 255;
                pixels[p + 3] = 255;
                //pixels[p + 3] += pixels[p + 3] + 255 * scale <= 255 ? (int)(255 * scale) : 255;
            }
        }

        tex.update(pixels);

        window.clear();
        window.draw(sp);

        if (drawBH) {
            vector<BH_tree*> nodes = vtree->getNodes();

            for (int i = 0; i < nodes.size(); i++) {
                BH_tree* cur = nodes[i];
                float width;
                if (cur->node_width < FLT_MAX) {
                    width = cur->node_width;
                }
                else {
                    width = FLT_MAX;
                }
                RectangleShape rect(Vector2f(width * scale, width * scale));
                rect.setPosition(Vector2f(W / 2 - (width / 2 - cur->center[0] + posX) * scale, H / 2 - (width / 2 - cur->center[1] + posY) * scale));
                rect.setFillColor(Color(0, 0, 0, 0));
                rect.setOutlineThickness(0.3f);
                rect.setOutlineColor(Color(255, 255, 255, 255));
                if (cur->body_mass > 0) {
                    rect.setFillColor(Color(0, 0, 255, 40));
                }
                window.draw(rect);
            }
        }

        window.display();

        auto now = chrono::system_clock::now();
        chrono::duration<double> elapsed_seconds = now - start;
        double elaps = elapsed_seconds.count();
        if (elaps < 1) {
            frames++;
        }
        else {
            fps = frames / elaps;
            start = now;
            frames = 0;
        }
        window.setTitle(to_string(fps) + " / " + to_string(ups) + " / " + to_string(ups * DeltaT) + " / " + to_string(depth));
    }
}