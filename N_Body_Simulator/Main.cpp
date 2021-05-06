#include <SFML/Graphics.hpp>
#include <chrono>
#include <ctime>
#include <thread>
#include <fstream>
#include "BH_tree.cpp"
#include "omp.h"
#include "constants.h"
#include "Config.h"
#include "Simulation.cpp"

using namespace sf;
using namespace std;

bool drawBH = false;
double fps = 0;
double scale = 1;
bool centrilize = false;
volatile bool cptr_loaded = false;
Config config;
Simulation<CALCULATION_TYPE>* sim;

int main() {
    ContextSettings settings;
    settings.antialiasingLevel = 8;
    RenderWindow window(VideoMode(config.W, config.H), "simulation", Style::Default, settings);
    //window.setFramerateLimit(60);

    auto start = std::chrono::system_clock::now();

    srand(start.time_since_epoch().count());

    config.readConfig("config.cfg");

    BH_tree<CALCULATION_TYPE>* vtree = BH_tree<CALCULATION_TYPE>::newTree(config);

    sim = new Simulation<CALCULATION_TYPE>(config);

    thread th(&Simulation<CALCULATION_TYPE>::run, sim);
    th.detach();

    double frames = 0;

    double posX = 0, posY = 0;

    VertexArray pixels(Points, config.N);

    CALCULATION_TYPE* point_b = new CALCULATION_TYPE[config.N * 2];

#define point_b(i, j) point_b[i*2 + j]

    while (window.isOpen())
    {
        Event event;
        while (window.pollEvent(event))
        {
            int tics;
            switch (event.type) {
            case Event::Closed:
                sim->stop();
                while (sim->alive);
                window.close();
                break;
            case Event::MouseWheelMoved:
                tics = event.mouseWheel.delta;
                scale += tics * 0.1 * scale;
                if (scale <= 0) scale = 0.000000001;
                break;
            case Event::KeyPressed:
                if (event.key.code == Keyboard::W) {
                    posY -= config.scroll_speed / scale;
                }
                if (event.key.code == Keyboard::S) {
                    posY += config.scroll_speed / scale;
                }
                if (event.key.code == Keyboard::A) {
                    posX -= config.scroll_speed / scale;
                }
                if (event.key.code == Keyboard::D) {
                    posX += config.scroll_speed / scale;
                }
                if (event.key.code == Keyboard::B) {
                    drawBH = !drawBH;
                }
                if (event.key.code == Keyboard::C) {
                    centrilize = !centrilize;
                }
                break;
            case Event::Resized:
                Vector2u dims = window.getSize();
                config.W = dims.x;
                config.H = dims.y;
                View newView(Vector2f(config.W/2, config.H/2), Vector2f(config.W, config.H));
                window.setView(newView);
            }
        }

        memcpy(point_b, sim->points, sizeof(CALCULATION_TYPE) * config.N * 2);

        if (drawBH || centrilize) {
            CALCULATION_TYPE maxD = 0;
            for (int i = 0; i < config.N; i++) {
                if (sim->skip[i]) continue;
                maxD = abs(point_b(i, 0)) > maxD ? abs(point_b(i, 0)) : maxD;
                maxD = abs(point_b(i, 1)) > maxD ? abs(point_b(i, 1)) : maxD;
            }
            maxD *= 2;
            maxD += 100;

            vtree->clear();
            vtree->setNew(0, 0, maxD);
            for (int i = 0; i < config.N; i++) {
                if (sim->skip[i]) continue;
                vtree->add(point_b + i * 2, sim->masses[i]);
            }
        }

        if (centrilize) {
            CALCULATION_TYPE center[2]{ 0, 0 };

            //for (int i = 0; i < N; i++) {
            //    center[0] += point_b[i][0];
            //    center[1] += point_b[i][1];
            //}
            //center[0] /= N;
            //center[1] /= N;

            BH_tree<CALCULATION_TYPE>* maxMtree = vtree;
            int maxdpt = vtree->depth();
            for (int i = 0; i < maxdpt; i++) {
                int maxM = 0;
                if (maxMtree->hasNodes) {
                    BH_tree<CALCULATION_TYPE>* start = maxMtree->children;
                    for (BH_tree<CALCULATION_TYPE>* i = start; i < start + 4; i++) {
                        if (i->node_mass > maxM) {
                            maxM = i->node_mass;
                            maxMtree = i;
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

        for (int i = 0; i < config.N; i++) {
            int x = (point_b(i, 0) - posX) * scale + 0.5 * config.W;
            int y = (point_b(i, 1) - posY) * scale + 0.5 * config.H;
            pixels[i].position = Vector2f(x, y);
            pixels[i].color = Color(255, 255, 255, 255);
        }

        window.clear();
        window.draw(pixels);


        if (drawBH) {
            BH_tree<CALCULATION_TYPE>** nodes = vtree->getNodes();

            for (BH_tree<CALCULATION_TYPE>** i = nodes; i < nodes + vtree->activeNodeCount(); i++) {
                BH_tree<CALCULATION_TYPE>* cur = *i;
                float width;
                if (cur->node_width < FLT_MAX) {
                    width = cur->node_width;
                }
                else {
                    width = FLT_MAX;
                }
                if (width * scale >= 1) {
                    RectangleShape rect(Vector2f(width * scale, width * scale));
                    rect.setPosition(
                        Vector2f(
                            config.W / 2 - (width / 2 - cur->center[0] + posX) * scale,
                            config.H / 2 - (width / 2 - cur->center[1] + posY) * scale)
                    );
                    rect.setFillColor(Color(0, 0, 0, 0));
                    rect.setOutlineThickness(0.3f);
                    rect.setOutlineColor(Color(255, 255, 255, 255));
                    if (cur->body_mass > 0) {
                        rect.setFillColor(Color(0, 0, 255, 40));
                    }
                    window.draw(rect);
                }
            }
            delete[] nodes;
        }

        auto now = chrono::system_clock::now();
        chrono::duration<double> elapsed_seconds = now - start;
        double elaps = elapsed_seconds.count();
        if (elaps < 1) {
            frames++;
        }
        else {
            frames++;
            fps = frames / elaps;
            start = now;
            frames = 0;
        }


        window.display();
        window.setTitle(to_string(fps) + " / " + to_string(sim->ups) + " / " + to_string(sim->ups * config.DeltaT) + " / " + to_string(sim->treeDepth) + " / " + to_string(sim->totalTreeNodes) + " / " + to_string(sim->activeTreeNodes));
    }
}