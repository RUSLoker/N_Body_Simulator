#include <SFML/Graphics.hpp>
#include <chrono>
#include <ctime>
#include <thread>
#include <iomanip>
#include <fstream>
#include "BH_tree.cpp"
#include "omp.h"
#include "constants.h"
#include "Config.h"
#include "Simulation.cpp"


using namespace sf;
using namespace std;

bool drawBH = false;
volatile double fps = 0;
double scale = 1;
bool centrilize = false;
bool show_info = false;
bool show_config_info = false;
volatile bool cptr_loaded = false;
Config config;
Simulation<CALCULATION_TYPE>* sim;
RenderWindow* window; 
BH_tree<CALCULATION_TYPE>* vtree; 
Font font;
Text sim_info;
Text config_info;
Text warning;

_inline string to_string_round(CALCULATION_TYPE num) {
    string cur = to_string(num);
    return cur.substr(0, cur.size() - 4);
}

_inline String percent_bar(CALCULATION_TYPE p) {
    char16_t bar[] = { '[', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ']' };
    p *= 100;
    int int_p = (int)ceil(p);
    switch (int_p / 10)
    {
    case 10:
        bar[10] = 9608;
    case 9:
        bar[9] = 9608;
    case 8:
        bar[8] = 9608;
    case 7:
        bar[7] = 9608;
    case 6:
        bar[6] = 9608;
    case 5:
        bar[5] = 9608;
    case 4:
        bar[4] = 9608;
    case 3:
        bar[3] = 9608;
    case 2:
        bar[2] = 9608;
    case 1:
        bar[1] = 9608;
    }
    switch (int_p % 10)
    {
    case 1:
        bar[int_p / 10 + 1] = 9615;
        break;
    case 2:
        bar[int_p / 10 + 1] = 9614;
        break;
    case 3:
        bar[int_p / 10 + 1] = 9613;
        break;
    case 4:
        bar[int_p / 10 + 1] = 9612;
        break;
    case 5:
        bar[int_p / 10 + 1] = 9611;
        break;
    case 6:
    case 7:
        bar[int_p / 10 + 1] = 9610;
        break;
    case 8:
    case 9:
        bar[int_p / 10 + 1] = 9609;
        break;
    }
    return String::fromUtf16<char16_t*>(bar, bar + 12);
}

_inline string to_time_str(double time) {
    long long int_p = (long long)time;
    double mS = time - int_p;
    long long H = int_p / 3600;
    int_p %= 3600;
    long long M = int_p / 60;
    int_p %= 60;
    long long S = int_p;
    stringstream ss;
    if (H < 10)
        ss << setw(2) << setfill('0') << right;
    ss << H << ":" << setw(2) << setfill('0') << right << M << ":" << setw(2) << setfill('0') << right << S << to_string(mS).substr(1, 4);
    return ss.str();
}

_inline void ask_cache_increace() {
    bool made = false;
    while (window->isOpen() && !made)
    {
        Event event;
        while (window->pollEvent(event))
        {
            switch (event.type) {
            case Event::Closed:
                window->close();
                break;
            case Event::KeyPressed:
                if (event.key.code == Keyboard::Y) {
                    size_t true_size = config.N * 9 / 2 * sizeof(BH_tree<CALCULATION_TYPE>);
                    if (config.max_cache < true_size) {
                        config.max_cache = true_size;
                    }
                    else {
                        config.max_cache = config.max_cache * 3 / 2;
                    }
                    if (!sim->alive) {
                        delete sim;
                        config.read_capture = true;
                        sim = new Simulation<CALCULATION_TYPE>(config);

                        thread th(&Simulation<CALCULATION_TYPE>::run, sim);
                        th.detach();
                    }

                    vtree->~BH_tree();
                    vtree = BH_tree<CALCULATION_TYPE>::newTree(config);
                    made = true;
                }
                break;
            case Event::Resized:
                Vector2u dims = window->getSize();
                config.W = dims.x;
                config.H = dims.y;
                View newView(Vector2f(config.W / 2, config.H / 2), Vector2f(config.W, config.H));
                window->setView(newView);
                sim_info.setPosition(Vector2f(10 + config.W % 2 * 0.5, 10 + config.H % 2 * 0.5));
                FloatRect warn_bounds = warning.getLocalBounds();
                warning.setPosition(Vector2f(config.W * 0.5 + config.W % 2 * 0.5 - warn_bounds.width * 0.5, config.H * 0.5 + config.H % 2 * 0.5 - warn_bounds.height * 0.5));
                break;
            }
        }
        window->clear(Color(165, 0, 0, 255));
        window->draw(warning);
        window->display();
    }
}

int main() {
    ContextSettings settings;
    settings.antialiasingLevel = 10;
    window = new RenderWindow(VideoMode(config.W, config.H), "N-body Simulator", Style::Default, settings);
    window->setVerticalSyncEnabled(true);

    auto start = std::chrono::system_clock::now();
    auto now = chrono::system_clock::now();

    srand(start.time_since_epoch().count());

    config.readConfig("config.cfg");

    vtree = BH_tree<CALCULATION_TYPE>::newTree(config);

    sim = new Simulation<CALCULATION_TYPE>(config);

    double frames = 0;

    double posX = 0, posY = 0;

    VertexArray pixels(Points, config.N);

    CALCULATION_TYPE* point_b = new CALCULATION_TYPE[config.N * 2];

    font.loadFromFile("fonts/JetBrainsMono-Regular.ttf");
    sim_info.setPosition(Vector2f(10, 10));
    sim_info.setCharacterSize(15);
    sim_info.setFillColor(Color(255, 255, 255, 255));
    sim_info.setOutlineColor(Color(0, 0, 0, 200));
    sim_info.setOutlineThickness(3.5);
    sim_info.setFont(font);
    config_info = Text(sim_info);
    warning = Text(sim_info);
    warning.setString("Cache overflowed! \nPress Y to increase it's size.");
    FloatRect warn_bounds = warning.getLocalBounds();
    warning.setPosition(Vector2f(config.W * 0.5 + config.W % 2 * 0.5 - warn_bounds.width * 0.5, config.H * 0.5 + config.H % 2 * 0.5 - warn_bounds.height * 0.5));

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    string cudaInfo = "CUDA device properties\nname: " + string(props.name) + "\n" +
        "API_major: " + to_string(props.major) + "\n" +
        "API_minor: " + to_string(props.minor) + "\n" +
        "clockRate: " + to_string_round(props.clockRate / 1000.) + " MHz\n" +
        "multiProcessorCount: " + to_string(props.multiProcessorCount) + "\n";

    thread th(&Simulation<CALCULATION_TYPE>::run, sim);
    th.detach();

#define point_b(i, j) point_b[i*2 + j]

    while (window->isOpen())
    {
        Event event;
        while (window->pollEvent(event))
        {
            int tics;
            switch (event.type) {
            case Event::Closed:
                sim->stop();
                while (sim->alive);
                window->close();
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
                if (event.key.code == Keyboard::F3) {
                    show_info = !show_info;
                }
                if (event.key.code == Keyboard::F1) {
                    show_config_info = !show_config_info;
                }
                break;
            case Event::Resized:
                Vector2u dims = window->getSize();
                config.W = dims.x;
                config.H = dims.y;
                View newView(Vector2f(config.W/2, config.H/2), Vector2f(config.W, config.H));
                window->setView(newView);
                sim_info.setPosition(Vector2f(10 + config.W % 2 * 0.5, 10 + config.H % 2 * 0.5));
                FloatRect warn_bounds = warning.getLocalBounds();
                warning.setPosition(Vector2f(config.W * 0.5 + config.W % 2 * 0.5 - warn_bounds.width * 0.5, config.H * 0.5 + config.H % 2 * 0.5 - warn_bounds.height * 0.5));
                break;
            }
        }

        if (!sim->alive) {
            if (string(sim->except.what()) == string(CACHE_OVERFLOW_EXCEPT.what())) {
                ask_cache_increace();
            }
        }

        sim->getPoints(point_b);

        try {
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
        }
        catch (exception e) {
            ask_cache_increace();
        }

        if (centrilize) {
            CALCULATION_TYPE center[2]{ 0, 0 };
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
            float x = (point_b(i, 0) - posX) * scale + 0.5 * config.W;
            float y = (point_b(i, 1) - posY) * scale + 0.5 * config.H;
            pixels[i].position = Vector2f(x, y);
            pixels[i].color = Color(255, 255, 255, 255);
        }

        window->clear();
        window->draw(pixels);


        if (drawBH) {
            BH_tree<CALCULATION_TYPE>* nodes = vtree->getNodes();

            for (BH_tree<CALCULATION_TYPE>* i = nodes; i < nodes + vtree->activeNodeCount(); i++) {
                float width;
                if (i->node_width < FLT_MAX) {
                    width = i->node_width;
                }
                else {
                    width = FLT_MAX;
                }
                if (width * scale >= 1) {
                    RectangleShape rect(Vector2f(width * scale, width * scale));
                    rect.setPosition(
                        Vector2f(
                            config.W / 2 - (width / 2 - i->center[0] + posX) * scale,
                            config.H / 2 - (width / 2 - i->center[1] + posY) * scale)
                    );
                    rect.setFillColor(Color(0, 0, 0, 0));
                    rect.setOutlineThickness(0.3f);
                    rect.setOutlineColor(Color(255, 255, 255, 255));
                    if (i->body_mass > 0) {
                        rect.setFillColor(Color(0, 0, 255, 40));
                    }
                    window->draw(rect);
                }
            }
        }
        if (show_info) {
            unsigned long long usedCache = sim->totalTreeNodes * sizeof(BH_tree<CALCULATION_TYPE>);
            unsigned long long assumedCache = config.max_cache;
            String bar = percent_bar((CALCULATION_TYPE)usedCache / assumedCache);
            sim_info.setString(
                "FPS: " + to_string(fps) + "\n" +
                "UPS: " + to_string(sim->UPS) + "(" + to_string(sim->meanUPS()) + ")\n" +
                "evaluation_time: " + to_time_str(sim->evaluationTime()) + "\n" +
                "sim_time_per_real_second: " + to_string(sim->UPS * config.DeltaT) + "(" + to_string(sim->meanUPS() * config.DeltaT) + ")" + "\n" +
                "sim_tree_depth: " + to_string(sim->treeDepth) + "\n" +
                "total_sim_tree_nodes: " + to_string(sim->totalTreeNodes) + "\n" +
                "active_sim_tree_nodes: " + to_string(sim->activeTreeNodes) + "\n" +
                "tree_node_cache: " + to_string_round((CALCULATION_TYPE)usedCache / (1 << 20)) + "(" + to_string_round((CALCULATION_TYPE)assumedCache / (1 << 20)) + ") Mb" + "\n" +
                "                 " + to_string_round((CALCULATION_TYPE)usedCache / assumedCache * 100) + "%" + bar
            );
            window->draw(sim_info);
        }

        if (show_config_info) {
            config_info.setString(
                "window_width: " + to_string(config.W) + "\n" +
                "window_height: " + to_string(config.H) + "\n" +
                "body_number: " + to_string(config.N) + "\n" +
                "time_step: " + to_string(config.DeltaT) + "\n" +
                "record: " + (config.record ? "true" : "false") + "\n" +
                "max_cache: " + to_string_round((CALCULATION_TYPE)config.max_cache / (1 << 20)) + " Mb\n" +
                "caching_nodes_num: " + to_string(config.max_cache / sizeof(BH_tree<CALCULATION_TYPE>)) + "\n" +
                (config.useCUDA ? cudaInfo : "")
            );
            FloatRect bounds = config_info.getLocalBounds();
            config_info.setPosition(Vector2f(config.W - 10 + config.W % 2 * 0.5 - bounds.width, 10 + config.H % 2 * 0.5));
            window->draw(config_info);
        }

        window->display();

        now = chrono::system_clock::now();
        chrono::duration<double> elapsed_seconds = now - start;
        frames++;
        volatile double elaps = elapsed_seconds.count();
        if (elaps >= 1) {
            fps = frames / elaps;
            start = now;
            frames = 0;
        }
    }
}