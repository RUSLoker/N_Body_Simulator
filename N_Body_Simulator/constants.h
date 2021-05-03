#pragma once

static const unsigned int W = 800;
static const unsigned int H = 800;
static unsigned int N = 5000;
static const double G = 100000;
static const double K = 1000 * G;
static double DeltaT = 0.00001;
static double MAX_START_SPEED = 30;
static const double min_width = 10;
static const double theta = 0.3;
static const double scroll_speed = 10;
static const double min_accel = 0.05;
static const double max_accel = 1000000000;
static const double max_dist = 40000 * 40000;
static const bool useBH = true;
static bool record = true;

static const int configN = 3;