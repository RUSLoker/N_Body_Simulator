#pragma once

const unsigned int W = 800;
const unsigned int H = 800;
const unsigned int N = 5000;
const double G = 100000;
const double K = 1000 * G;
const double DeltaT = 0.00001;
const double MAX_START_SPEED = 30;
const double min_width = 10;
const double theta = 0.3;
const double scroll_speed = 10; 
const double min_accel = 0.05;
const double max_accel = 1000000000;
const double max_dist = 40000 * 40000;
const bool useBH_default = true;
const bool centrilize_default = false;
const bool record_default = true;