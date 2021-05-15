#include "Config.h"
#include "BH_tree.cuh"
#include <vector>
#include <fstream>
#include <string>
#include <algorithm> 

using namespace std;

void Config::readConfig(char* path) {
	fstream cfg;
	cfg.open(path, ios::in);
	bool exist = cfg.good();
	unsigned int cfg_s;
	cfg.seekg(0, cfg._Seekend);
	cfg_s = cfg.tellg();
	cfg.seekg(0, cfg._Seekbeg);
	vector<bool> config_readed(configN, false);
	while (cfg) {
		string s;
		cfg >> s;
		if (s == "MAX_START_SPEED:") {
			cfg >> MAX_START_SPEED;
			config_readed[0] = true;
		}
		else if (s == "RECORD:") {
			cfg >> record;
			config_readed[1] = true;
		}
		else if (s == "N:") {
			cfg >> N;
			config_readed[2] = true;
		}
		else if (s == "DeltaT:") {
			cfg >> DeltaT;
			config_readed[3] = true;
		}
		else if (s == "MAX_CACHE_ALLOC:") {
			double value;
			string type;
			cfg >> value >> type;
			transform(type.begin(), type.end(), type.begin(), ::toupper);
			char prep = type[0];
			if (type[1] == 'B' || type[0] == 'B') {
				switch (prep) {
				case 'P':
					value *= 1024;
				case 'T':
					value *= 1024;
				case 'G':
					value *= 1024;
				case 'M':
					value *= 1024;
				case 'K':
					value *= 1024;
					break;
				}
				max_cache = (size_t)value;
				if (max_cache < sizeof(BH_tree<CALCULATION_TYPE>)) {
					max_cache = sizeof(BH_tree<CALCULATION_TYPE>);
				}
				config_readed[4] = true;
			}
		}
		else if (s == "READ_CAPTURE:") {
			cfg >> read_capture;
			config_readed[5] = true;
		}
	}
	cfg.close();

	cfg.open("config.cfg", ios::in);
	cfg.seekg(-1, cfg._Seekend);
	char last_ch = 0;
	cfg.read(&last_ch, sizeof(last_ch));
	cfg.close();

	cfg.open("config.cfg", ios::out | ios::app);
	if (last_ch != '\n' && cfg_s > 0 && exist)
		cfg << endl;
	if (!config_readed[0]) {
		cfg << "MAX_START_SPEED: " << MAX_START_SPEED << endl;
	}
	if (!config_readed[1]) {
		cfg << "RECORD: " << record << endl;
	}
	if (!config_readed[2]) {
		cfg << "N: " << N << endl;
	}
	if (!config_readed[3]) {
		cfg << "DeltaT: " << DeltaT << endl;
	}
	if (!config_readed[4]) {
		cfg << "MAX_CACHE_ALLOC: " << (double)max_cache / (1 << 20) << " Mb" << endl;
	}
	if (!config_readed[5]) {
		cfg << "READ_CAPTURE: " << read_capture << endl;
	}
	cfg.close();
}