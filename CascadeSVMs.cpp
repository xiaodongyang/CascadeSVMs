#include <iostream>
#include "CascadeSVMsLib.h"

int main(int argc, char **argv) {
	if (argc == 1) {
		std::cerr << "./CascadeSVMs -c control file -p phase (train or test)\n";
		exit(1);
	}
	
	// read arguments
	int arg = 0;
	string fctrl;
	string phase;

	while (++arg < argc) {
		if (!strcmp(argv[arg], "-c"))
			fctrl = argv[++arg];
		if (!strcmp(argv[arg], "-p"))
			phase = argv[++arg];
	}

	bool flag;
	if (!phase.compare("train"))
		flag = true;
	else if (!phase.compare("test"))
		flag = false;
	else {
		std::cerr << "phase must be either <train> or <test>!\n";
		exit(1);
	}

	CascadeSVMs csvms;

	// read control file
	vector<string> files;
	string dir = csvms.readControlFile(fctrl, files);

	// training phase
	if (flag) {
		csvms.readTrainData(dir, files);
		csvms.learn();
		csvms.clearTrainData();	

	// testing phase
	} else {
		csvms.test(dir, files);
	}

	return 0;
}