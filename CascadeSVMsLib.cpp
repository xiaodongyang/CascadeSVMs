#include <iostream>
#include "CascadeSVMsLib.h"
#include "CascadeSVMsUtil.h"

CascadeSVMs::CascadeSVMs() {
}


CascadeSVMs::~CascadeSVMs() {
}


string CascadeSVMs::readControlFile(string ctrl, vector<string> &files) {
	FILE *fp = fopen(ctrl.c_str(), "r");
	if (fp == NULL) {
		std::cerr << "Cannot open the file: " << ctrl << "!\n";
		exit(1);
	}

	char buff[1000];

	// read event type
	fgets(buff, 1000, fp);
	fscanf(fp, "%d", &index);
	fgets(buff, 1000, fp);

	// read feature dimension
	fgets(buff, 1000, fp);
	fscanf(fp, "%d", &ndim);
	fgets(buff, 1000, fp);

	// read directory of feature files
	fgets(buff, 1000, fp);
	fgets(buff, 1000, fp);
	string dir = buff;
	dir.erase(dir.length() - 1, 1);

	// read directory of models
	fgets(buff, 1000, fp);
	fgets(buff, 1000, fp);
	mdir = buff;
	mdir.erase(mdir.length() - 1, 1);

	// read directory of predictions
	fgets(buff, 1000, fp);
	fgets(buff, 1000, fp);
	pdir = buff;
	pdir.erase(pdir.length() - 1, 1);

	// read number of feature files
	int num;
	fgets(buff, 1000, fp);
	fscanf(fp, "%d", &num);
	fgets(buff, 1000, fp);

	// read feature file names
	fgets(buff, 1000, fp);
	for (int i = 0; i < num; ++i) {
		fgets(buff, 1000, fp);
		string str = buff;
		str.erase(str.length() - 1, 1);
		files.push_back(str);
	}

	fclose(fp); fp = NULL;

	return dir;
}


void CascadeSVMs::readTrainData(string dir, vector<string> files) {
	int npos = (int)3E+4;
	int nneg = (int)2E+5;

	featPos.create(npos, ndim, CV_32FC1);
	featNeg.create(nneg, ndim, CV_32FC1);

	int idxpos = 0;
	int idxneg = 0;
	int nfile = (int)files.size();

	std::cout << "\n";

	for (int i = 0; i < nfile; ++i) {
		std::cout << "reading training data from " << files[i] << " ......\n";

		string fvfile = dir + "\\" + files[i] + ".fv";
		string lbfile = dir + "\\" + files[i] + ".lb";
		
		Mat featbuff, tarbuff;
		readBinaryData(fvfile, lbfile, featbuff, tarbuff);

		for (int j = 0; j < tarbuff.rows; ++j) {
			if (tarbuff.at<int>(j) == 1)
				featbuff.row(j).copyTo(featPos.row(idxpos++));
			else
				featbuff.row(j).copyTo(featNeg.row(idxneg++));
		}
	}

	featPos = featPos.rowRange(0, idxpos);
	featNeg = featNeg.rowRange(0, idxneg);

	std::cout << "\n";
	std::cout << "# positive samples: " << featPos.rows << "\n";
	std::cout << "# negative samples: " << featNeg.rows << "\n";
}


void CascadeSVMs::readBinaryData(string featfile, string tarfile, Mat &feat, Mat &tar) {
	int count = 0;
	char buff[100];

	FILE *ptar = fopen(tarfile.c_str(), "r");
	if (ptar == NULL) {
		std::cerr << "Cannot open the file: " << tarfile << " !\n";
		exit(1);
	}

	FILE *pfeat = fopen(featfile.c_str(), "rb");
	if (pfeat == NULL) {
		std::cerr << "Cannot open the file: " << featfile << " !\n";
		exit(1);
	}

	// read number of samples
	while (fgets(buff, 100, ptar) != NULL)
		++count;

	count /= 2;

	rewind(ptar);

	tar = Mat::zeros(count, 1, CV_32SC1);
	feat.create(count, ndim, CV_32FC1);
	
	for (int i = 0; i < count; ++i) {
		// read target
		fgets(buff, 100, ptar);
		fgets(buff, 100, ptar);

		char *lb = strtok(buff, " ");
		
		while (lb != NULL) {
			if (atoi(lb) == index) {
				tar.at<int>(i) = 1;
				break;
			}

			lb = strtok(NULL, " ");
		}

		// read out window position
		int dumb[2];
		fread(dumb, sizeof(int), 2, pfeat);

		// read feature
		float *ptr = feat.ptr<float>(i);
		fread(ptr, sizeof(float), ndim, pfeat);
	}

	fclose(ptar);  ptar = NULL;
	fclose(pfeat); pfeat = NULL;
}


void CascadeSVMs::readBinaryData(string featfile, vector<int> &sfrm, vector<int> &efrm) {
	FILE *fp = fopen(featfile.c_str(), "rb");
	if (fp == NULL) {
		std::cerr << "Cannot open the file: " << featfile << " !\n";
		exit(1);
	}

	int count = (int)1E+5; // might be small !!!!

	sfrm.resize(count);
	efrm.resize(count);
	featTest.create(count, ndim, CV_32FC1);
	
	int idx = 0;

	while (!feof(fp)) {
		// read out window position
		int win[2];
		fread(win, sizeof(int), 2, fp);

		sfrm[idx] = win[0];
		efrm[idx] = win[1];

		// read feature
		float *ptr = featTest.ptr<float>(idx);
		fread(ptr, sizeof(float), ndim, fp);

		++idx;
	}

	fclose(fp); fp = NULL;

	sfrm.resize(idx - 1);
	efrm.resize(idx - 1);
	featTest = featTest.rowRange(0, idx - 1);
}


void CascadeSVMs::readASCIIData(string featfile, string tarfile, Mat &feat, Mat &tar) {
	int count = 0;
	char buff[100];

	FILE *ptar = fopen(tarfile.c_str(), "r");
	if (ptar == NULL) {
		std::cerr << "Cannot open the file: " << tarfile << " !\n";
		exit(1);
	}

	FILE *pfeat = fopen(featfile.c_str(), "r");
	if (pfeat == NULL) {
		std::cerr << "Cannot open the file: " << featfile << " !\n";
		exit(1);
	}

	// read number of samples
	while (fgets(buff, 100, ptar) != NULL)
		++count;

	count /= 2;

	rewind(ptar);

	tar = Mat::zeros(count, 1, CV_32SC1);
	feat.create(count, ndim, CV_32FC1);
	
	for (int i = 0; i < count; ++i) {
		// read target
		fgets(buff, 100, ptar);
		fgets(buff, 100, ptar);

		char *lb = strtok(buff, " ");
		
		while (lb != NULL) {
			if (atoi(lb) == index) {
				tar.at<int>(i) = 1;
				break;
			}

			lb = strtok(NULL, " ");
		}

		// read feature
		float *ptr = feat.ptr<float>(i);

		int ival;
		fscanf(pfeat, "%d", &ival);
		fscanf(pfeat, "%d", &ival);

		for (int j = 0; j < ndim; ++j)
			fscanf(pfeat, "%f", ptr + j);
	}

	fclose(ptar);  ptar = NULL;
	fclose(pfeat); pfeat = NULL;
}


void CascadeSVMs::readASCIIData(string featfile, vector<int> &sfrm, vector<int> &efrm) {
	FILE *fp = fopen(featfile.c_str(), "r");
	if (fp == NULL) {
		std::cerr << "Cannot open the file: " << featfile << " !\n";
		exit(1);
	}

	int count = (int)1E+5; // might be samll !!!!

	sfrm.resize(count);
	efrm.resize(count);
	featTest.create(count, ndim, CV_32FC1);
	
	int idx = 0;

	while (!feof(fp)) {
		// read out window position
		int ival;

		fscanf(fp, "%d", &ival);
		sfrm[idx] = ival;
		
		fscanf(fp, "%d", &ival);
		efrm[idx] = ival;
	
		// read feature
		float *ptr = featTest.ptr<float>(idx);

		for (int i = 0; i < ndim; ++i)
			fscanf(fp, "%f", ptr + i);

		++idx;
	}

	fclose(fp); fp = NULL;

	sfrm.resize(idx - 1);
	efrm.resize(idx - 1);
	featTest = featTest.rowRange(0, idx - 1);
}


void CascadeSVMs::learn() {
	int iter = 0;
	int nPreEffNeg = featNeg.rows; 
	double thresh = THRESH_POS_STRONG;

	vector<struct model*> pool; // cascaded classifiers
	vector<int> labelNeg(featNeg.rows, 1);
	vector<double> scoreNeg(featNeg.rows, 0.0);
	
	do {
		vector<int> hardNeg = selectHardNegative(iter, scoreNeg, labelNeg);

		if ((int)hardNeg.size() == 0)
			break;

		std::cout << "\n********** at stage: " << iter << " **********\n";

		liblinear(iter, thresh, hardNeg, pool);
		int nEffNeg = updateNegative(pool, scoreNeg, labelNeg);

		if (nEffNeg == nPreEffNeg)
			thresh = THRESH_POS_WEAK;
		else
			thresh = THRESH_POS_STRONG;

		nPreEffNeg = nEffNeg;
		
		std::cout << "positive accuracy threshold: " << thresh << "\n";
		std::cout << nEffNeg << " effective negative samples left: " << (double)nEffNeg / featNeg.rows << "\n";

		++iter;

	} while (iter < ITER_MAX);

	freeClassifierPool(pool);
}


vector<int> CascadeSVMs::selectHardNegative(int iter, const vector<double> &scores, const vector<int> &labels) {
	vector<int> hards;
	int npos = featPos.rows;
	int nneg = featNeg.rows;
	
	// randomly select "npos" negative samples  
	if (iter == 0) {
		hards.resize(nneg);

		for (int i = 0; i < nneg; ++i)
			hards[i] = i;

		std::random_shuffle(hards.begin(), hards.end());

		hards.resize(npos);

		return hards;

	// select most inaccurately classified negative samples
	} else {
		vector<double> wscores;

		for (int i = 0; i < nneg; ++i) {
			if (labels[i] == 1)
				wscores.push_back(scores[i]);
		}

		if ((int)wscores.size() < npos)
			return hards;

		std::sort(wscores.begin(), wscores.end()); 
		std::reverse(wscores.begin(), wscores.end());
		double thresh = wscores[npos - 1];

		for (int i = 0; i < nneg; ++i) {
			if (labels[i] == 1 && scores[i] >= thresh)
				hards.push_back(i);
		}
		
		return hards;
	}

}


void CascadeSVMs::liblinear(int iter, double thresh, const vector<int> &hardNeg, vector<struct model*> &pool) {
	LibLinear lsvm;

	// set default parameters
	lsvm.setParams();

	// load training data according to the LIBLINEAR interface
	lsvm.loadData(featPos, featNeg, hardNeg);

	double wtpos = 1.0;     // positive class weight
	double wtneg = 1.0;     // negative class weight
	vector<double> accpos; 

	while (1) {
		lsvm.setClassWeights(wtpos, wtneg);
		std::cout << "positive class weight: " << lsvm.getPosWeight() << "\n";

		// training
		lsvm.trainModel();

		// evaluate classifier
		struct accuracy acc;
		lsvm.evalModel(acc);
		accpos.push_back(acc.pos);
		std::cout << "positive class accuracy: " << acc.pos << "\n";
		std::cout << "negative class accuracy: " << acc.neg << "\n";
		
		if (acc.pos >= thresh) {
			// save the current classifier
			char buff[10];
			sprintf(buff, "%d", iter);
			string name = mdir + "\\Model_" + buff;
			lsvm.saveModel(name);

			// add in the classifier pool
			lsvm.cascadeModel(name, pool);

			break;

		} else {
			wtpos += WEIGHT_POS_STEP;
		}

		if (wtpos > WEIGHT_POS_MAX) {
			// retrieve the positive weight that gives the best positive accuracy
			int idx = std::max_element(accpos.begin(), accpos.end()) - accpos.begin();
			wtpos = 1.0 + idx * WEIGHT_POS_STEP;

			lsvm.setClassWeights(wtpos, wtneg);

			// training
			lsvm.trainModel();

			// save the current classifier
			char buff[10];
			sprintf(buff, "%d", iter);
			string name = mdir + "\\Model_" + buff;
			lsvm.saveModel(name);

			// add in the classifier pool
			lsvm.cascadeModel(name, pool);

			break;
		}
	}

}


void CascadeSVMs::freeClassifierPool(vector<struct model*> &pool) {
	for (int i = 0; i < (int)pool.size(); ++i) {
		free_and_destroy_model(&pool[i]);
		pool[i] = NULL;
	}
}


int CascadeSVMs::updateNegative(const vector<struct model*> &pool, vector<double> &scores, vector<int> &labels) {
	int nneg = featNeg.rows;
	
	for (int i = 0; i < nneg; ++i) {
		if (labels[i] == 1) {
			vector<double> confs;
			int cls = predictByAllSVMs(featNeg.row(i), pool, confs);

			// compute average score
			// if "cls == 0", this "scores[i]" is not important because it will not be used anymore
			double dec = 0.0;
			for (int j = 0; j < (int)confs.size(); ++j)
				dec += confs[j];
			dec /= confs.size();

			scores[i] = dec;
			labels[i] = cls;
		}
	}

	// how many effective negative samples left
	int count = 0;
	for (int i = 0; i < nneg; ++i)
		count += labels[i];

	return count;
}


int CascadeSVMs::predictByAllSVMs(const Mat &vec, const vector<struct model*> &pool, vector<double> &scores) {
	int label;
	int nmodel = (int)pool.size();

	for (int i = 0; i < nmodel; ++i) {
		label = predictByOneSVM(vec, pool[i], scores);

		// if any classifier predicts as a negative sample, then assigned as negative
		if (label == 0)
			break;
	}

	return label;
}


int CascadeSVMs::predictByOneSVM(const Mat &vec, const struct model *classifier, vector<double> &scores) {
	int ncol;
	struct problem data;
	struct feature_node *xbuff = NULL;
	
	data.l = 1;

	if (classifier->bias >= 0) {
		data.n = vec.cols + 1;
		ncol = vec.cols + 2;
	} else {
		data.n = vec.cols;
		ncol = vec.cols + 1;
	}

	data.x = (struct feature_node**)calloc(1, sizeof(struct feature_node*));
	xbuff = (struct feature_node*)calloc(1 * ncol, sizeof(struct feature_node));

	data.x[0] = xbuff;
	const float *ptr = vec.ptr<float>(0);

	for (int i = 0; i < vec.cols; ++i) {
		xbuff[i].index = i + 1;
		xbuff[i].value = ptr[i];
	}

	if (classifier->bias >= 0) {
		xbuff[vec.cols].index = vec.cols + 1;
		xbuff[vec.cols].value = classifier->bias;
		xbuff[vec.cols + 1].index = -1;
		xbuff[vec.cols + 1].value = -1;
	} else {
		xbuff[vec.cols].index = -1;
		xbuff[vec.cols].value = -1;
	}
	
	// for binary classification, only one decision value
	// positive (negative) class - positive (negative) decision value
	double dec;
	int label = (int)predict_values(classifier, data.x[0], &dec);
	scores.push_back(dec);

	// clear data
	if (data.x != NULL) {
		free(data.x[0]);
		xbuff = NULL;
		free(data.x);
		data.x = NULL;
	}

	return label;
}


void CascadeSVMs::clearTrainData() {
	featPos.release();
	featNeg.release();
}


void CascadeSVMs::test(string dir, vector<string> files) {
	vector<struct model*> pool;
	loadClassifierPool(pool);
	
	int nfile = (int)files.size();

	for (int i = 0; i < nfile; ++i) {
		std::cout << "\n" << "reading testing data from " << files[i] << " ......\n";
		
		string fvfile = dir + "\\" + files[i] + ".fv";
		vector<int> sfrm, efrm;
		readBinaryData(fvfile, sfrm, efrm);

		std::cout << "predicting testing data from " << files[i] << " ......\n";

		string pdfile = pdir + "\\" + files[i] + ".pd";
		makePrediction(pdfile, sfrm, efrm, pool);

		featTest.release();
	}

	freeClassifierPool(pool);
}


void CascadeSVMs::loadClassifierPool(vector<struct model*> &pool) {
	int idx = 0;

	while (1) {
		char buff[10];
		sprintf(buff, "%d", idx);
		string name = mdir + "\\Model_" + buff;
		
		struct model *classifier = NULL;
		classifier = load_model(name.c_str());
		
		if (classifier != NULL)
			pool.push_back(classifier);
		else
			break;

		++idx;
	}
}


void CascadeSVMs::makePrediction(string name, const vector<int> &sfrm, const vector<int> &efrm, const vector<struct model*> &pool) {
	FILE *fp = fopen(name.c_str(), "w");
	if (fp == NULL) {
		std::cerr << "Cannot open the prediction file: " << name << "!\n";
		exit(1);
	}

	// header of output file
	fprintf(fp, "# StartFrame EndFrame LeftX TopY WidthX HeightY Score EventType\n");

	int nfeat = featTest.rows;
	vector<int> labels(nfeat);
	vector<double> scores(nfeat);

	for (int i = 0; i < nfeat; ++i) {
		vector<double> confs;
		labels[i] = predictByAllSVMs(featTest.row(i), pool, confs);
		
		// compute classification score
		// 1. use average score of cascaded models positive prediction
		// 2. use "confs.back()" for negative prediction
		double dec = 0.0;

		if (labels[i] == 0)
			dec = confs.back();
		else {
			for (int j = 0; j < (int)confs.size(); ++j)
				dec += confs[j];
			dec /= confs.size();
		}

		scores[i] = dec;

		// output predictions
		fprintf(fp, "%d %d ", sfrm[i], efrm[i]);   // detection window position
		fprintf(fp, "%d %d %d %d ", 0, 0, 0, 0);   // spatial region where event happens
		fprintf(fp, "%f ", scores[i]);             // classification score
		if (labels[i] == 1)                        // event type
			fprintf(fp, "%d\n", index);
		else
			fprintf(fp, "%d\n", 0);
	}

	fclose(fp); fp = NULL;
}