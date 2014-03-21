#ifndef CASCADE_SVMS_LIB_H
#define CASCADE_SVMS_LIB_H

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include "LIBLINEAR/linear.h"

using std::vector;
using std::string;
using cv::Mat;

const double THRESH_POS_STRONG = 0.95;  // strong positive accuracy threshold
const double THRESH_POS_WEAK = 0.9;     // weak positive accuracy threshold
const double WEIGHT_POS_MAX = 24.0;     // max weight of positive class
const double WEIGHT_POS_STEP = 1.0;		// step to increase weight of positive class
const double TRESH_NEG = 0.5;			// negative accuracy threshold
const int LEFT_NEG_MIN = 400;			// min number of left negative samples
const int ITER_MAX = 15;				// max number of cascaded SVM classifiers

class CascadeSVMs {
public:
	CascadeSVMs();
	
	~CascadeSVMs();

	// read control file
	string readControlFile(string ctrl, vector<string> &files);

	// read training data
	void readTrainData(string dir, vector<string> files);
	
	// the learning algorithm
	void learn();

	// clear training data
	void clearTrainData();

	// testing 
	void test(string dir, vector<string> files);

private:
	// read training data in binary format
	void readBinaryData(string featfile, string tarfile, Mat &feat, Mat &tar);

	// read training data in ASCII format
	void readASCIIData(string featfile, string tarfile, Mat &feat, Mat &tar);

	// read testing data in binary format
	void readBinaryData(string featfile, vector<int> &sfrm, vector<int> &efrm);

	// read testing data in ASCII format
	void readASCIIData(string featfile, vector<int> &sfrm, vector<int> &efrm);

	// hard negative samples mining
	vector<int> selectHardNegative(int iter, const vector<double> &scores, const vector<int> &labels);

	// use LIBLINEAR to traing a SVM classifier
	void liblinear(int iter, double threshPos, const vector<int> &hardNeg, vector<struct model*> &pool);

	// free SVM classifiers
	void freeClassifierPool(vector<struct model*> &pool);

	// load SVM classifiers
	void loadClassifierPool(vector<struct model*> &pool);
	
	// update negative samples using currently available SVM classifiers
	int updateNegative(const vector<struct model*> &pool, vector<double> &scores, vector<int> &labels);

	// predict a sample using all available SVM classifiers
	int predictByAllSVMs(const Mat &vec, const vector<struct model*> &pool, vector<double> &scores);

	// predict a sample using one avaialble SVM classifier
	int predictByOneSVM(const Mat &vec, const struct model *classifier, vector<double> &scores);

	// predict testing samples
	void makePrediction(string name, const vector<int> &sfrm, const vector<int> &efrm, const vector<struct model*> &pool);

	int index;     // event index
	int ndim;      // feature dimension
	Mat featTest;  // feature matrix (used in testing)
	Mat featPos;   // positive feature matrix (used in training)
	Mat featNeg;   // negative feature matrix (used in training)
	string mdir;   // directory to store classifiers
	string pdir;   // directory to store predictions

	vector<int> adims;      // active dimension of each feature in early fusion, e.g., 1st level in spatial pyramids
	vector<int> tdims;      // total dimension of each feature in early fusion
	vector<float> weights;  // weight of different features
};

#endif