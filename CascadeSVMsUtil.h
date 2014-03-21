#ifndef CASCADE_SVMS_UTIL_H
#define CASCADE_SVMS_UTIL_H

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include "LIBLINEAR/linear.h"
#include "LIBLINEAR/tron.h"

using std::string;
using std::vector;
using cv::Mat;

struct accuracy {
	double pos; // positive accuracy
	double neg; // negative accuracy
};


class LibLinear {
public:
	LibLinear();

	~LibLinear();

	// set parameters for LIBLINEAR
	void setParams();

	// load training data
	void loadData(const Mat &pos, const Mat &neg, const vector<int> &hards);

	// set class weights
	void setClassWeights(double wtpos, double wtneg);

	// return class weight
	inline double getPosWeight() {
		return param.weight[1];
	}

	// train a SVM classifier using LIBLINEAR
	void trainModel();

	// evaluate a SVM classifier based on current training data
	void evalModel(struct accuracy &acc);

	// save a SVM classifier
	void saveModel(string name);

	// concatenate a SVM classifier to the cascaded SVM classifiers
	void cascadeModel(string name, vector<model*> &pool);

private:
	struct problem data;       // first half is positive and second half is negative
	struct parameter param;    // training parameters
	struct model *classifier;  // trained classifier
};

#endif