#include "CascadeSVMsUtil.h"

#include <iostream>

void print_null(const char *s) {};


LibLinear::LibLinear() {
	param.weight = NULL;
}


LibLinear::~LibLinear() {
	if (data.x != NULL) {
		free(data.x[0]);
		free(data.x);
		data.x = NULL;
	}

	if (data.y != NULL) {
		free(data.y);
		data.y = NULL;
	}

	destroy_param(&param);

	free_and_destroy_model(&classifier);
	classifier = NULL;
}

void LibLinear::setParams() {
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = 0.1;
	param.p = 0.1;
	param.nr_weight = 2;

	param.weight_label = (int*)calloc(2, sizeof(int));
	param.weight_label[0] = 0;
	param.weight_label[1] = 1;

	void (*print_func)(const char*) = NULL;
	print_func = &print_null;
	set_print_string_function(print_func);
}


void LibLinear::loadData(const Mat &pos, const Mat &neg, const vector<int> &hards) {
	int npos = pos.rows;
	int nneg = (int)hards.size();

	// number of samples
	data.l = npos + nneg;

	// bias
	data.bias = 0.0;

	// feature dimension
	if (data.bias >= 0)
		data.n = pos.cols + 1;
	else
		data.n = pos.cols;

	// targets
	data.y = (double*)calloc(data.l, sizeof(double));
	for (int i = 0; i < npos; ++i)
		data.y[i] = 1.0;

	// features
	int ncol;
	if (data.bias >= 0)
		ncol = pos.cols + 2;
	else
		ncol = pos.cols + 1;

	data.x = (struct feature_node**)calloc(data.l, sizeof(struct feature_node*));
	struct feature_node *xbuff = (struct feature_node*)calloc( (size_t)data.l * (size_t)ncol, sizeof(struct feature_node) );
	
	// fill in positive data
	for (int i = 0; i < npos; ++i) {
		data.x[i] = &(xbuff[i * (size_t)ncol]);
		const float *ptr = pos.ptr<float>(i);

		for (int j = 0; j < pos.cols; ++j) {
			xbuff[i * (size_t)ncol + j].index = j + 1;
			xbuff[i * (size_t)ncol + j].value = ptr[j];
		}

		if (data.bias >= 0) {
			xbuff[i * (size_t)ncol + pos.cols].index = pos.cols + 1;
			xbuff[i * (size_t)ncol + pos.cols].value = data.bias;
			xbuff[i * (size_t)ncol + pos.cols + 1].index = -1;
			xbuff[i * (size_t)ncol + pos.cols + 1].value = -1;
		} else {
			xbuff[i * (size_t)ncol + pos.cols].index = -1;
			xbuff[i * (size_t)ncol + pos.cols].value = -1;
		}
	}

	// fill in negative data
	for (int i = 0; i < nneg; ++i) {
		data.x[i + npos] = &(xbuff[(i + npos) * (size_t)ncol]);
		const float *ptr = neg.ptr<float>(hards[i]); // index: "hards[i]" NOT "i"

		for (int j = 0; j < neg.cols; ++j) {
			xbuff[(i + npos) * (size_t)ncol + j].index = j + 1;
			xbuff[(i + npos) * (size_t)ncol + j].value = ptr[j];
		}

		if (data.bias >= 0) {
			xbuff[(i + npos) * (size_t)ncol + neg.cols].index = neg.cols + 1;
			xbuff[(i + npos) * (size_t)ncol + neg.cols].value = data.bias;
			xbuff[(i + npos) * (size_t)ncol + neg.cols + 1].index = -1;
			xbuff[(i + npos) * (size_t)ncol + neg.cols + 1].value = -1;
		} else {
			xbuff[(i + npos) * (size_t)ncol + neg.cols].index = -1;
			xbuff[(i + npos) * (size_t)ncol + neg.cols].value = -1;
		}
	}

}


void LibLinear::setClassWeights(double wtpos, double wtneg) {
	if (param.weight != NULL) {
		free(param.weight);
		param.weight = NULL;
	}

	param.weight = (double*)calloc(2, sizeof(double));
	param.weight[0] = wtneg;
	param.weight[1] = wtpos;
}


void LibLinear::trainModel() {
	classifier = train(&data, &param);
}


void LibLinear::evalModel(struct accuracy &acc) {
	double npos, nneg;
	npos = nneg = data.l / 2;

	double cpos, cneg;
	cpos = cneg = 0;

	double label;

	// predict positive samples
	for (int i = 0; i < (int)npos; ++i) {
		label = predict(classifier, data.x[i]);
		
		if (label == 1)
			++cpos;
	}

	// predict negative samples
	for (int i = (int)npos; i < data.l; ++i) {
		label = predict(classifier, data.x[i]);
	
		if (label == 0)
			++cneg;
	}

	acc.pos = cpos / npos;
	acc.neg = cneg / nneg;
}


void LibLinear::saveModel(string name) {
	save_model(name.c_str(), classifier);
}


void LibLinear::cascadeModel(string name, vector<model*> &pool) {
	struct model *amodel = load_model(name.c_str());
	pool.push_back(amodel);
}