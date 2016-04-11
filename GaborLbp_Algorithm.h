#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\opencv.hpp>

#include <cv.h>
#include <highgui.h>
#include <vector>

#include "ZGabor.h"
#include "LBPH.h"

using namespace cv;
using namespace std;

class GaborLbp_Algorithm
{
public:
	GaborLbp_Algorithm(void);
	~GaborLbp_Algorithm(void);
public:
	void train(InputArrayOfArrays _in_src, InputArray _in_labels);
	int predict(InputArray _src,double& minDis);
protected:
	vector<Mat> m_projection;
	Mat m_labels;
};

