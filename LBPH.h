
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\opencv.hpp>

#include <cv.h>
#include <highgui.h>
#include <vector>
#include <fstream>

using namespace cv;
using namespace std;

#ifndef _LBPH
#define _LBPH
class LBPH
{
public:
	Mat m_eigenvector;
	void LBPEncoding(InputArray src,Mat&histogram,int radius, int neighbors,int grid_x,int grid_y);
	void GetFeature(Mat src);
	
};
#endif

