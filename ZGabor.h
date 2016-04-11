
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

#define  MAX_PATH 255  

#ifndef _ZGABOR
#define _ZGABOR
class ZGabor  
{  
public:  
	//默认参数及指定参数的初始化  
	bool  InitGabor();  
	bool  InitGabor(Size ksize, double kmax, double f, double sigma);  

	//存放kernel的实部,虚部及幅值矩阵, 直接使用, 暂未封装   
	Mat    m_gaborMagKernel[5][8];
	Mat m_eigenvector;
	void GetFeature(Mat src,int radius,int neighbors,int grid_x,int grid_y);
protected:  
	bool  GetKernel(Size ksize, int scaleIdx, int angleIdx, Mat &realKernel);
	Mat   GetKernelMagnitude(const Mat &rekernel, const Mat&imgkernel);


	double m_kmax;  
	double m_f;  
	double m_sigma;  
	int    m_scaleNum;  
	int    m_angleNum; 
	int    m_scaleStart;
	int    m_angleStart;
	vector<Mat> m_histogram;
	Size   m_ksize;  
private:  

};  
#endif


