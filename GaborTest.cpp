// BaborTest.cpp : 定义控制台应用程序的入口点。
//

#include "Gabor.h"
#include "GaborKernel.h"
#include "ZGabor.h"
#include "LBPH.h"
#include "GaborLbp_Algorithm.h"
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\opencv.hpp>

#include <cv.h>
#include <highgui.h>
#include <vector>

using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator =';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message ="No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if(!path.empty()&&!classlabel.empty()) {
			Mat tt = imread(path, 0);
			Mat temp;
			if (tt.rows > 96 && tt.cols > 96)
			{
				temp = Mat::zeros(96,96,CV_32F);
				resize(tt,temp,temp.size());
			}
			else
			{
				temp = tt;
			}
			images.push_back(temp);
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}


void writetext(const string& filename,Mat data)
{
	std::fstream file(filename.c_str(), fstream::out);
	for (int i=0;i<data.rows;i++)
	{
		for (int j=0;j<data.cols;j++)
		{
			float f = data.at<float>(i,j);
			char tmpstr[256];
			memset(tmpstr,'\0',256);
			sprintf(tmpstr,"%f\r\n",f);
			string str = tmpstr;
			file<<str;
		}
	}
}

void writerowcol(const string& filename,int row,int col)
{
	std::fstream file(filename.c_str(), fstream::out|fstream::app);
	char tmpstr[256];
	memset(tmpstr,'\0',256);
	sprintf(tmpstr,"%d,%d\r\n",row,col);
	string str = tmpstr;
	file<<str;
}

void testGabor()
{
    /// 提取图像的Gabor特征
    /* 1. 原图像卷积Gabor 小波 Og,h(x, y) = I(x, y) ∗ ψg,h(x, y),
        2. 在多个尺度上进行下采样
        3. 归一化到0均值1方差
        4. 将各尺度变成行向量连接起来，最终的特征为scales*orientations*ImgWidth*ImgHeight/rou长的行向量
    */
 
    char name[] = "G:\\test_image\\zhaojuan.jpg";
    IplImage *img = cvLoadImage( name, CV_LOAD_IMAGE_GRAYSCALE );
    Mat src( img, 0 );
 
    /// 实际使用的窗口尺寸, 若高斯核参数为sigma，窗口一般选择为6*sigma；二维时，亦然
 //   param. width = 6 * param.gamma; 
 //   param.height=6 * param.etah; 
 
    /// 下采样
    float rou = 4;
 
    /// 求得的Gabor特征,将各尺度变成行向量连接起来，长度为 scales*orientations*ImgWidth*ImgHeight/rou 
 //   vector<float> gaborFeatures; 
  //  GaborFeaturesMultiscalesAndMultiOrientations( param, src,  rou, gaborFeatures );

	Gabor m_gabor;
	m_gabor.InitGabor(Size(6*sqrt(2.0),6*sqrt(2.0)),0.25,sqrtf(2.0),sqrtf(2.0),src,rou);
}

void testGaborKernel()
{
    char name[] = "G:\\test_image\\zhaojuan.jpg";
    IplImage *img = cvLoadImage( name, CV_LOAD_IMAGE_GRAYSCALE );
    Mat src( img, 0 );
    float rou = 4;
	GaborKernel m_gabor;
	m_gabor.InitGabor(src,rou);
}

void testZGabor()
{
	char name[] = "G:\\user-faces\\s4\\4.jpg";
	IplImage *img = cvLoadImage( name, CV_LOAD_IMAGE_GRAYSCALE );
	Mat src( img, 0 );

	int row = src.rows;
	int col = src.cols;
	writerowcol("G:\\row_col.txt",row,col);

	ZGabor m_gabor;
	m_gabor.InitGabor();
	m_gabor.GetFeature(src,1,8,8,8);
}

void testLBPH()
{
	char name[] = "G:\\test_image\\fengyundi.jpg";
	IplImage *img = cvLoadImage( name, CV_LOAD_IMAGE_GRAYSCALE );
	Mat src( img, 0 );

	LBPH m_lbph;
	m_lbph.GetFeature(src);
	writetext("G:\\eigenvector.txt",m_lbph.m_eigenvector);
}

void testAlgorithm()
{
	string fn_csv = "G:\\user-faces2.txt";
	vector<Mat> images;
	vector<int> labels;
	try {
		read_csv(fn_csv, images, labels);
	} catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	if(images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}
	//自己的图片
	Mat tt = imread("G:\\test_image\\user-faces_18.jpg", 0);
	Mat temp;
	if (tt.rows > 96 && tt.cols > 96)
	{
		temp = Mat::zeros(96,96,CV_32F);
		resize(tt,temp,temp.size());
	}
	else
	{
		temp = tt;
	}
	Mat testSample = temp;

	GaborLbp_Algorithm m_algorithm;
	m_algorithm.train(images,labels);
	double minDis = 0.0;
	int predictedLabel = m_algorithm.predict(testSample,minDis);
	putText(testSample,format("%d",predictedLabel),Point(60,60),0,0.8,Scalar(255,0,0));
	imshow("image",testSample);
	cout<<"Lable is "<<predictedLabel<<endl;
	cout<<"min Distance is "<<minDis<<endl;
	waitKey(0);
}

void testtest()
{
	vector<Mat> images;
	vector<int> labels;
	Mat src = imread("G:/user-faces/user-faces_9/1.jpg");
	images.push_back(src);
	int d = 1;
	labels.push_back(d);
	//自己的图片
	Mat tt = imread("G:\\test_image\\user-faces_18.jpg", 0);
	Mat temp;
	if (tt.rows > 96 && tt.cols > 96)
	{
		temp = Mat::zeros(96,96,CV_32F);
		resize(tt,temp,temp.size());
	}
	else
	{
		temp = tt;
	}
	Mat testSample = temp;

	GaborLbp_Algorithm m_algorithm;
	m_algorithm.train(images,labels);
	double minDis = 0.0;
	int predictedLabel = m_algorithm.predict(testSample,minDis);
	putText(testSample,format("%d",predictedLabel),Point(60,60),0,0.8,Scalar(255,0,0));
	imshow("image",testSample);
	cout<<"Lable is "<<predictedLabel<<endl;
	cout<<"min Distance is "<<minDis<<endl;
	waitKey(0);
}

int main(int argc, char* argv[])
{
	testAlgorithm();
	//testtest();
	return 0;
}

