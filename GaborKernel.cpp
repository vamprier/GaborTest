
#include "GaborKernel.h"

/**
*     函数公式：                                                                      
*                     ||Kμ,ν|| ^2    (-||Kμ,ν|| ^2*||z||^2/2*σ^2)  iKμ,νz    -σ^2/2 
*     ψμ,ν(x,y) = -----------------e                               [e         - e        ]
*                          σ^2
*
*     μ代表Gabor滤波器的方向，ν代表Gabor滤波器的尺度，μ ∈ {0, 1... ,7} and v ∈ {1, 2, 3, 4}
*     z = (x, y)
*                  iφu
*     Kμ,ν = Kνe     
*
*
*    Kμ,ν = (Kν*cos(φu),Kν*sin(φu))
*
*     Kν =  Kmax/f^ν,φu = πμ/8
*
*     Kmax代表频率的最大值，f代表频域中两个核之间的空间因子，σ = 2π, kmax = π/2 and f = √2
*
*
**/

void GaborKernel::InitGabor(Mat src,float rou)
{
	m_scaleNum = 5;  
	m_angleNum = 8;  
	m_ksize = Size(64,64);

	CalculateKernelMultiscalesAndMultiOrientations(m_ksize.width,m_ksize.height,m_scaleNum,m_angleNum,src,rou);
}

void GaborKernel::InitGabor(Size ksize,Mat src,float rou)
{
	m_scaleNum = 5;  
	m_angleNum = 8;  
	m_ksize = ksize;

	CalculateKernelMultiscalesAndMultiOrientations(m_ksize.width,m_ksize.height,m_scaleNum,m_angleNum,src,rou);
}

/**
*   输出固定尺度固定方向的Gabor核函数
*   
**/
void GaborKernel::CalculateKernel(int width,int height,int oritation,int scale,Mat &KernelReal, Mat &KernelImg)
{
	int x=0,y=0;
	for (x=-width/2;x<=width/2;x++)
	{
		for (y=-height/2;y<=height/2;y++)
		{
 			float kmax = CV_PI/2;
			float Kv = kmax/pow(sqrtf(2.0),scale);
			float faiu = (CV_PI*oritation)/8;
			float sigma = 2*CV_PI*2*CV_PI;
			float xishu = exp(-(Kv*Kv*(x*x+y*y)/(2*sigma)));
			float temp2 = cos(Kv*cos(faiu)*x+Kv*sin(faiu)*y)-exp(-sigma/2);
			float Ker_real = xishu*temp2*Kv*Kv/sigma;
			float temp3 = sin(Kv*cos(faiu)*x+Kv*sin(faiu)*y);
			float Ker_img = xishu*temp3*Kv*Kv/sigma;
			KernelReal.at<float>(x+width/2,y+height/2) = Ker_real;
			KernelImg.at<float>(x+width/2,y+height/2) = Ker_img;
		}
	}
}

void GaborKernel::CalculateKernelMultiscalesAndMultiOrientations(int width, int height,int scales,int orientations, const Mat &src, float rou)
{
	int cnt = 1;
	vector<float> gaborFeatures;
	Mat KernelReal = Mat::zeros(width+1, height+1, CV_32F );
	Mat KernelImg = Mat::zeros( width+1, height+1, CV_32F );
	Mat faceReal;
	Mat faceImg;
	Mat faceMag = Mat::zeros( src.rows, src.cols, CV_32F );
	Mat gaborImg = Mat::zeros( src.rows/rou, src.cols/rou, CV_32F );

	for ( int g=0; g<scales; g++ )
	{
		for ( int h=0; h<orientations; h++ )
		{
			CalculateKernel(width,height,g,h,KernelReal,KernelImg);
			normalize( KernelReal, KernelReal, 0, 255, NORM_MINMAX );
			///< 这里我为了测试加入了保存图像的过程，实际使用应将该部分注释，此外，我只保存了实部用以观察
			char buf[MAX_PATH] = {0};   
			itoa( cnt, buf, 10 );
			string name = "gaborKernel" ;
			name += buf;
			name += ".jpg";

			IplImage gaborWaveletImg = IplImage(KernelReal); 
			cvSaveImage( name.c_str(), &gaborWaveletImg );

			/// 卷积
			flip( KernelReal, KernelReal, -1 );
			filter2D( src, faceReal, -1, KernelReal, Point( -1, -1 ), 0, BORDER_REPLICATE ); 
			filter2D( src, faceImg, -1, KernelImg, Point( -1, -1 ), 0, BORDER_REPLICATE ); 

			/// 幅值
			faceReal = cv::Mat_<float>(faceReal);
			faceImg = cv::Mat_<float>(faceImg);
			magnitude( faceReal, faceImg, faceMag );        

			///< 这里我为了测试加入了保存图像的过程，实际使用应将该部分注释掉
			normalize( faceMag, faceMag, 0, 255, NORM_MINMAX );
			char buf1[MAX_PATH] = {0};   
			itoa( cnt, buf1, 10 );
			string name1 = "gaborKernelFeatures" ;
			name1 += buf1;
			name1 += ".jpg";

			IplImage gaborWaveletImg1 = IplImage(faceMag); 
			cvSaveImage( name1.c_str(), &gaborWaveletImg1 );
			cnt++;  

			/// 下采样到要求的尺寸
			resize( faceMag, gaborImg, gaborImg.size() );

			/// 归一化到0均值1方差
			Scalar m;
			Scalar stddev;
			meanStdDev( gaborImg, m, stddev ); 

			gaborImg = (gaborImg-m[0])/stddev[0];

			/// 得到特征
			for ( int i=0; i<gaborImg.rows; i++ )
			{
				for ( int j=0; j<gaborImg.cols; j++ )
				{
					gaborFeatures.push_back( gaborImg.at<float>(i,j) );
				}
			}
		}
	}
	
}