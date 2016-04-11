
#include "Gabor.h"


void Gabor::InitGabor(Mat src,float rou)
{
	m_kmax = 0.25;  
	m_sigma = sqrtf(2.0);  
	m_etah = sqrtf(2.0);
	m_scaleNum = 5;  
	m_angleNum = 8;  
	m_ksize = Size(64,64);
	GaborFeaturesMultiscalesAndMultiOrientations(m_ksize.width,m_ksize.height,m_scaleNum,m_angleNum,m_kmax,m_sigma,m_etah,src,rou);
}

void Gabor::InitGabor(Size ksize, double kmax, double sigma,double etah,Mat src,float rou)
{
	m_kmax = kmax;  
	m_sigma = sigma;  
	m_etah = etah;
	m_scaleNum = 5;  
	m_angleNum = 8;  
	m_ksize = ksize;
	GaborFeaturesMultiscalesAndMultiOrientations(m_ksize.width,m_ksize.height,m_scaleNum,m_angleNum,m_kmax,m_sigma,m_etah,src,rou);
}

/**
*    \brief  输出固定尺度固定方向的Gabor小波
*                           f^2
*    ψ(f,θ,γ,η)  = ------------- exp( - (f^2*Xt^2/γ^2 + f^2*Yt^2/η^2) ) exp( j*2pai*f*Xt )
*                         pai*γ*η
*    xt = x cos θ + y sin θ,
*    yt = −x sin θ + y cos θ
*    \param[in] width 小波窗口宽
*    \param[in] height 小波窗口高
*    \param[in] f 参数
*    \param[in] sita  参数
*    \param[in] gamma 参数
*    \param[in] etah  参数
*    \param[in][out] KernelReal Gabor小波实部
*    \param[in][out] KernelImg Gabor小波虚部
**/
void Gabor::TwoDimGaborwavelet( int width, int height, float f, float sita, float gamma, float etah, Mat &KernelReal, Mat &KernelImg  )
{
	for ( int x=-width/2; x<width/2; x++)
	{
		for ( int y=-height/2; y<height/2; y++ )
		{
			float xt = x * cos(sita) + y * sin(sita);
			float yt = -x * sin(sita) + y * cos(sita);

			float fai = f*f / (CV_PI*gamma*etah) * exp( - f*f* ( xt*xt/(gamma*gamma) + yt*yt/(etah*etah) ) ) ;

			KernelReal.at<float>(x+width/2, y+height/2) = fai * cos( 2*CV_PI*f*xt);
			KernelImg.at<float>(x+width/2, y+height/2) = fai * sin( 2*CV_PI*f*xt);
		}
	}
}

/**
*    \brief  提取多尺度、多方向的Gabor特征
*    1. 原图像卷积Gabor 小波 Og,h(x, y) = I(x, y) ∗ ψg,h(x, y),
*    2. 在多个尺度上进行下采样
*    3. 归一化到0均值1方差
*    4. 将各尺度变成行向量连接起来，最终的特征为scales*orientations*ImgWidth*ImgHeight/rou长的行向量
*
*    \param[in] param 参数，详见结构体说明
*    \param[in] src 原图像
*    \param[in] rou 下采样的比例
*    \param[in] gaborFeatures  求得的gabor特征
**/
void Gabor::GaborFeaturesMultiscalesAndMultiOrientations( int width, int height,int scales,int orientations,float fmax,float gamma,float etah,const Mat &src, float rou)
{
	int cnt = 1;
	vector<float> gaborFeatures;
	Mat KernelReal = Mat::zeros( width, height, CV_32F );
	Mat KernelImg = Mat::zeros( width, height, CV_32F );
	Mat faceReal;
	Mat faceImg;
	Mat faceMag = Mat::zeros( src.rows, src.cols, CV_32F );
	Mat gaborImg = Mat::zeros( src.rows/rou, src.cols/rou, CV_32F );

	for ( int g=0; g<scales; g++ )
	{
		for ( int h=0; h<orientations; h++ )
		{
			float f = fmax/pow( sqrtf(2.0), g );
			float sita = h * CV_PI / 8;
			TwoDimGaborwavelet( width, height, f, sita, gamma, etah, KernelReal, KernelImg );
			normalize( KernelReal, KernelReal, 0, 255, NORM_MINMAX );

			///< 这里我为了测试加入了保存图像的过程，实际使用应将该部分注释，此外，我只保存了实部用以观察
			char buf[MAX_PATH] = {0};   
			itoa( cnt, buf, 10 );
			string name = "gaborWaveletKernelReal" ;
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
			string name1 = "gaborFeaturesMag" ;
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