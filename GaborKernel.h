
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\opencv.hpp>

#include <cv.h>
#include <highgui.h>
#include <vector>

using namespace cv;
using namespace std;

#define  MAX_PATH 255
/**
*     函数公式：                                                                      
*                     ||Kμ,ν|| ^2    (-||Kμ,ν|| ^2*||z||^2/2*σ^2)  iKμ,νz    -σ^2/2 
*     ψμ,ν(x,y) = -----------------e                               [e         - e  ]
*                          σ^2
*
*                     ||Kμ,ν|| ^2    (-||Kμ,ν|| ^2*(x^2+y^2)/2*σ^2)  i()(x,y)    -σ^2/2 
*     ψμ,ν(x,y) = -----------------e                               [e         - e  ]
*                          σ^2
*
*     μ代表Gabor滤波器的方向，ν代表Gabor滤波器的尺度，μ ∈ {0, 1... ,7} and v ∈ {1, 2, 3, 4}
*     z = (x, y)
*                  iφu
*     Kμ,ν = Kνe
*
*     Kν =  Kmax/f^ν,φu = πμ/8
*
*     Kmax代表频率的最大值，f代表频域中两个核之间的空间因子，σ = 2π, kmax = π/2 and f = √2
*
*
**/
class GaborKernel
{
public:
	void InitGabor(Mat src,float rou);
	void InitGabor(Size ksize,Mat src,float rou);

protected:
	void CalculateKernel(int width,int height,int oritation,int scale,Mat &KernelReal, Mat &KernelImg);
	void CalculateKernelMultiscalesAndMultiOrientations(int width, int height,int scales,int orientations, const Mat &src, float rou);

	int    m_scaleNum;  
	int    m_angleNum;  
	Size   m_ksize;
};


