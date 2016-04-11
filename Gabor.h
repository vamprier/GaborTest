
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
*    \brief GaborWavelet 的参数
*   
*                           f^2
*    ψ(f,θ,γ,η)  = ------------- exp( - (f^2*Xt^2/γ^2 + f^2*Yt^2/η^2) ) exp( j*2pai*f*Xt )
*                         pai*γ*η
*    xt = x cos θ + y sin θ,
*    yt = −x sin θ + y cos θ
*
*    ψg,h(x, y) = ψ(fg,θh,γ,η),
*      fg = fmax/(sqrtf(2))^g
*      θh = h*π/8
*    g,h由scales, orientations 决定, 例如 scales=5, orientations=8时，g={0, . . . , 4}，h=h ∈ {0, . . . , 7}
*    
*      γ、η、fmax是预先设好的参数
**/

class Gabor
{
public:
	void InitGabor(Mat src,float rou);
	void InitGabor(Size ksize, double kmax, double sigma,double etah,Mat src,float rou);
  
protected:
	void TwoDimGaborwavelet(int width, int height, float f, float sita, float gamma, float etah, Mat &KernelReal, Mat &KernelImg);
	void GaborFeaturesMultiscalesAndMultiOrientations( int width, int height,int scales,int orientations,float fmax,float gamma,float etah, const Mat &src, float rou);

	double m_kmax;  
	double m_sigma;  
	double m_etah;
	int    m_scaleNum;  
	int    m_angleNum;  
	Size   m_ksize;
};
