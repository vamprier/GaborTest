
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
*     ������ʽ��                                                                      
*                     ||K��,��|| ^2    (-||K��,��|| ^2*||z||^2/2*��^2)  iK��,��z    -��^2/2 
*     �צ�,��(x,y) = -----------------e                               [e         - e  ]
*                          ��^2
*
*                     ||K��,��|| ^2    (-||K��,��|| ^2*(x^2+y^2)/2*��^2)  i()(x,y)    -��^2/2 
*     �צ�,��(x,y) = -----------------e                               [e         - e  ]
*                          ��^2
*
*     �̴���Gabor�˲����ķ��򣬦ʹ���Gabor�˲����ĳ߶ȣ��� �� {0, 1... ,7} and v �� {1, 2, 3, 4}
*     z = (x, y)
*                  i��u
*     K��,�� = K��e
*
*     K�� =  Kmax/f^��,��u = �Ц�/8
*
*     Kmax����Ƶ�ʵ����ֵ��f����Ƶ����������֮��Ŀռ����ӣ��� = 2��, kmax = ��/2 and f = ��2
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


