
#include "ZGabor.h"

template <typename _Tp> static
	inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors) {
		//get matrices
		Mat src = _src.getMat();
		// allocate memory for result
		_dst.create(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
		Mat dst = _dst.getMat();
		// zero
		dst.setTo(0);
		for(int n=0; n<neighbors; n++) {
			// sample points
			float x = static_cast<float>(radius * cos(2.0*CV_PI*n/static_cast<float>(neighbors)));
			float y = static_cast<float>(-radius * sin(2.0*CV_PI*n/static_cast<float>(neighbors)));
			// relative indices
			int fx = static_cast<int>(floor(x));
			int fy = static_cast<int>(floor(y));
			int cx = static_cast<int>(ceil(x));
			int cy = static_cast<int>(ceil(y));
			// fractional part
			float ty = y - fy;
			float tx = x - fx;
			// set interpolation weights
			float w1 = (1 - tx) * (1 - ty);
			float w2 =      tx  * (1 - ty);
			float w3 = (1 - tx) *      ty;
			float w4 =      tx  *      ty;
			// iterate through your data
			for(int i=radius; i < src.rows-radius;i++) {
				for(int j=radius;j < src.cols-radius;j++) {
					// calculate interpolated value
					float t = static_cast<float>(w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx));
					// floating point precision, so check some machine-dependent epsilon
					dst.at<int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) || (std::abs(t-src.at<_Tp>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
				}
			}
		}
}

static void elbp(InputArray src, OutputArray dst, int radius, int neighbors)
{
	int type = src.type();
	switch (type) {
	case CV_8SC1:   elbp_<char>(src,dst, radius, neighbors); break;
	case CV_8UC1:   elbp_<unsigned char>(src, dst, radius, neighbors); break;
	case CV_16SC1:  elbp_<short>(src,dst, radius, neighbors); break;
	case CV_16UC1:  elbp_<unsigned short>(src,dst, radius, neighbors); break;
	case CV_32SC1:  elbp_<int>(src,dst, radius, neighbors); break;
	case CV_32FC1:  elbp_<float>(src,dst, radius, neighbors); break;
	case CV_64FC1:  elbp_<double>(src,dst, radius, neighbors); break;
	default:
		string error_msg = format("Using Original Local Binary Patterns for feature extraction only works on single-channel images (given %d). Please pass the image data as a grayscale image!", type);
		CV_Error(CV_StsNotImplemented, error_msg);
		break;
	}
}

static Mat
	histc_(const Mat& src, int minVal=0, int maxVal=255, bool normed=false)
{
	Mat result;
	// Establish the number of bins.
	int histSize = maxVal-minVal+1;
	// Set the ranges.
	float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal+1) };
	const float* histRange = { range };
	// calc histogram
	calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &histRange, true, false);
	// normalize
	if(normed) {
		result /= (int)src.total();
	}
	return result.reshape(1,1);
}

static Mat histc(InputArray _src, int minVal, int maxVal, bool normed)
{
	Mat src = _src.getMat();
	switch (src.type()) {
	case CV_8SC1:
		return histc_(Mat_<float>(src), minVal, maxVal, normed);
		break;
	case CV_8UC1:
		return histc_(src, minVal, maxVal, normed);
		break;
	case CV_16SC1:
		return histc_(Mat_<float>(src), minVal, maxVal, normed);
		break;
	case CV_16UC1:
		return histc_(src, minVal, maxVal, normed);
		break;
	case CV_32SC1:
		return histc_(Mat_<float>(src), minVal, maxVal, normed);
		break;
	case CV_32FC1:
		return histc_(src, minVal, maxVal, normed);
		break;
	default:
		CV_Error(CV_StsUnmatchedFormats, "This type is not implemented yet."); break;
	}
	return Mat();
}


static Mat spatial_histogram(InputArray _src, int numPatterns,
	int grid_x, int grid_y, bool /*normed*/)
{
	Mat src = _src.getMat();
	// calculate LBP patch size
	int width = src.cols/grid_x;
	int height = src.rows/grid_y;
	// allocate memory for the spatial histogram
	Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
	// return matrix with zeros if no data was given
	if(src.empty())
		return result.reshape(1,1);
	// initial result_row
	int resultRowIdx = 0;
	// iterate through grid
	for(int i = 0; i < grid_y; i++) {
		for(int j = 0; j < grid_x; j++) {
			Mat src_cell = Mat(src, Range(i*height,(i+1)*height), Range(j*width,(j+1)*width));
			Mat cell_hist = histc(src_cell, 0, (numPatterns-1), true);
			// copy to the result matrix
			Mat result_row = result.row(resultRowIdx);
			cell_hist.reshape(1,1).convertTo(result_row, CV_32FC1);
			// increase row count in result matrix
			resultRowIdx++;
		}
	}
	// return result as reshaped feature vector
	return result.reshape(1,1);
}

static Mat elbp(InputArray src, int radius, int neighbors) {
	Mat dst;
	elbp(src, dst, radius, neighbors);
	return dst;
}

static Mat asRowMatrix(InputArrayOfArrays src, int rtype, double alpha=1, double beta=0) {
	// make sure the input data is a vector of matrices or vector of vector
	if(src.kind() != _InputArray::STD_VECTOR_MAT && src.kind() != _InputArray::STD_VECTOR_VECTOR) {
		string error_message = "The data is expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< vector<...> >).";
		CV_Error(CV_StsBadArg, error_message);
	}
	// number of samples
	size_t n = src.total();
	// return empty matrix if no matrices given
	if(n == 0)
		return Mat();
	// dimensionality of (reshaped) samples
	size_t d = src.getMat(0).total();
	// create data matrix
	Mat data((int)n, (int)d, rtype);
	// now copy data
	for(unsigned int i = 0; i < n; i++) {
		// make sure data can be reshaped, throw exception if not!
		if(src.getMat(i).total() != d) {
			string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src.getMat(i).total());
			CV_Error(CV_StsBadArg, error_message);
		}
		// get a hold of the current row
		Mat xi = data.row(i);
		// make reshape happy by cloning for non-continuous matrices
		if(src.getMat(i).isContinuous()) {
			src.getMat(i).reshape(1, 1).convertTo(xi, rtype, alpha, beta);
		} else {
			src.getMat(i).clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
		}
	}
	return data;
}
 
bool ZGabor::InitGabor()  
{  
    m_kmax = CV_PI/2;  
    m_f = sqrt(double(2));  
    m_sigma = 2*CV_PI;  
    m_scaleNum = 5;  
    m_angleNum = 8;  
	m_scaleStart = 0;
	m_angleStart = 0;
    m_ksize = Size(6*m_f,6*m_f);  
    return true;  
}  

bool ZGabor::InitGabor(Size ksize, double kmax, double f, double sigma)  
{  
    m_kmax = kmax;  
    m_f = f;  
    m_sigma = sigma;  
    m_scaleNum = 5;  
    m_angleNum = 8;  
	m_scaleStart = 0;
	m_angleStart = 0;
    m_ksize = ksize;  
    return true;  
}    
  
Mat  ZGabor::GetKernelMagnitude(const Mat &rekernel, const Mat&imgkernel)  
{  
    CV_Assert(rekernel.size() == imgkernel.size());  
    CV_Assert(rekernel.type() == imgkernel.type());  
    Mat mag;  
    magnitude(rekernel,imgkernel, mag);   
  
    return mag;  
}  

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
  
/** 
*    Real Part: 
*              G(k,x,y,θ)=k^2/σ^2*exp⁡(-(k^2 (x^2+y^2 ))/(2σ^2 ))*(cos(k(xcosθ+ysinθ))-exp⁡(-σ^2/2)) 
*    Imag Part: 
*              G(k,x,y,θ)=k^2/σ^2*exp⁡(-(k^2 (x^2+y^2 ))/(2σ^2 ))*(sin(k(xcosθ+ysinθ))) 
*    In:
*       ksize --kernel size 
*       scaleIdx --Scale index 
*       andeIdx  --angle index 
*    Out: 
*        realKernel  --real part of kernel 
*        imgKernel   --imagine part of kernel 
**/  
bool  ZGabor::GetKernel(Size ksize, int scaleIdx, int angleIdx, Mat &realKernel)  
{  
    //reset out para mat size  
    realKernel.create(ksize.width+1,ksize.height+1,CV_32FC1);  
 //   imgKernel.create(ksize.width+1,ksize.height+1,CV_32FC1);  

    for (int x=-ksize.width/2;x<=ksize.width/2;x++)  
    {  
        for (int y=-ksize.height/2;y<=ksize.height/2;y++)  
        {  
			float kmax = CV_PI/2;
			float Kv = kmax/pow(sqrtf(2.0),scaleIdx);
			float faiu = (CV_PI*angleIdx)/8;
			float sigma = 2*CV_PI*2*CV_PI;
			float xishu = exp(-(Kv*Kv*(x*x+y*y)/(2*sigma)));
			float temp2 = cos(Kv*cos(faiu)*x+Kv*sin(faiu)*y)-exp(-sigma/2);
			float Ker_real = xishu*temp2*Kv*Kv/sigma;
//			float temp3 = sin(Kv*cos(faiu)*x+Kv*sin(faiu)*y);
//			float Ker_img = xishu*temp3*Kv*Kv/sigma;
			realKernel.at<float>(x+ksize.width/2,y+ksize.height/2) = Ker_real;
//			imgKernel.at<float>(x+ksize.width/2,y+ksize.height/2) = Ker_img;
        }  
    }  
  
    return true;  
}  

void ZGabor::GetFeature(Mat src,int radius,int neighbors,int grid_x,int grid_y)  
{
	int cnt = 1;
	Mat reKernel;  
	Mat imgKernel;  

	for (int scaleIdx=m_scaleStart;scaleIdx<m_scaleNum;scaleIdx++)  
	{  
		for (int angleIdx=m_angleStart;angleIdx<m_angleNum;angleIdx++)  
		{  
			Mat m_gaborReKernel;
			Mat    m_gaborFaceFeature = Mat::zeros( src.rows, src.cols, CV_32F );
			GetKernel(m_ksize, scaleIdx, angleIdx, m_gaborReKernel);  			
			/// 卷积
			flip( m_gaborReKernel, m_gaborReKernel, -1 );
			filter2D( src, m_gaborFaceFeature, -1, m_gaborReKernel, Point( -1, -1 ), 0, BORDER_REPLICATE ); 
			/// 幅值
			m_gaborFaceFeature = cv::Mat_<float>(m_gaborFaceFeature);
			///LBP编码
			Mat lbp_image = elbp(m_gaborFaceFeature,radius,neighbors);
			Mat histogram = spatial_histogram(
				lbp_image, /* lbp_image */
				static_cast<int>(std::pow(2.0, static_cast<double>(neighbors))), /* number of possible patterns */
				grid_x, /* grid size x */
				grid_y, /* grid size y */
				true);
			m_histogram.push_back(histogram);
			m_gaborReKernel.release();
			m_gaborFaceFeature.release();
		}
	}   
	m_eigenvector = asRowMatrix(m_histogram, CV_32FC1);
}