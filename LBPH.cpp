
#include "LBPH.h"

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


void LBPH::LBPEncoding(InputArray src,Mat&histogram,int radius, int neighbors,int grid_x,int grid_y)
{
	Mat lbp_image = elbp(src,radius,neighbors);
 	histogram = spatial_histogram(
 		lbp_image, /* lbp_image */
 		static_cast<int>(std::pow(2.0, static_cast<double>(neighbors))), /* number of possible patterns */
 		grid_x, /* grid size x */
 		grid_y, /* grid size y */
 		true);
	int type = lbp_image.type();
	type = histogram.type();
}

void LBPH::GetFeature(Mat src)
{
	m_eigenvector = Mat::zeros( src.rows, src.cols, CV_32F );
	LBPEncoding(src,m_eigenvector,1,8,8,8);
// 	///< 这里我为了测试加入了保存图像的过程，实际使用应将该部分注释掉
// 	normalize( m_histogram, m_histogram, 0, 255, NORM_MINMAX );
// 	string name1 = "gaborFeaturesMag.jpg" ;
// 	IplImage gaborWaveletImg1 = IplImage(m_histogram); 
// 	cvSaveImage( name1.c_str(), &gaborWaveletImg1 );
}
