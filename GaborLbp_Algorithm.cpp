
#include "GaborLbp_Algorithm.h"


GaborLbp_Algorithm::GaborLbp_Algorithm(void)
{
}


GaborLbp_Algorithm::~GaborLbp_Algorithm(void)
{

}

void GaborLbp_Algorithm::train(InputArrayOfArrays _in_src, InputArray _in_labels)
{
	if(_in_src.kind() != _InputArray::STD_VECTOR_MAT && _in_src.kind() != _InputArray::STD_VECTOR_VECTOR) {
		string error_message = "The images are expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< vector<...> >).";
		CV_Error(CV_StsBadArg, error_message);
	}
	if(_in_src.total() == 0) {
		string error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
		CV_Error(CV_StsUnsupportedFormat, error_message);
	} else if(_in_labels.getMat().type() != CV_32SC1) {
		string error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _in_labels.type());
		CV_Error(CV_StsUnsupportedFormat, error_message);
	}
	vector<Mat> src;
	_in_src.getMatVector(src);
	Mat labels = _in_labels.getMat();
	if(labels.total() != src.size()) {
		string error_message = format("The number of samples (src) must equal the number of labels (labels). Was len(samples)=%d, len(labels)=%d.", src.size(), m_labels.total());
		CV_Error(CV_StsBadArg, error_message);
	}
	int i=0,j=0;
	double m_kmax = CV_PI/2;  
	double m_f = sqrt(double(2));  
	double m_sigma = 2*CV_PI;  
	for (size_t sampleIdx = 0;sampleIdx<src.size();sampleIdx++)
	{
		int row = src[sampleIdx].rows;
		int col = src[sampleIdx].cols;
		if (row <= 0 || col <= 0)
		{
			continue;
		}
		m_labels.push_back(labels.at<int>((int)sampleIdx));

		ZGabor m_gabor;
		m_gabor.InitGabor();		
		m_gabor.GetFeature(src[sampleIdx],1,8,8,8);
		cout<<sampleIdx<<endl;
		m_projection.push_back(m_gabor.m_eigenvector);
	}
}

int GaborLbp_Algorithm::predict(InputArray _src,double& minDis)
{
	int i=0,j=0;
	int which_label = -1;
	Mat src = _src.getMat();
	int row = src.rows;
	int col = src.cols;
	if (row <= 0 || col <= 0)
	{
		return which_label;
	}
	ZGabor m_gabor;
	m_gabor.InitGabor();
	m_gabor.GetFeature(src,1,8,8,8);
	Mat combinemat = m_gabor.m_eigenvector;

	/*double */minDis = DBL_MAX;
	for (size_t sampleIdx=0;sampleIdx<m_projection.size();sampleIdx++)
	{
		double dis = compareHist(m_projection[sampleIdx],combinemat,CV_COMP_CHISQR);
		if (dis < minDis)
		{
			minDis = dis;
			which_label = m_labels.at<int>((int)sampleIdx);
		}
	}
	return which_label;
}

double SameClassMean(int ClassNumber,int SampleNumber[],Mat SameClassHistogram[][10])
{
	double mean = 0.0;
	double C_sum = 0.0;
	for (int i=0;i<ClassNumber;i++)
	{
		double sum = 0.0;
		for (int k=1;k<SampleNumber[i];k++)
		{
			for (int j=0;j<k-1;j++)
			{
				double dis = compareHist(SameClassHistogram[i][j],SameClassHistogram[i][k],CV_COMP_CHISQR);
				sum+=dis;
			}
		}	
		double temp = (2*sum)/(SampleNumber[i]*(SampleNumber[i]-1));
		C_sum+=temp;
	}
	mean = C_sum/ClassNumber;
	return mean;
}

double SameClassSemblanceVariance(int ClassNumber,int SampleNumber[],Mat SameClassHistogram[][10],double mean)
{
	double Variance = 0.0;
	for (int i=0;i<ClassNumber;i++)
	{
		double sum = 0.0;
		for (int k=1;k<SampleNumber[i];k++)
		{
			for (int j=0;j<k-1;j++)
			{
				double dis = compareHist(SameClassHistogram[i][j],SameClassHistogram[i][k],CV_COMP_CHISQR);
				double cha = dis - mean;
				double temp = cha*cha;
				sum += temp;
			}
		}	
		Variance += sum;
	}
	return Variance;
}

double DistinctClassMean(int ClassNumber,int SampleNumber[],Mat SameClassHistogram[][10])
{
	double mean = 0.0;
	double C_sum = 0.0;
	for (int i=0;i<ClassNumber-1;i++)
	{
		for (int j=i+1;j<ClassNumber;j++)
		{
			double sum = 0.0;
			for (int k=0;k<SampleNumber[i];k++)
			{
				for (int l=0;l<SampleNumber[j];k++)
				{
					double dis = compareHist(SameClassHistogram[i][k],SameClassHistogram[j][l],CV_COMP_CHISQR);
					sum+= dis;
				}
			}
			double temp = sum/(SampleNumber[i]*SampleNumber[j]);
			C_sum+= temp;
		}
	}
	mean = (C_sum*2)/(ClassNumber*(ClassNumber-1));
	return mean;
}

double DistinctClassSemblanceVariance(int ClassNumber,int SampleNumber[],Mat SameClassHistogram[][10],double mean)
{
	double Variance = 0.0;
	for (int i=0;i<ClassNumber;i++)
	{
		for (int j=i+1;j<ClassNumber;j++)
		{
			double sum = 0.0;
			for (int k=0;k<SampleNumber[i];k++)
			{
				for (int l=0;l<SampleNumber[j];k++)
				{
					double dis = compareHist(SameClassHistogram[i][k],SameClassHistogram[j][l],CV_COMP_CHISQR);
					double cha = dis - mean;
					double temp = cha*cha;
					sum += temp;
				}
			}
			Variance+= sum;
		}
	}
	return Variance;
}

double GetWight(double SCMean,double DCMean,double SCSVariance,double DCSVariance)
{
	double Wight = (SCMean-DCMean)*(SCMean-DCMean)/(SCSVariance*SCSVariance+DCSVariance*DCSVariance);
	return Wight;
}
