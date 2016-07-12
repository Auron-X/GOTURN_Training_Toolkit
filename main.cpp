#include <opencv2\opencv.hpp>
#include <iostream>
#include <gtrUtils.hpp>
#include <opencv2\datasets\track_alov.hpp>
#include <caffe\caffe.hpp>
#include "utils.h"


using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace caffe;

void main()
{
	
	buildDB();
	trainNet();
	testNet();

	getchar();
}