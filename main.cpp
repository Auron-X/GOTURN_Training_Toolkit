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
	//Turn on GPU mode for Caffe
	Caffe::set_mode(Caffe::GPU);

	//Extract training samples, Convert and Save them to HDF5 dataset
	//buildDB();

	//Train GOTURN
	//trainNet();

	//Test GOTURN *
	testNet();

	getchar();
}