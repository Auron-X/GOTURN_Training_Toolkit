#include <opencv2\opencv.hpp>
#include <iostream>
#include <gtrUtils.hpp>
#include <opencv2\datasets\track_alov.hpp>
#include <opencv2\datasets\track_vot.hpp>
#include <opencv2\tracking.hpp>
#include <opencv2\dnn.hpp>
#include <caffe\caffe.hpp>
#include "utils.h"
#include <H5Cpp.h>
#include <hdf5.h>
#include <hdf5_hl.h>


using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace cv::dnn;
using namespace caffe;

#define INPUT_SIZE 227
#define NUM_CHANNELS 3

Rect2f points2rect(vector<Point2d> gtPoints)
{
	float minX = 99999, maxX = 0, minY = 99999, maxY = 0;
	for (int j = 0; j < (int)gtPoints.size(); j++)
	{
		if (maxX < gtPoints[j].x) maxX = gtPoints[j].x;
		if (maxY < gtPoints[j].y) maxY = gtPoints[j].y;
		if (minX > gtPoints[j].x) minX = gtPoints[j].x;
		if (minY > gtPoints[j].y) minY = gtPoints[j].y;
	}
	Rect2f gtBB(minX, minY, maxX - minX, maxY - minY);
	return gtBB;
}

int main()
{
	//Turn on GPU mode for Caffe
	Caffe::set_mode(Caffe::GPU);

	//Extract training samples, Convert and Save them to HDF5 dataset
	//buildDB();

	//Train GOTURN
	//trainNet();

	//Test GOTURN
	//srand(612);
	//buildH5Datasets("D:/ALOV300++/trainDataset.h5", 10);
	//testNet("goturn_iter_30000.caffemodel");

	////Test on video


	//////////////////////////////////////////////////////////////////



	Ptr<cv::datasets::TRACK_alov> dataset = TRACK_alov::create();
	dataset->load("D:/ALOV300++");

	printf("Datasets number: %d\n", dataset->getDatasetsNum());
	for (int i = 1; i <= dataset->getDatasetsNum(); i++)
		printf("\tDataset #%d size: %d\n", i, dataset->getDatasetLength(i));

	int datasetID = 1;

	Mat frame;
	Rect2d curBB, gtBB;
	VideoWriter outputVideo;
	// create the tracker
	Ptr<Tracker> tracker = Tracker::create("GOTURN");

	for (int i = 0; i < dataset->getDatasetLength(datasetID); i++)
	{
		dataset->getFrame(frame, datasetID, i + 1);
		gtBB = gtr::anno2rect(dataset->getGT(datasetID, i + 1));

		if (i == 0)
		{
			//Ground Truth from 1-st frame
			gtBB = gtr::anno2rect(dataset->getGT(datasetID, 1));

			//Initializes the tracker
			tracker->init(frame, gtBB);

			////Define the codec and create VideoWriter object
			//int width = frame.cols; // Declare width here
			//int height = frame.rows; // Declare height here
			//Size S = Size(width, height); // Declare Size structure

			//// Open up the video for writing
			//const string filename = "video.avi"; // Declare name of file here

			//// Declare FourCC code
			//int fourcc = CV_FOURCC('M', 'P', '4', 'V');

			//// Declare FPS here
			//int fps = 5;

			////Start writing video
			//outputVideo.open(filename, -1, fps, S);
		}
		else
		{
			if (tracker->update(frame, curBB))
			{
				rectangle(frame, curBB, Scalar(0, 0, 255));
			}
		}

		if (gtBB.x != 0)
			rectangle(frame, gtBB, Scalar(0, 255, 0));
		imshow("Tracking...", frame);
		//outputVideo.write(frame);
		waitKey(1);


	}

	getchar();
}