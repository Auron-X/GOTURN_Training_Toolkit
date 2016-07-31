#include <opencv2\opencv.hpp>
#include <iostream>
#include <gtrUtils.hpp>
#include <opencv2\datasets\track_alov.hpp>
#include <opencv2\datasets\track_vot.hpp>
#include <opencv2\tracking.hpp>
#include <caffe\caffe.hpp>
#include "utils.h"
#include <H5Cpp.h>
#include <hdf5.h>
#include <hdf5_hl.h>


using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace caffe;

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

void main()
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
	//Ptr<dnn::Importer> importer;

	//Ptr<cv::datasets::TRACK_vot> dataset = TRACK_vot::create();
	//dataset->load("D:/Dropbox/Projects/Matlab/Datasets/VOT 2015");
	//printf("Datasets number: %d\n", dataset->getDatasetsNum());
	//for (int i = 1; i <= dataset->getDatasetsNum(); i++)
	//	printf("\tDataset #%d size: %d\n", i, dataset->getDatasetLength(i));

	//int datasetID = 1;
	//dataset->initDataset(datasetID);

	//Mat prevFrame, curFrame, searchPatch, targetPatch;
	//Rect2f currBB, gtBB, prevBB;

	//for (int i = 0; i < dataset->getDatasetLength(datasetID); i++)
	//{
	//	prevFrame = curFrame.clone();
	//	prevBB = currBB;
	//	if (!dataset->getNextFrame(curFrame))
	//		break;
	//	//Draw Ground Truth BB
	//	vector <Point2d> gtPoints = dataset->getGT();
	//	Rect2f gtBB = points2rect(gtPoints);
	//	rectangle(curFrame, gtBB, Scalar(0, 255, 0));
	//	if (i == 0)
	//		currBB = gtBB;
	//	else
	//	{
	//		float padTarget = 2.0;
	//		float padSearch = 2.0;
	//		Rect2f searchPatchRect, targetPatchRect;
	//		Point2f currCenter, prevCenter;
	//		Mat prevFramePadded, curFramePadded;
	//		prevCenter.x = prevBB.x + prevBB.width / 2;
	//		prevCenter.y = prevBB.y + prevBB.height / 2;
	//		targetPatchRect.width = (float)(prevBB.width*padTarget);
	//		targetPatchRect.height = (float)(prevBB.height*padTarget);
	//		targetPatchRect.x = (float)(prevCenter.x - prevBB.width*padTarget / 2.0 + targetPatchRect.width);
	//		targetPatchRect.y = (float)(prevCenter.y - prevBB.height*padTarget / 2.0 + targetPatchRect.height);
	//		copyMakeBorder(prevFrame, prevFramePadded, targetPatchRect.height, targetPatchRect.height, targetPatchRect.width, targetPatchRect.width, BORDER_REPLICATE);
	//		targetPatch = prevFramePadded(targetPatchRect).clone();
	//		copyMakeBorder(curFrame, curFramePadded, targetPatchRect.height, targetPatchRect.height, targetPatchRect.width, targetPatchRect.width, BORDER_REPLICATE);
	//		searchPatch = curFramePadded(targetPatchRect).clone();

	//		imshow("target", targetPatch);
	//		imshow("search", searchPatch);

	//	}
	//	currBB = gtBB;

	//	imshow("VOT 2015 DATASET TEST...", curFrame);
	//	waitKey(0);


	//}

	getchar();
}