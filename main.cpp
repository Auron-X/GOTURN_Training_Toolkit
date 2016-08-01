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


	//////////////////////////////////////////////////////////////////

	String modelTxt = "goturn.prototxt";
	String modelBin = "goturn_iter_30000.caffemodel";
	Ptr<dnn::Importer> importer;
	try                                     //Try to import GOTURN model
	{
		importer = dnn::createCaffeImporter(modelTxt, modelBin);
	}
	catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
	{
		std::cerr << err.msg << std::endl;
	}
	if (!importer)
	{
		std::cerr << "Can't load network by using the following files: " << std::endl;
		std::cerr << "prototxt:   " << modelTxt << std::endl;
		std::cerr << "caffemodel: " << modelBin << std::endl;
		std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
		std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
		exit(-1);
	}
	dnn::Net net;
	importer->populateNet(net);
	importer.release();                     //We don't need importer anymore

	Ptr<cv::datasets::TRACK_alov> dataset = TRACK_alov::create();
	dataset->load("D:/ALOV300++");

	printf("Datasets number: %d\n", dataset->getDatasetsNum());
	for (int i = 1; i <= dataset->getDatasetsNum(); i++)
		printf("\tDataset #%d size: %d\n", i, dataset->getDatasetLength(i));

	int datasetID = 35;

	Mat prevFrame, curFrame, searchPatch, targetPatch;
	Rect2f currBB, gtBB, prevBB;
	VideoWriter outputVideo;
	for (int i = 0; i < dataset->getDatasetLength(datasetID); i++)
	{
		prevFrame = curFrame.clone();
		prevBB = currBB;
		dataset->getFrame(curFrame, datasetID, i+1);

		//Draw Ground Truth BB
		Rect2f gtBB = gtr::anno2rect(dataset->getGT(datasetID, i+1));

		if (i == 0)
		{
			currBB = gtBB;
			if (gtBB.x == 0) cout << "X=0!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";

			//Define the codec and create VideoWriter object
			int width = curFrame.cols; // Declare width here
			int height = curFrame.rows; // Declare height here
			Size S = Size(width, height); // Declare Size structure

			// Open up the video for writing
			const string filename = "video.avi"; // Declare name of file here

			// Declare FourCC code
			int fourcc = CV_FOURCC('M', 'P', '4', 'V');

			// Declare FPS here
			int fps = 5;

			outputVideo.open(filename, -1, fps, S);
		}
		else
		{
			float padTarget = 2.0;
			float padSearch = 2.0;
			Rect2f searchPatchRect, targetPatchRect;
			Point2f currCenter, prevCenter;
			Mat prevFramePadded, curFramePadded;

			prevCenter.x = prevBB.x + prevBB.width / 2;
			prevCenter.y = prevBB.y + prevBB.height / 2;

			targetPatchRect.width = (float)(prevBB.width*padTarget);
			targetPatchRect.height = (float)(prevBB.height*padTarget);
			targetPatchRect.x = (float)(prevCenter.x - prevBB.width*padTarget / 2.0 + targetPatchRect.width);
			targetPatchRect.y = (float)(prevCenter.y - prevBB.height*padTarget / 2.0 + targetPatchRect.height);

			copyMakeBorder(prevFrame, prevFramePadded, targetPatchRect.height, targetPatchRect.height, targetPatchRect.width, targetPatchRect.width, BORDER_REPLICATE);
			targetPatch = prevFramePadded(targetPatchRect).clone();

			copyMakeBorder(curFrame, curFramePadded, targetPatchRect.height, targetPatchRect.height, targetPatchRect.width, targetPatchRect.width, BORDER_REPLICATE);
			searchPatch = curFramePadded(targetPatchRect).clone();

			resize(targetPatch, targetPatch, Size(INPUT_SIZE, INPUT_SIZE));
			resize(searchPatch, searchPatch, Size(INPUT_SIZE, INPUT_SIZE));

			imshow("target", targetPatch);
			imshow("search", searchPatch);

			//Preprocess
			//Resize
			resize(targetPatch, targetPatch, Size(INPUT_SIZE, INPUT_SIZE));
			resize(searchPatch, searchPatch, Size(INPUT_SIZE, INPUT_SIZE));

			//Mean Subtract
			targetPatch = targetPatch - 128;
			searchPatch = searchPatch - 128;

			//Convert to Float type
			targetPatch.convertTo(targetPatch, CV_32FC1);
			searchPatch.convertTo(searchPatch, CV_32FC1);

			dnn::Blob targetBlob = dnn::Blob(targetPatch);
			dnn::Blob searchBlob = dnn::Blob(searchPatch);

			net.setBlob(".data1", targetBlob);
			net.setBlob(".data2", searchBlob);

			net.forward();
			dnn::Blob res = net.getBlob("scale");

			Mat resMat = res.matRefConst().reshape(1, 1);

			currBB.x = targetPatchRect.x + (resMat.at<float>(0) * targetPatchRect.width / INPUT_SIZE) - targetPatchRect.width;
			currBB.y = targetPatchRect.y + (resMat.at<float>(1) * targetPatchRect.height / INPUT_SIZE) - targetPatchRect.height;
			currBB.width = (resMat.at<float>(2) - resMat.at<float>(0)) * targetPatchRect.width / INPUT_SIZE;
			currBB.height = (resMat.at<float>(3) - resMat.at<float>(1)) * targetPatchRect.height / INPUT_SIZE;

			cout << resMat.at<float>(0) << " " << resMat.at<float>(1) << " " << resMat.at<float>(2) << " " << resMat.at<float>(3) << endl;
			cout << currBB.x << " " << currBB.y << " " << currBB.width << " " << currBB.height << endl;
			cout << endl;

		}

		rectangle(curFrame, currBB, Scalar(0, 0, 255));
		if (gtBB.x != 0)
			rectangle(curFrame, gtBB, Scalar(0, 255, 0));
		imshow("VOT 2015 DATASET TEST...", curFrame);
		outputVideo.write(curFrame);
		waitKey(1);


	}

	getchar();
}