#include "buildH5Dataset.h"
using namespace H5;
using namespace std;
using namespace cv;
using namespace caffe;

#define INPUT_SIZE 227
#define NUM_CHANNELS 3

void buildH5Datasets(string fileName)
{
	//Create HDF5 database file
	hid_t fileID = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	//Create DATA/LABEL pointers and allocate memory
	int width, height;
	width = INPUT_SIZE;
	height = INPUT_SIZE;
	vector <vector <Mat>> targetPatchesSplitted;
	vector <vector <Mat>> searchPatchesSplitted;

	//Extract patches & lables
	Mat prevFrame, currFrame;
	Ptr<cv::datasets::TRACK_alov> alovDataset = cv::datasets::TRACK_alov::create();
	vector <cv::gtr::TrainingSample> trainingSamples, tmpSamples;

	alovDataset->loadAnnotatedOnly("D:/ALOV300++");
	Rect2f prevGTBB, currGTBB;

	int datasetID = 1;
	cout << alovDataset->getDatasetLength(datasetID) << endl;

	for (int i = 0; i < alovDataset->getDatasetLength(datasetID)-1; i++)
	{
		prevGTBB = gtr::anno2rect(alovDataset->getGT(datasetID, i + 1));
		currGTBB = gtr::anno2rect(alovDataset->getGT(datasetID, i + 2));
		alovDataset->getFrame(prevFrame, datasetID, i + 1);
		alovDataset->getFrame(currFrame, datasetID, i + 2);
		tmpSamples = gtr::gatherFrameSamples(prevFrame, currFrame, prevGTBB, currGTBB);
		trainingSamples.insert(trainingSamples.end(), tmpSamples.begin(), tmpSamples.end());
	}
	int N = trainingSamples.size();


	//Shuffle data
	random_shuffle(trainingSamples.begin(), trainingSamples.end());

	/*for (int i = 0; i < trainingSamples.size(); i++)
	{
		cv::gtr::TrainingSample  sample = trainingSamples[i];
		rectangle(sample.searchPatch, sample.targetBB, Scalar(0, 0, 255));
		imshow("1", sample.targetPatch);
		imshow("2", sample.searchPatch);
		waitKey();
	}*/

	//Warp DATA/LABEL pointers to their data
	int fullVolume = N * NUM_CHANNELS * INPUT_SIZE * INPUT_SIZE;
	float* targetData = new float[fullVolume];
	float* searchData = new float[fullVolume];
	float* labelData = new float[N * 4];
	float* pointer;
	int offset = 0;

	for (int i = 0; i < N; i++)
	{
		offset = i*NUM_CHANNELS*INPUT_SIZE*INPUT_SIZE;

		vector <Mat> targetPatchSplitted;
		pointer = targetData + offset;
		for (int j = 0; j < 3; ++j)
		{
			Mat channel(height, width, CV_32FC1, pointer);
			targetPatchSplitted.push_back(channel);
			pointer += width * height;
		}
		targetPatchesSplitted.push_back(targetPatchSplitted);

		vector <Mat> searchPatchSplitted;
		pointer = searchData + offset;
		for (int j = 0; j < 3; ++j)
		{
			Mat channel(height, width, CV_32FC1, pointer);
			searchPatchSplitted.push_back(channel);
			pointer += width * height;
		}
		searchPatchesSplitted.push_back(searchPatchSplitted);
	}

	//Split channels and map to the DATA/LABELS
	for (int i = 0; i < N; i++)
	{
		Mat targetPatch, searchPatch;
		Rect2f relBB;
		targetPatch = trainingSamples[i].targetPatch;
		searchPatch = trainingSamples[i].searchPatch;
		relBB = trainingSamples[i].targetBB;

		//Scale parameters
		float dx = (float)INPUT_SIZE / searchPatch.cols;
		float dy = (float)INPUT_SIZE / searchPatch.rows;
		cout << dx <<  "   " << dy << endl;
		//Preprocess
		//Resize
		resize(targetPatch, targetPatch, Size(INPUT_SIZE, INPUT_SIZE));
		resize(searchPatch, searchPatch, Size(INPUT_SIZE, INPUT_SIZE));

		//Mean Subtract
		targetPatch = targetPatch-128;
		searchPatch = searchPatch-128;

		//Convert to Float type
		targetPatch.convertTo(targetPatch, CV_32FC1);
		searchPatch.convertTo(searchPatch, CV_32FC1);

		//Split data to mapped memory
		split(targetPatch, targetPatchesSplitted[i]);
		split(searchPatch, searchPatchesSplitted[i]);

		//Labels mapping
		labelData[i * 4] = dx * relBB.x;
		labelData[i * 4 + 1] = dy * relBB.y;
		labelData[i * 4 + 2] = dx * relBB.width;
		labelData[i * 4 + 3] = dy * relBB.height;
	}

	int numAxes = 4;
	hsize_t *dims = new hsize_t[numAxes];
	dims[0] = N;
	dims[1] = NUM_CHANNELS;
	dims[2] = INPUT_SIZE;
	dims[3] = INPUT_SIZE;

	H5LTmake_dataset_float(fileID, "data1", numAxes, dims, targetData);
	H5LTmake_dataset_float(fileID, "data2", numAxes, dims, searchData);

	numAxes = 2;
	dims[0] = N;
	dims[1] = 4;

	H5LTmake_dataset_float(fileID, "label", numAxes, dims, labelData);

	delete[] dims;

	H5Fclose(fileID);
}