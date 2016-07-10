#include "buildH5Dataset.h"
using namespace H5;
using namespace std;
using namespace cv;
using namespace caffe;

#define INPUT_SIZE 227
#define NUM_CHANNELS 3

void buildH5Datasets(int N, string fileName)
{
	//Create HDF5 database file
	hid_t fileID = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	//Create DATA/LABEL pointers and allocate memory
	int fullVolume = N * NUM_CHANNELS * INPUT_SIZE * INPUT_SIZE;
	float* targetData = new float[fullVolume];
	float* searchData = new float[fullVolume];
	float* label = new float[N*4];

	int width, height;
	width = INPUT_SIZE;
	height = INPUT_SIZE;
	vector <vector <Mat>> targetPatchesSplitted;
	vector <vector <Mat>> searchPatchesSplitted;

	//Warp DATA/LABEL pointers to their data
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

	//Extract patches & lables
	vector <cv::gtr::TrainingSample> trainingSamples;
	

	for (int i = 0; i < N; i++)
	{

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

	H5LTmake_dataset_float(fileID, "label", numAxes, dims, label);

	delete[] dims;

	H5Fclose(fileID);
}