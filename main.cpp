#include <opencv2\opencv.hpp>
#include <iostream>
#include <gtrUtils.hpp>
#include <opencv2\datasets\track_alov.hpp>
#include <caffe\caffe.hpp>
#include "buildH5Dataset.h"

using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace caffe;

void buildDB(int n)
{
	buildH5Datasets(100, "trainDataset.h5");
	buildH5Datasets(10, "testDataset.h5");
}

void testNet()
{
	Caffe::set_mode(Caffe::GPU);
	Net<float> net("goturnDeploy.prototxt", TEST);
	net.CopyTrainedLayersFrom(".caffemodel");

	boost::shared_ptr<Blob <float>> data1Layer;
	boost::shared_ptr<Blob <float>> data2Layer;
	boost::shared_ptr<Blob <float>> labelLayer;
	Blob <float>* outputLayer;

	outputLayer = net.output_blobs()[0];
	data1Layer = net.blob_by_name("data1");
	data2Layer = net.blob_by_name("data2");
	labelLayer = net.blob_by_name("label");

	net.Forward();

	float *out = outputLayer->mutable_cpu_data();
	float *data1 = data1Layer->mutable_cpu_data();
	float *data2 = data2Layer->mutable_cpu_data();
	float *label = labelLayer->mutable_cpu_data();

	for (int i = 0; i < 10; i++)
		cout << data1[i] << " " << data2[i] << " " << label[i] << " " << out[i] << endl;
}

void trainNet()
{
	SolverParameter solverPar;
	ReadSolverParamsFromTextFileOrDie("goturnSolver.prototxt", &solverPar);

	boost::shared_ptr<caffe::Solver<float>> solver(caffe::SolverRegistry<float>::CreateSolver(solverPar));
	solver->Solve();

	getchar();
}

void main()
{
	buildDB(1000);
	trainNet();
	testNet();





	getchar();


	/*outputLayer = net.output_blobs()[0];
	float* outputData = outputLayer->mutable_cpu_data();
	vector <int> dims = outputLayer->shape();

	float max = -9999;
	int maxID = 0;
	for (int i = 0; i < 1000; i++)
	{
		if (max < outputData[i])
		{
			max = outputData[i];
			maxID = i;
		}
		cout << outputData[i] << endl;
	}

	cout << maxID+1 << "	" << outputData[maxID] << endl;




	Ptr<TRACK_alov> alovDataset = TRACK_alov::create();
	alovDataset->loadAnnotatedOnly("D:/ALOV300++");
	alovDataset->getFrame(img, 1, 1);
	imshow("X", orig);

	waitKey();*/
}