#include "utils.h"

using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace caffe;

void trainNet()
{
	boost::shared_ptr <Net <float>> net;
	NetParameter caffenetParam, goturnParam;
	SolverParameter solverParam;
	ReadSolverParamsFromTextFileOrDie("goturnSolver.prototxt", &solverParam);
	ReadNetParamsFromBinaryFileOrDie("bvlc_reference_caffenet.caffemodel", &caffenetParam);
	boost::shared_ptr<caffe::Solver<float>> solver(caffe::SolverRegistry<float>::CreateSolver(solverParam));

	//Replace CONV layers weights with Caffenet values 
	int numSourceLayers = caffenetParam.layer_size();
	net = solver->net();
	for (int i = 0; i < numSourceLayers; i++)
	{
		const LayerParameter& sourceLayer = caffenetParam.layer(i);
		boost::shared_ptr <Layer<float>> targetLayer[2];

		if (sourceLayer.name() == "conv1")
		{
			targetLayer[0] = net->layer_by_name("conv11");
			targetLayer[1] = net->layer_by_name("conv21");
		}
		else if (sourceLayer.name() == "conv2")
		{
			targetLayer[0] = net->layer_by_name("conv12");
			targetLayer[1] = net->layer_by_name("conv22");
		}
		else if (sourceLayer.name() == "conv3")
		{
			targetLayer[0] = net->layer_by_name("conv13");
			targetLayer[1] = net->layer_by_name("conv23");
		}
		else if (sourceLayer.name() == "conv4")
		{
			targetLayer[0] = net->layer_by_name("conv14");
			targetLayer[1] = net->layer_by_name("conv24");
		}
		else if (sourceLayer.name() == "conv5")
		{
			targetLayer[0] = net->layer_by_name("conv15");
			targetLayer[1] = net->layer_by_name("conv25");
		}
		else
			continue;

		if (sourceLayer.blobs_size() != 2) continue;

		if (targetLayer[0]->blobs().size() != 2 ||
			targetLayer[1]->blobs().size() != 2)
		{
			cout << "Target Blobs are not Conv";
			getchar();
			break;
		}

		for (int j = 0; j < sourceLayer.blobs_size(); j++)
		{
			targetLayer[0]->blobs()[j]->FromProto(sourceLayer.blobs(j));
			targetLayer[1]->blobs()[j]->FromProto(sourceLayer.blobs(j));
		}

		cout << "Replaced by: " << sourceLayer.name() << endl;
		cout << "Replaced by: " << sourceLayer.name() << endl;
	}

	solver->Solve();
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

void buildDB()
{
	buildH5Datasets("trainDataset.h5");
	buildH5Datasets("testDataset.h5");
}