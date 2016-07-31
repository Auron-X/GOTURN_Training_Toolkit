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

		for (int j = 0; j < targetLayer[0]->blobs().size(); ++j) {
			if (!targetLayer[0]->blobs()[j]->ShapeEquals(sourceLayer.blobs(j))) {
				Blob<float > source_blob;
				const bool kReshape = true;
				source_blob.FromProto(sourceLayer.blobs(j), kReshape);
			}
			const bool kReshape = false;
			targetLayer[0]->blobs()[j]->FromProto(sourceLayer.blobs(j), kReshape);
		}
		/*for (int j = 0; j < sourceLayer.blobs_size(); j++)
		{
			targetLayer[0]->blobs()[j]->FromProto(sourceLayer.blobs(j));
			targetLayer[1]->blobs()[j]->FromProto(sourceLayer.blobs(j));
		}*/

		//cout << "Replaced by: " << sourceLayer.name() << endl;
		//cout << "Replaced by: " << sourceLayer.name() << endl;
	}

	solver->Solve();
}

void testNet(string modelPath)
{
	Caffe::set_mode(Caffe::GPU);
	Net<float> net("goturnDeploy.prototxt", TEST);
	net.CopyTrainedLayersFrom(modelPath);

	boost::shared_ptr<Blob <float>> data1Layer;
	boost::shared_ptr<Blob <float>> data2Layer;
	boost::shared_ptr<Blob <float>> labelLayer;
	boost::shared_ptr<Blob <float>> outputLayer;

	outputLayer = net.blob_by_name("out");
	data1Layer = net.blob_by_name("data1");
	data2Layer = net.blob_by_name("data2");
	labelLayer = net.blob_by_name("label");

	net.Forward();

	float *out = outputLayer->mutable_cpu_data();
	float *data1 = data1Layer->mutable_cpu_data();
	float *data2 = data2Layer->mutable_cpu_data();
	float *label = labelLayer->mutable_cpu_data();

	for (int k = 0; k < 100; k++)
	{
		//Print GTBB
		cout << label[k * 4 + 0] << " " << label[k * 4 + 1] << " " << label[k * 4 + 2] << " " << label[k * 4 + 3] << endl;
		//Print GOTURN_BB
		cout << out[k * 4 + 0] << " " << out[k * 4 + 1] << " " << out[k * 4 + 2] << " " << out[k * 4 + 3] << endl;

		//Make GTBB and GOTURN_BB
		Rect2f gtbb, res_bb;
		gtbb.x = label[k * 4 + 0];
		gtbb.y = label[k * 4 + 1];
		gtbb.width = label[k * 4 + 2] - label[k * 4 + 0];
		gtbb.height = label[k * 4 + 3] - label[k * 4 + 1];

		res_bb.x = out[k * 4 + 0];
		res_bb.y = out[k * 4 + 1];
		res_bb.width = out[k * 4 + 2] - out[k * 4 + 0];
		res_bb.height = out[k * 4 + 3] - out[k * 4 + 1];

		//Construct Target/Search patches from data1/data2
		vector <Mat> channelsTargetPatch;
		vector <Mat> channelsSearchPatch;
		Mat targetPatch;
		Mat searchPatch;

		for (int i = 0; i < 3; i++)
		{
			Mat channelTarget(227, 227, CV_32FC1, data1 + i * 227 * 227 + k * 3 * 227 * 227);
			Mat channelSearch(227, 227, CV_32FC1, data2 + i * 227 * 227 + k * 3 * 227 * 227);
			channelsTargetPatch.push_back(channelTarget);
			channelsSearchPatch.push_back(channelSearch);
		}
		//RGB -> BGR and Merge
		//reverse(channelsTargetPatch.begin(), channelsTargetPatch.end());
		//reverse(channelsSearchPatch.begin(), channelsSearchPatch.end());

		merge(channelsTargetPatch, targetPatch);
		merge(channelsSearchPatch, searchPatch);

		//Add mean
		targetPatch = targetPatch + 128;
		searchPatch = searchPatch + 128;

		targetPatch.convertTo(targetPatch, CV_8U);
		searchPatch.convertTo(searchPatch, CV_8U);

		//Draw GT/GOTURN bounding boxes and show patches
		rectangle(searchPatch, gtbb, Scalar(0, 255, 0));
		rectangle(searchPatch, res_bb, Scalar(0, 0, 255));
		imshow("Target", targetPatch);
		imshow("Search", searchPatch);
		waitKey();
	}
}

void buildDB()
{
	//Generate training datasets
	for (int i = 1; i <= 10; i++)
	{
		string fileName = "D:/ALOV300++/trainDataset_" + to_string(i) + ".h5";
		buildH5Datasets(fileName, 500);
	}
}