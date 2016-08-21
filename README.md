# GOTURN Training Toolkit

This is the code for training of GOTURN tracker implemented inside OpenCV.

Original GOTURN paper:

**[Learning to Track at 100 FPS with Deep Regression Networks](http://davheld.github.io/GOTURN/GOTURN.html)**,
<br>
[David Held](http://davheld.github.io/),
[Sebastian Thrun](http://robots.stanford.edu/),
[Silvio Savarese](http://cvgl.stanford.edu/silvio/),
<br>
European Conference on Computer Vision (ECCV), 2016 (In press)

## Installation

### Install dependencies:

* Install Caffe and compile using the CMake build instructions:
http://caffe.berkeleyvision.org/installation.html

* Install OpenCV
```
sudo apt-get install libopencv-dev
```

### Building HDF5 dataset
For GOTURN training, first HDF5 dataset shold be generated. It can be done by simple function call:

```
buildDB()
```
By default it generates 10 HDF5 datasets every with 500x10 samples (10 crop samples per image as proposed by authors). As alternative more low-level function can be used:

```
buildH5Datasets(datasetName, numberOfSamples)
```
### Training network
GOTURN training requires a "bvlc_reference_caffenet.caffemodel" for GOTURN network weights inialization:

http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

Training is launched by next line:
```
trainNet()
```
All hyperparameters are configured in goturnSolver.prototxt, for more details refer Caffe documentation.

### Evaluate the tracker
In order to visualize results, GOTURN tracker can be tested on test dataset:

```
buildH5Datasets("D:/ALOV300++/trainDataset.h5", 10);
testNet("goturn_iter_30000.caffemodel");
```
First command generate a new small test dataset, and second launching a test procedure with visualization.

### Pretrained GOTURN model

Also there is pretrained GOTURN model is available in OpenCV_extra repository

https://github.com/opencv/opencv_extra

