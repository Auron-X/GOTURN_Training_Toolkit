#include <H5Cpp.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <opencv2\opencv.hpp>
#include <gtrUtils.hpp>
#include <opencv2\datasets\track_alov.hpp>
#include <iostream>
#include <caffe\caffe.hpp>
#include <algorithm>

void buildH5Datasets(std::string fileName);