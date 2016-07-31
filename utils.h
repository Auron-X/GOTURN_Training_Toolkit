#include <opencv2\opencv.hpp>
#include <iostream>
#include <gtrUtils.hpp>
#include <opencv2\datasets\track_alov.hpp>
#include <caffe\caffe.hpp>
#include "buildH5Dataset.h"

void trainNet();
void testNet(string modelPath);
void buildDB();