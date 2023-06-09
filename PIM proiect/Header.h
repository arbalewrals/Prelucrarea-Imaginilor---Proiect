#pragma once
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

unsigned char* processImage(unsigned char* img, int w, int h);
unsigned char* thresholdImage(unsigned char* img, int w, int h);
void watershed_v1(unsigned char* img, int w, int h);
void watershed_v2(unsigned char* img, int w, int h);



