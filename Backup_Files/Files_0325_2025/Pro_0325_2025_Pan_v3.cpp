#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int filterSwitchFlag = 1;
int fsize = 17; // spot: 27, tippinngu: 17
int absoluteFlag = 0;
int threshVal = 10; // spot: 63, tippinngu: 7
int dilateTimes = 3;
int aspectOffset = 0;
int contourPixNums = 2; // spot: 1, tippinngu: 7 (4 bits)

void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

void differenceProcess(Mat postImg, Mat preImg, Mat& resImg) {
	resImg = Mat::zeros(Size(postImg.cols, postImg.rows), CV_8UC1);
	for (int j = 0; j < postImg.rows; j++)
	{
		for (int i = 0; i < postImg.cols; i++) {
			int diffVal = postImg.at<uchar>(j, i) - preImg.at<uchar>(j, i);
			if (diffVal < 0) {
				if (absoluteFlag != 0) {
					diffVal = abs(diffVal);
				}
				else {
					diffVal = 0;
				}
			}
			resImg.at<uchar>(j, i) = diffVal;
		}
	}
}

void contourProcess(Mat& metaImg, Mat& resImg, int aspectRatio, int pixNums) {
	vector<vector<Point>> contours;
	findContours(metaImg, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	Mat mask = Mat::zeros(metaImg.size(), CV_8UC1);
	for (const auto& contour : contours) {
		Rect bounding_box = boundingRect(contour);
		double aspect_ratio = static_cast<double>(bounding_box.width) / bounding_box.height;
		if ((aspect_ratio <= (1 - aspectRatio * 0.1) || aspect_ratio > (1 + aspectRatio * 0.1)) && cv::contourArea(contour) < pixNums) {
			drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(255), -1);
		}
	}
	// imgShow("mask", mask);

	for (int y = 0; y < resImg.rows; y++) {
		for (int x = 0; x < resImg.cols; x++) {
			if (mask.at<uchar>(y, x) == 255) {
				resImg.at<uchar>(y, x) = 255;
			}
		}
	}
}

int main(void) {
	Mat oriImg = imread("./imgs_0327_2025_v1/input/oriImg_08.png", IMREAD_GRAYSCALE);

	// imgShow("ori", oriImg);

	Mat blurImg;
	Mat diffImg;
	Mat biImg;
	Mat labelImg;

	// bluring
	if (filterSwitchFlag) {
		medianBlur(oriImg, blurImg, fsize); // marked
	}
	else {
		blur(oriImg, blurImg, Size(fsize, fsize));
	}
	// imgShow("res", blurImg);

	differenceProcess(blurImg, oriImg, diffImg);
	// imgShow("res", diffImg);

	threshold(diffImg, biImg, threshVal, 255, THRESH_BINARY);
	// imgShow("res", biImg);

	bitwise_not(biImg, biImg);
	// imgShow("res", biImg);

	// Å™ easiest way of spot-detect task

	Mat maskImg = biImg.clone();
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	for (int idxET = 0; idxET < dilateTimes; idxET++) {
		erode(maskImg, maskImg, kernel);
	}
	// imgShow("res", maskImg);

	contourProcess(maskImg, biImg, aspectOffset, 100 * contourPixNums);
	imgShow("res", biImg);
	// imwrite("./imgs_0327_2025_v1/input/tarImg_08.png", biImg);

	return 0;
}