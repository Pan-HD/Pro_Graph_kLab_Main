#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// the name of decision-variables
// ["filterSwitchFlag", "fsize", "absoluteFlag", "threshVal"]
// ["dilateTimes", "aspectOffset", "contourPixNum"]
int info_val_dv[10] = { 1, 17, 0, 9, 3, 0, 2, 3, 0, 5 };

void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

void differenceProcess(Mat postImg, Mat preImg, Mat& resImg, int absoluteFlag) {
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
	for (int y = 0; y < resImg.rows; y++) {
		for (int x = 0; x < resImg.cols; x++) {
			if (mask.at<uchar>(y, x) == 255) {
				resImg.at<uchar>(y, x) = 255;
			}
		}
	}
}

void imgSingleProcess(Mat& oriImg, Mat& resImg, int arr_val_dv[]) {
	Mat blurImg;
	Mat diffImg;
	Mat biImg;
	Mat labelImg;
	if (arr_val_dv[0]) {
		medianBlur(oriImg, blurImg, arr_val_dv[1]);
	}
	else {
		blur(oriImg, blurImg, Size(arr_val_dv[1], arr_val_dv[1]));
	}
	differenceProcess(blurImg, oriImg, diffImg, arr_val_dv[2]);
	threshold(diffImg, biImg, arr_val_dv[3], 255, THRESH_BINARY);
	bitwise_not(biImg, biImg);
	Mat maskImg_01 = biImg.clone();
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	for (int idxET = 0; idxET < arr_val_dv[4]; idxET++) {
		erode(maskImg_01, maskImg_01, kernel);
	}
	contourProcess(maskImg_01, biImg, arr_val_dv[5], 100 * arr_val_dv[6]);

	Mat maskImg_02 = biImg.clone();
	for (int idxET = 0; idxET < arr_val_dv[7]; idxET++) {
		erode(maskImg_02, maskImg_02, kernel);
	}
	contourProcess(maskImg_02, biImg, arr_val_dv[8], 100 * arr_val_dv[9]);
	resImg = biImg.clone();
}

int main(void) {
	Mat oriImg = imread("./imgs_0422_2025_v2/input/oriImg_14.png", IMREAD_GRAYSCALE);
	Mat resImg;
	// imgShow("ori", oriImg);
	imgSingleProcess(oriImg, resImg, info_val_dv);
	imgShow("resImg", resImg);
	return 0;
}