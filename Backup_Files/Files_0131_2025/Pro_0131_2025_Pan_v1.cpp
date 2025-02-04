#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int thresh = 17; // [0, 255] -> dv01 - 8bit
int sizeGaussian = 17; // (n * 2 + 1) 3, 5, 7, 9, ..., 31 -> dv02 - 4bit
int offset = 9; // [0, 15] -> dv03 - 4bit
int erodeFlag = 0; // -> dv04 - 1bit
int erodeTimes = 2; // -> dv05 - 2bit 
int aspectRatio = 1; // [0, 7] -> dv06 - 3bit
int contPixNums = 7; // [0, 7] -> dv07 - 3bit
int erodeTimes_s2 = 1; // -> dv08 - 2bit
int contPixNums_s2 = 3; // -> dv09 - 3bit

void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

void gradCal(Mat& srcImg, Mat& dstImg) {
	// Sobel Detecting
	Mat sobelX, sobelY, gradientMagnitude;
	Sobel(srcImg, sobelX, CV_64F, 1, 0, 1);
	Sobel(srcImg, sobelY, CV_64F, 0, 1, 1);
	magnitude(sobelX, sobelY, gradientMagnitude);
	normalize(gradientMagnitude, dstImg, 0, 255, NORM_MINMAX, CV_8U);
}

vector<Vec3f> circleDetect(Mat img) {
	Mat blurred;
	GaussianBlur(img, blurred, Size(sizeGaussian, sizeGaussian), 0, 0);

	// imgShow("GauBlur in circleDetect", blurred);

	vector<Vec3f> circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, blurred.rows / 8, 200, 100, 0, 0);
	return circles;
}

int comDistance(int y, int x, Vec3f circle) {
	int centerX = (int)circle[0];
	int centerY = (int)circle[1];
	int radius = (int)circle[2];
	int distance = (int)sqrt(pow((double)(x - centerX), 2) + pow((double)(y - centerY), 2));
	if (distance > radius) {
		return 0;
	}
	else if (distance > radius - offset && distance <= radius) {
		return 1;
	}
	else {
		return 2;
	}
}

void contourProcess(Mat& metaImg, Mat& resImg, int aspectRatio, int pixNums, vector<Vec3f> circles, int aspectFlag) {
	vector<vector<Point>> contours;
	findContours(metaImg, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	Mat mask = Mat::zeros(metaImg.size(), CV_8UC1);
	for (const auto& contour : contours) {
		Rect bounding_box = boundingRect(contour);
		double aspect_ratio = static_cast<double>(bounding_box.width) / bounding_box.height;
		if (aspectFlag) {
			if ((aspect_ratio < (1 - aspectRatio * 0.1) || aspect_ratio >(1 + aspectRatio * 0.1)) && cv::contourArea(contour) < pixNums) {
				drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(255), -1);
			}
		}
		else { // just need to focus on the proportions(areas), since the probability of the aspect_ratio equals "1.0000"
			// when the second time to find contours (without long-slender ribon inside)
			if (cv::contourArea(contour) < pixNums) {
				drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(255), -1);
			}
		}

	}
	// imgShow("mask", mask);

	if (circles.size() != 0) {
		for (int y = 0; y < resImg.rows; y++) {
			for (int x = 0; x < resImg.cols; x++) {
				if (comDistance(y, x, circles[0]) == 2) {
					if (mask.at<uchar>(y, x) == 255) {
						resImg.at<uchar>(y, x) = 255;
					}
				}
			}
		}
	}
	// imgShow("res", resImg);
}

int main(void) {
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

	Mat oriImg = imread("./imgs_1225_v1/input/oriImg_01.png", IMREAD_GRAYSCALE);
	// imgShow("res", oriImg);

	Mat edges_s1;
	gradCal(oriImg, edges_s1); // stat-01 -> Sobel
	// imgShow("res", edges_s1);

	Mat biImg;
	threshold(edges_s1, biImg, thresh, 255, THRESH_BINARY); // stat-02 -> threshold
	// imgShow("res", biImg);

	vector<Vec3f> circles = circleDetect(biImg); // GaussianSize
	if (circles.size() != 0) { // stat-03
		for (int y = 0; y < biImg.rows; y++) {
			for (int x = 0; x < biImg.cols; x++) {
				if (comDistance(y, x, circles[0]) == 0) { // offset
					biImg.at<uchar>(y, x) = 0;
				}
				else if (comDistance(y, x, circles[0]) == 1) {
					biImg.at<uchar>(y, x) = 255;
				}
				else {
					biImg.at<uchar>(y, x) = biImg.at<uchar>(y, x) == 0 ? 255 : 0;
				}
			}
		}
	}
	// Note-biImg(in stat-01): Cannot divide the large-spot and the noise, each part have small pieces, 
	//                         long-slender ribon and each part with same proportion.
	// imgShow("test", biImg); 

	Mat blurImg_mask_s1;
	medianBlur(biImg, blurImg_mask_s1, 3);
	if (erodeFlag) {
		// For img-02
		Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
		for (int i = 0; i < erodeTimes; i++) {
			erode(blurImg_mask_s1, blurImg_mask_s1, kernel);
		}
	}
	// imgShow("test", blurImg_mask_s1);

	// Note-blurImg_mask_s1: the small noise been removed, and the connection between long-slender ribon and large-spot
	//                       been cut done 
	contourProcess(blurImg_mask_s1, biImg, aspectRatio, 100 * contPixNums, circles, 1);
	// imgShow("res", biImg);

	// Note-biImg(in stat-02): in the resImg, only the large-spot and several small noise remains, so it could be 
	//                         processed by findContours directly.
	Mat metaImg = biImg.clone();
	for (int i = 0; i < erodeTimes_s2; i++) {
		erode(metaImg, metaImg, kernel);
	}
	// imgShow("meta", metaImg);

	contourProcess(metaImg, biImg, 0, 100 * contPixNums_s2, circles, 0);
	imgShow("res", biImg);

	return 0;
}