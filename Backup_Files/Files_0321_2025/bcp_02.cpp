#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// the declaration of 8 decision variables (with the sum-fVal of 3.5645)
int fsize = 51;
int binary = 63;
int linear = 5;
int filterswitch_flag = 0;
int erodedilate_times = 5;
int erodedilate_sequence = 1;
int abusolute_flag = 1;
int pixellabelingmethod = 0;

void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

void sabun(Mat &input1, Mat &input2, Mat &resImg) {
	int i, j;
	Mat output;
	output = cv::Mat::zeros(cv::Size(input2.cols, input2.rows), CV_8UC3);//8UC3は3チャンネルに変えるタイプだ
	cvtColor(output, output, COLOR_RGB2GRAY);//グレースケール

	for (j = 0; j < input1.rows; j++)
	{
		for (i = 0; i < input1.cols; i++) {
			output.at<unsigned char>(j, i) = input2.at<unsigned char>(j, i) - input1.at<unsigned char>(j, i);
			if (input2.at<unsigned char>(j, i) - input1.at<unsigned char>(j, i) < 0)
			{
				if (abusolute_flag == 0) {
					output.at<unsigned char>(j, i) = abs(output.at<unsigned char>(j, i));
				}
				else {
					output.at<unsigned char>(j, i) = 0;
				}
			}
		}
	}
	resImg = output.clone();
}

Mat labeling(Mat img, int connectivity) {
	Mat img_con;
	Mat stats, centroids;

	int label_x, label_y;
	int label_longer;
	int label_areaall;
	double label_cal;

	int i, j, label_num;

	label_num = cv::connectedComponentsWithStats(img, img_con, stats, centroids, connectivity, CV_32S);
	// colors: for storing the color of background and every connected areas 
	vector<Vec3b>colors(label_num + 1);
	colors[0] = Vec3b(0, 0, 0);
	colors[1] = Vec3b(255, 255, 255);

	for (i = 2; i <= label_num; i++)
	{
		// colors[i] = Vec3b(0, 0, 0);
		label_areaall = stats.at<int>(i, CC_STAT_AREA);
		label_x = stats.at<int>(i, CC_STAT_WIDTH);
		label_y = stats.at<int>(i, CC_STAT_HEIGHT);

		label_longer = label_x > label_y ? label_x : label_y;
		label_cal = label_longer * label_longer;

		// (int)(label_cal / label_areaall) < linear -> detected area is not a fold-area
		//  In fold-detect task: discard -> colors[i] = Vec3b(255, 255, 255);
		if ((int)(label_cal / label_areaall) < linear + 127)
		{
			colors[i] = Vec3b(0, 0, 0); // in spot-detect task
		}
		else {
			colors[i] = Vec3b(255, 255, 255); // in spot-detect task
		}
	}

	// CV_8UC3丗3 channels
	Mat img_color = Mat::zeros(img_con.size(), CV_8UC3);
	for (j = 0; j < img_con.rows; j++) {
		for (i = 0; i < img_con.cols; i++)
		{
			int label = img_con.at<int>(j, i);
			CV_Assert(0 <= label && label <= label_num); // make sure the num of label is leagal
			img_color.at<Vec3b>(j, i) = colors[label];
		}
	}
	return img_color;
}

Mat Morphology(Mat img, int isDilFirst) {
	Mat dst;
	dst.create(img.size(), img.type());
	if (isDilFirst) {
		dilate(img, dst, Mat());
		erode(dst, dst, Mat());
	}
	else {
		erode(img, dst, Mat());
		dilate(dst, dst, Mat());
	}
	return dst;
}

int main(void) {
	Mat oriImg = imread("./imgs_1209_v1/input/oriImg_01.png", IMREAD_GRAYSCALE);
	// imgShow("ori", oriImg);

	Mat blurImg;
	Mat sabunImg;
	Mat biImg;
	Mat labelImg;

	// bluring
	if (filterswitch_flag) {
		medianBlur(oriImg, blurImg, fsize);
	}
	else {
		blur(oriImg, blurImg, Size(fsize, fsize));
	}
	imgShow("res", blurImg);

	sabun(oriImg, blurImg, sabunImg);
	// imgShow("res", sabunImg);

	threshold(blurImg, biImg, binary, 255, THRESH_BINARY);
	// imgShow("res", biImg);

	if (!pixellabelingmethod)
	{
		// biImg with 1-channel has been changed to 3-channel
		labelImg = labeling(biImg, 4);
	}
	else
	{
		labelImg = labeling(biImg, 8);
	}

	// Morphology
	if (!erodedilate_sequence)
	{
		if (erodedilate_times != 0) {
			for (int idx_edt = 0; idx_edt < erodedilate_times; idx_edt++)
			{
				labelImg = Morphology(labelImg, 1);
			}
		}
	}
	else
	{
		if (erodedilate_times != 0) {
			for (int idx_edt = 0; idx_edt < erodedilate_times; idx_edt++)
			{
				labelImg = Morphology(labelImg, 0);
			}
		}
	}

	// imgShow("res", labelImg);
	// imwrite("./imgs_Ver.1.1/resImg_04.png", labelImg);

	return 0;
}