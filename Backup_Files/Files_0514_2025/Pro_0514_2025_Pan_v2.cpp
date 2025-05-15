#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define numSets 8 // the num of sets(pairs)
#define idSet 1 // for mark the selected set if the numSets been set of 1
#define numDV 17 // the nums of decision-variables
#define chLen 54 // the length of chromosome
#define num_ind 100 // the nums of individuals in the group
#define num_gen 100 // the nums of generation of the GA algorithm
#define cross 0.8 // the rate of cross
#define mut 0.05 // the rate of mutation

void imgShow(const string& name, const Mat& img);
void make();
void phenotype();
void import_para(int idxInd);
double calculateF1Score(double precision, double recall);
void calculateMetrics(Mat metaImg_g[], Mat tarImg_g[], int numInd, int numGen, int flagEB);
void fitness(int numGen);
int roulette();
void crossover();
void mutation();
void elite_back(Mat imgArr[][3], Mat resImg[], Mat tarImg[], int numGen);
void differenceProcess(Mat postImg, Mat preImg, Mat& resImg, int absoluteFlag);
void conPro_singleTime(Mat& metaImg, Mat& resImg, int arr_info_cps[]);
void processOnGenLoop(Mat imgArr[][3], Mat resImg[], Mat tarImg[], int numGen, int flagEB);
void imgSingleProcess(Mat& oriImg, Mat& resImg, int arr_val_dv[]);
void multiProcess(Mat imgArr[][3]);

typedef struct {
	int chrom[chLen];
	double f_value;
}indInfoType;

// the info of each generation, including the info of elite individual and the info of the group
typedef struct {
	double eliteFValue;
	double genMinFValue; // the min value of each gen, (max value is eliteFValue)
	double genAveFValue;
	double genDevFValue;
	int eliteChrom[chLen];
	int arr_val_dv[numDV]; // for storing the value of DV of elite-ind in each generation
}genInfoType;

indInfoType group[num_ind];
genInfoType genInfo[num_gen];

// for storing the index of the individual with max f-value
int curMaxFvalIdx = 0;

// the name of decision-variables
// ["filterSwitchFlag", "fsize", "absoluteFlag", "threshVal", "ConProTimes"]
// [, "dilateTimes_01", "aspectOffset_01", "contourPixNum_01"]
// [, "dilateTimes_02", "aspectOffset_02", "contourPixNum_02"]
// [, "dilateTimes_03", "aspectOffset_03", "contourPixNum_03"]
// [, "dilateTimes_04", "aspectOffset_04", "contourPixNum_04"]
int info_len_dv[numDV] = { 1, 6, 1, 8, 2, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4 };
int groupDvMapArr[num_ind][numDV];
int info_val_dv[numDV];
int groupDvInfoArr[num_ind][numDV];

// for storing the f-value of every individual in the group
double indFvalInfo[num_ind][numSets + 1];

int main(void) {
	Mat imgArr[numSets][3]; // imgArr -> storing all images 2(2 pairs) * 3(ori, tar, mask)
	char inputPathName_ori[256];
	char inputPathName_tar[256];

	if (numSets == 1) {
		sprintf_s(inputPathName_ori, "./imgs_0514_2025_v2/input/oriImg_0%d.png", idSet);
		sprintf_s(inputPathName_tar, "./imgs_0514_2025_v2/input/tarImg_0%d.png", idSet);
		for (int j = 0; j < 2; j++) {
			if (j == 0) {
				imgArr[0][j] = imread(inputPathName_ori, 0);
			}
			else {
				imgArr[0][j] = imread(inputPathName_tar, 0);
			}
		}
	}
	else {
		for (int i = 0; i < numSets; i++) {
			sprintf_s(inputPathName_ori, "./imgs_0514_2025_v2/input/oriImg_0%d.png", i + 1);
			sprintf_s(inputPathName_tar, "./imgs_0514_2025_v2/input/tarImg_0%d.png", i + 1);
			for (int j = 0; j < 2; j++) {
				if (j == 0) {
					imgArr[i][j] = imread(inputPathName_ori, 0);
				}
				else {
					imgArr[i][j] = imread(inputPathName_tar, 0);
				}
			}
		}
	}

	multiProcess(imgArr);
	return 0;
}

void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

void make()
{
	for (int idxInd = 0; idxInd < num_ind; idxInd++) {
		for (int idxCh = 0; idxCh < chLen; idxCh++) {
			group[idxInd].chrom[idxCh] = rand() > ((RAND_MAX + 1) / 2) ? 1 : 0;
		}
	}
}

/*
  function: Convert the decision variable information in the chromosome corresponding to each individual
			from binary to decimal and store it in the h array
*/
void phenotype()
{
	for (int idxInd = 0; idxInd < num_ind; idxInd++) { // the loop of inds
		int curIdx_chrom = 0;
		for (int idx_dv = 0; idx_dv < numDV; idx_dv++) {
			int len_curDv = info_len_dv[idx_dv];
			int sum_val = 0;
			for (int idx = curIdx_chrom + len_curDv - 1; idx >= curIdx_chrom; idx--) {
				sum_val += group[idxInd].chrom[idx] * (int)pow(2.0, (double)(len_curDv - (idx - curIdx_chrom) - 1));
			}
			groupDvMapArr[idxInd][idx_dv] = sum_val;
			curIdx_chrom += len_curDv;
		}
	}
}

void import_para(int idxInd) {
	for (int idxDV = 0; idxDV < numDV; idxDV++) {
		info_val_dv[idxDV] = groupDvMapArr[idxInd][idxDV];
	}

	info_val_dv[0] = 1;

	if (info_val_dv[1] % 2 == 0) {
		info_val_dv[1] += 1;
	}

	info_val_dv[2] = 0;

	// info_val_dv[3] = 5;

	info_val_dv[4] += 2;
	if (info_val_dv[4] > 4) {
		info_val_dv[4] = 4;
	}

	info_val_dv[5] = 3;

}

double calculateF1Score(double precision, double recall) {
	if (precision + recall == 0) return 0.0;
	return 2.0 * (precision * recall) / (precision + recall);
}

// for calculating the fValue of the ind and writting the organized info into group-arr and groupDvInfoArr
void calculateMetrics(Mat metaImg_g[], Mat tarImg_g[], int numInd, int numGen, int flagEB) {
	double f1_score[numSets];
	for (int idxSet = 0; idxSet < numSets; idxSet++) {
		int tp = 0, fp = 0, fn = 0;
		for (int i = 0; i < metaImg_g[idxSet].rows; i++) {
			for (int j = 0; j < metaImg_g[idxSet].cols; j++) {
				if (metaImg_g[idxSet].at<uchar>(i, j) == 0 && tarImg_g[idxSet].at<uchar>(i, j) == 0) {
					tp += 1;
				}
				if (metaImg_g[idxSet].at<uchar>(i, j) == 0 && tarImg_g[idxSet].at<uchar>(i, j) == 255) {
					fp += 1;
				}
				if (metaImg_g[idxSet].at<uchar>(i, j) == 255 && tarImg_g[idxSet].at<uchar>(i, j) == 0) {
					fn += 1;
				}
			}
		}
		if (tp == 0) tp += 1;
		if (fp == 0) fp += 1;
		if (fn == 0) fn += 1;
		double precision = (tp + fp > 0) ? tp / double(tp + fp) : 0.0;
		double recall = (tp + fn > 0) ? tp / double(tp + fn) : 0.0;
		f1_score[idxSet] = calculateF1Score(precision, recall);
	}
	double sum_f1 = 0.0;
	for (int idxSet = 0; idxSet < numSets; idxSet++) {
		// if cur gen is the last gen, then writting the detail-info into the indFvalInfo-arr
		if (numGen == num_gen - 1) {
			indFvalInfo[numInd][idxSet] = f1_score[idxSet];
		}
		sum_f1 += f1_score[idxSet];
	}
	group[numInd].f_value = sum_f1;

	if (numGen == num_gen - 1) {
		// if cur gen is the last gen, then writting the detail-info into the indFvalInfo-arr
		indFvalInfo[numInd][numSets] = sum_f1;
	}

	// flagEB: the flag of elite-back (func been called in the elite-back)
	if (flagEB) {
		for (int idxDV = 0; idxDV < numDV; idxDV++) {
			groupDvInfoArr[numInd][idxDV] = info_val_dv[idxDV];
		}
	}
}

// for organizing the infomation of one generation
void fitness(int numGen)
{
	int i = 0, j = 0;
	double minFValue = group[0].f_value;
	double maxFValue = group[0].f_value;
	double aveFValue = 0.0;
	double deviation = 0.0;
	double variance = 0.0;
	double sumFValue = 0.0;

	// for getting maxFValue, curMaxFvalIdx, minFValue, sumFValue in cur generation
	for (int idxInd = 0; idxInd < num_ind; idxInd++) {
		sumFValue += group[idxInd].f_value;
		if (group[idxInd].f_value > maxFValue) {
			maxFValue = group[idxInd].f_value;
			curMaxFvalIdx = idxInd;
		}
		if (group[idxInd].f_value < minFValue) {
			minFValue = group[idxInd].f_value;
		}
	}

	// for writting the info of cur generation to the genInfo-arr
	genInfo[numGen].eliteFValue = maxFValue;
	for (int idxCh = 0; idxCh < chLen; idxCh++) {
		genInfo[numGen].eliteChrom[idxCh] = group[curMaxFvalIdx].chrom[idxCh];
	}
	aveFValue = sumFValue / num_ind;
	genInfo[numGen].genMinFValue = minFValue;
	genInfo[numGen].genAveFValue = aveFValue;
	for (int idxInd = 0; idxInd < num_ind; idxInd++)
	{
		double diff = group[idxInd].f_value - aveFValue;
		variance += diff * diff;
	}
	deviation = sqrt(variance / num_ind);
	genInfo[numGen].genDevFValue = deviation;
	for (int idxDV = 0; idxDV < numDV; idxDV++) {
		genInfo[numGen].arr_val_dv[idxDV] = groupDvInfoArr[curMaxFvalIdx][idxDV];
	}
}

int roulette() {
	double sumGroupFValue = 0.0;
	double sumRandMaxProportion = 0.0;
	int random = rand();
	int selInd = 0;
	double indFValProportion[num_ind];

	for (int idxInd = 0; idxInd < num_ind; idxInd++) {
		sumGroupFValue += group[idxInd].f_value;
	}
	for (int idxInd = 0; idxInd < num_ind; idxInd++) {
		indFValProportion[idxInd] = group[idxInd].f_value / sumGroupFValue;
	}
	for (int idxInd = 0; idxInd < num_ind; idxInd++) {
		sumRandMaxProportion += RAND_MAX * indFValProportion[idxInd];
		if (random <= (int)sumRandMaxProportion) {
			selInd = idxInd;
			break;
		}
	}
	return selInd;
}

void crossover() {
	int groupChromArrTmp[num_ind][chLen];
	int selInd_01, selInd_02;
	int idxCross;
	for (int idxInd = 0; idxInd < num_ind; idxInd += 2) {
		if (rand() <= RAND_MAX * cross) {
			selInd_01 = roulette();
			selInd_02 = roulette();
			idxCross = (int)(rand() * ((chLen - 2) - 1 + 1.0) / (1.0 + RAND_MAX) + 1);
			for (int idxCh = 0; idxCh < chLen; idxCh++) {
				if (idxCh < idxCross) {
					groupChromArrTmp[idxInd][idxCh] = group[selInd_01].chrom[idxCh];
					groupChromArrTmp[idxInd + 1][idxCh] = group[selInd_02].chrom[idxCh];
				}
				else {
					groupChromArrTmp[idxInd][idxCh] = group[selInd_02].chrom[idxCh];
					groupChromArrTmp[idxInd + 1][idxCh] = group[selInd_01].chrom[idxCh];
				}
			}
		}
		else {
			selInd_01 = roulette();
			selInd_02 = roulette();
			for (int idxCh = 0; idxCh < chLen; idxCh++) {
				groupChromArrTmp[idxInd][idxCh] = group[selInd_01].chrom[idxCh];
				groupChromArrTmp[idxInd + 1][idxCh] = group[selInd_02].chrom[idxCh];
			}
		}
	}
	for (int idxInd = 0; idxInd < num_ind; idxInd++) {
		for (int idxCh = 0; idxCh < chLen; idxCh++) {
			group[idxInd].chrom[idxCh] = groupChromArrTmp[idxInd][idxCh];
		}
	}
}

void mutation() {
	int idxMut;
	for (int idxInd = 0; idxInd < num_ind; idxInd++) {
		if (rand() <= RAND_MAX * mut) {
			idxMut = (int)(rand() * ((chLen - 1) + 1.0) / (1.0 + RAND_MAX));
			group[idxInd].chrom[idxMut] = group[idxInd].chrom[idxMut] == 0 ? 1 : 0;
		}
	}
}

void elite_back(Mat imgArr[][3], Mat resImg[], Mat tarImg[], int numGen) {
	processOnGenLoop(imgArr, resImg, tarImg, numGen, 1);
	int idxMinFVal = 0;
	double minFVal = genInfo[numGen].eliteFValue;
	for (int idxInd = 0; idxInd < num_ind; idxInd++) {
		if (group[idxInd].f_value < minFVal) {
			idxMinFVal = idxInd;
			minFVal = group[idxInd].f_value;
		}
	}
	for (int idxCh = 0; idxCh < chLen; idxCh++) {
		group[idxMinFVal].chrom[idxCh] = genInfo[numGen].eliteChrom[idxCh];
	}
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

void conPro_singleTime(Mat& metaImg, Mat& resImg, int arr_info_cps[]) {
	Mat maskImg = metaImg.clone();
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	for (int idxET = 0; idxET < arr_info_cps[0]; idxET++) {
		erode(maskImg, maskImg, kernel);
	}
	vector<vector<Point>> contours;
	findContours(maskImg, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	Mat mask = Mat::zeros(maskImg.size(), CV_8UC1);
	for (const auto& contour : contours) {
		Rect bounding_box = boundingRect(contour);
		double aspect_ratio = static_cast<double>(bounding_box.width) / bounding_box.height;
		if ((aspect_ratio <= (1 - arr_info_cps[1] * 0.1) || aspect_ratio > (1 + arr_info_cps[1] * 0.1)) && cv::contourArea(contour) < 100 * arr_info_cps[2]) {
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

void processOnGenLoop(Mat imgArr[][3], Mat resImg[], Mat tarImg[], int numGen, int flagEB) {
	phenotype();
	for (int numInd = 0; numInd < num_ind; numInd++) {
		import_para(numInd);
		for (int i = 0; i < numSets; i++) {
			imgSingleProcess(imgArr[i][0], resImg[i], info_val_dv);
		}
		for (int i = 0; i < numSets; i++) {
			tarImg[i] = imgArr[i][1];
		}
		calculateMetrics(resImg, tarImg, numInd, numGen, flagEB);
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

	for (int idxCPT = 0; idxCPT < arr_val_dv[4]; idxCPT++) {
		int arr_info_cps[3];
		for (int idxCps = 0; idxCps < 3; idxCps++) {
			arr_info_cps[idxCps] = arr_val_dv[5 + idxCPT * 3 + idxCps];
		}
		conPro_singleTime(biImg, biImg, arr_info_cps);
	}
	resImg = biImg.clone();
}

void multiProcess(Mat imgArr[][3]) {
	Mat resImg[numSets];
	Mat tarImg[numSets];

	char imgName_pro[numSets][256];
	char imgName_final[numSets][256];

	// for recording the f_value of every generation (max, min, ave, dev)
	FILE* fl_fValue = nullptr;
	errno_t err = fopen_s(&fl_fValue, "./imgs_0514_2025_v2/output/f_value.txt", "w");
	if (err != 0 || fl_fValue == nullptr) {
		perror("Cannot open the file");
		return;
	}

	// for recording the decision varibles
	FILE* fl_params = nullptr;
	errno_t err1 = fopen_s(&fl_params, "./imgs_0514_2025_v2/output/params.txt", "w");
	if (err1 != 0 || fl_params == nullptr) {
		perror("Cannot open the file");
		return;
	}

	// for recording the f_value of elite-ind in last gen (setX1, setX2, ..., Max)
	FILE* fl_maxFval = nullptr;
	errno_t err2 = fopen_s(&fl_maxFval, "./imgs_0514_2025_v2/output/maxFvalInfo_final.txt", "w");
	if (err2 != 0 || fl_maxFval == nullptr) {
		perror("Cannot open the file");
		return;
	}

	srand((unsigned)time(NULL));
	make();

	for (int numGen = 0; numGen < num_gen; numGen++) {
		cout << "-------generation: " << numGen + 1 << "---------" << endl;
		processOnGenLoop(imgArr, resImg, tarImg, numGen, 0);

		fitness(numGen);
		printf("f_value: %.4f\n", genInfo[numGen].eliteFValue);

		// preparing for next generation
		if (numGen < num_gen - 1) {
			crossover();
			mutation();
			elite_back(imgArr, resImg, tarImg, numGen);
		}
	}

	Mat resImg_01;
	Mat resImg_02;
	Mat res;
	for (int idxGen = 0; idxGen < num_gen; idxGen++) {
		if ((idxGen + 1) % 10 == 0) {
			if (numSets == 1) {
				imgSingleProcess(imgArr[0][0], resImg_01, genInfo[idxGen].arr_val_dv);
				sprintf_s(imgName_pro[0], "./imgs_0514_2025_v2/output/img_0%d/Gen-%d.png", idSet, idxGen + 1);
				imwrite(imgName_pro[0], resImg_01);
				if (idxGen == num_gen - 1) {
					vector<Mat> images = { resImg_01, imgArr[0][1] };
					hconcat(images, res);
					sprintf_s(imgName_final[0], "./imgs_0514_2025_v2/output/img_0%d/imgs_final.png", idSet);
					imwrite(imgName_final[0], res);
				}
			}
			else {
				for (int idxSet = 0; idxSet < numSets; idxSet++) {
					imgSingleProcess(imgArr[idxSet][0], resImg_02, genInfo[idxGen].arr_val_dv);
					sprintf_s(imgName_pro[idxSet], "./imgs_0514_2025_v2/output/img_0%d/Gen-%d.png", idxSet + 1, idxGen + 1);
					imwrite(imgName_pro[idxSet], resImg_02);
					if (idxGen == num_gen - 1) {
						vector<Mat> images = { resImg_02, imgArr[idxSet][1] };
						hconcat(images, res);
						sprintf_s(imgName_final[idxSet], "./imgs_0514_2025_v2/output/img_0%d/imgs_final.png", idxSet + 1);
						imwrite(imgName_final[idxSet], res);
					}
				}
			}
		}
	}
	for (int i = 0; i < num_gen; i++) {
		fprintf(fl_fValue, "%.4f %.4f %.4f %.4f\n", genInfo[i].eliteFValue, genInfo[i].genMinFValue, genInfo[i].genAveFValue, genInfo[i].genDevFValue);
	}
	for (int idxDV = 0; idxDV < numDV; idxDV++) {
		fprintf(fl_params, "%d ", genInfo[num_gen - 1].arr_val_dv[idxDV]);
	}
	fprintf(fl_params, "\n");

	for (int i = 0; i <= numSets; i++) {
		fprintf(fl_maxFval, "%.4f ", indFvalInfo[curMaxFvalIdx][i]);
	}
	fprintf(fl_maxFval, "\n");

	fclose(fl_fValue);
	fclose(fl_params);
	fclose(fl_maxFval);
}