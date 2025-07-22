#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define sysRunTimes 1
#define numSets 8 // the num of sets(pairs)
#define idSet 1 // for mark the selected set if the numSets been set of 1
#define POP_SIZE 20
#define GENERATIONS 1000
#define OFFSPRING_COUNT 10
#define MUTATION_RATE 0.9

void imgShow(const string& name, const Mat& img);
double calculateF1Score(double precision, double recall);
double calculateMetrics(Mat metaImg_g[], Mat tarImg_g[]);
// void crossover(shared_ptr<TreeNode>& a, shared_ptr<TreeNode>& b);
void differenceProcess(Mat postImg, Mat preImg, Mat& resImg, int absoluteFlag);
void conPro_singleTime(Mat& metaImg, Mat& resImg, int arr_info_cps[]);
void imgSingleProcess(Mat& oriImg, Mat& resImg, int arr_val_dv[]);
void multiProcess(Mat imgArr[][2]);

enum FilterType {
	TERMINAL_INPUT,
	BLUR,
	MED_BLUR,
	DIFF_PROCESS, // with 2 input imgs
	THRESHOLD,
	BITWISE_NOT,
	MORPHOLOGY_EX,
	CON_PRO_SINGLE_TIME,
};

struct TreeNode {
	FilterType type;
	vector<shared_ptr<TreeNode>> children;
};

int main(void) {
	Mat imgArr[numSets][2]; // imgArr -> storing all images numSets(numSets pairs) * 2(ori, tar)
	char inputPathName_ori[256];
	char inputPathName_tar[256];

	if (numSets == 1) {
		sprintf_s(inputPathName_ori, "./imgs_0721_2025_v1/input/oriImg_0%d.png", idSet);
		sprintf_s(inputPathName_tar, "./imgs_0721_2025_v1/input/tarImg_0%d.png", idSet);
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
			sprintf_s(inputPathName_ori, "./imgs_0721_2025_v1/input/oriImg_0%d.png", i + 1);
			sprintf_s(inputPathName_tar, "./imgs_0721_2025_v1/input/tarImg_0%d.png", i + 1);
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

random_device rd;
mt19937 rng(rd());
uniform_real_distribution<> prob(0.0, 1.0);

shared_ptr<TreeNode> generateRandomTree(int depth = 0, int maxDepth = 4) {
	if (depth >= maxDepth || prob(rng) < 0.3) {
		return make_shared<TreeNode>(TreeNode{ TERMINAL_INPUT, {} });
	}

	FilterType t = static_cast<FilterType>(1 + (rng() % 3));
	auto node = make_shared<TreeNode>(TreeNode{ t, {} });

	int numChildren = (t == DIFF_PROCESS ? 2 : 1);
	for (int i = 0; i < numChildren; ++i) {
		node->children.push_back(generateRandomTree(depth + 1, maxDepth));
	}
	return node;
}

shared_ptr<TreeNode> cloneTree(const shared_ptr<TreeNode>& node) {
	if (!node) return nullptr;
	auto newNode = make_shared<TreeNode>(TreeNode{ node->type, {} });
	for (auto& child : node->children) {
		newNode->children.push_back(cloneTree(child));
	}
	return newNode;
}

Mat blurFunc(const Mat& img) {
	Mat out;
	blur(img, out, Size(19, 19));
	return out;
}

Mat medBlurFunc(const Mat& img) {
	Mat out;
	medianBlur(img, out, 19);
	return out;
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

Mat diffProcess(const Mat postImg, const Mat preImg) {
	int absoluteFlag = 0;
	Mat resImg = Mat::zeros(Size(postImg.cols, postImg.rows), CV_8UC1);
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
	return resImg;
}

Mat threshFunc(const Mat& img) {
	Mat out;
	threshold(img, out, 9, 255, THRESH_BINARY);
	return out;
}

Mat bitWiseFunc(const Mat& img) {
	Mat out;
	bitwise_not(img, out);
	return out;
}

Mat morphFunc(const Mat& img) {
	Mat out;
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(img, out, MORPH_CLOSE, kernel);
	return out;
}

Mat conPro_singleTime(const Mat& img) {
	Mat out;
	Mat maskImg = img.clone();
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	for (int idxET = 0; idxET < 3; idxET++) {
		erode(maskImg, maskImg, kernel);
	}
	vector<vector<Point>> contours;
	findContours(maskImg, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	Mat mask = Mat::zeros(maskImg.size(), CV_8UC1);
	for (const auto& contour : contours) {
		Rect bounding_box = boundingRect(contour);
		double aspect_ratio = static_cast<double>(bounding_box.width) / bounding_box.height;
		if ((aspect_ratio <= (1 - 1 * 0.1) || aspect_ratio > (1 + 1 * 0.1)) && cv::contourArea(contour) < 100 * 2) {
			drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(255), -1);
		}
	}
	for (int y = 0; y < out.rows; y++) {
		for (int x = 0; x < out.cols; x++) {
			if (mask.at<uchar>(y, x) == 255) {
				out.at<uchar>(y, x) = 255;
			}
		}
	}
	return out;
}

Mat executeTree(const shared_ptr<TreeNode>& node, Mat& input) {
	switch (node->type) {
	case TERMINAL_INPUT:
		return input.clone();
	case BLUR:
		return blurFunc(executeTree(node->children[0], input));
	case MED_BLUR:
		return (executeTree(node->children[0], input));
	case DIFF_PROCESS:
		return diffProcess(executeTree(node->children[0], input), input);
	case THRESHOLD:
		return threshFunc(executeTree(node->children[0], input));
	case BITWISE_NOT:
		return bitWiseFunc(executeTree(node->children[0], input));
	case MORPHOLOGY_EX:
		return morphFunc(executeTree(node->children[0], input));
	case CON_PRO_SINGLE_TIME:
		return conPro_singleTime(executeTree(node->children[0], input));
	default:
		return input;
	}
}

void collectNodes(const shared_ptr<TreeNode>& node, vector<shared_ptr<TreeNode>>& nodes) {
	if (!node) return;
	nodes.push_back(node);
	for (auto& child : node->children) {
		collectNodes(child, nodes);
	}
}

using NodeWithParent = pair<shared_ptr<TreeNode>, shared_ptr<TreeNode>>;

void collectNodesWithParents(const shared_ptr<TreeNode>& node,
	const shared_ptr<TreeNode>& parent,
	vector<NodeWithParent>& result) {
	if (!node) return;
	result.emplace_back(node, parent);
	for (auto& child : node->children) {
		collectNodesWithParents(child, node, result);
	}
}

void crossover(shared_ptr<TreeNode>& a, shared_ptr<TreeNode>& b) {
	vector<NodeWithParent> nodesA, nodesB;
	collectNodesWithParents(a, nullptr, nodesA);
	collectNodesWithParents(b, nullptr, nodesB);

	// 排除根节点（parent 为 nullptr）
	vector<NodeWithParent> validA, validB;
	for (const auto& np : nodesA) {
		if (np.second) validA.push_back(np);
	}
	for (const auto& np : nodesB) {
		if (np.second) validB.push_back(np);
	}

	if (validA.empty() || validB.empty()) return;

	int idxA = rng() % validA.size();
	int idxB = rng() % validB.size();

	shared_ptr<TreeNode> nodeA = validA[idxA].first;
	shared_ptr<TreeNode> parentA = validA[idxA].second;

	shared_ptr<TreeNode> nodeB = validB[idxB].first;
	shared_ptr<TreeNode> parentB = validB[idxB].second;

	// 找到 nodeA 在 parentA->children 中的位置
	auto& childrenA = parentA->children;
	auto itA = find(childrenA.begin(), childrenA.end(), nodeA);

	// 找到 nodeB 在 parentB->children 中的位置
	auto& childrenB = parentB->children;
	auto itB = find(childrenB.begin(), childrenB.end(), nodeB);

	// 交换子节点
	if (itA != childrenA.end() && itB != childrenB.end()) {
		swap(*itA, *itB);
	}
}

void mutate(shared_ptr<TreeNode>& node, int maxDepth = 4) {
	vector<shared_ptr<TreeNode>> nodes;
	collectNodes(node, nodes);
	if (nodes.empty()) return;
	int idx = rng() % nodes.size();
	nodes[idx] = generateRandomTree(0, maxDepth);
}

double calculateF1Score(double precision, double recall) {
	if (precision + recall == 0) return 0.0;
	return 2.0 * (precision * recall) / (precision + recall);
}

// for calculating the fValue of the ind and writting the organized info into group-arr and groupDvInfoArr
double calculateMetrics(Mat metaImg_g[], Mat tarImg_g[]) {
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
		//if (numGen == num_gen - 1) {
		//	indFvalInfo[numInd][idxSet] = f1_score[idxSet];
		//}
		sum_f1 += f1_score[idxSet];
	}
	return sum_f1;
}

//bool coin(double p) {
//	static uniform_real_distribution<> dist(0.0, 1.0);
//	static mt19937 rng(random_device{}());
//	return dist(rng) < p;
//}

void multiProcess(Mat imgArr[][2]) {
	Mat resImg[numSets];
	Mat tarImg[numSets];

	char imgName_pro[numSets][256];
	char imgName_final[numSets][256];

	// for recording the f_value of every generation (max, min, ave, dev)
	FILE* fl_fValue = nullptr;
	errno_t err = fopen_s(&fl_fValue, "./imgs_0721_2025_v1/output/f_value.txt", "w");
	if (err != 0 || fl_fValue == nullptr) {
		perror("Cannot open the file");
		return;
	}

	// for recording the decision varibles
	FILE* fl_params = nullptr;
	errno_t err1 = fopen_s(&fl_params, "./imgs_0721_2025_v1/output/params.txt", "w");
	if (err1 != 0 || fl_params == nullptr) {
		perror("Cannot open the file");
		return;
	}

	// for recording the f_value of elite-ind in last gen (setX1, setX2, ..., Max)
	FILE* fl_maxFval = nullptr;
	errno_t err2 = fopen_s(&fl_maxFval, "./imgs_0721_2025_v1/output/maxFvalInfo_final.txt", "w");
	if (err2 != 0 || fl_maxFval == nullptr) {
		perror("Cannot open the file");
		return;
	}

	for (int idxProTimes = 0; idxProTimes < sysRunTimes; idxProTimes++) {

		vector<shared_ptr<TreeNode>> population;
		for (int i = 0; i < POP_SIZE; ++i) {
			population.push_back(generateRandomTree());
		}

		shared_ptr<TreeNode> best;
		double bestFitness = -1;

		//srand((unsigned)time(NULL));
		//make();

		for (int numGen = 0; numGen < GENERATIONS; numGen++) {
			cout << "---------idxProTimes: " << idxProTimes + 1 << ", generation: " << numGen + 1 << "---------" << endl;
			int idx1 = rng() % POP_SIZE;
			int idx2 = rng() % POP_SIZE;
			while (idx2 == idx1) idx2 = rng() % POP_SIZE;

			auto parent1 = cloneTree(population[idx1]);
			auto parent2 = cloneTree(population[idx2]);

			vector<pair<double, shared_ptr<TreeNode>>> family;

			for (int i = 0; i < numSets; i++) {
				tarImg[i] = imgArr[i][1];
			}

			for (int i = 0; i < numSets; i++) {
				resImg[i] = executeTree(parent1, imgArr[i][0]);
			}
			double score1 = calculateMetrics(resImg, tarImg);

			for (int i = 0; i < numSets; i++) {
				resImg[i] = executeTree(parent2, imgArr[i][0]);
			}
			double score2 = calculateMetrics(resImg, tarImg);

			family.push_back({ score1, parent1 });
			family.push_back({ score2, parent2 });

			for (int k = 0; k < OFFSPRING_COUNT; ++k) {
				auto childA = cloneTree(parent1);
				auto childB = cloneTree(parent2);
				crossover(childA, childB);

				if (prob(rng) < MUTATION_RATE) mutate(childA);
				if (prob(rng) < MUTATION_RATE) mutate(childB);

				auto chosen = (prob(rng) < 0.5) ? childA : childB;

				for (int i = 0; i < numSets; i++) {
					resImg[i] = executeTree(chosen, imgArr[i][0]);
				}
				double fit = calculateMetrics(resImg, tarImg);
				family.push_back({ fit, chosen });
			}


			for (const auto& f : family) {

				if (f.first > bestFitness) {
					bestFitness = f.first;
					best = cloneTree(f.second);
					cout << "[Gen " << numGen << "] 最佳适应度: " << bestFitness << endl;
				}

			}
			sort(family.rbegin(), family.rend());
			auto elite = family[0];
			double total = 0;
			for (const auto& f : family) total += f.first;
			double r = prob(rng) * total, accum = 0;
			shared_ptr<TreeNode> rouletteSelected = family[1].second; // fallback
			for (const auto& f : family) {
				accum += f.first;
				if (accum >= r) {
					rouletteSelected = f.second;
					break;
				}
			}
			population[idx1] = cloneTree(elite.second);
			population[idx2] = cloneTree(rouletteSelected);
		}
		printf("--------END--------------\n");
	}

	fclose(fl_fValue);
	fclose(fl_params);
	fclose(fl_maxFval);
}