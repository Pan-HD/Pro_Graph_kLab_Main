#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>

#define mutateRate 0.3
#define numMeType 3
#define numBitSingleMethod 2
#define numPopulations 100
#define lenMeSeqConChroms 9

using namespace cv;
using namespace std;

// the name of methods
// ["blur", "kernel", "absoluteFlag", "threshVal", "dilateTimes", "aspectOffset", "contourPixNum"]

enum OperationType {
	BLUR_TYPE,
	KERNEL_SIZE,
	THRESHOLD
};

struct GeneNode {
	OperationType opType;
	float param;
	GeneNode* left;
	GeneNode* right;
};

int meSeqConArr[numPopulations][lenMeSeqConChroms];

void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

Mat applyGeneTree(const Mat& image, GeneNode* root, int showFlag) {
	Mat result = image.clone();
	int filterType = 0; // 0: average, 1: Median
	int kernelSize = 3;
	int thresholdValue = 128;

	queue<GeneNode*> q;
	q.push(root);
	while (!q.empty()) {
		GeneNode* current = q.front();
		q.pop();
		if (current->opType == BLUR_TYPE) {
			filterType = static_cast<int>(current->param);
		}
		else if (current->opType == KERNEL_SIZE) {
			kernelSize = static_cast<int>(current->param);
			if (kernelSize % 2 == 0) kernelSize += 1;
		}
		else if (current->opType == THRESHOLD) {
			thresholdValue = static_cast<int>(current->param);
			thresholdValue = max(0, min(thresholdValue, 255));
		}
		if (current->left) q.push(current->left);
		if (current->right) q.push(current->right);
	}

	if (filterType == 1) {
		medianBlur(result, result, kernelSize);
	}
	else {
		blur(result, result, Size(kernelSize, kernelSize));
	}

	threshold(result, result, thresholdValue, 255, THRESH_BINARY);

	return result;
}

float calculatePrecision(int TP, int FP) { return (TP + FP == 0) ? 0 : static_cast<float>(TP) / (TP + FP); }
float calculateRecall(int TP, int FN) { return (TP + FN == 0) ? 0 : static_cast<float>(TP) / (TP + FN); }
float calculateFValue(float precision, float recall) { return (precision + recall == 0) ? 0 : 2 * precision * recall / (precision + recall); }

double calculateF1Score(double precision, double recall) {
	if (precision + recall == 0) return 0.0;
	return 2.0 * (precision * recall) / (precision + recall);
}

float evaluateGeneTree(const Mat& image, GeneNode* node, const Mat& targetImage) {
	Mat processedImage = applyGeneTree(image, node, 0); // processedImg, tarImg
	int tp = 0, fp = 0, fn = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (processedImage.at<uchar>(i, j) == 0 && targetImage.at<uchar>(i, j) == 0) {
				tp += 1;
			}
			if (processedImage.at<uchar>(i, j) == 0 && targetImage.at<uchar>(i, j) == 255) {
				fp += 1;
			}
			if (processedImage.at<uchar>(i, j) == 255 && targetImage.at<uchar>(i, j) == 0) {
				fn += 1;
			}
		}
	}
	if (tp == 0) tp += 1;
	if (fp == 0) fp += 1;
	if (fn == 0) fn += 1;
	double precision = (tp + fp > 0) ? tp / double(tp + fp) : 0.0;
	double recall = (tp + fn > 0) ? tp / double(tp + fn) : 0.0;
	float score = (float)calculateF1Score(precision, recall);
	return score;
}

GeneNode* crossover(GeneNode* parent1, GeneNode* parent2) {
	if (!parent1 || !parent2) return nullptr;
	GeneNode* child = new GeneNode(*parent1);
	child->param = (rand() % 2 == 0) ? parent1->param : parent2->param;
	child->left = crossover(parent1->left, parent2->left);
	child->right = crossover(parent1->right, parent2->right);
	return child;
}

void mutate(GeneNode* node) {
	if (!node) return;
	if (rand() <= RAND_MAX * mutateRate) {
		if (node->opType == BLUR_TYPE) node->param = rand() % 2;
		else if (node->opType == KERNEL_SIZE) node->param = (rand() % 5) * 2 + 3;
		else if (node->opType == THRESHOLD) node->param = rand() % 256;
	}
	mutate(node->left);
	mutate(node->right);
}

GeneNode* chromBasedTreeGenerating(int idxPop) { // running in init process
	GeneNode* rootNode = new GeneNode{ BLUR_TYPE, static_cast<float>(rand() % 2), nullptr, nullptr };

	int zeroFlag = 1;
	for (int idxChrom = 0; idxChrom < numMeType; idxChrom++) {
		if (meSeqConArr[idxPop][idxChrom] == 1) zeroFlag = 0;
	}
	if (zeroFlag) meSeqConArr[idxPop][0] = 1;

	for (int idxMeFlagChrom = 0; idxMeFlagChrom < numMeType; idxMeFlagChrom++) {
		if (meSeqConArr[idxPop][idxMeFlagChrom]) {
			int startIdx = numMeType + idxMeFlagChrom * numBitSingleMethod;
			int sumVal = 0;
			for (int idxMeValChrom = startIdx + numBitSingleMethod - 1; idxMeValChrom >= startIdx; idxMeValChrom--) {
				sumVal += meSeqConArr[idxPop][idxMeValChrom] * (int)pow(2.0, (double)(numBitSingleMethod - (idxMeValChrom - startIdx) - 1));
			}
			if (sumVal >= numMeType) {
				sumVal = rand() % numMeType;
			}
			// GeneNode* curNode = new GeneNode{};

		}
	}

	return rootNode;
}

vector<GeneNode*> initializePopulation(int populationSize) {
	for (int idxPop = 0; idxPop < numPopulations; idxPop++) {
		for (int idxChroms = 0; idxChroms < lenMeSeqConChroms; idxChroms++) {
			meSeqConArr[idxPop][idxChroms] = rand() > ((RAND_MAX + 1) / 2) ? 1 : 0;
		}
	}

	vector<GeneNode*> population;
	for (int i = 0; i < populationSize; i++) {

		GeneNode* filterNode = new GeneNode{ BLUR_TYPE, static_cast<float>(rand() % 2), nullptr, nullptr };
		GeneNode* kernelNode = new GeneNode{ KERNEL_SIZE, static_cast<float>((rand() % 5) * 2 + 3), nullptr, nullptr };
		GeneNode* thresholdNode = new GeneNode{ THRESHOLD, static_cast<float>(rand() % 256), nullptr, nullptr };

		filterNode->left = kernelNode;
		kernelNode->left = thresholdNode;
		population.push_back(filterNode);
	}
	return population;
}

GeneNode* geneticAlgorithm(const Mat& image, const Mat& targetImage, int generations, int populationSize) {
	vector<GeneNode*> population = initializePopulation(populationSize);
	vector<float> fitnessScores(populationSize);
	int bestIndex = 0;
	for (int gen = 0; gen < generations; gen++) {
		for (int i = 0; i < populationSize; i++) {
			fitnessScores[i] = evaluateGeneTree(image, population[i], targetImage);
		}
		bestIndex = max_element(fitnessScores.begin(), fitnessScores.end()) - fitnessScores.begin();
		GeneNode* bestIndividual = population[bestIndex];
		cout << "Generation " << gen + 1 << ": Best F-value = " << fitnessScores[bestIndex]
			<< ", Filter Type: " << (bestIndividual->param == 1 ? "Median" : "Average")
			<< ", Kernel Size: " << bestIndividual->left->param
			<< ", Threshold: " << bestIndividual->left->left->param << endl;

		if (gen != generations - 1) {
			GeneNode* child = crossover(bestIndividual, population[rand() % populationSize]);
			mutate(child);
			population[rand() % populationSize] = child;
		}
	}
	return population[bestIndex];
}
int main() {
	srand(static_cast<unsigned int>(time(0)));
	Mat image = imread("./imgs_0326_2025_v1/input/oriImg_01.png", IMREAD_GRAYSCALE);
	Mat targetImage = imread("./imgs_0326_2025_v1/input/tarImg_01.png", IMREAD_GRAYSCALE);

	GeneNode* bestSolution = geneticAlgorithm(image, targetImage, 100, 100);
	Mat processedImage = applyGeneTree(image, bestSolution, 1);
	imshow("Best Solution", processedImage);
	waitKey(0);
	return 0;
}