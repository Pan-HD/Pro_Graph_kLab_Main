#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>

using namespace cv;
using namespace std;

#define mutateRate 0.3
#define numOpType 3
#define numBitSingleOp 2
#define numGens 100
#define numPops 100
#define lenOpSeqConChroms 9

void imgShow(const string& name, const Mat& img);

int main() {
	srand(static_cast<unsigned int>(time(0)));

	return 0;
}

void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}