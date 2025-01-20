#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// #define RAND_MAX 32767 // the max value of random number
#define chLen 24 // the length of chromosome
#define num_ind 100 // the nums of individuals in the group
#define num_gen 100 // the nums of generation of the GA algorithm
#define cross 0.8 // the rate of cross
#define mut 0.05 // the rate of mutation
#define numSets 2 // the num of sets(pairs)

// for storing the index of the individual with max f-value
int curMaxFvalIdx = 0;

// the declaration of 6 decision variables
int offsetOuter = 0; // [0, 5] -> 3 bits
int offsetInner = 0; // [0, 5] -> 3 bits
int thresh = 0; // [0, 255] -> 8 bits
int sigmaVal = 0; // [0, 10] -> 4 bits
int sizeSobel = 1; // 1, 3, 5, 7 -> 2 bits
int sizeGaussian = 3; // 3, 5, 7, 9, 31 -> 4 bits

typedef struct {
    int ch[chLen]; // defining chromosomes by ch-array
    int fitness;
    double f_value; // the harmonic mean of precision and recall of the individual
}gene;

// the info of each generation, including the info of elite individual and the info of the group
typedef struct {
    double eliteFValue;
    double genMinFValue; // the min value of each gen, (max value is eliteFValue)
    double genAveFValue;
    double genDevFValue;
}genInfoType;

// for storing the fitness value of 6 decision variables
gene h[num_ind][6];

// for storing the info of each generation
genInfoType genInfo[num_gen];

// for storing the f-value of every individual in the group
double indFvalInfo[num_ind][numSets + 1];

void imgShow(const string& name, const Mat& img) {
    imshow(name, img);
    waitKey(0);
    destroyAllWindows();
}

Vec3f circleDetect(Mat img) {
    Mat blurred;
    GaussianBlur(img, blurred, Size(sizeGaussian, sizeGaussian), sigmaVal, sigmaVal);
    vector<Vec3f> circles;
    HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, blurred.rows / 8, 200, 100, 0, 0);
    return circles[0];
}

/*
    Ret: 0 -> padding of the box -> set to black
         1 -> outer of the circle -> set to white
         2 -> inner of the circle -> color swapping
*/
int comDistance(int y, int x, Vec3f circle) {
    int centerX = (int)circle[0];
    int centerY = (int)circle[1];
    int radius = (int)circle[2];
    int distance = (int)sqrt(pow((double)(x - centerX), 2) + pow((double)(y - centerY), 2));
    if (distance > radius + offsetOuter) {
        return 0;
    }
    else if (distance > radius - offsetInner && distance <= radius + offsetOuter) {
        return 1;
    }
    else {
        return 2;
    }
}

void make(gene* g)
{
    for (int j = 0; j < num_ind; j++) {
        for (int i = 0; i < chLen; i++) {
            if (rand() > (RAND_MAX + 1) / 2) g[j].ch[i] = 1;
            else g[j].ch[i] = 0;
        }
    }
}

void phenotype(gene* g)
{
    int i = 0, j = 0, k = 0;
    // initializing the fitness in h-array by assigning 0
    for (j = 0; j < num_ind; j++) {
        for (i = 0; i < 6; i++) {
            h[j][i].fitness = 0;
        }
    }

    for (j = 0; j < num_ind; j++) {
        // offsetOuter - 3 bits
        i = 3;
        for (k = 0; k < 3; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][0].fitness += (int)pow(2.0, (double)i);
            }
        }
        // offsetInner - 3 bits
        i = 3;
        for (k = 3; k < 6; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][1].fitness += (int)pow(2.0, (double)i);
            }
        }

        // thresh - 8 bit
        i = 8;
        for (k = 6; k < 14; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][2].fitness += (int)pow(2.0, (double)i);
            }
        }

        // sigmaVal - 4 bit
        i = 4;
        for (k = 14; k < 18; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][3].fitness += (int)pow(2.0, (double)i);
            }
        }
        // sizeSobel - 2 bits
        i = 2;
        for (k = 18; k < 20; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][4].fitness += (int)pow(2.0, (double)i);
            }
        }
        // sizeGaussian - 4 bit
        i = 4;
        for (k = 20; k < 24; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][5].fitness += (int)pow(2.0, (double)i);
            }
        }
    }

}

void import_para(int ko) {
    offsetOuter = h[ko][0].fitness;
    offsetInner = h[ko][1].fitness;
    thresh = h[ko][2].fitness;
    sigmaVal = h[ko][3].fitness;
    sizeSobel = 1 + 2 * h[ko][4].fitness;
    sizeGaussian = 1 + 2 * h[ko][5].fitness;
}

double calculateF1Score(double precision, double recall) {
    if (precision + recall == 0) return 0.0;
    return 2.0 * (precision * recall) / (precision + recall);
}

void calculateMetrics(Mat metaImg[], Mat tarImg[], Mat maskImg[], int numInd, int numGen) {
    Mat metaImg_g[numSets];
    Mat tarImg_g[numSets];
    Mat maskImg_g[numSets];
    for (int i = 0; i < numSets; i++) {
        cvtColor(metaImg[i], metaImg_g[i], cv::COLOR_BGR2GRAY);
        cvtColor(tarImg[i], tarImg_g[i], cv::COLOR_BGR2GRAY);
        cvtColor(maskImg[i], maskImg_g[i], cv::COLOR_BGR2GRAY);
    }

    double f1_score[numSets];

    for (int k = 0; k < numSets; k++) { // k: the index of the set being processed
        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < maskImg_g[k].rows; i++) {
            for (int j = 0; j < maskImg_g[k].cols; j++) {
                if (maskImg_g[k].at<uchar>(i, j) == 0) {
                    continue;
                }
                if (metaImg_g[k].at<uchar>(i, j) == 0 && tarImg_g[k].at<uchar>(i, j) == 0) {
                    tp += 1;
                }
                if (metaImg_g[k].at<uchar>(i, j) == 0 && tarImg_g[k].at<uchar>(i, j) == 255) {
                    fp += 1;
                }
                if (metaImg_g[k].at<uchar>(i, j) == 255 && tarImg_g[k].at<uchar>(i, j) == 0) {
                    fn += 1;
                }
            }
        }
        if (tp == 0) tp += 1;
        if (fp == 0) fp += 1;
        if (fn == 0) fn += 1;
        double precision = (tp + fp > 0) ? tp / double(tp + fp) : 0.0;
        double recall = (tp + fn > 0) ? tp / double(tp + fn) : 0.0;
        f1_score[k] = calculateF1Score(precision, recall);
    }
    double sum_f1 = 0.0;
    for (int i = 0; i < numSets; i++) {
        indFvalInfo[numInd][i] = f1_score[i];
        sum_f1 += f1_score[i];
    }

    // h[numInd][0].f_value = sum_f1 / numSets;
    h[numInd][0].f_value = sum_f1;
    indFvalInfo[numInd][numSets] = sum_f1;
}

void fitness(gene* g, gene* elite, int se) // for storing the info of elite individual
{
    int i = 0, j = 0;
    double minFValue = h[0][0].f_value;
    double maxFValue = h[0][0].f_value;
    double aveFValue = 0.0;
    double deviation = 0.0;
    double variance = 0.0;
    double sumFValue = 0.0;
    int maxFValueIndex = 0;

    for (i = 0; i < num_ind; i++) {
        sumFValue += h[i][0].f_value;
        if (h[i][0].f_value > maxFValue) {
            maxFValue = h[i][0].f_value;
            maxFValueIndex = i;
        }
        if (h[i][0].f_value < minFValue) {
            minFValue = h[i][0].f_value;
        }
    }
    curMaxFvalIdx = maxFValueIndex;
    elite[1].f_value = maxFValue;
    for (j = 0; j < chLen; j++) {
        elite[1].ch[j] = g[maxFValueIndex].ch[j];
    }
    aveFValue = sumFValue / num_ind;
    genInfo[se - 1].eliteFValue = h[maxFValueIndex][0].f_value;
    genInfo[se - 1].genMinFValue = minFValue;
    genInfo[se - 1].genAveFValue = aveFValue;
    for (i = 0; i < num_ind; i++)
    {
        double diff = h[i][0].f_value - aveFValue;
        variance += diff * diff;
    }
    deviation = sqrt(variance / num_ind);
    genInfo[se - 1].genDevFValue = deviation;
}

int roulette()
{
    int i = 0, r = 0;
    int num = 0;
    double sum = 0.0;
    double p[num_ind];

    for (i = 0; i < num_ind; i++) {
        sum += h[i][0].f_value;
    }
    for (i = 0; i < num_ind; i++) {
        p[i] = h[i][0].f_value / sum;
    }

    sum = 0;
    r = rand();
    for (i = 0; i < num_ind; i++) {
        sum += RAND_MAX * p[i];
        if (r <= sum) {
            num = i;
            break;
        }
    }
    if (num < 0)	num = roulette();
    return(num);
}

void crossover(gene* g) {
    gene g2[num_ind];
    int num = 0;
    int n1 = 0;
    int n2 = 0;
    int p = 0;
    int i, j;
    for (num = 0; num < num_ind; num += 2) {
        n1 = rand() % 10;
        n2 = rand() % 10;
        if (rand() <= RAND_MAX * cross) {
            n1 = roulette();
            n2 = roulette();
            p = (int)(rand() * ((chLen - 2) - 1 + 1.0) / (1.0 + RAND_MAX) + 1);
            for (i = 0; i < p; i++) {
                g2[num].ch[i] = g[n1].ch[i];
            }
            for (i = p; i < chLen; i++) {
                g2[num].ch[i] = g[n2].ch[i];
            }

            for (i = 0; i < p; i++) {
                g2[num + 1].ch[i] = g[n2].ch[i];
            }
            for (i = p; i < chLen; i++) {
                g2[num + 1].ch[i] = g[n1].ch[i];
            }
        }
        else {
            for (i = 0; i < chLen; i++) {
                n1 = roulette();
                n2 = roulette();
                g2[num].ch[i] = g[n1].ch[i];
                g2[num + 1].ch[i] = g[n2].ch[i];
            }
        }
    }
    for (j = 0; j < num_ind; j++) {
        for (i = 0; i < chLen; i++) {
            g[j].ch[i] = g2[j].ch[i];
        }
    }
}

void mutation(gene* g)
{
    int num = 0;
    int r = 0;
    int i = 0;
    int p = 0;
    for (num = 0; num < num_ind; num++) {
        if (rand() <= RAND_MAX * mut) {
            p = (int)(rand() * ((chLen - 1) + 1.0) / (1.0 + RAND_MAX));
            for (i = 0; i < chLen; i++) {
                if (i == p) {
                    if (g[num].ch[i] == 0) g[num].ch[i] = 1;
                    else				g[num].ch[i] = 0;
                }
            }
            p = 0;
        }
    }
}

void elite_back(gene* g, gene* elite) {
    int i = 0, j = 0;
    double ave = 0.0;
    double min1 = 1.0;
    int tmp = 0;
    for (i = 0; i < num_ind; i++) {
        if (h[i][0].f_value < min1) {
            min1 = h[i][0].f_value;
            tmp = i;
        }
    }
    for (j = 0; j < chLen; j++) {
        g[tmp].ch[j] = elite[1].ch[j];
    }
    h[tmp][0].f_value = elite[1].f_value;
}

void multiProcess(Mat imgArr[][3]) {
    Mat sobelX[numSets];
    Mat sobelY[numSets];
    Mat gradientMagnitude[numSets];
    Mat normalizedGradient[numSets];
    Vec3f circleInfo[numSets];
    Mat biImg[numSets];

    char imgName_pro[numSets][256];
    char imgName_final[numSets][256];

    // for recording the f_value of every generation (max, min, ave, dev)
    FILE* fl_fValue = nullptr;
    errno_t err = fopen_s(&fl_fValue, "./imgs_1225_v1/output/f_value.txt", "a");
    if (err != 0 || fl_fValue == nullptr) {
        perror("Cannot open the file");
        return;
    }

    // for recording the decision varibles
    FILE* fl_params = nullptr;
    errno_t err1 = fopen_s(&fl_params, "./imgs_1225_v1/output/params.txt", "a");
    if (err1 != 0 || fl_params == nullptr) {
        perror("Cannot open the file");
        return;
    }

    // for recording the f_value of elite-ind in last gen (setX1, setX2, ..., Max)
    FILE* fl_maxFval = nullptr;
    errno_t err2 = fopen_s(&fl_maxFval, "./imgs_1225_v1/output/maxFvalInfo_final.txt", "a");
    if (err2 != 0 || fl_maxFval == nullptr) {
        perror("Cannot open the file");
        return;
    }

    srand((unsigned)time(NULL));
    gene g[num_ind]; // For storing the group of individuals
    gene elite[10]; // For storing the elite individual of each generation
    elite[1].f_value = 0.0;
    make(g); // Initializing the info of chrom of 100 individuals

    for (int numGen = 1; numGen <= num_gen; numGen++) {
        cout << "-------generation: " << numGen << "---------" << endl;
        phenotype(g);
        for (int numInd = 0; numInd < num_ind; numInd++) {
            import_para(numInd);
            for (int i = 0; i < numSets; i++) {

                Sobel(imgArr[i][0], sobelX[i], CV_64F, 1, 0, sizeSobel);
                Sobel(imgArr[i][0], sobelY[i], CV_64F, 0, 1, sizeSobel);
                magnitude(sobelX[i], sobelY[i], gradientMagnitude[i]);
                normalize(gradientMagnitude[i], normalizedGradient[i], 0, 255, NORM_MINMAX, CV_8U);
                printf("ENTER-TESTING\n");
                imgShow("test1", normalizedGradient[i]);
                circleInfo[i] = circleDetect(normalizedGradient[i]);
                threshold(normalizedGradient[i], biImg[i], thresh, 255, THRESH_BINARY);

                for (int y = 0; y < biImg[i].rows; y++) {
                    for (int x = 0; x < biImg[i].cols; x++) {
                        if (comDistance(y, x, circleInfo[i]) == 0) {
                            biImg[i].at<uchar>(y, x) = 0;
                        }
                        else if (comDistance(y, x, circleInfo[i]) == 1) {
                            biImg[i].at<uchar>(y, x) = 255;
                        }
                        else {
                            biImg[i].at<uchar>(y, x) = biImg[i].at<uchar>(y, x) == 0 ? 255 : 0;
                        }
                    }
                }
                // cvtColor(biImg[i], biImg[i], COLOR_GRAY2BGR);
            }

            Mat tarImg[numSets];
            Mat maskImg[numSets];
            for (int i = 0; i < numSets; i++) {
                tarImg[i] = imgArr[i][1];
                maskImg[i] = imgArr[i][2];
            }

            calculateMetrics(biImg, tarImg, maskImg, numInd, numGen);
        }

        fitness(g, elite, numGen);
        crossover(g);
        mutation(g);
        elite_back(g, elite);
        printf("f_value: %.4f\n", elite[1].f_value);
        if (numGen % 10 == 0) {
            for (int i = 0; i < numSets; i++) {
                sprintf_s(imgName_pro[i], "./imgs_1225_v1/output/img_0%d/Gen-%d.png", i + 1, numGen);
                imwrite(imgName_pro[i], biImg[i]);
            }
        }
    }
    for (int i = 0; i < numSets; i++) {
        vector<Mat> images = { biImg[i], imgArr[i][1], imgArr[i][2] };
        Mat res;
        hconcat(images, res);
        sprintf_s(imgName_final[i], "./imgs_1225_v1/output/img_0%d/imgs_final.png", i + 1);
        imwrite(imgName_final[i], res);
    }

    for (int i = 0; i < num_gen; i++) {
        fprintf(fl_fValue, "%.4f %.4f %.4f %.4f\n", genInfo[i].eliteFValue, genInfo[i].genMinFValue, genInfo[i].genAveFValue, genInfo[i].genDevFValue);
    }
    fprintf(fl_params, "%d %d %d %d %d %d\n", offsetOuter, offsetInner, thresh, sigmaVal, sizeSobel, sizeGaussian);
    for (int i = 0; i <= numSets; i++) {
        fprintf(fl_maxFval, "%.4f ", indFvalInfo[curMaxFvalIdx][i]);
    }
    fprintf(fl_maxFval, "\n");

    fclose(fl_fValue);
    fclose(fl_params);
    fclose(fl_maxFval);
}

int main(void) {
    Mat imgArr[numSets][3]; // imgArr -> storing all images 2(2 pairs) * 3(ori, tar, mask)
    char inputPathName_ori[256];
    char inputPathName_tar[256];
    char inputPathName_mask[256];

    for (int i = 0; i < numSets; i++) {
        sprintf_s(inputPathName_ori, "./imgs_1225_v1/input/oriImg_0%d.png", i + 1);
        sprintf_s(inputPathName_tar, "./imgs_1225_v1/input/tarImg_0%d.png", i + 1);
        sprintf_s(inputPathName_mask, "./imgs_1225_v1/input/maskImg_general.png");

        for (int j = 0; j < 3; j++) {
            if (j == 0) {
                imgArr[i][j] = imread(inputPathName_ori, 0);
            }
            else if (j == 1) {
                imgArr[i][j] = imread(inputPathName_tar);
            }
            else {
                imgArr[i][j] = imread(inputPathName_mask);
            }
        }
    }
    multiProcess(imgArr);
    return 0;
}