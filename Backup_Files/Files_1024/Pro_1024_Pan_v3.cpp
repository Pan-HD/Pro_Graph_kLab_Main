#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define RAND_MAX 32767 // the max value of random number
#define chLen 17 // the length of chromosome
#define num_ind 100 // the nums of individuals in the group
#define num_gen 200 // the nums of generation of the GA algorithm
#define cross 0.95 // the rate of cross
#define mut 0.25 // the rate of mutation

// the declaration of 7 decision variables
int fsize = 0;
int binary = 0;
int abusolute_flag = 0;
int erodedilate_sequence = 0;
int filterswitch_flag;
int erodedilate_times;
int pixellabelingmethod = 0;

typedef struct {
    int ch[chLen]; // defining chromosomes by ch-array
    // float fitness; // the fitness of 7 decision variables
    int fitness;
    float f_value; // the harmonic mean of precision and recall of the individual
}gene;

// for storing the fitness value of 7 decision variables
gene h[num_ind][7];

void imgShow(const string& name, const Mat& img) {
    imshow(name, img);
    waitKey(0);
    destroyAllWindows();
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
        for (i = 0; i < 7; i++) {
            h[j][i].fitness = 0;
        }
    }

    for (j = 0; j < num_ind; j++) {
        // fsize - 2 bits
        i = 2;
        for (k = 0; k < 2; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][0].fitness += (int)pow(2.0, (double)i);
            }
        }
        // binary - 8 bits
        i = 8;
        for (k = 2; k < 10; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][1].fitness += (int)pow(2.0, (double)i);
            }
        }
        // filterswich_flag - 1 bit
        i = 1;
        for (k = 10; k < 11; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][2].fitness += (int)pow(2.0, (double)i);
            }
        }
        // erodedilate_times - 3 bits
        i = 3;
        for (k = 11; k < 14; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][3].fitness += (int)pow(2.0, (double)i);
            }
        }
        // erodedilate_sequence - 1 bit
        i = 1;
        for (k = 14; k < 15; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][4].fitness += (int)pow(2.0, (double)i);
            }
        }
        // abusolute_flag - 1 bit
        i = 1;
        for (k = 15; k < 16; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][5].fitness += (int)pow(2.0, (double)i);
            }
        }
        // pixellabelingmethod - 1bit
        i = 1;
        for (k = 16; k < 17; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][6].fitness += (int)pow(2.0, (double)i);
            }
        }
    }
}

void import_para(int ko) {
    fsize = 0;
    binary = 0;
    filterswitch_flag = 0;
    abusolute_flag = 0;
    fsize = 1 + 2 * h[ko][0].fitness;
    binary = h[ko][1].fitness;
    filterswitch_flag = h[ko][2].fitness;
    erodedilate_times = h[ko][3].fitness;
    erodedilate_sequence = h[ko][4].fitness;
    abusolute_flag = h[ko][5].fitness;
    pixellabelingmethod = h[ko][6].fitness;
}

Mat sabun(Mat input1, Mat input2) {
    int i, j;
    int res; // the diff value between input2 and input1
    Mat output;
    output = Mat::zeros(Size(input2.cols, input2.rows), CV_8UC3);
    cvtColor(output, output, COLOR_RGB2GRAY);
    for (j = 0; j < input1.rows; j++)
    {
        for (i = 0; i < input1.cols; i++) {
            res = input2.at<unsigned char>(j, i) - input1.at<unsigned char>(j, i);
            if (abusolute_flag) {
                output.at<unsigned char>(j, i) = res >= 0 ? res : (-1) * res;
            }
            else {
                output.at<unsigned char>(j, i) = res >= 0 ? res : 0;
            }
        }
    }
    return output;
}

Mat labeling(Mat img, int connectivity) {
    Mat img_con;
    Mat stats, centroids;
    int i, j, label_num;

    //int label_x, label_y;
    //int label_longer;
    //double label_cal;
    //int label_areaall;

    label_num = cv::connectedComponentsWithStats(img, img_con, stats, centroids, 8, CV_32S);
    vector<Vec3b>colors(label_num + 1); // ���x�����Ƃɂǂ̐F���g�������i�[���邽�߂̂��̂ł�
    colors[0] = Vec3b(0, 0, 0); // �w�i�̐F�����ɐݒ�
    colors[1] = Vec3b(255, 255, 255);
    for (i = 2; i < label_num; i++) // ���x�����Ƃ̏���
    {
        // colors[i] = Vec3b(255, 255, 255);
        colors[i] = Vec3b(0, 0, 0);
    }
    // CV_8UC3�F3�`�����l���E�E�E
    Mat img_color = Mat::zeros(img_con.size(), CV_8UC3);
    for (j = 0; j < img_con.rows; j++) {
        for (i = 0; i < img_con.cols; i++)
        {
            int label = img_con.at<int>(j, i);
            CV_Assert(0 <= label && label <= label_num);
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
        erode(dst, dst, Mat());
        dilate(img, dst, Mat());
    }

    return dst;
}

double calculateF1Score(double precision, double recall) {
    if (precision + recall == 0) return 0.0;
    return 2.0 * (precision * recall) / (precision + recall);
}

void calculateMetrics(Mat metaImg, Mat tarImg, Mat maskImg, int numInd, int numGen) {
    Mat metaImg_g;
    Mat tarImg_g;
    Mat maskImg_g;
    cvtColor(metaImg, metaImg_g, cv::COLOR_BGR2GRAY);
    cvtColor(tarImg, tarImg_g, cv::COLOR_BGR2GRAY);
    cvtColor(maskImg, maskImg_g, cv::COLOR_BGR2GRAY);

    int tp = 0, fp = 0, fn = 0;

    for (int i = 0; i < maskImg_g.rows; i++) {
        for (int j = 0; j < maskImg_g.cols; j++) {
            if (maskImg_g.at<uchar>(i, j) == 0) {
                continue;
            }
            if (metaImg_g.at<uchar>(i, j) == 0 && tarImg_g.at<uchar>(i, j) == 0) {
                tp += 1;
            }
            if (metaImg_g.at<uchar>(i, j) == 0 && tarImg_g.at<uchar>(i, j) == 255) {
                fp += 1;
            }
            if (metaImg_g.at<uchar>(i, j) == 255 && tarImg_g.at<uchar>(i, j) == 0) {
                fn += 1;
            }
        }
    }
    // cout << "----gen: " << numGen << ", ind: " << numInd << ", tp fp fn: " << tp << " " << fp << " " << fn << "-------" << endl;
    if (tp == 0) tp += 1;
    if (fp == 0) fp += 1;
    if (fn == 0) fn += 1;
    double precision = (tp + fp > 0) ? tp / double(tp + fp) : 0.0;
    double recall = (tp + fn > 0) ? tp / double(tp + fn) : 0.0;
    double f1_score = calculateF1Score(precision, recall);
    h[numInd][0].f_value = f1_score;
}

void fitness(gene* g, gene* elite, int se) // �K���x�̌v�Z(�G���[�g�ۑ�)
{
    int i = 0, j = 0;
    //double ave = 0.0;
    //double deviation = 0.0;
    //double variance = 0.0;

    // int sum1 = 0; // ������

    // �G���[�g�ۑ�
    for (i = 0; i < num_ind; i++) {
        // sum1 += h[i][0].f_value;
        if (h[i][0].f_value > elite[1].f_value) {
            // �G���[�g���ꊷ��
            elite[1].f_value = h[i][0].f_value;
            for (j = 0; j < chLen; j++) {
                elite[1].ch[j] = g[i].ch[j];
            }
        }
    }

    //float min_value = 1.1;
    //elite[2].f_value = 1.1;
    //for (i = 0; i < num_ind; i++) {
    //    if (h[i][0].f_value < min_value) {
    //        min_value = h[i][0].f_value;
    //    }
    //}

    //elite[3].f_value = 0.0; // ������
    //for (i = 0; i < num_ind; i++) {
    //    elite[3].f_value += h[i][0].f_value;
    //}

    //ave = (double)(elite[3].f_value) / (double)num_ind;

    //for (i = 0; i < num_ind; i++)
    //{
    //    double diff = h[i][0].f_value - ave;
    //    variance += diff * diff;
    //}

    //deviation = sqrt(variance / num_ind);


    // fprintf(fp4, "%d	%.2f	%.2f	���ρF%.2f\n", se, elite[1].valu, elite[2].valu, ave);
    // fprintf(fp4, "%d	�ő�F%.2f	�ŏ��F%.2f�@���ϒl�F%.2f  ���ϕ΍��l�F%.2f\n", se, elite[1].valu, min_value, ave, deviation);
    // printf("%d	%.2f	%.2f	���ρF%.2f  ���ϕ΍��l�F%.2f\n", se, elite[1].valu, min_value, ave, deviation);

}

int roulette() // ���[���b�g�I��
{
    int i = 0, r = 0;
    int num = 0;
    float sum = 0.0;
    //float* p;
    //p = (float*)malloc(sizeof(int) * num_ind);

    float p[num_ind];

    //sum = 0;
    for (i = 0; i < num_ind; i++) {
        sum += h[i][0].f_value; // ���ׂĂ̍��v
    }
    for (i = 0; i < num_ind; i++) {
        p[i] = h[i][0].f_value / sum; // �̓K���x / �Q�̓K���x
    }

    sum = 0;
    r = rand();
    for (i = 0; i < num_ind; i++) {
        sum += RAND_MAX * p[i]; // 1
        if (r <= sum) {
            num = i;
            break;
        }
    }
    if (num < 0)	num = roulette(); // �G���[�̂��߂̏���
    free(p);
    return(num);
}

void crossover(gene* g) {
    gene g2[num_ind]; // �V�����̌Q���ꎞ�I�Ɋi�[���邽�߂̔z��
    int num = 0; // �������̌̔ԍ��B2���������܂��B
    // n1, n2: �������邽�߂ɑI�΂ꂽ2�̐e�̃C���f�b�N�X
    int n1 = 0;
    int n2 = 0;
    int p = 0; // �������s����`�q�̈ʒu
    int i, j;

    for (num = 0; num < num_ind; num += 2) {
        n1 = rand() % 10;
        n2 = rand() % 10;
        if (rand() <= RAND_MAX * cross) { // �����m���𖞂����ꍇ
            n1 = roulette();
            n2 = roulette();
            // �����͈͎̔w������F(int)( rand() * (�ő�l - �ŏ��l + 1.0) / (1.0 + RAND_MAX) )
            p = (int)(rand() * ((chLen - 2) - 1 + 1.0) / (1.0 + RAND_MAX) + 1);
            // g[n1], g[n2]: 2�̐e�@g2[num], g2[num+1]: 2�̎q
            // �����d�g�݁F�e1�̍ŏ���p�r�b�g���p�����A�c��̃r�b�g��e2����p�����܂�
            // �qA
            for (i = 0; i < p; i++) {
                g2[num].ch[i] = g[n1].ch[i];
            }
            for (i = p; i < chLen; i++) {
                g2[num].ch[i] = g[n2].ch[i];
            }

            // �qB
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

    // �V�����̌Qg2���A���̌̌Qg�ɏ㏑�����Ď�����̌̌Q�Ƃ��܂�
    for (j = 0; j < num_ind; j++) {
        for (i = 0; i < chLen; i++) {
            g[j].ch[i] = g2[j].ch[i]; // g[]���X�V
        }
    }
}

void mutation(gene* g) // �ˑR�ψ�
{
    int num = 0;
    int r = 0;
    int i = 0;
    int p = 0;
    for (num = 0; num < num_ind; num++) {
        if (rand() <= RAND_MAX * mut) { // �ˑR�ψيm���𖞂����ꍇ�C1�̈�`�q��I��
            p = (int)(rand() * ((chLen - 1) + 1.0) / (1.0 + RAND_MAX));
            for (i = 0; i < chLen; i++) { // 1��0���t�]
                if (i == p) {
                    if (g[num].ch[i] == 0) g[num].ch[i] = 1;
                    else				g[num].ch[i] = 0;
                }
            }
            p = 0;
        }
    }
}

void elite_back(gene* g, gene* elite) { // �G���[�g�̂�valu�ŏ��̂�����
    int i = 0, j = 0;
    float ave = 0.0;
    float min1 = 1.0;
    int tmp = 0; // �J�E���^�[�̏�����
    for (i = 0; i < num_ind; i++) { // �ŏ��l�T��
        if (h[i][0].f_value < min1) {
            min1 = h[i][0].f_value;
            tmp = i;
        }
    }
    for (j = 0; j < chLen; j++) {
        g[tmp].ch[j] = elite[1].ch[j]; // �ŏ��l�ƃG���[�g������
    }
    h[tmp][0].f_value = elite[1].f_value; // �G���[�g�̕]���l�ƌ���
    // ave = sum1 / kotai; // ���v�l�̌v�Z
}

void print_h() {
    cout << "fsize, binary, filterFlag, eroDilTimes, eroDilSeq, abuFlag, labelMethod" << endl;
    for (int i = 0; i < num_ind; i++) {
        for (int j = 0; j < 7; j++) {
            printf("%d  ", h[i][j].fitness);
        }
        printf("\n");
    }
}

int main(void) {
    // preparation of images
    Mat oriImg = imread("./imgs_Pro_GA/oriImg.png");
    Mat oriImg_g = imread("./imgs_Pro_GA/oriImg.png", 0);
    Mat maskImg = imread("./imgs_Pro_GA/maskImg.png");
    Mat tarImg = imread("./imgs_Pro_GA/tarImg.png");
    Mat blurImg;
    Mat diffImg;
    Mat biImg;
    Mat labelImg;

    // Initializing �E�E�E
    srand((unsigned)time(NULL));
    gene g[num_ind]; // For storing the group of individuals 
    gene elite[10]; // For storing the elite individual of each generation
    elite[1].f_value = 0.0;
    make(g);

    //char imgFileName[] = "./imgs_Pro_GA/Output/"

    // Main Part
    for (int numGen = 1; numGen <= num_gen; numGen++) {
        //for (int numGen = 1; numGen <= 1; numGen++) {
        cout << "-------generation: " << numGen << "---------" << endl;
        phenotype(g);
        for (int numInd = 0; numInd < num_ind; numInd++) {
            //for (int numInd = 0; numInd < 1; numInd++) {
                // global value of decision variables been written
            import_para(numInd);
            // bluring
            if (filterswitch_flag)
            {
                medianBlur(oriImg_g, blurImg, fsize);
            }
            else
            {
                blur(oriImg_g, blurImg, Size(fsize, fsize));
            }

            //diffImg = sabun(oriImg_g, blurImg);
            threshold(blurImg, biImg, binary, 255, THRESH_BINARY);

            if (!pixellabelingmethod)
            {
                // biImg with 1-channel has been changed to 3-channel
                labelImg = labeling(biImg, 4);
            }
            else
            {
                labelImg = labeling(biImg, 8);
            }

            //vector<Mat> images_01 = { labelImg, tarImg};
            //Mat res01;
            //hconcat(images_01, res01);
            //imgShow("res_01", res01);

            // bitwise_not(labelImg, labelImg);

            // Morphology
            if (!erodedilate_sequence)
            {
                if (erodedilate_times != 0) {
                    for (int i = 0; i < erodedilate_times; i++)
                    {
                        labelImg = Morphology(labelImg, 1);
                    }
                }
            }
            else
            {
                if (erodedilate_times != 0) {
                    for (int i = 0; i < erodedilate_times; i++)
                    {
                        labelImg = Morphology(labelImg, 0);
                    }
                }
            }
            calculateMetrics(labelImg, tarImg, maskImg, numInd, numGen);
        }
        fitness(g, elite, numGen);
        crossover(g);
        mutation(g);
        elite_back(g, elite);
        cout << "binary: " << binary << "  f_value: " << elite[1].f_value << endl;
    }
    vector<Mat> images = { labelImg, tarImg, maskImg };
    Mat res;
    hconcat(images, res);
    imgShow("res_p1", res);
}