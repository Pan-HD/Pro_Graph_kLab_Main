/*
    To do list:
        01. ���܂̃}�X�N�摜�̍쐬
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define RAND_MAX 32767 // �����̍ő�l
#define chLen 21 // �̂̐��F�̂̒���
#define num_ind 100 // the nums of individuals in the group
#define num_gen 100 // the nums of generation of the GA algorithm
#define cross 0.8 // ������
#define mut 0.05 // �ˑR�ψٗ�

typedef struct {
    int ch[chLen]; // defining chromosomes by ch-array
    float fitness; // the fitness of 7 decision variables
    float f_value; // the harmonic mean of precision and recall of the individual
}gene;

//typedef struct {
//    double precision;
//    double recall;
//    double f_value;
//}eliteValue;

// for mask-generate and target-generate
int circlePoint_x = 0;
int circlePoint_y = 0;
int circleRadius = 0;
// for storing the fitness value of 7 decision variables
gene h[num_ind][7];
// the input 3 images in the beginning
Mat oriImg;
Mat maskImg;
Mat tarImg;
// the declaration of 7 decision variables
int fsize = 0;
int binary = 0;
int abusolute_flag = 0;
int erodedilate_sequence = 0;
int filterswitch_flag;
int erodedilate_times;
int pixellabelingmethod = 0;
// 
FILE* fp;

void imgShow(const string& name, const Mat& img) {
    imshow(name, img);
    waitKey(0);
    destroyAllWindows();
}

/*
    Func: Independent function, for generating mask picture.
          Only for pictures of pills(����)
*/
void maskGenerate() {
    Mat img_ori = imread("./imgs_Pro_GA/oriImg.png");
    Mat img_ori_g = imread("./imgs_Pro_GA/oriImg.png", 0);
    GaussianBlur(img_ori_g, img_ori_g, cv::Size(9, 9), 2, 2);
    vector<Vec3f> circles;

    HoughCircles(img_ori_g, circles, HOUGH_GRADIENT, 1,
        img_ori_g.rows / 16,
        100, 30, 0, 0
    );

    Mat maskImg = cv::Mat::zeros(img_ori.rows, img_ori.cols, CV_8UC1);
    circlePoint_x = circles[0][0];
    circlePoint_y = circles[0][1];
    circleRadius = circles[0][2];

    for (int y = 0; y < img_ori.rows; y++) {
        for (int x = 0; x < img_ori.cols; x++) {
            int dx = x - circlePoint_x;
            int dy = y - circlePoint_y;
            if (dx * dx + dy * dy <= circleRadius * circleRadius) {
                maskImg.at<uchar>(y, x) = 255;
            }
        }
    }
    imwrite("./imgs_Pro_GA/maskImg.png", maskImg);
}

/*
    Func: Independent function, for generating target picture.
          Only for pictures of pills(����)
*/
void targetGenerate() {
    Mat img_ori = imread("./imgs_Pro_GA/oriImg.png");
    Mat img_ori_g = imread("./imgs_Pro_GA/oriImg.png", 0);
    Mat img_mask2tar = imread("./imgs_Pro_GA/maskImg.png");

    GaussianBlur(img_ori_g, img_ori_g, cv::Size(9, 9), 2, 2);
    Mat img_ori_bi;
    threshold(img_ori_g, img_ori_bi, 205, 255, THRESH_BINARY);

    vector<Vec3f> circles;
    HoughCircles(img_ori_bi, circles, HOUGH_GRADIENT, 1,
        1,
        100, 20, 0, 0
    );

    //cout << "inner radius: " << circles[0][2] << endl;

    int squ_out = circleRadius * circleRadius;
    int squ_in = circles[0][2] * circles[0][2] - 300;

    for (int y = 0; y < img_ori.rows; y++) {
        for (int x = 0; x < img_ori.cols; x++) {
            int dx = x - circlePoint_x;
            int dy = y - circlePoint_y;
            if (dx * dx + dy * dy >= squ_in && dx * dx + dy * dy <= squ_out) {
                img_ori_bi.at<uchar>(y, x) = 255;
            }
        }
    }
    imwrite("./imgs_Pro_GA/tarImg.png", img_ori_bi);
}

/*
    Func: Initializing the elements of the chromosomes of every individual
*/
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
            h[j][i].fitness = 0.0;
        }
    }

    for (j = 0; j < num_ind; j++) {
        // fsize - 6 bits
        i = 6;
        for (k = 0; k < 6; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][0].fitness += (float)pow((double)g[j].ch[k] * 2, (double)i);
            }
        }
        // binary - 8 bits
        i = 8;
        for (k = 6; k < 14; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][1].fitness += (float)pow((double)g[j].ch[k] * 2, (double)i);
            }
        }
        // filterswich_flag - 1 bit
        i = 1;
        for (k = 14; k < 15; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][2].fitness += (float)pow((double)g[j].ch[k] * 2, (double)i);
            }
        }
        // erodedilate_times - 3 bits
        i = 3;
        for (k = 15; k < 18; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][3].fitness += (float)pow((double)g[j].ch[k] * 2, (double)i);
            }
        }
        // erodedilate_sequence - 1 bit
        i = 1;
        for (k = 18; k < 19; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][4].fitness += (float)pow((double)g[j].ch[k] * 2, (double)i);
            }
        }
        // abusolute_flag - 1 bit
        i = 1;
        for (k = 19; k < 20; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][5].fitness += (float)pow((double)g[j].ch[k] * 2, (double)i);

            }
        }
        // pixellabelingmethod - 1bit
        i = 1;
        for (k = 20; k < 21; k++) {
            i--;
            if (g[j].ch[k] == 1) {
                h[j][6].fitness += (float)pow((double)g[j].ch[k] * 2, (double)i);

            }
        }
    }
}

void import_para(int ko) {
    fsize = 0;
    binary = 0;
    filterswitch_flag = 0;
    abusolute_flag = 0;
    fsize = (int)(3 + 2 * h[ko][0].fitness);
    binary = (int)(1 * h[ko][1].fitness);
    filterswitch_flag = (int)(h[ko][2].fitness);
    erodedilate_times = (int)(h[ko][3].fitness);
    erodedilate_sequence = (int)(h[ko][4].fitness);
    abusolute_flag = (int)(h[ko][5].fitness);
    pixellabelingmethod = (int)(h[ko][6].fitness);
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

Mat labeling_new4(Mat img_sabun) {
    Mat img_con;
    Mat stats, centroids;
    int i, j, label_num;

    int label_x, label_y;
    int label_longer;
    double label_cal;
    int label_areaall;
    label_num = connectedComponentsWithStats(img_sabun, img_con, stats, centroids, 4, 4);
    vector<Vec3b>colors(label_num + 1); // ���x�����Ƃɂǂ̐F���g�������i�[���邽�߂̂��̂ł�
    colors[0] = Vec3b(0, 0, 0); // �w�i�̐F�����ɐݒ�
    for (i = 1; i < label_num; i++) // ���x�����Ƃ̏���
    {
        colors[i] = Vec3b(255, 255, 255);

        //label_areaall = stats.at<int>(i, CC_STAT_AREA);
        //label_x = stats.at<int>(i, CC_STAT_WIDTH);
        //label_y = stats.at<int>(i, CC_STAT_HEIGHT);
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

Mat labeling_new8(Mat img_sabun) { // �ߖT8��f�ŉ�f�̉�����x�����O���@
    Mat img_con;
    Mat stats, centroids; // �A�ʋ��̑���
    int i, j, label_num; // �A�ʋ��̐�

    int label_x, label_y;
    int label_longer;
    double label_cal;
    int label_areaall; // ���x����t�������̉�f��
    label_num = connectedComponentsWithStats(img_sabun, img_con, stats, centroids, 8, 4);
    vector<Vec3b>colors(label_num + 1);
    colors[0] = Vec3b(0, 0, 0);

    for (i = 1; i < label_num; i++)
    {
        colors[i] = Vec3b(255, 255, 255);
        //label_areaall = stats.at<int>(i, CC_STAT_AREA);
        //label_x = stats.at<int>(i, CC_STAT_WIDTH);
        //label_y = stats.at<int>(i, CC_STAT_HEIGHT);
    }
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

Mat dilate_erode(Mat src1) { // �I�[�v�j���O�N���[�W���O����
    Mat dst;
    dst.create(src1.size(), src1.type());
    // �N���[�W���O������C�I�[�v�j���O�����D�X�ɖc������
    dilate(src1, dst, Mat()); // �c������
    erode(dst, dst, Mat()); // ���k����

    return dst;
}

Mat erode_dilate(Mat src1) { // �I�[�v�j���O�N���[�W���O����
    Mat dst;
    dst.create(src1.size(), src1.type());
    // �N���[�W���O������C�I�[�v�j���O�����D�X�ɖc������
    erode(dst, dst, Mat()); // ���k����
    dilate(src1, dst, Mat()); // �c������

    return dst;
}

double calculateF1Score(double precision, double recall) {
    if (precision + recall == 0) return 0.0;
    return 2.0 * (precision * recall) / (precision + recall);
}

void calculateMetrics(Mat image1, Mat image2, Mat mask, int numInd) {
    // tp: True Positive, ���������o�̃s�N�Z����
    // fp: False Positive, �댟�o�̃s�N�Z����
    // fn: False Negative, �����o�̃s�N�Z����
    int tp = 0, fp = 0, fn = 0;
    int mask_pixel_count = 0;  // �}�X�N���Œl��255�̃s�N�Z�������J�E���g����ϐ�

    for (int y = 0; y < image1.rows; y++) {
        for (int x = 0; x < image1.cols; x++) {
            // �}�X�N�����݂���ꍇ�A�}�X�N�̔��������̂݌v�Z
            if (mask.at<uchar>(y, x) != 255) {
                continue;
            }

            if (mask.at<uchar>(y, x) == 255) {
                mask_pixel_count++;
            }

            bool isImage1White = (image1.at<uchar>(y, x) == 255);
            bool isImage2White = (image2.at<uchar>(y, x) == 255);

            if (!isImage1White && !isImage2White) {
                tp++; // �^�z��
            }
            else if (!isImage1White && isImage2White) {
                fp++; // �U�z��
            }
            else if (isImage1White && !isImage2White) {
                fn++; // �U�A��
            }
        }
    }

    // precision: ���o�����̒��A���m�ȕ����̊���
    double precision = (tp + fp > 0) ? tp / double(tp + fp) : 0.0;
    // recall: ���o���ׂ������̒��A���ۂ̌��o�����̊���
    double recall = (tp + fn > 0) ? tp / double(tp + fn) : 0.0;
    double f1_score = calculateF1Score(precision, recall);
    h[numInd][0].f_value = f1_score;

    // ���ʂ�\��
    //if (mask) {
    //    std::cout << "�}�X�N�����̌���:" << std::endl;
    //    std::cout << "�}�X�N����255�̃s�N�Z����: " << mask_pixel_count << std::endl;
    //}
    //else {
    //    std::cout << "�S�̉摜�̌���:" << std::endl;
    //}

    //std::cout << std::fixed << std::setprecision(20); // �Œ�\���Ə����_�ȉ�20���ɐݒ�
    //std::cout << "�K���� (Precision): " << precision << std::endl;
    //std::cout << "�Č��� (Recall): " << recall << std::endl;
    //std::cout << "F�l (F1 Score): " << f1_score << std::endl;
    //std::cout << "���� (True Positives): " << tp << std::endl;
    //std::cout << "������f�� (False Negatives): " << fn << std::endl;
}

void fitness(gene* g, gene* elite, int se) // �K���x�̌v�Z(�G���[�g�ۑ�)
{
    int i = 0, j = 0;
    double ave = 0.0;
    double deviation = 0.0;
    double variance = 0.0;

    int sum1 = 0; // ������

    // �G���[�g�ۑ�
    for (i = 0; i < num_ind; i++) {
        sum1 += h[i][0].f_value;
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
    float* p;

    p = (float*)malloc(sizeof(int) * num_ind);

    sum = 0;
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
    gene g2[1000] = { 0 }; // �V�����̌Q���ꎞ�I�Ɋi�[���邽�߂̔z��
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

int main(void) {
    oriImg = imread("./imgs_Pro_GA/oriImg.png");
    maskImg = imread("./imgs_Pro_GA/maskImg.png");
    tarImg = imread("./imgs_Pro_GA/tarImg.png");
    Mat blurImg;
    Mat diffImg;
    Mat labelImg;
    if (maskImg.empty() || tarImg.empty())
    {
        maskGenerate();
        targetGenerate();
        Mat maskImg = imread("./imgs_Pro_GA/maskImg.png");
        Mat tarImg = imread("./imgs_Pro_GA/tarImg.png");
    }
    // Initializing �E�E�E
    srand((unsigned)time(NULL));
    gene g[num_ind]; // For storing the group of individuals 
    gene elite[10]; // For storing the elite individual of each generation
    elite[1].f_value = 0.0;

    //eliteValue eValue; // for storing the value-info of elite individual of each generation
    //eValue.precision = 0.0;
    //eValue.recall = 0.0;
    //eValue.f_value = 0.0;

    make(g); // Initializing the group of individuals
    //if ((fp = fopen("process_log.txt", "w")) == NULL) { // for working log
    //    printf("Opening Failed!\n");
    //}

    // Main Part
    for (int numGen = 1; numGen <= num_gen; numGen++) { // the loop of generation
        cout << "-------generation: " << numGen << "---------" << endl;
        // in the start of each generation, first calculate the fitness of the decision variable and storing it in h-array 
        phenotype(g);
        for (int numInd = 0; numInd < num_ind; numInd++) { // the loop of individual
            import_para(numInd); // read in the fitness of h-array and write it to the global variables
            // the process of bluring
            if (filterswitch_flag == 0)
            {
                medianBlur(oriImg, blurImg, fsize);
            }
            else
            {
                blur(oriImg, blurImg, Size(fsize, fsize));
            }
            // the differential process of oriImg and blurImg
            diffImg = sabun(oriImg, blurImg);
            threshold(diffImg, diffImg, binary, 255, THRESH_BINARY);
            if (pixellabelingmethod == 0)
            {
                labelImg = labeling_new4(diffImg);
            }
            else
            {
                labelImg = labeling_new8(diffImg);
            }
            pixellabelingmethod = 0;
            cv::bitwise_not(labelImg, labelImg);
            if (erodedilate_sequence == 0)
            {
                if (erodedilate_times != 0) {
                    for (int i = 0; i < erodedilate_times; i++)
                    {
                        labelImg = dilate_erode(labelImg);
                    }
                }
            }
            else if (erodedilate_sequence == 1)
            {
                if (erodedilate_times != 0) {
                    for (int i = 0; i < erodedilate_times; i++)
                    {
                        labelImg = erode_dilate(labelImg);
                    }
                }
            }
            calculateMetrics(labelImg, tarImg, maskImg, numInd);
        }
        fitness(g, elite, numGen);
        crossover(g);
        mutation(g);
        elite_back(g, elite);
    }
    vector<Mat> images = { oriImg, tarImg, labelImg };
    Mat res;
    hconcat(images, res);
    imgShow("res_p1", res);
    // imwrite("./imgs/final_opt.jpg", res);
    return 0;
}