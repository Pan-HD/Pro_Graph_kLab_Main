#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<conio.h> //getch()
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/opencv.hpp"
#define sedai 100//���㐔
#define kotai 30//�̐�
#define	length 26//��`�q��(=�r�b�g��) 
#define cross 0.8//������(70�`90%)
#define mut 0.1//�ˑR�ψٗ�(0.1�`5%)
#define RAND_MAX 32767//�����̍ő�l
#define maxnoise 999999//�m�C�Y�����̌��x
#define hs 200;//�␳�̊�l
using namespace cv;
using namespace std;
short** gazo;//�摜�����ɗp����摜�̔z��
short** gazoold;//���͉摜��ۊǂ���z��
short** gazo2;//�摜�����ɗp����摜2�̔z��
short** gazoold2;//���͉摜2��ۊǂ���z��
short** gazo3;
short** sabun_g;//����������̉摜�̉�f�l
int x, y;//���͉摜1��X,Y�����̑傫��
int x2, y2;//���͉摜2��X,Y�����̑傫��
int** label;//��f���Ƃ̃��x��
int label_num[maxnoise];//���x�����Ƃ̓_�̐�
int label_sum[maxnoise];//���x�����Ƃ̑���f�l
int lx1[maxnoise];//���x����X�����̑傫�������߂邽�߂̕ϐ�
int lx2[maxnoise];//���x����X�����̑傫�������߂邽�߂̕ϐ�
int ly1[maxnoise];//���x����Y�����̑傫�������߂邽�߂̕ϐ�
int ly2[maxnoise];//���x����Y�����̑傫�������߂邽�߂̕ϐ�
int label_area[maxnoise];//���x�����Ƃ̐����`�̖ʐ�
int table[maxnoise][2];//���b�N�A�b�v�e�[�u��
int tmp = 0;//�J�E���^�[�̏�����
float sum1 = 0.0;//���v�l
int ekotai = 0;
int esedai = 0;
float evalu = 0.0;
FILE* fp, * fp2, * fp3, * fp4, * fp5, * fp6;//�t�@�C����`
int fsize = 0;//���f�B�A���t�B���^
int binary = 0;//2�l���������l
int ccc = 0;//�Œ��f��
double linear = 0.0;//����x
int eee = 0;//����x*�L�Y�Z�x3
int abusolute_flag = 0;//����x*�L�Y����f��
int erodedilate_sequence = 0;//�c������k�̏��ԁ@
int filterswitch_flag;//�t�H���_��؂�ւ���t���O
int erodedilate_times;//�c�����k�����̌J��Ԃ���
int pixellabelingmethod = 0;//8��4��

Mat img_label, img_bitwise, img_output;//���x���͘A�ʋ������o�����摜�@img_bitwise�͔������]�����摜
Mat img1, img2, img3;//��̓��͉摜 img1�͌��摜�̐؂蔲���Cimg2�͋��t�摜�ɐ؂蔲��
Mat img1_before, img2_before;//�O������̐؂蔲��
Mat img1ths;
Mat img_origin, img_origincopy, img_target, img_targetcopy;

Point sp(-1, -1);
Point ep(-1, -1);

typedef struct gene {
public:
	int ge[length];//���F��
	float tekioudo;//�K���x
	float valu;//�]��
}gene;

gene h[kotai][8];//GENE�Ƃ����\���̍\�z���@��h�Ƃ����\���̂��`����

// Adding Part
void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

void make(gene* g)//�����̌Q�̐���
{
	int i = 0;
	int j = 0;

	for (j = 0; j < kotai; j++) {
		//fprintf(fp,"��:%2d�Ԗ�",j+1);//�̔ԍ��̕\��
		for (i = 0; i < length; i++) {
			if (rand() > (RAND_MAX + 1) / 2) g[j].ge[i] = 1;
			else g[j].ge[i] = 0;

			fprintf(fp, "%d", g[j].ge[i]); //�̂̕\��
			// printf("%d", g[j].ge[i]);
		}
		puts("\n");
		fprintf(fp, "\n");
	}
}

void phenotype(gene* g)//�\���n�v�Z(2�i����10�i����)
{
	int i = 0, j = 0, k = 0;

	for (j = 0; j < kotai; j++) {
		for (i = 0; i < 6; i++) {
			h[j][i].tekioudo = 0.0;//������
		}
	}

	for (j = 0; j < kotai; j++) {
		i = 6;
		for (k = 0; k < 6; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][0].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i); //���f�B�A���t�B���^
			}
		}

		i = 8;
		for (k = 6; k < 14; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][1].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);//2�l���������l
			}
		}


		i = 5;
		for (k = 14; k < 19; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][2].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);//����x臒l
			}
		}

		i = 1;
		for (k = 19; k < 20; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][3].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);//�g�p����t�B���^�̎��
			}
		}

		i = 3;
		for (k = 20; k < 23; k++) {
			i--;


			if (g[j].ge[k] == 1) {
				h[j][4].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);//���x�����O臒l
			}
		}

		i = 1;
		for (k = 23; k < 24; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][5].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);//�c������k�̏���

			}
		}

		i = 1;
		for (k = 24; k < 25; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][6].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);//��Βl�̑I��

			}
		}

		i = 1;
		for (k = 25; k < 26; k++) {
			i--;
			if (g[j].ge[24] == 1) {
				h[j][7].tekioudo = 1;//�ߖT8��f��������4��f�̑I��
			}
			else
			{
				h[j][7].tekioudo = 0;//�ߖT8��f��������4��f�̑I��
			}
		}

	}
}

void fitness(gene* g, gene* elite, int se)//�K���x�̌v�Z(�G���[�g�ۑ�)
{
	int i = 0, j = 0;
	double ave = 0.0;
	double deviation = 0.0;
	double variance = 0.0;

	sum1 = 0;//������

	//�G���[�g�ۑ�
	for (i = 0; i < kotai; i++) {
		sum1 += h[i][0].valu;
		if (h[i][0].valu > elite[1].valu) {
			elite[1].valu = h[i][0].valu;//�G���[�g���ꊷ��
			for (j = 0; j < length; j++) {
				elite[1].ge[j] = g[i].ge[j];
			}
		}
	}

	float min_value = 1.1;
	elite[2].valu = 1.1;
	for (i = 0; i < kotai; i++) {
		if (h[i][0].valu < min_value) {
			min_value = h[i][0].valu;
		}
	}

	elite[3].valu = 0.0;//������
	for (i = 0; i < kotai; i++) {
		elite[3].valu += h[i][0].valu;
	}

	ave = (double)(elite[3].valu) / (double)kotai;

	for (i = 0; i < kotai; i++)
	{
		double diff = h[i][0].valu - ave;
		variance += diff * diff;
	}

	deviation = sqrt(variance / kotai);


	//fprintf(fp4, "%d	%.2f	%.2f	���ρF%.2f\n", se, elite[1].valu, elite[2].valu, ave);
	fprintf(fp4, "%d	�ő�F%.2f	�ŏ��F%.2f�@���ϒl�F%.2f  ���ϕ΍��l�F%.2f\n", se, elite[1].valu, min_value, ave, deviation);
	// printf("%d	%.2f	%.2f	���ρF%.2f  ���ϕ΍��l�F%.2f\n", se, elite[1].valu, min_value, ave, deviation);

}

void elite_back(gene* g, gene* elite) {//�G���[�g�̂ƓK���x�ŏ��̂�����

	int i = 0, j = 0;
	float ave = 0.0;
	float min1 = 1.0;

	tmp = 0;//�J�E���^�[�̏�����

	for (i = 0; i < kotai; i++) {//�ŏ��l�T��
		if (h[i][0].valu < min1) {
			min1 = h[i][0].valu;
			tmp = i;
		}
	}

	for (j = 0; j < length; j++) {
		g[tmp].ge[j] = elite[1].ge[j];//�ŏ��l�ƃG���[�g������
	}

	h[tmp][0].valu = elite[1].valu;//�G���[�g�̕]���l�ƌ���
	ave = sum1 / kotai;//���v�l�̌v�Z
}

int roulette()//���[���b�g�I��
{
	int i = 0, r = 0;
	int num = 0;
	float sum = 0.0;
	float* p;

	p = (float*)malloc(sizeof(int) * kotai);

	sum = 0;
	for (i = 0; i < kotai; i++) {
		sum += h[i][0].valu;//���ׂĂ̍��v
	}
	for (i = 0; i < kotai; i++) {
		p[i] = h[i][0].valu / sum;//�K���x(��)
	}

	sum = 0;
	r = rand();
	for (i = 0; i < kotai; i++) {
		sum += RAND_MAX * p[i];//1
		if (r <= sum) {
			num = i;
			break;
		}
	}
	if (num < 0)	num = roulette();//�G���[�̂��߂̏���
	free(p);
	return(num);
}

void crossover(gene* g) {//��_����
	gene g2[1000] = { 0 };
	int num = 0;
	int n1 = 0;
	int n2 = 0;
	int p = 0;
	int i, j;

	for (num = 0; num < kotai; num += 2) {
		n1 = rand() % 10;
		n2 = rand() % 10;
		if (rand() <= RAND_MAX * cross) {//�����m���𖞂����ꍇ
			n1 = roulette();
			n2 = roulette();
			//�����͈͎̔w������F(int)( rand() * (�ő�l - �ŏ��l + 1.0) / (1.0 + RAND_MAX) )
			p = (int)(rand() * ((length - 2) - 1 + 1.0) / (1.0 + RAND_MAX) + 1);

			//�qA
			for (i = 0; i < p; i++) {
				g2[num].ge[i] = g[n1].ge[i];
			}
			for (i = p; i < length; i++) {
				g2[num].ge[i] = g[n2].ge[i];
			}

			//�qB
			for (i = 0; i < p; i++) {
				g2[num + 1].ge[i] = g[n2].ge[i];
			}
			for (i = p; i < length; i++) {
				g2[num + 1].ge[i] = g[n1].ge[i];
			}
		}
		else {
			for (i = 0; i < length; i++) {
				n1 = roulette();
				n2 = roulette();
				g2[num].ge[i] = g[n1].ge[i];
				g2[num + 1].ge[i] = g[n2].ge[i];
			}
		}
	}

	for (j = 0; j < kotai; j++) {
		for (i = 0; i < length; i++) {
			g[j].ge[i] = g2[j].ge[i];//g[]���X�V
		}
	}
}

void mutation(gene* g)//�ˑR�ψ�
{
	int num = 0;
	int r = 0;
	int i = 0;
	int p = 0;
	for (num = 0; num < kotai; num++) {
		if (rand() <= RAND_MAX * mut) {//�ˑR�ψيm���𖞂����ꍇ�C1�̈�`�q��I��
			p = (int)(rand() * ((length - 1) + 1.0) / (1.0 + RAND_MAX));
			for (i = 0; i < length; i++) {//1��0���t�]
				if (i == p) {
					if (g[num].ge[i] == 0) g[num].ge[i] = 1;
					else				g[num].ge[i] = 0;
				}
			}
			p = 0;
		}
	}
}

void import_image() {

	img_bitwise.copyTo(img_output);
}

void import_para(int ko) {//�p�����[�^�̏o��
	fsize = 0;
	binary = 0;
	ccc = 0;
	linear = 0.0;
	filterswitch_flag = 0;
	abusolute_flag = 0;

	fsize = (int)(3 + 2 * h[ko][0].tekioudo);
	binary = (int)(1 * h[ko][1].tekioudo);
	//ccc = (int)(4 * h[ko][2].tekioudo);
	linear = (double)(1.0 + 0.5 * h[ko][2].tekioudo);
	filterswitch_flag = (int)(h[ko][3].tekioudo);
	erodedilate_times = (int)(h[ko][4].tekioudo);
	erodedilate_sequence = (int)(h[ko][5].tekioudo);
	abusolute_flag = (int)(h[ko][6].tekioudo);
	pixellabelingmethod = (int)(h[ko][7].tekioudo);

	fprintf(fp2, "fsize�F%5d	binary�F%5d	filterswitch_flag�F%5d	linear�F%2.2f	erodedilate_times:%7d�@ erodedilate_sequence:%d pixellabelingmethod:%7d\n", fsize, binary, filterswitch_flag, linear, erodedilate_times, erodedilate_sequence, pixellabelingmethod);
}

void noiz_kessonnew(int ko, int se) {
	int i = 0, j = 0;
	int m = 0, n = 0, tm = 0, tn = 0;
	float ks = 0.0, nz = 0.0;
	float v = 0.0;
	int flg = 0;
	h[ko][0].valu = 0.0;//������
	for (j = 0; j < y2; j++) {
		for (i = 0; i < x2; i++) {
			if (img2.at<unsigned char>(j, i) == 0)
			{
				tm++;
			}
			else tn++;
		}
	}
	//�m�C�Y���C������
	for (j = 0; j < y2; j++) {
		for (i = 0; i < x2; i++) {
			if (img2.at<unsigned char>(Point(i, j)) == 0 && img_bitwise.at<unsigned char>(Point(i, j)) == 255) {//�m�C�Y��
				m++;
			}
			if (img2.at<unsigned char>(Point(i, j)) == 255 && img_bitwise.at<unsigned char>(Point(i, j)) == 0) {//������
				n++;
			}
		}
	}
	if (m == 0)
	{
		m = 1;
	}
	if (n == 0)
	{
		n = 1;
	}

	ks = (float)((float)m / (float)tm);
	//printf("ks:%f\n", ks);
	nz = (float)((float)n / (float)tn);
	h[ko][0].valu = 1.0 - sqrt((0.2 * ks * ks) + (0.8 * nz * nz));
	v = 1.0 - sqrt((0.2 * ks * ks) + (0.8 * nz * nz));

	if (ko == 0) {
		import_image();
		ekotai = ko;
		esedai = se;
		evalu = v;
		fprintf(fp5, "����F%d �́F%d\n", se, ko);
	}
	if ((ko != 0) /*&& (v > evalu)*/) {
		flg = 1;
		import_image();
		ekotai = ko;
		esedai = se;
		evalu = v;
		//printf("v:%f\n", v);
		fprintf(fp5, "����F%d �́F%d\n", se, ko);
	}
	fprintf(fp3, "nz * 100:%.4f	ks * 100�F%.2f\n", nz * 100, ks * 100);
}

void noiz_kessonFvalue(int ko, int se) {
	int i = 0, j = 0;
	int TP = 0, FP = 0, TN = 0, FN = 0;
	int m = 0, n = 0, tm = 0, tn = 0;
	float ks = 0.0, nz = 0.0;
	float precision = 0.0, recall = 0.0;//precision�K�����Crecall�Č���
	float v = 0.0;
	int flg = 0;
	h[ko][0].valu = 0.0; //������

	// �m�C�Y���C������
	for (j = 0; j < y2; j++) {
		for (i = 0; i < x2; i++) {
			// img3�̔��������i255�j�ł̂݌v�Z
			if (img3.at<unsigned char>(j, i) == 255) {
				// �^�z��: img2��img_bitwise�������Ƃ����i0�j�̏ꍇ
				if (img2.at<unsigned char>(Point(i, j)) == 0 && img_bitwise.at<unsigned char>(Point(i, j)) == 0) {
					TP++;
				}
				// �U�z��: img2�����i255�j�ŁAimg_bitwise�����i0�j�̏ꍇ
				else if (img2.at<unsigned char>(Point(i, j)) == 255 && img_bitwise.at<unsigned char>(Point(i, j)) == 0) {
					FP++;
				}
				// �U�A��: img2�����i0�j�ŁAimg_bitwise�����i255�j�̏ꍇ
				else if (img2.at<unsigned char>(Point(i, j)) == 0 && img_bitwise.at<unsigned char>(Point(i, j)) == 255) {
					FN++;
				}
			}
		}
	}

	// ���x��Č����̌v�Z
	if (TP == 0) { TP = 1; } // TP��0�̏ꍇ�̏���
	if (FP == 0) { FP = 1; } // FP��0�̏ꍇ�̏���
	if (FN == 0) { FN = 1; } // FN��0�̏ꍇ�̏���

	precision = ((float)TP / ((float)TP + (float)FP)); // �K����
	recall = ((float)TP / ((float)TP + (float)FN));    // �Č���
	v = (2 * precision * recall) / (precision + recall); // F�l�̌v�Z

	// ���ʂ̕ۑ��Əo��
	h[ko][0].valu = v;
	if (ko == 0) {
		import_image();
		ekotai = ko;
		esedai = se;
		evalu = v;
	}

	if (ko != 0) {
		flg = 1;
		import_image();
		ekotai = ko;
		esedai = se;
		evalu = v;
		fprintf(fp5, "����F%d �́F%d\n", se, ko);
		fprintf(fp5, "v:%f\n	\n", v);
	}

	// Precision, Recall, F�l�̃��O�o��
	fprintf(fp3, "precision:%.4f	recall�F%.6f\n  Fvalue�F%.2f\n", precision, recall, v);

	// F�l�Ɋ�Â���������
	if (v < 0.51) {
		fprintf(fp6, "NO");
	}
	else {
		fprintf(fp6, "YES");
	}
}


int tyuuou()//�摜�̒����l�����߂�
{
	int i = 0, j = 0, n = 0;
	int sum = 0;
	int c = 0;
	for (j = 0; j <= y - 1; j++) {
		for (i = 0; i <= x - 1; i++) {
			sum += gazoold[i][j];
			c++;
		}
	}
	return (sum / (float)c);
}

void lookup(int look, int min) {
	int tmp;
	if (table[look][1] > min) {
		tmp = table[look][1];
		table[look][1] = min;
		lookup(tmp, min);
	}
}

Mat labeling_new4(Mat img_sabun, double linear) {//�ߖT4��f�ŉ�f�̉�����x�����O���@
	Mat img_con;
	Mat stats, centroids;//�A�ʋ��̑���
	int i, j, label_num;//�A�ʋ��̐�
	int label_x, label_y;
	int label_longer;
	double label_cal;
	int label_areaall;//���x����t�������̉�f��
	label_num = connectedComponentsWithStats(img_sabun, img_con, stats, centroids, 4, 4);
	vector<Vec3b>colors(label_num + 1);
	colors[0] = Vec3b(0, 0, 0);

	for (i = 1; i < label_num; i++)
	{
		colors[i] = Vec3b(255, 255, 255);
		label_areaall = stats.at<int>(i, CC_STAT_AREA);
		label_x = stats.at<int>(i, CC_STAT_WIDTH);
		label_y = stats.at<int>(i, CC_STAT_HEIGHT);

		if (label_x > label_y)label_longer = label_x;
		else label_longer = label_y;

		label_cal = label_longer * label_longer;//��蒷���ӂ̓����v�Z����

		if (label_cal / label_areaall < linear)
		{
			colors[i] = Vec3b(0, 0, 0);
		}
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
Mat labeling_new8(Mat img_sabun, double linear) {//�ߖT8��f�ŉ�f�̉�����x�����O���@
	Mat img_con;
	Mat stats, centroids;//�A�ʋ��̑���
	int i, j, label_num;//�A�ʋ��̐�
	int label_x, label_y;
	int label_longer;
	double label_cal;
	int label_areaall;//���x����t�������̉�f��
	label_num = connectedComponentsWithStats(img_sabun, img_con, stats, centroids, 8, 4);
	vector<Vec3b>colors(label_num + 1);
	colors[0] = Vec3b(0, 0, 0);

	for (i = 1; i < label_num; i++)
	{
		colors[i] = Vec3b(255, 255, 255);
		label_areaall = stats.at<int>(i, CC_STAT_AREA);
		label_x = stats.at<int>(i, CC_STAT_WIDTH);
		label_y = stats.at<int>(i, CC_STAT_HEIGHT);
		if (label_x > label_y)label_longer = label_x;
		else label_longer = label_y;
		label_cal = label_longer * label_longer;//��蒷���ӂ̓����v�Z����
		if (label_cal / label_areaall < linear)
		{
			colors[i] = Vec3b(0, 0, 0);
		}
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

Mat sabun(Mat input1, Mat input2) {
	int i, j;
	Mat output;
	output = cv::Mat::zeros(cv::Size(input2.cols, input2.rows), CV_8UC3);//8UC3��3�`�����l���ɕς���^�C�v��
	cvtColor(output, output, COLOR_RGB2GRAY);//�O���[�X�P�[��

	for (j = 0; j < input1.rows; j++)
	{
		for (i = 0; i < input1.cols; i++) {
			output.at<unsigned char>(j, i) = input2.at<unsigned char>(j, i) - input1.at<unsigned char>(j, i);
			if (input2.at<unsigned char>(j, i) - input1.at<unsigned char>(j, i) < 0)
			{
				output.at<unsigned char>(j, i) = 0;
			}
		}
	}
	return output;
}

Mat dilate_erode(Mat src1) {//�I�[�v�j���O�N���[�W���O����
	Mat dst;
	dst.create(src1.size(), src1.type());
	//�N���[�W���O������C�I�[�v�j���O�����D�X�ɖc������
	dilate(src1, dst, Mat());//�c������
	erode(dst, dst, Mat());//���k����

	return dst;
}

Mat erode_dilate(Mat src1) {//�I�[�v�j���O�N���[�W���O����
	Mat dst;
	dst.create(src1.size(), src1.type());
	//�N���[�W���O������C�I�[�v�j���O�����D�X�ɖc������
	erode(dst, dst, Mat());//���k����
	dilate(src1, dst, Mat());//�c������

	return dst;
}

int main(int argc, char* argv[])
{
	Mat img1_median, img1_sabun;//��̓��͉摜
	unsigned char r1, g1, b1, r2, g2, b2;
	short count = 0;
	short tyuu;
	short v2;
	int i = 0, j = 0, k = 0;//for���p�ϐ�

	// int picture_loading_mode = 0;//�V���Ȑ؂蔲��������̂��O�̉摜�̂܂܎g���̂��@0�Ȃ�V�����؂蔲�����쐬�@1�Ȃ�

	int num;
	char filename1[256];//�t�@�C�������͗p�z��
	char filename2[256];//�t�@�C�������͗p�z��
	char filename3[256];//�t�@�C�������͗p�z��
	char decision_data[256];//�t�@�C�������͗p�z�� ����ؗp
	char sw = 0;
	clock_t start, end;
	start = clock();

	if ((fp = fopen("./imgs_1024_v1/�̂̈�`�q�^.txt", "w")) == NULL) {
		printf("error:�t�@�C�����I�[�v���ł��܂���B\n");
	}
	if ((fp2 = fopen("./imgs_1024_v1/parameter.txt", "w")) == NULL) {
		printf("error:�t�@�C�����I�[�v���ł��܂���B\n");
	}
	if ((fp3 = fopen("./imgs_1024_v1/noiz_kesson.txt", "w")) == NULL) {
		printf("error:�t�@�C�����I�[�v���ł��܂���B\n");
	}
	if ((fp4 = fopen("./imgs_1024_v1/max_ave_min_deviation.txt", "w")) == NULL) {
		printf("error:�t�@�C�����I�[�v���ł��܂���B\n");
	}
	if ((fp5 = fopen("./imgs_1024_v1/max_para.txt", "w")) == NULL) {
		printf("error:�t�@�C�����I�[�v���ł��܂���B\n");
	}
	if ((fp6 = fopen("./imgs_1024_v1/decision_tree_dataset.txt", "w")) == NULL) {
		printf("error:�t�@�C�����I�[�v���ł��܂���B\n");
	}

	srand((unsigned)time(NULL));//�����ɂ�闐���̏�����
	gene g[kotai];
	gene elite[10];//�G���[�g�ۑ���
	elite[1].valu = 0.0;//�G���[�g�̂̏�����

	//img_origin = imread("./imgs_1024_v1/�V��1.jpg");
	//img_target = imread("./imgs_1024_v1/mask1.jpg");

	/*puts("�O������̐؂蔲�����g���܂����H0/1\n");
	puts("0�������@�V���ȋ��Ŏ�������\n");
	puts("1�͂��@�O�̉摜���g������\n");
	scanf("%d", &picture_loading_mode);
	printf("picture_loading_mode:%d\n", picture_loading_mode);*/

	img1 = imread("./imgs_1024_v1/image1.png");
	img2 = imread("./imgs_1024_v1/kyoushi1.png");
	img3 = imread("./imgs_1024_v1/maskImg1.png");
	cvtColor(img1, img1ths, COLOR_RGB2GRAY);
	if (img1.empty())
	{
		printf("�摜��ǂݍ��݂ł��Ȃ�");
		return -1;
	}

	x = img1.cols;//x�����̉摜�T�C�Y
	y = img1.rows;//y�����̉摜�T�C�Y
	x2 = img2.cols;//x�����̉摜�T�C�Y
	y2 = img2.rows;//y�����̉摜�T�C�Y
	make(g);//�����̌Q�̐���

	for (num = 1; num <= sedai; num++) {//����
		printf("======= ��%d���� =======\n", num);
		phenotype(g);//�\���n�v�Z(2�i����10�i����)
		// fprintf(fp2, "======= ��%d���� =======\n", num);

		for (k = 0; k < kotai; k++) {
			import_para(k);
			// fprintf(fp6, "%d,%d,%2.2f", fsize, binary, linear);
			sprintf_s(decision_data, "%d", fsize);
			sprintf_s(decision_data, ",%d", binary);

			if (filterswitch_flag == 0)
			{
				medianBlur(img1ths, img1_median, fsize);
				// fprintf(fp6, ",median-filter");

			}
			else
			{
				blur(img1ths, img1_median, Size(fsize, fsize));
				// fprintf(fp6, ",average-filter");
			}

			if (abusolute_flag == 0)
			{
				// fprintf(fp6, ",abusolute");
			}
			else
			{
				// fprintf(fp6, ",non-abusolute");
			}

			img1_sabun = sabun(img1ths, img1_median);

			waitKey(0);
			threshold(img1_sabun, img1_sabun, binary, 255, THRESH_BINARY);//��l������
			if (pixellabelingmethod == 0)
			{
				//printf("pixelmehod's value:%d\n", pixellabelingmethod);
				img_label = labeling_new4(img1_sabun, linear);//���x�����O+�m�C�Y����
				// fprintf(fp6, ",4pixel");
			}
			else
			{
				img_label = labeling_new8(img1_sabun, linear);//���x�����O+�m�C�Y����
				// fprintf(fp6, ",8pixel");

			}
			pixellabelingmethod = 0;//0�ɖ߂�
			bitwise_not(img_label, img_bitwise);//�������]����
			if (erodedilate_sequence == 0)
			{
				// fprintf(fp6, ",dilatefirst");
				if (erodedilate_times != 0) {
					for (i = 0; i < erodedilate_times; i++)
					{
						img_bitwise = dilate_erode(img_bitwise);
					}
				}
				// fprintf(fp6, ",%d,", erodedilate_times);
			}
			else if (erodedilate_sequence == 1)
			{
				// fprintf(fp6, ",erodefirst");
				if (erodedilate_times != 0) {
					for (i = 0; i < erodedilate_times; i++)
					{
						img_bitwise = erode_dilate(img_bitwise);
					}
				}
				// fprintf(fp6, ",%d,", erodedilate_times);
			}
			noiz_kessonFvalue(k, num);//�]��
			// fprintf(fp6, "\n");
		}
		//output();//�摜�o��
		// sprintf(filename3, "sedai%d�G���[�g.png", esedai);//�o�͉摜������
		// imwrite(filename3, img_output);
		esedai = 0;
		ekotai = 0;
		evalu = 0.0;
		fitness(g, elite, num);//�K���x�̌v�Z(�G���[�g�ۑ�)
		crossover(g);//��_����
		mutation(g);//�ˑR�ψ�
		elite_back(g, elite);//�G���[�g�̂ƓK���x�ŏ��̂�����
		// Adding Part
		cout << "F1-gen-" << num << ": " << elite[1].valu << endl;
	}
	// Adding Part
	vector<Mat> images = { img1, img2, img_bitwise };
	Mat res;
	hconcat(images, res);
	imgShow("res_p1", res);
	printf("========= �I�� =========\n");
	//�t�@�C���I��
	fclose(fp);
	fclose(fp2);
	fclose(fp3);
	fclose(fp4);
	fclose(fp5);
	fclose(fp6);
	//�����ɂ�����������
	end = clock();
	int minute = 0;//����������
	double second = 0;//���������b
	minute = ((double)(end - start) / CLOCKS_PER_SEC) / 60;
	second = ((double)(end - start) / CLOCKS_PER_SEC) - 60 * minute;
	printf("%d��%.3f�b\n", minute, second);
	//�������J��
	free(gazo);
	free(gazoold);
	free(gazo2);
	free(gazoold2);
	free(sabun_g);
	free(label);
	_getch();
	return(0);
}
