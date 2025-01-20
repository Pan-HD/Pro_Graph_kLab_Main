#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<conio.h>  // getch()
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/opencv.hpp"

#define sedai 100
#define kotai 30
#define length 26 // 遺伝子の長さ
#define cross 0.8 // 交叉率
#define mut 0.05 // 突然変異率
#define RAND_MAX 32767 // 乱数の最大値
#define maxnoise 999999
#define hs 200 // 補正の基準値 _Q01

using namespace cv;
using namespace std;

Mat img_origin, img_target;
Mat img_origincopy, img_targetcopy;
// img1: img_originのキャプチャ部分, img2: img_targetのキャプチャ部分 (Part1 - キャプチャ)
Mat img1, img2;
Mat img1ths; // img1のグレース化画像 (Part1 - キャプチャ処理後の際)
Mat img1_before, img2_before;
Point sp(-1, -1); // start point
Point ep(-1, -1); // end point
// 入力した2つの画像(キャプチャ)に対して・・・
int x, y; // 入力した img1 の広さと高さ
int x2, y2; // 入力した img2 の広さと高さ

int fsize = 0; // メディアンフィルタ
int binary = 0; // 2値化しきい値
int ccc = 0; // 最低画素数
double linear = 0.0; // 線状度
int eee = 0; // 線状度*キズ濃度3
int abusolute_flag = 0; // 線状度*キズ総画素数
int erodedilate_sequence = 0; // 膨張や収縮の順番　
int filterswitch_flag; // フォルダを切り替えるフラグ
int erodedilate_times; // 膨張収縮処理の繰り返す回数
int pixellabelingmethod = 0; // 8か4か
// ラベルは連通区域を検出した画像　img_bitwiseは白黒反転した画像
Mat img_label, img_bitwise, img_output;

int ekotai = 0;
int esedai = 0;
float evalu = 0.0;
float sum1 = 0.0; // 合計値
int tmp = 0; // カウンターの初期化

/*
	説明：個体を表現するための構造体 -> "01.png"
*/
typedef struct gene {
public:
	int ge[length]; // 染色体
	float tekioudo; // 適応度
	float valu; // 評価
}gene;

void imgShow(const string& name, const Mat& img) {
	imshow(name, img);
	waitKey(0);
	destroyAllWindows();
}

Mat imgResize(Mat img, int width, int height) {
	Mat resImg;
	Size size(width, height);
	resize(img, resImg, size);
	return resImg;
}

/*
	役割：各個体に対して、それに対応する8つの決定変数のtekioudoを保存するための二次元配列
	　　　h[j][k] -> j:各個体のindex、k:定変数のindex
	　　　k: 0 -> fsize, 1 -> binary, 2 -> linear, 3 -> filterswitch_flag
	　　　   4 -> erodedilate_times, 5 -> erodedilate_sequence, 6 -> abusolute_flag
	　　　   7 -> pixellabelingmethod
*/
gene h[kotai][8];

/*
	役割：生成した個体群の初期化 -> 各個体に対してその遺伝子・・・
		  g: 個体群を保存する配列, gene g[kotai]
*/
void make(gene* g) // 
{
	int i = 0;
	int j = 0;

	for (j = 0; j < kotai; j++) {
		// fprintf(fp,"個体:%2d番目",j+1); // 個体番号の表示
		for (i = 0; i < length; i++) { // length: 遺伝子の長さ
			if (rand() > (RAND_MAX + 1) / 2) g[j].ge[i] = 1;
			else g[j].ge[i] = 0;
			// 今回の初期化情報をfpとstdoutに書き込む。
			// fprintf(fp, "%d", g[j].ge[i]);  // 個体の表示
			// printf("%d", g[j].ge[i]);
		}
		puts("\n");
		// fprintf(fp, "\n");
	}
}

/*
	役割：遺伝子型（2進数）から表現型（10進数）を生成する処理
		  具体的には、遺伝子データ（g[j].ge[k]）に基づいて、
		  各個体の tekioudo を計算しています。
	追加説明：h[j]は、第j番の個体の適応度を含む配列(h[j][k])
*/
void phenotype(gene* g)
{
	int i = 0, j = 0, k = 0;
	// 各個体に対して前の六つ決定変数の適応度を初期化する
	for (j = 0; j < kotai; j++) {
		for (i = 0; i < 6; i++) { // i < 8 ?
			h[j][i].tekioudo = 0.0;
		}
	}
	for (j = 0; j < kotai; j++) {
		i = 6;
		for (k = 0; k < 6; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][0].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);
			}
		}

		i = 8;
		for (k = 6; k < 14; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][1].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);
			}
		}


		i = 5;
		for (k = 14; k < 19; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][2].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);
			}
		}

		i = 1;
		for (k = 19; k < 20; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][3].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);
			}
		}

		i = 3;
		for (k = 20; k < 23; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][4].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);
			}
		}
		i = 1;
		for (k = 23; k < 24; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][5].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);

			}
		}
		i = 1;
		for (k = 24; k < 25; k++) {
			i--;
			if (g[j].ge[k] == 1) {
				h[j][6].tekioudo += (float)pow((double)g[j].ge[k] * 2, (double)i);

			}
		}
		i = 1;
		for (k = 25; k < 26; k++) {
			i--;
			if (g[j].ge[24] == 1) {
				h[j][7].tekioudo = 1;
			}
			else
			{
				h[j][7].tekioudo = 0;
			}
		}
	}
}

/*
	役割：各世代で各個体に対して、その10進数のtekioudoを特定の関数式で
	　　　実際のパラメーター値にマッピングします。
	ko: 特定の個体の番号(index)
*/
void import_para(int ko) { // パラメータの出力
	fsize = 0;
	binary = 0;
	ccc = 0;
	linear = 0.0;
	filterswitch_flag = 0;
	abusolute_flag = 0;

	fsize = (int)(3 + 2 * h[ko][0].tekioudo);
	binary = (int)(1 * h[ko][1].tekioudo);
	// ccc = (int)(4 * h[ko][2].tekioudo);
	linear = (double)(1.0 + 0.5 * h[ko][2].tekioudo);
	filterswitch_flag = (int)(h[ko][3].tekioudo);
	erodedilate_times = (int)(h[ko][4].tekioudo);
	erodedilate_sequence = (int)(h[ko][5].tekioudo);
	abusolute_flag = (int)(h[ko][6].tekioudo);
	pixellabelingmethod = (int)(h[ko][7].tekioudo);
	//fprintf(fp2, "fsize：%5d	binary：%5d	filterswitch_flag：%5d	linear：%2.2f	erodedilate_times:%7d　 erodedilate_sequence:%d pixellabelingmethod:%7d\n", fsize, binary, filterswitch_flag, linear, erodedilate_times, erodedilate_sequence, pixellabelingmethod);
}

/*
	役割：入力した2つの画像の差分を計算して、その結果をoutputに保存して返します。
*/
Mat sabun(Mat input1, Mat input2) {
	int i, j;
	Mat output;
	output = cv::Mat::zeros(cv::Size(input2.cols, input2.rows), CV_8UC3); // 8UC3は3チャンネルに変えるタイプだ
	cvtColor(output, output, COLOR_RGB2GRAY); // グレースケール

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

/*
	役割：ブロブの検出
	Para: img_sabun: 差分結果
		  linear: ラベリングされた領域が条件を満たすかどうかを決定するための閾値
		  img_color: ラベリング後の画像 ー＞ ブロブを検出した部分は白い、背景の部分は黒いで・・・
*/
Mat labeling_new4(Mat img_sabun, double linear) { // 近傍4画素で画素の塊をラベリング方法
	Mat img_con;
	Mat stats, centroids; // 連通区域の属性
	int i, j, label_num; // 連通区域の数

	int label_x, label_y;
	int label_longer;
	double label_cal;
	int label_areaall; // ラベルを付けた区域の画素数
	// img_con: 各画素に対して、どの領域に属するかがラベルとして img_con に保存されます。
	label_num = connectedComponentsWithStats(img_sabun, img_con, stats, centroids, 4, 4);
	vector<Vec3b>colors(label_num + 1); // ラベルごとにどの色を使うかを格納するためのものです
	colors[0] = Vec3b(0, 0, 0); // 背景の色を黒に設定

	for (i = 1; i < label_num; i++) // ラベルごとの処理
	{
		colors[i] = Vec3b(255, 255, 255);

		label_areaall = stats.at<int>(i, CC_STAT_AREA);
		label_x = stats.at<int>(i, CC_STAT_WIDTH);
		label_y = stats.at<int>(i, CC_STAT_HEIGHT);

		if (label_x > label_y)label_longer = label_x;
		else label_longer = label_y;

		label_cal = label_longer * label_longer; // より長い辺の二乗を計算する

		// ラベリングされた領域のうち、形状が細長くない（長辺の二乗が領域の面積に対して小さい）
		// 領域に対して、ラベルを無効化（黒色に設定）しています
		if (label_cal / label_areaall < linear)
		{
			colors[i] = Vec3b(0, 0, 0); // 条件を満たさない領域を黒にする
		}
	}
	// CV_8UC3：3チャンネル・・・
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
Mat labeling_new8(Mat img_sabun, double linear) { // 近傍8画素で画素の塊をラベリング方法
	Mat img_con;
	Mat stats, centroids; // 連通区域の属性
	int i, j, label_num; // 連通区域の数

	int label_x, label_y;
	int label_longer;
	double label_cal;
	int label_areaall; // ラベルを付けた区域の画素数
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
		label_cal = label_longer * label_longer; // より長い辺の二乗を計算する
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

Mat dilate_erode(Mat src1) { // オープニングクロージング処理
	Mat dst;
	dst.create(src1.size(), src1.type());
	// クロージング処理後，オープニング処理．更に膨張処理
	dilate(src1, dst, Mat()); // 膨張処理
	erode(dst, dst, Mat()); // 収縮処理

	return dst;
}

Mat erode_dilate(Mat src1) { // オープニングクロージング処理
	Mat dst;
	dst.create(src1.size(), src1.type());
	// クロージング処理後，オープニング処理．更に膨張処理
	erode(dst, dst, Mat()); // 収縮処理
	dilate(src1, dst, Mat()); // 膨張処理

	return dst;
}

void import_image() {
	img_bitwise.copyTo(img_output);
}

/*
	役割：ある世代で各個体に対して、その 精度と再現率の調和平均(v) を計算して、
		  結果はh[ko][0].valu に保存します。
	パラメーター：ko -> 個体、se -> 世代
*/
void noiz_kessonFvalue(int ko, int se) {
	int i = 0, j = 0;
	// TP: True Positive, FP: False Positive
	// TN: True Negative, FN: False Negative
	int TP = 0, FP = 0, TN = 0, FN = 0;
	int m = 0, n = 0, tm = 0, tn = 0;
	float ks = 0.0, nz = 0.0;
	// recall: です。
	float precision = 0.0, recall = 0.0;
	// v: 精度（precision）と再現率（recall）を組み合わせた評価指標で、
	//    分類問題においてモデルのパフォーマンスを測る・・・
	float v = 0.0;
	int flg = 0;
	h[ko][0].valu = 0.0; // 初期化
	// ノイズ率，欠損率
	// すべてのピクセルをスキャンして、TP FP FN を計算します。
	for (j = 0; j < y2; j++) {
		for (i = 0; i < x2; i++) {
			if (img2.at<unsigned char>(Point(i, j)) == 0 && img_bitwise.at<unsigned char>(Point(i, j)) == 0) { // TruePositive
				TP++; // 正しく検出のピクセル数
			}
			else if (img2.at<unsigned char>(Point(i, j)) == 255 && img_bitwise.at<unsigned char>(Point(i, j)) == 0) { // FalsePositive
				FP++; // 誤検出のピクセル数
			}
			else if (img2.at<unsigned char>(Point(i, j)) == 0 && img_bitwise.at<unsigned char>(Point(i, j)) == 255) { // FalseNegative
				FN++; // 未検出のピクセル数
			}
		}
	}
	if (TP == 0)
	{
		TP = 1;
	}
	if (FP == 0)
	{
		FP = 1;
	}
	if (FN == 0)
	{
		FN = 1;
	}

	// 適合率：検出部分の中、正確な部分の割合
	precision = ((float)TP / ((float)TP + (float)FP));
	// 再現率：検出すべき部分の中、実際の検出部分の割合
	recall = (float)((float)TP / ((float)TP + (float)FN));
	// v 値は、精度と再現率の調和平均を取ることで、両者のバランスを取った指標として機能します。
	//         調和平均を取ることで、片方だけが高くてもF値は高くなりません。
	v = (2 * precision * recall) / (precision + recall);
	h[ko][0].valu = v;

	// ko が 0 の場合、画像のインポートと評価値の初期化が行われます。
	if (ko == 0) {
		import_image(); // img_bitwise.copyTo(img_output);
		ekotai = ko;
		esedai = se;
		evalu = v;
	}

	if ((ko != 0) /*&& (v > evalu)*/) {
		flg = 1;
		import_image(); // img_bitwise.copyTo(img_output);
		ekotai = ko;
		esedai = se;
		evalu = v;
		// printf("v:%f\n", v);
		//fprintf(fp5, "世代：%d 個体：%d\n", se, ko);
		//fprintf(fp5, "v:%f\n	\n", v);
	}
	//fprintf(fp3, "precision:%.4f	recall：%.6f\n  Fvalue：%.2f\n", precision, recall, v);
	//if (v < 0.51)
	//{
	//	fprintf(fp6, "NO");
	//}
	//else
	//{
	//	fprintf(fp6, "YES");
	//}
}

/*
	役割：h[j][0].valuに基づいて、最も優れた個体(ge, valu)を選択して、elite[1] に保存します。
*/
void fitness(gene* g, gene* elite, int se) // 適応度の計算(エリート保存)
{
	int i = 0, j = 0;
	double ave = 0.0;
	double deviation = 0.0;
	double variance = 0.0;

	sum1 = 0; // 初期化

	// エリート保存
	for (i = 0; i < kotai; i++) {
		sum1 += h[i][0].valu;
		if (h[i][0].valu > elite[1].valu) {
			// エリート入れ換え
			elite[1].valu = h[i][0].valu;
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

	elite[3].valu = 0.0; // 初期化
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


	// fprintf(fp4, "%d	%.2f	%.2f	平均：%.2f\n", se, elite[1].valu, elite[2].valu, ave);
	// fprintf(fp4, "%d	最大：%.2f	最小：%.2f　平均値：%.2f  平均偏差値：%.2f\n", se, elite[1].valu, min_value, ave, deviation);
	// printf("%d	%.2f	%.2f	平均：%.2f  平均偏差値：%.2f\n", se, elite[1].valu, min_value, ave, deviation);

}

int roulette() // ルーレット選択
{
	int i = 0, r = 0;
	int num = 0;
	float sum = 0.0;
	float* p;

	p = (float*)malloc(sizeof(int) * kotai);

	sum = 0;
	for (i = 0; i < kotai; i++) {
		sum += h[i][0].valu; // すべての合計
	}
	for (i = 0; i < kotai; i++) {
		p[i] = h[i][0].valu / sum; // 個体適応度 / 群体適応度
	}

	sum = 0;
	r = rand();
	for (i = 0; i < kotai; i++) {
		sum += RAND_MAX * p[i]; // 1
		if (r <= sum) {
			num = i;
			break;
		}
	}
	if (num < 0)	num = roulette(); // エラーのための処理
	free(p);
	return(num);
}

/*
	役割：いまの個体群に基づいて、kotai個の交叉ペア（g[n1], g[n2]）を選択して、
		  交叉処理を行い、そして、得られた kotai 個の新しい個体を更新された個体と見なされます。
*/
void crossover(gene* g) { // 一点交叉
	gene g2[1000] = { 0 }; // 新しい個体群を一時的に格納するための配列
	int num = 0; // 処理中の個体番号。2つずつ処理します。
	// n1, n2: 交叉するために選ばれた2つの親のインデックス
	int n1 = 0;
	int n2 = 0;
	int p = 0; // 交叉を行う遺伝子の位置
	int i, j;

	for (num = 0; num < kotai; num += 2) {
		n1 = rand() % 10;
		n2 = rand() % 10;
		if (rand() <= RAND_MAX * cross) { // 交叉確率を満たす場合
			n1 = roulette();
			n2 = roulette();
			// 乱数の範囲指定公式：(int)( rand() * (最大値 - 最小値 + 1.0) / (1.0 + RAND_MAX) )
			p = (int)(rand() * ((length - 2) - 1 + 1.0) / (1.0 + RAND_MAX) + 1);
			// g[n1], g[n2]: 2つの親　g2[num], g2[num+1]: 2つの子
			// 交叉仕組み：親1の最初のpビットを継承し、残りのビットを親2から継承します
			// 子A
			for (i = 0; i < p; i++) {
				g2[num].ge[i] = g[n1].ge[i];
			}
			for (i = p; i < length; i++) {
				g2[num].ge[i] = g[n2].ge[i];
			}

			// 子B
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

	// 新しい個体群g2を、元の個体群gに上書きして次世代の個体群とします
	for (j = 0; j < kotai; j++) {
		for (i = 0; i < length; i++) {
			g[j].ge[i] = g2[j].ge[i]; // g[]を更新
		}
	}
}

/*
	役割：すべての個体をスキャンして、各個体に対して、変異確率に基づいて、ランダムに選択した
		  ビットを反転操作を行うかどうかを決定します。
*/
void mutation(gene* g) // 突然変異
{
	int num = 0;
	int r = 0;
	int i = 0;
	int p = 0;
	for (num = 0; num < kotai; num++) {
		if (rand() <= RAND_MAX * mut) { // 突然変異確率を満たす場合，1つの遺伝子を選択
			p = (int)(rand() * ((length - 1) + 1.0) / (1.0 + RAND_MAX));
			for (i = 0; i < length; i++) { // 1と0を逆転
				if (i == p) {
					if (g[num].ge[i] == 0) g[num].ge[i] = 1;
					else				g[num].ge[i] = 0;
				}
			}
			p = 0;
		}
	}
}

/*
	役割：すべての個体をスキャンして、最小valu値を持つ個体を見つけます。
		  個体群の中、elite個体とvalu最小個体を交換します。
*/
void elite_back(gene* g, gene* elite) { // エリート個体とvalu最小個体を交換

	int i = 0, j = 0;
	float ave = 0.0;
	float min1 = 1.0;

	tmp = 0; // カウンターの初期化

	for (i = 0; i < kotai; i++) { // 最小値探索
		if (h[i][0].valu < min1) {
			min1 = h[i][0].valu;
			tmp = i;
		}
	}

	for (j = 0; j < length; j++) {
		g[tmp].ge[j] = elite[1].ge[j]; // 最小値とエリートを交換
	}

	h[tmp][0].valu = elite[1].valu; // エリートの評価値と交換
	ave = sum1 / kotai; // 合計値の計算
}

static void on_draw(int event, int x, int y, int flags, void* userdata) {
	Mat image = *((Mat*)userdata); // -> img_origin
	if (event == EVENT_LBUTTONDOWN) {      // マウスの左ボタンを押すとき
		// sp: クリックした位置の座標
		sp.x = x;
		sp.y = y;
		// std::cout << "start point:" << sp << std::endl;
	}
	else if (event == EVENT_LBUTTONUP) { // マウスの左ボタンを離すとき
		ep.x = x;
		ep.y = y;
		// dx, dy: マウスの左ボタンを上げた位置と指した位置の距離
		int dx = abs(ep.x - sp.x);
		int dy = abs(ep.y - sp.y);
		Rect box;
		// 選定してるかどうかを判定: 指した位置と上げた位置が異なる場合、選択した部分を描画します。
		if (dx > 0 && dy > 0) {
			if ((ep.x - sp.x) > 0 && (ep.y - sp.y) > 0) {			 // 左上から初めて　右下まで選定
				box.x = sp.x;
				box.y = sp.y;
				box.width = dx;
				box.height = dy;
				cv::rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);    // 処理区域を選定
			}
			else if ((ep.x - sp.x) > 0 && (ep.y - sp.y) < 0) {		 // 左下から初めて　右上まで選定
				box.x = sp.x;
				box.y = sp.y - dy;
				box.width = dx;
				box.height = dy;
				cv::rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
			}
			else if ((ep.x - sp.x) < 0 && (ep.y - sp.y) > 0) {			 // 右上から初めて　左下まで
				box.x = sp.x - dx;
				box.y = sp.y;
				box.width = dx;
				box.height = dy;
				cv::rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
			}
			else {												 // 右下から初めて　左上まで選定以右下角为起点，左上角为结点框选
				{
					box.x = sp.x - dx;
					box.y = sp.y - dy;
					box.width = dx;
					box.height = dy;
					cv::rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
				}
			}
			img_origincopy.copyTo(image);
			img_targetcopy.copyTo(img_target);
			imshow("鼠标绘制", image);
			// image(box)やimg_target(box)は選択された矩形領域を切り取った部分画像です
			imshow("ROI区域_origin", image(box));
			imshow("ROI区域_target", img_target(box));
			imwrite("./imgs/origin_out.jpg", image(box));
			imwrite("./imgs/target_out.jpg", img_target(box));
			img1 = image(box);
			img2 = img_target(box);
			sp.x = -1;
			sp.y = -1;
		}
	}
	else if (event == EVENT_MOUSEMOVE) { // マウスの移動イベント　選択区域の線は動的に見えるように
		if (sp.x > 0 && sp.y > 0) {
			ep.x = x;
			ep.y = y;
			int dx = abs(ep.x - sp.x);
			int dy = abs(ep.y - sp.y);
			Rect box;
			if ((ep.x - sp.x) > 0 && (ep.y - sp.y) > 0) {
				box.x = sp.x;
				box.y = sp.y;
				box.width = dx;
				box.height = dy;
				img_origincopy.copyTo(image);
				img_targetcopy.copyTo(img_target);
				cv::rectangle(img_origin, box, Scalar(0, 0, 255), 2, 8, 0);
				cv::rectangle(img_target, box, Scalar(0, 0, 255), 2, 8, 0);

			}
			else if ((ep.x - sp.x) > 0 && (ep.y - sp.y) < 0) {
				box.x = sp.x;
				box.y = sp.y - dy;
				box.width = dx;
				box.height = dy;
				img_origincopy.copyTo(image);
				img_targetcopy.copyTo(img_target);
				cv::rectangle(img_origin, box, Scalar(0, 0, 255), 2, 8, 0);
				cv::rectangle(img_target, box, Scalar(0, 0, 255), 2, 8, 0);

			}
			else if ((ep.x - sp.x) < 0 && (ep.y - sp.y) > 0) {
				box.x = sp.x - dx;
				box.y = sp.y;
				box.width = dx;
				box.height = dy;
				img_origincopy.copyTo(image);
				img_targetcopy.copyTo(img_target);
				cv::rectangle(img_origin, box, Scalar(0, 0, 255), 2, 8, 0);
				cv::rectangle(img_target, box, Scalar(0, 0, 255), 2, 8, 0);
			}
			else {
				box.x = sp.x - dx;
				box.y = sp.y - dy;
				box.width = dx;
				box.height = dy;
				img_origincopy.copyTo(image);
				img_targetcopy.copyTo(img_target);
				cv::rectangle(img_origin, box, Scalar(0, 0, 255), 2, 8, 0);
				cv::rectangle(img_target, box, Scalar(0, 0, 255), 2, 8, 0);

			}
			imshow("鼠标绘制", image);
			imshow("鼠标绘制", img_target);
		}
	}
}

/*
	役割：img_originとimg_target2つの画像に対して、動的に範囲を選択して、
	　　　その部分をキャプチャして出力します。( origin_out.jpg, target_out.jpg )
*/
void mouse_drawing_demo(Mat& image, Mat& image_target) {
	namedWindow("鼠标绘制", WINDOW_FREERATIO);
	setMouseCallback("鼠标绘制", on_draw, (void*)(&image));
	imshow("鼠标绘制", image);
	img_origincopy = image.clone();
	img_targetcopy = image_target.clone();
}

int main(void) {
	Mat img1_median, img1_sabun;
	int picture_loading_mode = 0;
	int num; // 今の世代数
	int i = 0, j = 0, k = 0; // for 用変数
	char decision_data[256];
	srand((unsigned)time(NULL));

	gene g[kotai]; // 個体群を格納するための配列
	gene elite[10]; // 最大適応度を持つ個体を格納するための配列
	elite[1].valu = 0.0;
	img_origin = imread("./imgs/oriImg.png");
	img_target = imread("./imgs/maskImg.jpg");
	img_origin = imgResize(img_origin, 512, 512);
	img_target = imgResize(img_target, 512, 512);

	/*
		vector<Mat> images = {img_origin, img_target};
		Mat res;
		hconcat(images, res);
		imgShow("res", res);
	*/

	printf("pic_loading_modeを入力してください。\n");
	printf("------0: new area; 1: old area-------\n");
	scanf("%d", &picture_loading_mode);

	if (picture_loading_mode == 0) { // 新たな区域で実験する時
		// 入力画像に対してのキャプチャ処理
		mouse_drawing_demo(img_origin, img_target);
		waitKey(0);
		destroyAllWindows();
		// img1ths: img1のグレース化画像
		cvtColor(img1, img1ths, COLOR_RGB2GRAY);
		img1_before = img1.clone();
		img2_before = img2.clone();
		imwrite("./imgs/originout_before.jpg", img1_before);
		imwrite("./imgs/targetout_before.jpg", img2_before);
	}
	else {
		// この前キャプチャした部分
		img1 = imread("./imgs/originout_before.jpg"); // ori
		img2 = imread("./imgs/targetout_before.jpg"); // tar
		cvtColor(img1, img1ths, COLOR_RGB2GRAY);
		if (img1.empty())
		{
			printf("画像を読み込みできない");
			return -1;
		}
	}
	// (x, y) -> originout_before
	x = img1.cols; // x方向の画像サイズ
	y = img1.rows; // y方向の画像サイズ
	// (x2, y2) -> targetout_before
	x2 = img2.cols; // x方向の画像サイズ
	y2 = img2.rows; // y方向の画像サイズ

	// ----------MAIN PART------------
	make(g);
	for (num = 1; num <= sedai; num++) {
		printf("-------第%d世代：-------\n", num);

		phenotype(g); // S2_1
		for (k = 0; k < kotai; k++) {
			// h 配列の中の決定変数を変換して宣言した変数に代入します。
			import_para(k);

			/* 役割？
			sprintf_s(decision_data, "%d", fsize);
			sprintf_s(decision_data + strlen(decision_data), sizeof(decision_data) - strlen(decision_data), ",%d", binary);
			*/

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

			/* 役割？
			if (abusolute_flag == 0)
			{
				fprintf(fp6, ",abusolute");
			}
			else
			{
				fprintf(fp6, ",non-abusolute");
			}
			*/

			// img1ths: 元のグレースケール画像, img1_median: ブラー画像
			// img1_sabun: img1thsとimg1_medianの差分結果を保存するための Mat
			//             ブラー処理前後の比較するため・・・
			img1_sabun = sabun(img1ths, img1_median);

			/* 役割？
			waitKey(0);
			*/
			threshold(img1_sabun, img1_sabun, binary, 255, THRESH_BINARY); // 二値化処理
			if (pixellabelingmethod == 0)
			{
				// printf("pixelmehod's value:%d\n", pixellabelingmethod);
				// img_label: ラベリングの結果を保存します。（白：検出したブロブ）
				img_label = labeling_new4(img1_sabun, linear); // ラベリング+ノイズ除去
				// fprintf(fp6, ",4pixel");
			}
			else
			{
				img_label = labeling_new8(img1_sabun, linear); // ラベリング+ノイズ除去
				// fprintf(fp6, ",8pixel");
			}
			pixellabelingmethod = 0; // ０に戻す
			bitwise_not(img_label, img_bitwise); // 白黒反転処理
			if (erodedilate_sequence == 0)
			{
				// fprintf(fp6, ",dilatefirst");
				if (erodedilate_times != 0) {
					for (i = 0; i < erodedilate_times; i++)
					{
						// CLOSING
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
						// OPENING (BETTER)
						img_bitwise = erode_dilate(img_bitwise);
					}
				}
				// fprintf(fp6, ",%d,", erodedilate_times);
			}
			//vector<Mat> images = { img2, img_bitwise };
			//Mat res;
			//hconcat(images, res);
			//imgShow("res_p1", res);

			noiz_kessonFvalue(k, num);
		}
		esedai = 0;
		ekotai = 0;
		evalu = 0.0;
		// S2_3: 計算された個体(valu)の中、最適な個体をエリートに保存します。
		fitness(g, elite, num); // 適応度の計算(エリート保存)
		// S3 -> S4
		crossover(g); // 一点交叉(roulette操作を含む)
		mutation(g); // 突然変異
		// S6
		elite_back(g, elite); // エリート個体と適応度最小個体を交換
	}
	vector<Mat> images = { img1, img2, img_bitwise };
	Mat res;
	hconcat(images, res);
	imgShow("res_p1", res);
	imwrite("./imgs/final_opt.jpg", res);
}