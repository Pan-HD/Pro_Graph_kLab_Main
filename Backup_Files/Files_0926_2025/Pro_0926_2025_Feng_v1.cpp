#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace std;
using namespace cv;



// ���C���֐�
int main(void)
{

    //Mat Src1 = imread("�g����_��4.jpeg");
    Mat Src1 = imread("13.png");
    //cv::Mat mat = (cv::Mat_<double>(2, 3) << 1, 0, 4032 / 2 -320, 0, 1, 1512 / 2 - 240);
    //cv::warpAffine(Src1, Src1, mat, cv::Size(4032, 1512), CV_INTER_LINEAR, cv::BORDER_TRANSPARENT);
    //Mat Src4 = imread("�{�d��.jpeg");
    //cv::imwrite("�^��_�ϊ���.png", Src1);

    Mat Src2 = imread("1007�w�����x����_11��_ORB_90_����.png");
    //cv::Mat mat = (cv::Mat_<double>(2, 3) << 1, 0, 4032 / 2 -320, 0, 1, 1512 / 2 - 240);
    //cv::warpAffine(Src1, Src1, mat, cv::Size(4032, 1512), CV_INTER_LINEAR, cv::BORDER_TRANSPARENT);
    Mat image, image1;
    //cv::Mat dst1 = AddNoise(Src1);
    //cv::Mat dst2 = AddNoise(Src2);
    //cv::imwrite("�{1_�C���p���X�m�C�Y_10%.png", dst1);
    //cv::imwrite("�{�d��_�C���p���X�m�C�Y_10%.png", dst2);

    // �摜�̕���\������
    int width = Src2.cols;

    // �摜�̍�����\������
    int height = Src2.rows;

    //�L�[�|�C���g���o�Ɠ����ʋL�q

    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    Mat descriptors1, descriptors2;
    cv::Ptr<cv::ORB> akaze = cv::ORB::create();
    akaze->detectAndCompute(Src1, cv::Mat(), keypoints1, descriptors1);
    akaze->detectAndCompute(Src2, cv::Mat(), keypoints2, descriptors2);



    //�}�b�`���O(knn�}�b�`���O)
    vector<vector<cv::DMatch>> knnmatch_points;
    cv::BFMatcher match(cv::NORM_HAMMING);
    match.knnMatch(descriptors1, descriptors2, knnmatch_points, 2);

    //�Ή��_���i��
    const double match_par = 0.60; //���_���c���ꍇ�̂������l
    vector<cv::DMatch> goodMatch;
    //KeyPoint -> Point2d
    vector<cv::Point2f> match_point1;
    vector<cv::Point2f> match_point2;
    for (size_t i = 0; i < knnmatch_points.size(); ++i) {
        double distance1 = knnmatch_points[i][0].distance;
        double distance2 = knnmatch_points[i][1].distance;


        //�����_���狗���l������Ă���_�̂ݒ��o�i�����_�����c���j
        if (distance1 <= distance2 * match_par) {
            goodMatch.push_back(knnmatch_points[i][0]);
            match_point1.push_back(keypoints1[knnmatch_points[i][0].queryIdx].pt);
            match_point2.push_back(keypoints2[knnmatch_points[i][0].trainIdx].pt);
        }
    }

    for (size_t i = 0; i < goodMatch.size(); ++i) {
        double distance1 = knnmatch_points[i][0].distance;
        double distance2 = knnmatch_points[i][1].distance;


        //�����_���狗���l������Ă���_�̂ݒ��o�i�����_�����c���j
        if (distance1 <= distance2 * match_par) {
            goodMatch.push_back(knnmatch_points[i][0]);
            match_point1.push_back(keypoints1[knnmatch_points[i][0].queryIdx].pt);
            match_point2.push_back(keypoints2[knnmatch_points[i][0].trainIdx].pt);
        }
    }

    //�z���O���t�B�s�񐄒�
    cv::Mat masks;
    cv::Mat H = cv::findHomography(match_point1, match_point2, masks, cv::RANSAC);
    //cv::Mat H = cv::findHomography(match_point1, match_point2, masks, cv::LMEDS);
    //cv::Mat H = cv::findHomography(match_point1, match_point2, masks);
    warpPerspective(Src1, image, H, cv::Size(width, height));

    //RANSAC�Ŏg��ꂽ�Ή��_�̂ݒ��o
    vector<cv::DMatch> inlinerMatch;
    for (size_t i = 0; i < masks.rows; ++i) {
        uchar* inliner = masks.ptr<uchar>(i);
        if (inliner[0] == 1) {
            inlinerMatch.push_back(goodMatch[i]);
        }

    }


    //�Ή��_�̕\��
    cv::Mat drawmatch;
    cv::drawMatches(Src1, keypoints1, Src2, keypoints2, goodMatch, drawmatch);
    imwrite("1007�w�����x����_1��_ORB_90_match_point.jpg", drawmatch);

    //�C�����C�A�̑Ή��_�̂ݕ\��
    cv::Mat drawMatch_inliner;
    cv::drawMatches(Src1, keypoints1, Src2, keypoints2, inlinerMatch, drawMatch_inliner);
    imwrite("1007�w�����x����_1��_ORB_90_match_inliner.jpg", drawMatch_inliner);

    //imshow("DrawMatch", drawmatch);
    //imshow("Inliner", drawMatch_inliner);
    imwrite("1007�w�����x����_1��_ORB_90_�ϊ���.png", image);

    //Mat image2 = imread("307���@��2_5��_�ϊ���.png");

    cv::threshold(image, image1, 1, 255, cv::THRESH_BINARY);
    imwrite("1007�w�����x����_1��_ORB_90_mask.png", image1);

    //Mat image3 = imread("307���@��2_5��_mask.png");

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int B1 = image1.at<Vec3b>(y, x)[0];
            int G1 = image1.at<Vec3b>(y, x)[1];
            int R1 = image1.at<Vec3b>(y, x)[2];
            if (B1 != 0 && G1 != 0 && R1 != 0) {
                int B2 = Src2.at<Vec3b>(y, x)[0];
                int G2 = Src2.at<Vec3b>(y, x)[1];
                int R2 = Src2.at<Vec3b>(y, x)[2];
                if (B2 == 0 && G2 == 0 && R2 == 0) {
                    int B = image.at<Vec3b>(y, x)[0];
                    int G = image.at<Vec3b>(y, x)[1];
                    int R = image.at<Vec3b>(y, x)[2];
                    Src2.at<Vec3b>(y, x)[0] = B;
                    Src2.at<Vec3b>(y, x)[1] = G;
                    Src2.at<Vec3b>(y, x)[2] = R;
                }
            }

        }

    }

    /*
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int B1 = image3.at<Vec3b>(y, x)[0];
            int G1 = image3.at<Vec3b>(y, x)[1];
            int R1 = image3.at<Vec3b>(y, x)[2];
            if (B1 != 0 && G1 != 0 && R1 != 0) {
                int B = image2.at<Vec3b>(y, x)[0];
                int G = image2.at<Vec3b>(y, x)[1];
                int R = image2.at<Vec3b>(y, x)[2];
                Src2.at<Vec3b>(y, x)[0] = B;
                Src2.at<Vec3b>(y, x)[1] = G;
                Src2.at<Vec3b>(y, x)[2] = R;
            }

        }

    }
    */

    //cv::bitwise_and(Src2, Src2, image, image1);

    //imshow("result", Src2);

    cv::imwrite("1007�w�����x����_12��_ORB_90_����.png", Src2);


    cv::waitKey(0);

    return 0;
}