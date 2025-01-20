#include <opencv2/opencv.hpp>

void createImageWithCircle(int width, int height, int radius) {
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Point center(width / 2, height / 2);
    cv::circle(image, center, radius, cv::Scalar(255, 255, 255), -1);
    cv::imshow("Image with Circle", image);
    cv::waitKey(0);
    cv::imwrite("./imgs_1104_v1/output.png", image);
}

int main() {
    int width = 116;
    int height = 118;
    int radius = 50;

    createImageWithCircle(width, height, radius);

    return 0;
}
