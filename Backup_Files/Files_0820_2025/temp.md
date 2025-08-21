### Diff type of Functions

---

- ブラー処理

  - ```
    cv::GaussianBlur(src, dst, Size(5,5), 1.5);
    cv::medianBlur(src, dst, 5);
    cv::blur(src, dst, Size(3,3));
    cv::bilateralFilter(src, dst, 9, 75, 75);
    ```

- エッジ検出・差分処理

  - ```
    cv::Sobel(src, dst, CV_8U, 1, 0, 3);
    cv::Sobel(src, dst, CV_8U, 0, 1, 3);
    cv::Canny(src, dst, 100, 200);
    Mat diffProcess(const Mat postImg, const Mat preImg);
    ```

- 二値化処理

  - ```
    cv::threshold(src, dst, 9, 255, cv::THRESH_BINARY);
    cv::threshold(src, dst, 31, 255, cv::THRESH_BINARY);
    cv::threshold(src, dst, 63, 255, cv::THRESH_BINARY);
    cv::threshold(src, dst, 127, 255, cv::THRESH_BINARY);
    ```

- モルフォロジー

  - ```
    cv::erode(src, dst, kernel);
    cv::dilate(src, dst, kernel);
    ```

- 特徴抽出

  - ```
    Mat conPro_singleTime(const Mat& img);
    ```

- 論理・算術演算

  - ```
    cv::bitwise_and(img1, img2, dst);
    cv::bitwise_or(img1, img2, dst);
    cv::bitwise_not(img, dst);
    cv::bitwise_xor(img1, img2, dst);
    ```