#include <opencv2/opencv.hpp>
  #include <torch/script.h>

  torch::Tensor warp_perspective(torch::Tensor image, torch::Tensor warp) {
    cv::Mat image_mat(image.size(0), image.size(1), CV_32FC1, image.data<float>());
    // cv::Mat image_mat(image.sizes().vec(), CV_32FC1, image.data<float>());
    cv::Mat warp_mat(3, 3, CV_32FC1, warp.data<float>());

    cv::Mat output_mat;
    cv::warpPerspective(image_mat, output_mat, warp_mat, {64, 64});

    return torch::from_blob(output.ptr<float>(), {64, 64}).clone();
  }
