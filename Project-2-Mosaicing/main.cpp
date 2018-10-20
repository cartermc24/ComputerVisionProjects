#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/FrameReader.h"
#include "utils/ImageShower.h"
#include <unistd.h>

cv::Mat get_normalized_correlation(cv::Mat F, cv::Mat G);
cv::Mat harris_corner_detector(cv::Mat img, uint8_t window_size);

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: ./CVProject2 [image_dir]" << std::endl;
        return 0;
    }
    std::string image_dir(argv[1]);
    std::cout << "This is Project 2: Mosaicing" << std::endl;
    FrameReader reader(image_dir + "/");

    /*
    cv::Mat img = reader.getNextFrame(), img2;
    cv::cvtColor(img, img2, cv::COLOR_BGR2GRAY, img.type());
    cv::Mat corrnerdHarri(img.rows, img.cols, CV_32FC1, cv::Scalar(0));
    cv::cornerHarris(img2, corrnerdHarri, 3, 3, 0.04);
    corrnerdHarri *= 1000000;
    std::cout << corrnerdHarri << std::endl;
    ImageShower shower("Harris");
    shower.showImage(corrnerdHarri);
    */

    cv::Mat output = harris_corner_detector(reader.getNextFrame(), 3);

    std::cout << output << std::endl;

    return 0;
}

cv::Mat get_normalized_correlation(cv::Mat F, cv::Mat G) {
    auto g_anchor_pt = (uint8_t)floor(G.rows/2);

    // ------------- Find Normalized Correlation between f and g ----------------
    // Find Fhat and Ghat
    cv::Mat Fhat(F.rows, F.cols, CV_64FC1, cv::Scalar(0));
    cv::Mat Ghat(G.rows, G.cols, CV_64FC1, cv::Scalar(0));
    // Fhat
    for (uint32_t x = g_anchor_pt; x < F.rows-g_anchor_pt; x++) {
        for (uint32_t y = g_anchor_pt; y < F.cols-g_anchor_pt; y++) {
            // Run "corr convolution" over F
            uint32_t f_x_start = x - g_anchor_pt;
            uint32_t f_y_start = y - g_anchor_pt;

            double_t fhat_val = 0;
            for (uint32_t g_x = 0; g_x < G.rows; g_x++) {
                for (uint32_t g_y = 0; g_y < G.cols; g_y++) {
                    fhat_val += pow(F.at<uint8_t>(f_x_start+g_x, f_y_start+g_y), 2);
                }
            }

            if (fhat_val > 0) {
                Fhat.at<double_t>(x, y) = F.at<uint8_t>(x, y) / sqrt(fhat_val);
            } else {
                Fhat.at<double_t>(x, y) = 0;
            }
        }
    }

    // Ghat
    double ghat_val = 0;
    for (uint32_t g_x = 0; g_x < G.rows; g_x++) {
        for (uint32_t g_y = 0; g_y < G.cols; g_y++) {
            ghat_val += pow(G.at<uint8_t>(g_x, g_y), 2);
        }
    }
    ghat_val = sqrt(ghat_val);
    for (uint32_t g_x = 0; g_x < G.rows; g_x++) {
        for (uint32_t g_y = 0; g_y < G.cols; g_y++) {
            Ghat.at<double_t>(g_x, g_y) = G.at<uint8_t>(g_x, g_y) / ghat_val;
        }
    }

    cv::Mat ncorr(F.rows, F.cols, CV_64FC1, cv::Scalar(0));
    // Ignore borders
    for (uint32_t x = g_anchor_pt; x < F.rows-g_anchor_pt; x++) {
        for (uint32_t y = g_anchor_pt; y < F.cols-g_anchor_pt; y++) {
            // Run "corr convolution" over F
            uint32_t f_x_start = x - g_anchor_pt;
            uint32_t f_y_start = y - g_anchor_pt;

            double_t ncorr_val = 0;
            for (uint32_t g_x = 0; g_x < G.rows; g_x++) {
                for (uint32_t g_y = 0; g_y < G.cols; g_y++) {
                    ncorr_val += Fhat.at<double_t>(f_x_start+g_x, f_y_start+g_y) \
                               * Ghat.at<double_t>(g_x, g_y);
                }
            }

            ncorr.at<double_t>(x, y) = ncorr_val;
        }
    }
    return ncorr;
}

cv::Mat harris_corner_detector(cv::Mat img, uint8_t window_size) {
    /********* This function isn't tested yet **************/
    const double_t K = 0.05;
    const double_t R_thresh = 0.5;
    const double_t R_thresh_maxval = 1000;

    // Convert to gray if image is color
    if (img.channels() > 1) {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY, img.type());
        img = gray;
    }

    uint8_t b_anchor = (uint8_t)floor(window_size / 2);

    cv::Mat harris_r(img.rows, img.cols, CV_64F, cv::Scalar(0));
    for (uint32_t x = b_anchor; x < (img.rows - window_size); x++) {
        for (uint32_t y = b_anchor; y < (img.cols - window_size); y++) {
            cv::Mat subimage(img, cv::Rect(y-b_anchor, x-b_anchor, window_size, window_size));

            cv::Mat img_sobel_x, img_sobel_y;
            cv::Sobel(subimage, img_sobel_x, CV_64F, 1, 0);
            cv::Sobel(subimage, img_sobel_y, CV_64F, 0, 1);

            double_t grad_x_sum = 0, grad_y_sum = 0, grad_xy_sum = 0;
            for (int i = 0; i < window_size; i++) {
                for (int j = 0; j < window_size; j++) {
                    grad_x_sum += pow(img_sobel_x.at<double_t>(i, j), 2);
                    grad_y_sum += pow(img_sobel_y.at<double_t>(i, j), 2);
                    grad_xy_sum += img_sobel_x.at<double_t>(i, j) * img_sobel_y.at<double_t>(i, j);
                }
            }

            // Average values across window
            grad_x_sum /= window_size*window_size;
            grad_y_sum /= window_size*window_size;
            grad_xy_sum /= window_size*window_size;

            cv::Mat C(2, 2, CV_64FC1, cv::Scalar(0));
            C.at<double_t>(0, 0) = grad_x_sum;
            C.at<double_t>(1, 1) = grad_y_sum;
            C.at<double_t>(0, 1) = grad_xy_sum;
            C.at<double_t>(1, 0) = grad_xy_sum;

            harris_r.at<double_t>(x, y) = cv::determinant(C) - K*pow(cv::trace(C)[0], 2);
        }
    }

    cv::Mat harris_r_thresh;
    cv::threshold(harris_r, harris_r_thresh, 10000, std::numeric_limits<double_t>::max(), cv::THRESH_TOZERO);

    /************* Non-max suppression **************/
    cv::Mat post_supression(harris_r_thresh.rows, harris_r_thresh.cols, CV_64FC1, cv::Scalar(0));
    for (uint32_t x = b_anchor; x < (harris_r_thresh.rows - window_size); x++) {
        for (uint32_t y = b_anchor; y < (harris_r_thresh.cols - window_size); y++) {
            cv::Mat subimage(img, cv::Rect(y-b_anchor, x-b_anchor, window_size, window_size));

            double_t val_at_pos = harris_r_thresh.at<double_t>(x, y);
            bool found_greater = false;
            for (int i = 0; i < window_size; i++) {
                for (int j = 0; j < window_size; j++) {
                    if (subimage.at<double_t>(i, j) > val_at_pos) {
                        found_greater = true;
                        i = window_size;
                        break;
                    }
                }
            }

            if (found_greater) {
                post_supression.at<double_t>(x, y) = 0;
            } else {
                post_supression.at<double_t>(x, y) = harris_r_thresh.at<double_t>(x, y);
            }
        }
    }

    return post_supression;
}