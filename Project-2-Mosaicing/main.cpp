#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/FrameReader.h"
#include "utils/ImageShower.h"
#include <unistd.h>
#include <tuple>
#include <algorithm>

std::vector<std::tuple<cv::Point, cv::Point, double_t>> get_normalized_correlation(cv::Mat F, cv::Mat G, uint8_t window_size);
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

    ImageShower frame1("F");
    ImageShower frame2("G");

    cv::Mat scaled1, scaled2;
    cv::resize(reader.getNextFrame(), scaled1, cv::Size(), 0.25, 0.25);
    cv::resize(reader.getNextFrame(), scaled2, cv::Size(), 0.25, 0.25);

    std::cout << "Running harris on image 1" << std::endl;
    cv::Mat image1 = harris_corner_detector(scaled1, 3);
    std::cout << "Running harris on image 2" << std::endl;
    cv::Mat image2 = harris_corner_detector(scaled2, 3);

    std::cout << "Running ncorr on image 1 & 2" << std::endl;
    std::vector<std::tuple<cv::Point, cv::Point, double_t>> corr_pts = get_normalized_correlation(image1, image2, 3);

    frame1.showImage(image1);
    frame2.showImage(image2);

    for (auto corr : corr_pts) {
        std::cout << "-> Ncorr: F: " << std::get<0>(corr) << " G:" << std::get<1>(corr) << " Val: " << std::get<2>(corr) << std::endl;
    }



    return 0;
}

std::vector<std::tuple<cv::Point, cv::Point, double_t>> get_normalized_correlation(cv::Mat F, cv::Mat G, uint8_t window_size) {
    auto anchor_pt = (uint8_t)floor(window_size/2);

    // ------------- Find Normalized Correlation between f and g ----------------
    // Find Fhat and Ghat
    cv::Mat Fhat(F.rows, F.cols, CV_64FC1, cv::Scalar(0));
    cv::Mat Ghat(G.rows, G.cols, CV_64FC1, cv::Scalar(0));
    // Fhat
    std::cout << "\t-> Creating Fhat for ncorr" << std::endl;
#pragma omp parallel for
    for (uint32_t x = anchor_pt; x < F.rows-anchor_pt; x++) {
        for (uint32_t y = anchor_pt; y < F.cols-anchor_pt; y++) {
            uint32_t f_x_start = x - anchor_pt;
            uint32_t f_y_start = y - anchor_pt;

            double_t fhat_val = 0;
            for (uint32_t f_x = 0; f_x < F.rows; f_x++) {
                for (uint32_t f_y = 0; f_y < F.cols; f_y++) {
                    fhat_val += pow(F.at<uint8_t>(f_x_start+f_x, f_y_start+f_y), 2);
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
    std::cout << "\t-> Creating Ghat for ncorr" << std::endl;
#pragma omp parallel for
    for (uint32_t x = anchor_pt; x < G.rows-anchor_pt; x++) {
        for (uint32_t y = anchor_pt; y < G.cols-anchor_pt; y++) {
            uint32_t g_x_start = x - anchor_pt;
            uint32_t g_y_start = y - anchor_pt;

            double_t ghat_val = 0;
            for (uint32_t g_x = 0; g_x < G.rows; g_x++) {
                for (uint32_t g_y = 0; g_y < G.cols; g_y++) {
                    ghat_val += pow(G.at<uint8_t>(g_x_start+g_x, g_y_start+g_y), 2);
                }
            }

            if (ghat_val > 0) {
                Ghat.at<double_t>(x, y) = G.at<uint8_t>(x, y) / sqrt(ghat_val);
            } else {
                Ghat.at<double_t>(x, y) = 0;
            }
        }
    }

    std::vector<std::tuple<cv::Point, cv::Point, double_t>> corr_points;
    // Ignore borders
    // Iterate through all points of G
    std::cout << "\t-> Starting ncorr scan" << std::endl;
    for (uint32_t g_x = anchor_pt; g_x < Ghat.rows-anchor_pt; g_x++) {
        for (uint32_t g_y = anchor_pt; g_y < Ghat.cols-anchor_pt; g_y++) {
            cv::Mat g_region(Ghat, cv::Rect(g_y-anchor_pt, g_x-anchor_pt, window_size, window_size));

            double_t max_ncorr_val = 0;
            uint32_t max_f_x = 0, max_f_y = 0;

            // Iterate through all points of F
            for (uint32_t f_x = anchor_pt; f_x < Fhat.rows - anchor_pt; f_x++) {
                for (uint32_t f_y = anchor_pt; f_y < Fhat.cols - anchor_pt; f_y++) {

                    uint32_t f_x_start = f_x - anchor_pt;
                    uint32_t f_y_start = f_y - anchor_pt;

                    // Do the ncorr operation on R[g] and R[f]
                    double_t ncorr_val = 0;
                    for (uint32_t scan_x = 0; scan_x < g_region.rows; scan_x++) {
                        for (uint32_t scan_y = 0; scan_y < g_region.cols; scan_y++) {
                            ncorr_val += Fhat.at<double_t>(f_x_start + scan_x, f_y_start + scan_y) \
                                       * g_region.at<double_t>(scan_x, scan_y);
                        }
                    }

                    if (ncorr_val > max_ncorr_val) {
                        max_f_x = f_x;
                        max_f_y = f_y;
                        max_ncorr_val = ncorr_val;
                    }
                }
            }

            if (max_ncorr_val > 0) {
                cv::Point f_point(max_f_x, max_f_y);
                cv::Point g_point(g_x, g_y);

                corr_points.emplace_back(f_point, g_point, max_ncorr_val);
            }
        }
    }


    std::cout << "\t-> Found num ncorrs: " << corr_points.size() << std::endl;
    return corr_points;
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