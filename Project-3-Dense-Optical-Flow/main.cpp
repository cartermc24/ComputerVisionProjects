#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/FrameReader.h"
#include "utils/ImageShower.h"
#include <unistd.h>
#include <tuple>
#include <algorithm>
#include <random>

std::pair<cv::Mat, cv::Mat> lukas_kanade(const cv::Mat &image1, const cv::Mat &image2);
void display_optical_flow(const cv::Mat &image, const cv::Mat &u_vectors, const cv::Mat &v_vectors);

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: ./CVProject3 [image_dir]" << std::endl;
        return 0;
    }
    std::string image_dir(argv[1]);
    std::cout << "This is Project 3: Dense Optical Flow" << std::endl;
    FrameReader reader(image_dir + "/");

    ImageShower shower1("img1");
    ImageShower shower2("img2");

    cv::Mat i1 = reader.getNextFrame();
    cv::Mat i2 = reader.getNextFrame();

    std::pair<cv::Mat, cv::Mat> flow_vectors = lukas_kanade(i1, i2);
    display_optical_flow(i1, flow_vectors.first, flow_vectors.second);

    return 0;
}

void display_optical_flow(const cv::Mat &image, const cv::Mat &u_vectors, const cv::Mat &v_vectors) {
    // <<< Equivalent of the Matlab quiver function >>>
    double_t QUIVER_THRESHOLD = 0.8;

    cv::Mat dsp_image = image.clone();
    for (uint32_t row = 0; row < image.rows; row += 5) {
        for (uint32_t col = 0; col < image.cols; col += 5) {
            if (abs(u_vectors.at<double_t>(row, col)) < QUIVER_THRESHOLD && abs(v_vectors.at<double_t>(row, col)) < QUIVER_THRESHOLD) {
                continue;
            }

            cv::Point src_point(row, col);
            cv::Point dst_point((int)std::round(row+u_vectors.at<double_t>(row, col)),
                                (int)std::round(col+v_vectors.at<double_t>(row, col)));

            cv::arrowedLine(dsp_image, src_point, dst_point, cv::Scalar(255, 0, 0), 1);
        }
    }

    ImageShower shower("Quiver");
    shower.showImage(dsp_image);
}

std::pair<cv::Mat, cv::Mat> lukas_kanade(const cv::Mat &image1, const cv::Mat &image2) {
    cv::Mat grayImg1, grayImg2;
    cv::cvtColor(image1, grayImg1, CV_BGR2GRAY);
    cv::cvtColor(image2, grayImg2, CV_BGR2GRAY);
    grayImg1.convertTo(grayImg1, CV_64FC1, 1.0/255.0);
    grayImg2.convertTo(grayImg2, CV_64FC1, 1.0/255.0);

    // Smooth the images
    const double_t SMOOTHING_SIGMA = 1;
    cv::Mat smooth1, smooth2;
    cv::GaussianBlur(grayImg1, smooth1, cv::Size(), SMOOTHING_SIGMA);
    cv::GaussianBlur(grayImg2, smooth2, cv::Size(), SMOOTHING_SIGMA);
    grayImg1 = smooth1;
    grayImg2 = smooth2;

    // Get spacial intensity gradients for both images in x and y directions
    cv::Mat space_int_1_x, space_int_1_y, space_int_2_x, space_int_2_y;
    cv::Sobel(grayImg1, space_int_1_x, CV_64F, 1, 0);
    cv::Sobel(grayImg1, space_int_1_y, CV_64F, 0, 1);
    cv::Sobel(grayImg2, space_int_2_x, CV_64F, 1, 0);
    cv::Sobel(grayImg2, space_int_2_y, CV_64F, 0, 1);

    // Calculate the temporal gradients for both images
    cv::Mat temporal_deriv(image1.rows, image2.cols, CV_64FC1, cv::Scalar(0));
    for (int row = 0; row < image1.rows; row++) {
        for (int cols = 0; cols < image1.cols; cols++) {
            temporal_deriv.at<double_t>(row, cols) = grayImg2.at<double_t>(row, cols) - grayImg1.at<double_t>(row, cols);
        }
    }

    // Generate LK matrix elements
    const uint32_t WINDOW_SIZE = 5;
    const uint32_t anchor_pt = (uint8_t)floor(WINDOW_SIZE/2);

    cv::Mat Ix2(image1.rows, image2.cols, CV_64FC1, cv::Scalar(0)),
            IxIy(image1.rows, image2.cols, CV_64FC1, cv::Scalar(0)),
            Iy2(image1.rows, image2.cols, CV_64FC1, cv::Scalar(0)),
            IxIt(image1.rows, image2.cols, CV_64FC1, cv::Scalar(0)),
            IyIt(image1.rows, image2.cols, CV_64FC1, cv::Scalar(0));

    for (uint32_t row = anchor_pt; row < image1.rows-anchor_pt; row++) {
        for (uint32_t col = anchor_pt; col < image1.cols - anchor_pt; col++) {
            uint32_t f_row_start = row - anchor_pt;
            uint32_t f_col_start = col - anchor_pt;

            double_t ix2sum = 0;
            double_t iy2sum = 0;
            double_t ixiysum = 0;
            double_t ixitsum = 0;
            double_t iyitsum = 0;

            for (uint8_t f_row = 0; f_row < WINDOW_SIZE; f_row++) {
                for (uint8_t f_col = 0; f_col < WINDOW_SIZE; f_col++) {
                    ix2sum += pow(space_int_2_x.at<double_t>(f_row_start+f_row, f_col_start+f_col), 2);
                    iy2sum += pow(space_int_2_y.at<double_t>(f_row_start+f_row, f_col_start+f_col), 2);
                    ixiysum += space_int_2_x.at<double_t>(f_row_start+f_row, f_col_start+f_col) * \
                               space_int_2_y.at<double_t>(f_row_start+f_row, f_col_start+f_col);
                    ixitsum += space_int_2_x.at<double_t>(f_row_start+f_row, f_col_start+f_col) * \
                               temporal_deriv.at<double_t>(f_row_start+f_row, f_col_start+f_col);
                    iyitsum += space_int_2_y.at<double_t>(f_row_start+f_row, f_col_start+f_col) * \
                               temporal_deriv.at<double_t>(f_row_start+f_row, f_col_start+f_col);
                }
            }

            Ix2.at<double_t>(row, col) = ix2sum;
            Iy2.at<double_t>(row, col) = iy2sum;
            IxIy.at<double_t>(row, col) = ixiysum;
            IxIt.at<double_t>(row, col) = ixitsum;
            IyIt.at<double_t>(row, col) = iyitsum;
        }
    }

    // Solve for flow vectors u and v
    cv::Mat u_flow_vector(image1.rows, image2.cols, CV_64FC1, cv::Scalar(0)),
            v_flow_vector(image1.rows, image2.cols, CV_64FC1, cv::Scalar(0));
    cv::Mat l_matrix(2, 2, CV_64FC1),
            r_matrix(2, 1, CV_64FC1);

    for (uint32_t row = anchor_pt; row < image1.rows-anchor_pt; row++) {
        for (uint32_t col = anchor_pt; col < image1.cols - anchor_pt; col++) {
            l_matrix.at<double_t>(0, 0) = Ix2.at<double_t>(row, col);
            l_matrix.at<double_t>(0, 1) = IxIy.at<double_t>(row, col);
            l_matrix.at<double_t>(1, 0) = IxIy.at<double_t>(row, col);
            l_matrix.at<double_t>(1, 1) = Iy2.at<double_t>(row, col);

            r_matrix.at<double_t>(0, 0) = -IxIt.at<double_t>(row, col);
            r_matrix.at<double_t>(1, 0) = -IyIt.at<double_t>(row, col);

            cv::Mat l_matrix_inv = l_matrix.inv(cv::DECOMP_SVD);
            cv::Mat uv_mat = l_matrix_inv * r_matrix;

            u_flow_vector.at<double_t>(row, col) = uv_mat.at<double_t>(0, 0);
            v_flow_vector.at<double_t>(row, col) = uv_mat.at<double_t>(1, 0);
        }
    }

    return std::make_pair(u_flow_vector, v_flow_vector);
}