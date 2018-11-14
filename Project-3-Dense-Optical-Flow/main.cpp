#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/FrameReader.h"
#include "utils/ImageShower.h"
#include <unistd.h>
#include <tuple>
#include <algorithm>
#include <random>

std::pair<cv::Mat, cv::Mat> lukas_kanade(const cv::Mat &image1, const cv::Mat &image2);
void display_optical_flow(const cv::Mat &image, const cv::Mat &u_vectors, const cv::Mat &v_vectors, double_t scale);
void display_optical_flow_hsv(const cv::Mat &image, const cv::Mat &u_vectors, const cv::Mat &v_vectors, double_t scale);
void display_optical_flow_vectors(const cv::Mat &u, const cv::Mat &v, double_t scale);

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

    cv::Mat quarter_res1, quarter_res2;
    cv::resize(i1, quarter_res1, cv::Size(0, 0), 0.25, 0.25);
    cv::resize(i2, quarter_res2, cv::Size(0, 0), 0.25, 0.25);

    // Base pyramid level
    std::pair<cv::Mat, cv::Mat> flow_vectors = lukas_kanade(i1, i2);
    shower1.showImage(i1);
    display_optical_flow_vectors(flow_vectors.first, flow_vectors.second, 1);
    display_optical_flow(i2, flow_vectors.first, flow_vectors.second, 1);
    display_optical_flow_hsv(i1, flow_vectors.first, flow_vectors.second, 1);

    // Second pyramid level
    flow_vectors = lukas_kanade(quarter_res1, quarter_res2);
    shower1.showImage(quarter_res1);
    display_optical_flow_vectors(flow_vectors.first, flow_vectors.second, 0.25);
    display_optical_flow(quarter_res2, flow_vectors.first, flow_vectors.second, 0.25);
    display_optical_flow_hsv(quarter_res2, flow_vectors.first, flow_vectors.second, 0.25);

    return 0;
}

void display_optical_flow_vectors(const cv::Mat &u, const cv::Mat &v, double_t scale) {
    // Display the vectors
    ImageShower u_s("U Vector");
    ImageShower v_s("V Vector");

    cv::Mat u_scaled, v_scaled;
    if (scale != 1) {
        cv::resize(u, u_scaled, cv::Size(0, 0), 1/scale, 1/scale);
        cv::resize(v, v_scaled, cv::Size(0, 0), 1/scale, 1/scale);
    } else {
        u_scaled = u;
        v_scaled = v;
    }

    u_s.showImage(u_scaled);
    v_s.showImage(v_scaled);
}

void display_optical_flow(const cv::Mat &image, const cv::Mat &u_vectors, const cv::Mat &v_vectors, double_t scale) {
    // <<< Equivalent of the Matlab quiver function >>>
    double_t QUIVER_THRESHOLD = 0;

    double_t adjustment_factor = 1/scale;
    cv::Mat dsp_image = image.clone();
    if (scale != 1) {
        cv::Mat resized;
        cv::resize(dsp_image, resized, cv::Size(0, 0), adjustment_factor, adjustment_factor);
        dsp_image = resized;
    }

    uint32_t jump_figure = scale == 1 ? 5 : 1;

    for (uint32_t row = 0; row < image.rows; row += jump_figure) {
        for (uint32_t col = 0; col < image.cols; col += jump_figure) {
            if (abs(u_vectors.at<double_t>(row, col)) < QUIVER_THRESHOLD && abs(v_vectors.at<double_t>(row, col)) < QUIVER_THRESHOLD) {
                continue;
            }

            uint32_t row_adj = (uint32_t)(row * adjustment_factor);
            uint32_t col_adj = (uint32_t)(col * adjustment_factor);

            cv::Point src_point(col_adj, row_adj);
            cv::Point dst_point((int)std::round(col_adj+(v_vectors.at<double_t>(row, col)*adjustment_factor)),
                                (int)std::round(row_adj+(u_vectors.at<double_t>(row, col)*adjustment_factor)));

            cv::arrowedLine(dsp_image, src_point, dst_point, cv::Scalar(255, 0, 0), 1);
        }
    }

    ImageShower shower("Quiver");
    shower.showImage(dsp_image);
}

void display_optical_flow_hsv(const cv::Mat &image, const cv::Mat &u_vectors, const cv::Mat &v_vectors, double_t scale) {
    cv::Mat hsv_img(image.rows, image.cols, CV_8UC3, cv::Scalar(0)), final_img;


    for (uint32_t row = 0; row < image.rows; row++) {
        for (uint32_t col = 0; col < image.cols; col++) {
            double_t vec_length = sqrt(pow(u_vectors.at<double_t>(row, col), 2) + pow(v_vectors.at<double_t>(row, col), 2));
            double_t angle = atan2(v_vectors.at<double_t>(row, col), u_vectors.at<double_t>(row, col)) + 3.14159;

            //printf("[%i,%i]: Angle: %f, length: %f\n", row, col, angle, vec_length);

            hsv_img.at<cv::Vec3b>(row, col)[0] = (uint8_t)((angle) * (180/3.14159/2));
            hsv_img.at<cv::Vec3b>(row, col)[1] = (uint8_t)std::min<double_t>(vec_length*4, 255);
            hsv_img.at<cv::Vec3b>(row, col)[2] = (uint8_t)255;
        }
    }

    cv::cvtColor(hsv_img, final_img, CV_HSV2BGR);

    if (scale != 1) {
        cv::Mat resized;
        cv::resize(final_img, resized, cv::Size(0, 0), 1/scale, 1/scale);
        final_img = resized;
    }

    ImageShower shower("HSV Visualization");
    shower.showImage(final_img);
}

std::pair<cv::Mat, cv::Mat> lukas_kanade(const cv::Mat &image1, const cv::Mat &image2) {
    cv::Mat grayImg1, grayImg2;
    cv::cvtColor(image1, grayImg1, CV_BGR2GRAY);
    cv::cvtColor(image2, grayImg2, CV_BGR2GRAY);
    grayImg1.convertTo(grayImg1, CV_64FC1, 1.0/255.0);
    grayImg2.convertTo(grayImg2, CV_64FC1, 1.0/255.0);

    // Smooth the images
    const double_t SMOOTHING_SIGMA = 2;
    cv::Mat smooth1, smooth2;
    cv::GaussianBlur(grayImg1, smooth1, cv::Size(), SMOOTHING_SIGMA);
    cv::GaussianBlur(grayImg2, smooth2, cv::Size(), SMOOTHING_SIGMA);

    // Get spacial intensity gradients for both images in x and y directions
    cv::Mat space_int_2_x, space_int_2_y;
    cv::Sobel(grayImg2, space_int_2_x, CV_64F, 1, 0);
    cv::Sobel(grayImg2, space_int_2_y, CV_64F, 0, 1);

    // Calculate the temporal gradients for both images
    cv::Mat temporal_deriv(image1.rows, image2.cols, CV_64FC1, cv::Scalar(0));
    for (int row = 0; row < image1.rows; row++) {
        for (int cols = 0; cols < image1.cols; cols++) {
            temporal_deriv.at<double_t>(row, cols) = smooth2.at<double_t>(row, cols) - smooth1.at<double_t>(row, cols);
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

#pragma omp parallel for
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
    cv::Mat u_flow_vector(image1.rows, image1.cols, CV_64FC1, cv::Scalar(0)),
            v_flow_vector(image1.rows, image1.cols, CV_64FC1, cv::Scalar(0));
    cv::Mat l_matrix(2, 2, CV_64FC1),
            r_matrix(2, 1, CV_64FC1);

#pragma omp parallel for
    for (uint32_t row = anchor_pt; row < image1.rows - anchor_pt; row++) {
        for (uint32_t col = anchor_pt; col < image1.cols - anchor_pt; col++) {
            l_matrix.at<double_t>(0, 0) = Ix2.at<double_t>(row, col);
            l_matrix.at<double_t>(0, 1) = IxIy.at<double_t>(row, col);
            l_matrix.at<double_t>(1, 0) = IxIy.at<double_t>(row, col);
            l_matrix.at<double_t>(1, 1) = Iy2.at<double_t>(row, col);

            r_matrix.at<double_t>(0, 0) = -IxIt.at<double_t>(row, col);
            r_matrix.at<double_t>(1, 0) = -IyIt.at<double_t>(row, col);

            cv::Mat l_matrix_inv = l_matrix.inv();
            cv::Mat uv_mat = l_matrix_inv * r_matrix;

            u_flow_vector.at<double_t>(row, col) = uv_mat.at<double_t>(0, 0);
            v_flow_vector.at<double_t>(row, col) = uv_mat.at<double_t>(1, 0);
        }
    }

    return std::make_pair(u_flow_vector, v_flow_vector);
}
