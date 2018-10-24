#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/FrameReader.h"
#include "utils/ImageShower.h"
#include <unistd.h>
#include <tuple>
#include <algorithm>
#include <random>

std::vector<std::tuple<cv::Point, cv::Point>> get_normalized_correlation(cv::Mat F, cv::Mat G, cv::Mat harris_F, cv::Mat harris_G, uint8_t window_size, double_t ncorr_threshold);
cv::Mat harris_corner_detector(cv::Mat img, uint8_t window_size);
cv::Mat non_max_supression(cv::Mat src, uint8_t window_size);
cv::Mat generateHomographyFromPoints(std::vector<std::tuple<cv::Point, cv::Point>> points);
cv::Mat warpImageFromHomography(cv::Mat baseImg, cv::Mat warpImg, const cv::Mat &H);
void showNcorrPts(const cv::Mat &img1, const cv::Mat &img2, std::vector<std::tuple<cv::Point, cv::Point>> points, double_t scale_factor);
void showCorners(const cv::Mat &corner_img, const cv::Mat &disp_img, int32_t scale_factor);
std::vector<std::tuple<cv::Point, cv::Point>> findRANSACHomographyPoints(
        std::vector<std::tuple<cv::Point, cv::Point>> corrs,
        uint8_t iter,
        double_t thresh_dist
);


int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: ./CVProject2 [image_dir]" << std::endl;
        return 0;
    }
    std::string image_dir(argv[1]);
    std::cout << "This is Project 2: Mosaicing" << std::endl;
    FrameReader reader(image_dir + "/");

    cv::Mat i1 = reader.getNextFrame();
    cv::Mat i2 = reader.getNextFrame();

    cv::Mat scaled1, scaled2;
    cv::resize(i1, scaled1, cv::Size(), 1, 1);
    cv::resize(i2, scaled2, cv::Size(), 1, 1);

    std::cout << "Running harris on image 1" << std::endl;
    cv::Mat image1 = harris_corner_detector(scaled1, 11);

    std::cout << "Running harris on image 2" << std::endl;
    cv::Mat image2 = harris_corner_detector(scaled2, 11);

    // Run non-max supression
    image1 = non_max_supression(image1, 11);
    image2 = non_max_supression(image2, 11);

    showCorners(image1, i1, 1);
    showCorners(image2, i2, 1);

    std::cout << "Running ncorr on image 1 & 2" << std::endl;
    std::vector<std::tuple<cv::Point, cv::Point>> corr_pts = get_normalized_correlation(scaled1, scaled2, image1, image2, 15, 0.2);


    for (auto corr : corr_pts) {
        std::cout << "-> NCORR: F: " << std::get<0>(corr) << " G:" << std::get<1>(corr) << std::endl;
    }

    showNcorrPts(i1, i2, corr_pts, 1);

    // Show Matches
    //std::vector<std::tuple<cv::Point, cv::Point>> largest_canidate_pool = findRANSACHomographyPoints(
    //        corr_pts,
    //        10,
    //        50);

    //cv::Mat homography = generateHomographyFromPoints(largest_canidate_pool);
    cv::Mat homography = generateHomographyFromPoints(corr_pts);
    std::cout << "Identified best homography as:\n" << homography << std::endl;

    //showNcorrPts(i1, i2, largest_canidate_pool, 1);

    cv::Mat finalImage = warpImageFromHomography(scaled1, scaled2, homography);
    ImageShower frame1("Final Stitched Image");
    frame1.showImage(finalImage);

    return 0;
}

void showCorners(const cv::Mat &corner_img, const cv::Mat &disp_img, int32_t scale_factor) {
    ImageShower shower("Corners");

    cv::Mat c_img(disp_img);

    for (int x = 0; x < corner_img.rows; x++) {
        for (int y = 0; y < corner_img.cols; y++) {
            if (corner_img.at<double_t>(x, y) > 0) {
                cv::circle(c_img, cv::Point(y*scale_factor, x*scale_factor), 1, cv::Scalar(255, 0, 0), -1);
            }
        }
    }

    shower.showImage(c_img);
}

void showNcorrPts(const cv::Mat &img1, const cv::Mat &img2, std::vector<std::tuple<cv::Point, cv::Point>> points, double_t scale_factor) {
    ImageShower shower("Ncorrelated Points");
    cv::Mat dispImg;
    cv::hconcat(img1, img2, dispImg);

    int32_t right_img_offset = img1.cols;

    for (const auto &point : points) {
        cv::Point left_point = std::get<0>(point);
        cv::Point right_point = std::get<1>(point);

        left_point *= scale_factor;
        right_point *= scale_factor;
        right_point.x += right_img_offset;

        cv::line(dispImg, left_point, right_point, cv::Scalar(0, 255, 0));
    }

    shower.showImage(dispImg);
}

std::string opencvmattype(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

cv::Mat warpImageFromHomography(cv::Mat baseImg, cv::Mat warpImg, const cv::Mat &H) {
    // Calculate how large the image will be by calculating extreme corner
    uint32_t max_dist_col, max_dist_row;

    double_t denom = warpImg.cols*H.at<double_t>(2,0)+warpImg.rows*H.at<double_t>(2,1)+H.at<double_t>(2,2);
    double_t x_numer = warpImg.cols*H.at<double_t>(0,0)+warpImg.rows*H.at<double_t>(0,1)+H.at<double_t>(0,2);
    double_t y_numer = warpImg.cols*H.at<double_t>(1,0)+warpImg.rows*H.at<double_t>(1,1)+H.at<double_t>(1,2);

    max_dist_col = (uint32_t)std::ceil(x_numer/denom);
    max_dist_row = (uint32_t)std::ceil(y_numer/denom);

    std::cout << "Final image size: (" << max_dist_col << "x" << max_dist_row << ")" << std::endl;

    cv::Mat output;
    cv::warpPerspective(warpImg, output, H, cv::Size(max_dist_col, max_dist_row), CV_INTER_LINEAR);

    baseImg.copyTo(output(cv::Rect(0, 0, baseImg.cols, baseImg.rows)));

    std::cout << "Output image is: " << opencvmattype(output.type()) << std::endl;
    std::cout << "Warp image is: " << opencvmattype(warpImg.type()) << std::endl;
    std::cout << "Base image is: " << opencvmattype(baseImg.type()) << std::endl;

    std::cout << "Blending output image..." << std::endl;
    // Blend the images together
    uint32_t border_range_limit = 20;
    for (uint32_t col = 0; col < warpImg.cols; col++) {
        for (uint32_t row = 0; row < warpImg.rows; row++) {
            double_t hdenom = col*H.at<double_t>(2,0)+row*H.at<double_t>(2,1)+H.at<double_t>(2,2);
            double_t hx_numer = col*H.at<double_t>(0,0)+row*H.at<double_t>(0,1)+H.at<double_t>(0,2);
            double_t hy_numer = col*H.at<double_t>(1,0)+row*H.at<double_t>(1,1)+H.at<double_t>(1,2);

            uint32_t projected_col = (uint32_t)std::ceil(hx_numer/hdenom);
            uint32_t projected_row = (uint32_t)std::ceil(hy_numer/hdenom);

            if (projected_col < baseImg.cols && projected_row < baseImg.rows \
                && projected_col > baseImg.cols-border_range_limit && projected_row > baseImg.rows-border_range_limit) {
                std::cout << "Projected col: " << projected_col << " / projected row: " << projected_row << std::endl;
                uint8_t B = (uint8_t)(((uint32_t)warpImg.at<cv::Vec3b>(row, col)[0] + \
                                   (uint32_t)baseImg.at<cv::Vec3b>(projected_row, projected_col)[0]) / 2);
                uint8_t G = (uint8_t)(((uint32_t)warpImg.at<cv::Vec3b>(row, col)[1] + \
                                   (uint32_t)baseImg.at<cv::Vec3b>(projected_row, projected_col)[1]) / 2);
                uint8_t R = (uint8_t)(((uint32_t)warpImg.at<cv::Vec3b>(row, col)[2] + \
                                   (uint32_t)baseImg.at<cv::Vec3b>(projected_row, projected_col)[2]) / 2);

                output(cv::Rect(projected_col, projected_row, 1, 1)) = cv::Scalar(B, G, R);
            }
        }
    }


    return output;
}

cv::Mat generateHomographyFromPoints(std::vector<std::tuple<cv::Point, cv::Point>> points) {
    std::vector<cv::Point2f> src_points;
    std::vector<cv::Point2f> dst_points;

    for (auto point : points) {
        src_points.emplace_back(std::get<0>(point));
        dst_points.emplace_back(std::get<1>(point));
    }

    return cv::findHomography(src_points, dst_points, CV_RANSAC);
}

std::vector<std::tuple<cv::Point, cv::Point>> findRANSACHomographyPoints(
        std::vector<std::tuple<cv::Point, cv::Point>> corrs,
        uint8_t iter,
        double_t thresh_dist
) {
    std::cout << "Running RANSAC to find valid point pool" << std::endl;
    std::vector<std::tuple<cv::Point, cv::Point>> largest_point_pool;

    for (int j = 0; j < iter; j++) {
        // Choose 4 random points
        int32_t rand_idx[4] = { -1 };
        for (int k = 0; k < 4;) {
            uint32_t rand_cani = (uint32_t)(std::rand()%corrs.size());
            bool idx_exists = false;
            for (int l = k; l < 4; l++) {
                if (rand_cani == rand_idx[l]) {
                    idx_exists = true;
                    break;
                }
            }
            if (!idx_exists) {
                rand_idx[k] = rand_cani;
                k++;
            }
        }

        // Format those points into vectors
        std::vector<cv::Point2f> src_pts;
        std::vector<cv::Point2f> dst_pts;

        for (int32_t k : rand_idx) {
            cv::Point2f src_pt(std::get<0>(corrs[k]));
            cv::Point2f dst_pt(std::get<1>(corrs[k]));
            src_pts.emplace_back(src_pt);
            dst_pts.emplace_back(dst_pt);
        }

        // Calculate test homography
        cv::Mat canidate_homography = cv::findHomography(src_pts, dst_pts);

        if (canidate_homography.cols != 3 || canidate_homography.rows != 3) {
            std::cout << "\t!> RANSAC: Could not build homography from selection" << std::endl;
            continue;
        }

        // Compute inliers
        std::vector<std::tuple<cv::Point, cv::Point>> inliers;
        double_t average_error = 0;
        for (auto point : corrs) {
            cv::Mat P(3, 1, CV_64F, cv::Scalar(1));
            P.at<double_t>(0, 0) = std::get<0>(point).x;
            P.at<double_t>(1, 0) = std::get<0>(point).y;

            // P' = H*P
            cv::Mat P_prime = canidate_homography * P;

            // Dist = sqrt( (x2-x1)^2 + (y2-y1)^2 )
            double_t dist_err = sqrt( pow(P_prime.at<double_t>(0, 0) - std::get<0>(point).x, 2) +
                                      pow(P_prime.at<double_t>(1, 0) - std::get<0>(point).y, 2) );

            double_t actual_l_dist = sqrt( pow(std::get<0>(point).x - std::get<1>(point).x, 2) +
                                           pow(std::get<0>(point).y - std::get<1>(point).y, 2) );


            average_error += dist_err;

            if (abs(dist_err - actual_l_dist) < thresh_dist) {
                inliers.emplace_back(std::make_tuple(std::get<0>(point), std::get<1>(point)));
            }
        }

        std::cout << "\t-> Average error for batch [" << j << "] is: " << (average_error/corrs.size()) << std::endl;

        if (inliers.size() > largest_point_pool.size()) {
            largest_point_pool = inliers;
        }
    }

    std::cout << "\t-> RANSAC: Largest point pool is: " << largest_point_pool.size() << std::endl;

    return largest_point_pool;
}

std::vector<std::tuple<cv::Point, cv::Point>> get_normalized_correlation(cv::Mat F, cv::Mat G, cv::Mat harris_F, cv::Mat harris_G, uint8_t window_size, double_t ncorr_threshold) {
    uint8_t anchor_pt = (uint8_t)floor(window_size/2);

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

            double_t mean_val = 0;
            for (uint32_t f_x = 0; f_x < window_size; f_x++) {
                for (uint32_t f_y = 0; f_y < window_size; f_y++) {
                    mean_val += F.at<uint8_t>(f_x_start+f_x, f_y_start+f_y);
                }
            }
            mean_val /= window_size*window_size;

            double_t fhat_val = 0;
            for (uint32_t f_x = 0; f_x < window_size; f_x++) {
                for (uint32_t f_y = 0; f_y < window_size; f_y++) {
                    fhat_val += pow((double_t)F.at<uint8_t>(f_x_start+f_x, f_y_start+f_y)-mean_val, 2);
                }
            }

            Fhat.at<double_t>(x, y) = (F.at<uint8_t>(x, y)-mean_val) / sqrt(fhat_val);
        }
    }

    // Ghat
    std::cout << "\t-> Creating Ghat for ncorr" << std::endl;
#pragma omp parallel for
    for (uint32_t x = anchor_pt; x < G.rows-anchor_pt; x++) {
        for (uint32_t y = anchor_pt; y < G.cols-anchor_pt; y++) {
            uint32_t g_x_start = x - anchor_pt;
            uint32_t g_y_start = y - anchor_pt;

            double_t mean_val = 0;
            for (uint32_t g_x = 0; g_x < window_size; g_x++) {
                for (uint32_t g_y = 0; g_y < window_size; g_y++) {
                    mean_val += F.at<uint8_t>(g_x_start+g_x, g_y_start+g_y);
                }
            }
            mean_val /= window_size*window_size;

            double_t ghat_val = 0;
            for (uint32_t g_x = 0; g_x < window_size; g_x++) {
                for (uint32_t g_y = 0; g_y < window_size; g_y++) {
                    ghat_val += pow((double_t)G.at<uint8_t>(g_x_start+g_x, g_y_start+g_y)-mean_val, 2);
                }
            }

            Ghat.at<double_t>(x, y) = (G.at<uint8_t>(x, y)-mean_val) / sqrt(ghat_val);
        }
    }

    std::vector<std::tuple<cv::Point, cv::Point>> corr_points;
    // Ignore borders
    // Iterate through all corners of F
    std::cout << "\t-> Starting ncorr scan" << std::endl;
    for (uint32_t f_x = anchor_pt; f_x < Fhat.rows-anchor_pt; f_x++) {
        for (uint32_t f_y = anchor_pt; f_y < Fhat.cols-anchor_pt; f_y++) {
            if (harris_F.at<double_t>(f_x, f_y) == 0) {
                continue;
            }

            uint32_t f_x_start = f_x - anchor_pt;
            uint32_t f_y_start = f_y - anchor_pt;

            double_t max_ncorr_val = 0;
            uint32_t max_g_x = 0, max_g_y = 0;

            // Iterate through all corners of G
            for (uint32_t g_x = anchor_pt; g_x < Ghat.rows - anchor_pt; g_x++) {
                for (uint32_t g_y = anchor_pt; g_y < Ghat.cols - anchor_pt; g_y++) {
                    if (harris_G.at<double_t>(g_x, g_y) == 0) {
                        continue;
                    }

                    uint32_t g_x_start = g_x - anchor_pt;
                    uint32_t g_y_start = g_y - anchor_pt;

                    // Create window
                    cv::Mat subimage_F(F, cv::Rect(f_y_start, f_x_start, window_size, window_size));
                    cv::Mat subimage_G(G, cv::Rect(g_y_start, g_x_start, window_size, window_size));

                    // Run NCC on image 2
                    cv::Mat post_ncorr;
                    cv::matchTemplate(subimage_G, subimage_F, post_ncorr, CV_TM_CCORR_NORMED);

                    // Find best match in G
                    double min_val, max_val;
                    cv::Point min_point, max_point; // We only care about max point
                    cv::minMaxLoc(post_ncorr, &min_val, &max_val, &min_point, &max_point, cv::Mat());

                    if (max_val > max_ncorr_val) {
                        max_g_x = g_x;
                        max_g_y = g_y;
                        max_ncorr_val = max_val;
                    }
                }
            }

            if (max_ncorr_val > ncorr_threshold) {
                printf("\t-> Best for point F(%u, %u) and G(%u, %u) ncorr val is: %f\n", f_x, f_y, max_g_x, max_g_y, max_ncorr_val);
                cv::Point f_point(f_y, f_x);
                cv::Point g_point(max_g_y, max_g_x);

                corr_points.emplace_back(g_point, f_point);
            }
        }
    }

    std::cout << "\t-> Found num ncorrs: " << corr_points.size() << std::endl;
    return corr_points;
}

cv::Mat non_max_supression(cv::Mat src, uint8_t window_size) {
    std::cout << "Running non-maximum supression" << std::endl;
    /************* Non-max suppression **************/
    uint8_t b_anchor = (uint8_t)floor(window_size / 2);

    cv::Mat post_supression(src.rows, src.cols, CV_64FC1, cv::Scalar(0));
#pragma omp parallel for
    for (uint32_t x = b_anchor; x < (src.rows - window_size); x++) {
        for (uint32_t y = b_anchor; y < (src.cols - window_size); y++) {
            cv::Mat subimage(src, cv::Rect(y-b_anchor, x-b_anchor, window_size, window_size));

            double_t val_at_pos = src.at<double_t>(x, y);
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
                post_supression.at<double_t>(x, y) = src.at<double_t>(x, y);
            }
        }
    }
    return post_supression;
}


cv::Mat harris_corner_detector(cv::Mat img, uint8_t window_size) {
    const double_t K = 0.04;

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

    return harris_r_thresh;
}
