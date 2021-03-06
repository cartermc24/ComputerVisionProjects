#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/FrameReader.h"
#include "utils/ImageShower.h"
#include <unistd.h>


void simpleTemporalFilter(FrameReader &frame_reader);
cv::Mat applyTemporalDerivFilter(cv::Mat *images, int start_idx, cv::Mat filter);
void runTemporalDifferenceRun(cv::Mat *input, int num_images, int threshold);
void partB1(FrameReader &frame_reader);
void partB2(FrameReader &frame_reader);
cv::Mat getGaussianKernel(double tsigma);
void estimateNoise(FrameReader reader);
uint8_t getThresholdNiblack(double est_noise_sigma, double k, cv::Mat frame);

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: ./CVProject1 [image_dir]" << std::endl;
        return 0;
    }
    std::string image_dir(argv[1]);
    
    std::cout << "This is Project 1: Motion Detection" << std::endl;

    FrameReader reader(image_dir + "/");

    estimateNoise(reader);

    simpleTemporalFilter(reader);
    partB1(reader);
    partB2(reader);

    return 0;
}

/*
 * This image takes a FrameReader and returns an array of OpenCV Mats in video order
 */
cv::Mat* getImagesAsGrayscale(FrameReader &frame_reader) {
    frame_reader.resetFramePointer();
    auto *images = new cv::Mat[frame_reader.getNumTotalFrames()];

    // Convert images to gray and save them into the image array
    for (int i = 0; i < frame_reader.getNumTotalFrames(); i++) {
        cv::Mat grayscaleImage;
        cv::cvtColor(frame_reader.getNextFrame(), grayscaleImage, cv::COLOR_BGR2GRAY);
        images[i] = grayscaleImage;
    }

    return images;
}

/*
 * This function evaluates the linear temporal & 1D Derivative of a Gaussian filter
 */
void partB1(FrameReader &frame_reader) {
    std::cout << "Running part 2.b.i - Temporal deriv filters" << std::endl;

    int num_images = frame_reader.getNumTotalFrames();

    if (num_images < 2) {
        std::cerr << "Less than 2 frames, can't apply temporal filter" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::Mat *images = getImagesAsGrayscale(frame_reader);
    cv::Mat prev, current, next, diff, comb, post_threshold;
    int threshold = 75;

    runTemporalDifferenceRun(images, num_images, threshold);

    delete[] images;
}

/*
 * This function implements the 2D spacial smoothing filter implementation
 */
void partB2(FrameReader &frame_reader) {
    std::cout << "Running part 2.b.ii - 2D smoothing filters" << std::endl;

    int num_images = frame_reader.getNumTotalFrames();

    if (num_images < 2) {
        std::cerr << "Less than 2 frames, can't apply temporal filter" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::Mat *smoothed_images = new cv::Mat[num_images];
    cv::Mat *images = getImagesAsGrayscale(frame_reader);
    int threshold = 87;

    // --------------------- 3x3 BOX FILTER ------------------------------

    std::cout << "Running with 3x3 Box filter" << std::endl;
    // Apply 3x3 box filter to images to smooth out noise
    for (int i = 0; i < num_images; i++) {
        cv::Mat post_gauss;
        cv::blur(images[i], post_gauss, cv::Size(3, 3));
        smoothed_images[i] = post_gauss;
    }

    runTemporalDifferenceRun(smoothed_images, num_images, threshold);

    // --------------------- 5x5 BOX FILTER ------------------------------

    std::cout << "Running with 5x5 Box filter" << std::endl;
    // Apply 3x3 box filter to images to smooth out noise
    for (int i = 0; i < num_images; i++) {
        cv::Mat post_gauss;
        cv::blur(images[i], post_gauss, cv::Size(5, 5));
        smoothed_images[i] = post_gauss;
    }

    runTemporalDifferenceRun(smoothed_images, num_images, threshold);

    // ----------------- GAUSSIAN FILTER ------------------------------

    std::cout << "Running with Gaussian smoothing" << std::endl;

    double ssigma = 1.5;

    // Apply Gaussian Filter to images to smooth out noise
    for (int i = 0; i < num_images; i++) {
        cv::Mat post_gauss;
        cv::GaussianBlur(images[i], post_gauss, cv::Size(0, 0), ssigma);
        smoothed_images[i] = post_gauss;
    }

    runTemporalDifferenceRun(smoothed_images, num_images, threshold);

    delete[] images;
    delete[] smoothed_images;
}

/*
 * Implements the EST_NOISE algorithm
 */
void estimateNoise(FrameReader reader) {
    cv::Mat *images = getImagesAsGrayscale(reader);
    int32_t num_images = 10;//reader.getNumTotalFrames();
    int32_t img_width = images[0].cols;
    int32_t img_height = images[0].rows;

    cv::Mat weightedAverage = cv::Mat::zeros(img_width, img_height, CV_64F);
    for (int i = 0; i < img_width; i++) {
        for (int j = 0; j < img_height; j++) {
            double interframeSum = 0;
            for (int k = 0; k < num_images; k++) {
                interframeSum += images[k].at<uint8_t>(i, j);
            }
            double average = interframeSum/num_images;
            weightedAverage.at<double>(i, j) = average;
        }
    }

    cv::Mat sigma = cv::Mat::zeros(img_width, img_height, CV_64F);
    for (int i = 0; i < img_width; i++) {
        for (int j = 0; j < img_height; j++) {
            double sum = 0;

            for (int k = 0; k < num_images; k++) {
                sum += pow((weightedAverage.at<double>(i, j) - images[k].at<uint8_t>(i, j)), 2);
            }

            double_t average = sqrt((sum/(num_images-1)));
            sigma.at<double>(i, j) = average;
        }
    }

    double sigmaAverage = 0;
    for (int i = 0; i < img_width; i++) {
        for (int j = 0; j < img_height; j++) {
            sigmaAverage += sigma.at<double>(i, j);
        }
    }
    sigmaAverage /= (img_height*img_width);

    std::cout << "Estimated average for EST_NOISE: " << sigmaAverage << std::endl;
}

/*
 * Implements the modified version of Niblack's Technique to calculate dynamic thresholds per frame
 */
uint8_t getThresholdNiblack(double est_noise_sigma, double k, cv::Mat frame) {
    double mean = 0;
    for (int i = 0; i < frame.cols; i++) {
        for (int j = 0; j < frame.rows; j++) {
            mean += frame.at<uint8_t>(i, j);
        }
    }
    mean /= frame.total();

    std::cout << "Calculated threshold is: " << mean << std::endl;

    return (uint8_t)floor(mean + k*est_noise_sigma);
}

/*
 * Returns a 1D derivative of a Gaussian temporal filter kernel
 */
cv::Mat getGaussianKernel(double tsigma) {
    double k_size = 5;
    double center = floor(k_size/2); //floor((5*tsigma)/2);
    cv::Mat kernel(1, (int)k_size, CV_64F);
    for (int i = 0; i < k_size; i++){
        double exponent = -(i-center)*(i-center)/(2*tsigma*tsigma);
        double dergau = 0.0;

        dergau = -(i-center)*exp(exponent)/(tsigma*tsigma*tsigma*sqrt(2*3.14159));

        kernel.at<double>(0, i) = dergau;
    }
    std::cout << "Gaussian Kernel with tsigma=" << tsigma << " is: " << kernel << std::endl;
    return kernel;
}

/*
 * Sweeps through and displays frames using a chosen kernel & threshold
 */
void runTemporalDifferenceRun(cv::Mat *input, int num_images, int threshold) {
    cv::Mat prev, current, next, diff, comb, post_threshold;

    ImageShower mask("Mask");
    ImageShower mask_with_img("Frame Data with Mask");
    ImageShower original_img("Original Image");

    int max_val = 255;

    // Create simple 0.5[-1, 0, 1] filter
    cv::Mat simple(1, 3, CV_64F);
    simple.at<double_t>(0, 0) = -0.5;
    simple.at<double_t>(0, 1) = 0;
    simple.at<double_t>(0, 2) = 0.5;

    // Gaussian sigma=1.5
    cv::Mat kernel = getGaussianKernel(1.5);

    bool useNiblack = true;

    // Apply temporal deriv operation
    for (int i = 0; i < num_images-kernel.cols; i++) {
        // Compute temporal derivative
        // Change the last argument to select a different kernel
        diff = applyTemporalDerivFilter(input, i, kernel);

        //Set the useNiblack variable to enable dynamic thresholding
        if (useNiblack) {
            threshold = getThresholdNiblack(1.33, 0.5, input[i]);
        }

        // Threshold the difference
        cv::threshold(diff, post_threshold, threshold, max_val, 0);

        // Add mask to original image
        cv::max(input[i], post_threshold, comb);

        mask_with_img.showImage(comb);
        mask.showImage(post_threshold);
        original_img.showImage(input[i]);

        usleep(10000);
    }
}

/*
 * Calculates the 1D temporal derivative using a given kernel and outputs a motion mask as a cv::Mat
 */
cv::Mat applyTemporalDerivFilter(cv::Mat *images, int start_idx, cv::Mat filter) {
    cv::Mat deriv(images[0].rows, images[0].cols, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < images[0].rows; i++) {
        for (int j = 0; j < images[0].cols; j++) {
            double accum = 0;

            for (int k = 0; k < filter.cols; k++) {
                accum += filter.at<double>(0, k) * images[start_idx+k].at<uint8_t>(i, j);
            }

            deriv.at<uint8_t>(i, j) = (uint8_t)(accum);
        }
    }
    return deriv;
}

/*
 * Applies a simple difference operator to calculate a temporal derivative (part 2.a)
 */
void simpleTemporalFilter(FrameReader &frame_reader) {
    std::cout << "Running part 2.a - Simple Temporal Filter" << std::endl;

    ImageShower image_shower("Part A: Simple Temporal Filter");
    ImageShower mask_shower("Part A: Mask Value");

    int threshold = 18;
    int max_val = 255;

    if (frame_reader.getNumFramesLeft() < 2) {
        std::cerr << "Less than 2 frames, can't apply temporal filter" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::Mat prev, current, diff, post_threshold, combined;
    prev = frame_reader.getNextFrame();
    cv::cvtColor(prev, prev, cv::COLOR_BGR2GRAY);
    
    while (frame_reader.getNumFramesLeft() > 0) {
        current = frame_reader.getNextFrame();
        cv::cvtColor(current, current, cv::COLOR_BGR2GRAY);

        cv::absdiff(prev, current, diff);

        // Apply binary threshold at 20 and set to 255
        cv::threshold(diff, post_threshold, threshold, max_val, 0);

        // Combine the mask with the original image
        cv::max(current, post_threshold, combined);

        image_shower.showImage(combined);
        mask_shower.showImage(post_threshold);
        usleep(10000);

        prev = current;
    }
}
