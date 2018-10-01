#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/FrameReader.h"
#include "utils/ImageShower.h"
#include <unistd.h>


void simpleTemporalFilter(FrameReader &frame_reader);
cv::Mat applyTemporalDeriv1by3Filter(cv::Mat prev, cv::Mat cur, cv::Mat next, cv::Mat filter);
void runTemporalDifferenceRun(cv::Mat *input, int num_images, int threshold);
void partB1(FrameReader &frame_reader);
void partB2(FrameReader &frame_reader);

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: ./CVProject1 [image_dir]" << std::endl;
        return 0;
    }
    std::string image_dir(argv[1]);
    
    std::cout << "This is project 1: Motion Detection" << std::endl;

    FrameReader reader(image_dir + "/");

    partB2(reader);

    //simpleTemporalFilter(reader);

    return 0;
}

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

void partB1(FrameReader &frame_reader) {
    std::cout << "Running part 2.b.i - Temporal deriv filters" << std::endl;

    int num_images = frame_reader.getNumTotalFrames();

    if (num_images < 2) {
        std::cerr << "Less than 2 frames, can't apply temporal filter" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::Mat *images = getImagesAsGrayscale(frame_reader);
    int threshold = 18;

    runTemporalDifferenceRun(images, num_images, threshold);

    delete[] images;
}

void partB2(FrameReader &frame_reader) {
    std::cout << "Running part 2.b.ii - 2D smoothing filters" << std::endl;

    int num_images = frame_reader.getNumTotalFrames();

    if (num_images < 2) {
        std::cerr << "Less than 2 frames, can't apply temporal filter" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::Mat *smoothed_images = new cv::Mat[num_images];
    cv::Mat *images = getImagesAsGrayscale(frame_reader);
    int threshold = 18;

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

    uint8_t ssigma = 2;

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

void runTemporalDifferenceRun(cv::Mat *input, int num_images, int threshold) {
    cv::Mat prev, current, next, diff, comb, post_threshold;

    ImageShower mask("Mask");
    ImageShower mask_with_img("Frame Data with Mask");
    ImageShower original_img("Original Image");

    int max_val = 255;

    // Create simple 0.5[-1, 0, 1] filter
    cv::Mat kernel(1, 3, CV_64F);
    kernel.at<double_t>(0, 0) = -0.5;
    kernel.at<double_t>(0, 1) = 0;
    kernel.at<double_t>(0, 2) = 0.5;

    // Apply temporal deriv operation
    for (int i = 1; i < num_images-1; i++) {
        prev = input[i-1];
        current = input[i];
        next = input[i+1];

        // Compute temporal derivative
        diff = applyTemporalDeriv1by3Filter(prev, current, next, kernel);

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


cv::Mat applyTemporalDeriv1by3Filter(cv::Mat prev, cv::Mat cur, cv::Mat next, cv::Mat filter) {
    cv::Mat deriv(prev.rows, prev.cols, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < prev.rows; i++) {
        for (int j = 0; j < prev.cols; j++) {
            double left = filter.at<double>(0, 0) * prev.at<uint8_t>(i, j);
            double center = filter.at<double>(0, 1) * cur.at<uint8_t>(i, j);
            double right = filter.at<double>(0, 2) * next.at<uint8_t>(i, j);

            deriv.at<uint8_t>(i, j) = (uint8_t)(left + center + right);
        }
    }
    return deriv;
}

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
