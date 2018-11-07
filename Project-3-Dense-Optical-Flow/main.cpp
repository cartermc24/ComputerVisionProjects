#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/FrameReader.h"
#include "utils/ImageShower.h"
#include <unistd.h>
#include <tuple>
#include <algorithm>
#include <random>

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

    shower1.showImage(i1);
    shower2.showImage(i2);

    return 0;
}
