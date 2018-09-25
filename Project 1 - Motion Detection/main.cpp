#include <iostream>
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>
#include "utils/FrameReader.h"
#include "utils/ImageShower.h"
#include <unistd.h>

int main() {
    std::cout << "This is project 1: Motion Detection" << std::endl;

    FrameReader reader("videos/Office/");
    ImageShower shower("Hay there");

    while (reader.getFramesLeft() > 0) {
        cv::Mat c_frame = reader.getNextFrame();
        shower.show_image(c_frame);
        usleep(10000);
    }

    return 0;
}