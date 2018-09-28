#include <iostream>
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>
#include "utils/FrameReader.h"
#include "utils/ImageShower.h"
#include <unistd.h>


void simpleTemporalFilter(FrameReader &frame_reader, ImageShower &image_shower);

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: ./CVProject1 [image_dir]" << std::endl;
        return 0;
    }

    std::string image_dir(argv[1]);

    std::cout << "This is project 1: Motion Detection" << std::endl;

    FrameReader reader(image_dir + "/");
    ImageShower shower("Hay there");

    /*
    while (reader.getFramesLeft() > 0) {
        cv::Mat c_frame = reader.getNextFrame();
        cv::cvtColor(c_frame, c_frame, cv::COLOR_BGR2GRAY);
        shower.show_image(c_frame);
        usleep(10000);
    }
    */

    simpleTemporalFilter(reader, shower);

    return 0;
}


void simpleTemporalFilter(FrameReader &frame_reader, ImageShower &image_shower) {
    std::cout << "Running part a - Simple Temporal Filter" << std::endl;
    
    int threshold = 18;
    int max_val = 255;

    if (frame_reader.getFramesLeft() < 2) {
        std::cout << "Less than 2 frames, can't apply temporal filter" << std::endl;    
    }

    cv::Mat prev, current, diff, post_threshold;
    prev = frame_reader.getNextFrame();
    cv::cvtColor(prev, prev, cv::COLOR_BGR2GRAY);
    
    while (frame_reader.getFramesLeft() > 0) {
        current = frame_reader.getNextFrame();
        cv::cvtColor(current, current, cv::COLOR_BGR2GRAY);

        cv::absdiff(prev, current, diff);

        // Apply binary threshold at 20 and set to 255
        cv::threshold(diff, post_threshold, threshold, max_val, 0);
        
        image_shower.show_image(post_threshold);
        usleep(10000);

        prev = current;
    }
}
