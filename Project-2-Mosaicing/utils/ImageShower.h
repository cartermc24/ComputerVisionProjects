#include <utility>

//
// Created by Carter McCardwell on 9/23/18.
//

#ifndef CVPROJECT1_IMAGESHOWER_H
#define CVPROJECT1_IMAGESHOWER_H

#include <opencv2/opencv.hpp>

class ImageShower {
public:
    ImageShower(std::string frame_name) {
        framename = std::move(frame_name);
    };

    void showImage(cv::Mat image) {
        if (!image.empty()) {
            cv::imshow(framename.c_str(), image);
            char c = (char) cv::waitKey(0);
        }
    };

private:
    std::string framename;
};


#endif //CVPROJECT1_IMAGESHOWER_H
