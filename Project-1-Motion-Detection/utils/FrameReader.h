//
// Created by Carter McCardwell on 9/23/18.
//

#ifndef CVPROJECT1_FRAMEREADER_H
#define CVPROJECT1_FRAMEREADER_H

#include <errno.h>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <vector>
#include <string>
#include <algorithm>

class FrameReader {
public:
    FrameReader(std::string picture_folder);
    cv::Mat getNextFrame();
    int getNumFramesLeft();
    int getNumTotalFrames();
    void resetFramePointer();
private:
    int getdir(std::string dir);

    std::vector<std::string> file_list;
    int file_index;
    std::string filepath;
};


#endif //CVPROJECT1_FRAMEREADER_H
