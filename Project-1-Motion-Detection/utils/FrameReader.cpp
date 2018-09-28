//
// Created by Carter McCardwell on 9/23/18.
//

#include "FrameReader.h"

FrameReader::FrameReader(std::string picture_folder) {
    if (getdir(picture_folder) != 0) {
        std::cerr << "[FrameReader]: Couldn't open folder: " << picture_folder << std::endl;
        exit(EXIT_FAILURE);
    }
    filepath = picture_folder;

    std::cout << "[FrameReader]: Found " << file_list.size() << " pictures" << std::endl;
    std::sort(file_list.begin(), file_list.end());
    file_index = 0;
}

void FrameReader::resetFramePointer() {
    std::cout << "[FrameReader]: Frame index reset" << std::endl;
    file_index = 0;
}

int FrameReader::getFramesLeft() {
    return file_list.size() - file_index;
}

int FrameReader::getTotalFrames() {
    return file_list.size();
}

cv::Mat FrameReader::getNextFrame() {
    cv::Mat m;

    if (file_index > file_list.size()) {
        std::cout << "[FrameReader]: Note: End of stream but frame requested, returning empty Mat" << std::endl;
        return m;
    }

    std::string path = filepath + file_list[file_index];

    m = cv::imread(path.c_str(), CV_LOAD_IMAGE_COLOR);

    file_index++;

    return m;
}

int FrameReader::getdir(std::string dir)
{
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(dir.c_str())) == NULL) {
        std::cout << "Error(" << errno << ") opening " << dir << std::endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        // Ignore "dot" files
        if (dirp->d_name[0] != '.') {
            file_list.push_back(std::string(dirp->d_name));
        }
    }
    closedir(dp);
    return 0;
}
