#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include <iostream>
#include <chrono>
using namespace std;
using namespace cv;
bool divide_images = false;
//Stitcher::Mode mode = Stitcher::PANORAMA;
//vector<Mat> imgs;
string result_name = "result.jpg";

#define GD "0.0.0.0"



#include <opencv2/opencv.hpp>
#include <fstream>
int main(int argc, char* argv[])
{

    cv::Rect rects = { 0,0,250,250 };
    std::string first = GD;
    std::string gh = "appsrc !\
        videoconvert !video / x - raw, format = (string)I420 !omxh264enc control - rate = 2 bitrate = 8000000 !video / x - h264, stream - format = byte - stream !\
        rtph264pay mtu = 1500 !udpsink host ="+ first +" port = 5000 sync = false async = false";
    std::cout << gh<< " \n";
    std::vector<cv::Mat> images;
    std::string name;
    std::vector<std::string> im_patch{"outscreen_0.png","outscreen_1.png"};
    for (const auto& p : im_patch) {
        auto img = cv::imread(p);
        img.empty();
        std::cout << img.rows << "\n";
        std::cout << img.cols << "\n";
        std::cout << img.empty() << "\n";
        std::cout << img.type() << "\n";
        //cv::format();
        //cv::imread();
        if (img.empty()) {
            std::cerr << "Problem loading: " << name << std::endl;
            exit(-1);
        }
        images.emplace_back(img);

    }


    cv::Mat pano;
    Stitcher::Mode mode = Stitcher::PANORAMA;
    auto stitcher = cv::Stitcher::create(mode);
    auto begin = std::chrono::steady_clock::now();
    
    auto status = stitcher->stitch(images, pano);
    //auto mask = stitcher->resultMask();
    //std::cout << mask << "\n";
    
    
    auto end = std::chrono::steady_clock::now();

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "The time: " << elapsed_ms.count() << " ms\n";


    if (status != decltype(status)::OK) {
        std::cerr << "stitch failed" << std::endl;
        exit(-1);
    }
    cv::imshow("Stitching Example", pano);
    cv::waitKey(0);
    cv::imwrite("1.bmp", pano);

    return 0;
}