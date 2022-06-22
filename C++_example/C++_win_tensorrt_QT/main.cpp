#include <iostream>
#include <cmath>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#include "treacker.h"


using namespace cv;
using namespace cv::dnn;

const char* keys =
"{ help     h  |   | Print help message }"
"{ input    i  | 0 | Full path to input video folder, the specific camera index. (empty for camera 0) }"
"{ net         | dasiamrpn_model.onnx | Path to onnx model of net}"
"{ kernel_cls1 | dasiamrpn_kernel_cls1.onnx | Path to onnx model of kernel_r1 }"
"{ kernel_r1   | dasiamrpn_kernel_r1.onnx | Path to onnx model of kernel_cls1 }"
"{ output      | output.avi | output file name }"
"{ backend     | 5 | Choose one of computation backends: "
"0: automatically (by default), "
"1: Halide language (http://halide-lang.org/), "
"2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
"3: OpenCV implementation, "
"4: VKCOM, "
"5: CUDA },"
"{ target      | 7 | Choose one of target computation devices: "
"0: CPU target (by default), "
"1: OpenCL, "
"2: OpenCL fp16 (half-float precision), "
"3: VPU, "
"4: Vulkan, "
"6: CUDA, "
"7: CUDA fp16 (half-float preprocess) }"
;

static
int run(int argc, char** argv)
{
    // Parse command line arguments.
    CommandLineParser parser(argc, argv, keys);


    std::string model_ = "dasiamrpn_model_271.onnx";
    std::string kernel_cls1_ = "dasiamrpn_kernel_cls1.onnx";
    std::string kernel_r1_ = "dasiamrpn_kernel_r1.onnx";
    auto da_treacker = std::make_unique<Da_treacker>(model_,kernel_cls1_,kernel_r1_);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    std::string inputName = parser.get<String>("input");
    std::string net = parser.get<String>("net");
    std::string kernel_cls1 = parser.get<String>("kernel_cls1");
    std::string kernel_r1 = parser.get<String>("kernel_r1");
    int backend = parser.get<int>("backend");
    int target = parser.get<int>("target");
    std::string output_file = parser.get<String>("output");
    bool treck_result = false;
    auto color = Scalar(0, 255, 0);



    const std::string winName = "DaSiamRPN";
    namedWindow(winName, WINDOW_AUTOSIZE);

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;

    if (inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1))
    {
        int c = inputName.empty() ? 0 : inputName[0] - '0';
        std::cout << "Trying to open camera #" << c << " ..." << std::endl;
        if (!cap.open(c))
        {
            std::cout << "Capture from camera #" << c << " didn't work. Specify -i=<video> parameter to read from video file" << std::endl;
            return 2;
        }
         cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
         cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
         
    }
    else if (inputName.size())
    {
        inputName = samples::findFileOrKeep(inputName);
        if (!cap.open(inputName))
        {
            std::cout << "Could not open: " << inputName << std::endl;
            return 2;
        }
    }

    // Read the first image.
    Mat image;
    cap >> image;

    if (image.empty())
    {
        std::cerr << "Can't capture frame!" << std::endl;
        return 2;
    }
    cv::Size S = image.size();

    Mat image_select = image.clone();
    putText(image_select, "Select initial bounding box you want to track.", Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    putText(image_select, "And Press the ENTER key.", Point(0, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

    Rect selectRect = selectROI(winName, image_select);
    std::cout << "ROI=" << selectRect << std::endl;

    da_treacker->init(image, selectRect);
    treck_result = true;

    TickMeter tickMeter, updateMeter;
  
    cv::VideoWriter outputVideo;
    outputVideo.open(output_file,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        30, S, true);

    for (int count = 0; ; ++count)
    {
        cap >> image;
        if (image.empty())
        {
            std::cerr << "Can't capture frame " << count << ". End of video stream?" << std::endl;
            break;
        }

        Rect rect;

        tickMeter.start();
        bool ok = da_treacker->update(image, rect);
        tickMeter.stop();

        //-------------------------------------------//
        float score = da_treacker->getTrackingScore();
        if (score < 0.9) {
            color = Scalar(255, 0, 255);
        }
        if (score > 0.99) {
            color = Scalar(0, 255, 0);
        }
        if (score < 0.4) {
            ok = false;
        }
        //-------------------------------------------//


        std::cout << "frame " << count <<
            ": predicted score=" << score <<
            "  rect=" << rect <<
            "  time=" << tickMeter.getTimeMilli() << "ms" <<
            std::endl;

        Mat render_image = image.clone();

        if (ok)
        {
            rectangle(render_image, rect, Scalar(0, 255, 0), 2);

            std::string timeLabel = format("Inference time: %.2f ms", tickMeter.getTimeMilli());
            std::string scoreLabel = format("Score: %f", score);
            putText(render_image, timeLabel, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, color);
            putText(render_image, scoreLabel, Point(0, 35), FONT_HERSHEY_SIMPLEX, 0.5, color);
        }

        imshow(winName, render_image);
        outputVideo.write(render_image);

        tickMeter.reset();

        int c = waitKey(1);
        if (c == 27 /*ESC*/)
            break;
        if (c == 113 /*q*/){
            Rect selectRect = selectROI(winName, render_image);
            std::cout << "ROI=" << selectRect << std::endl;
            updateMeter.start();
            da_treacker->init(image, selectRect);
            updateMeter.stop();
            std::cout << "frame " << count <<
                 "  rect= " << selectRect <<
                "  time=" << updateMeter.getTimeMilli() << "ms" <<
                std::endl;
        }
    }

    std::cout << "Exit" << std::endl;
    outputVideo.release();
    return 0;
}


int main(int argc, char** argv)
{
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "FATAL: C++ exception: " << e.what() << std::endl;
        return 1;
    }
}