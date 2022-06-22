/**
 * Copyright (c) 2021 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */


#include <chrono>
#include <memory>
#include <string>
#include <opencv2/video.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <NvInfer.h>

#include "buffers.h"
#include "object_detector_fp16.h"




struct trackerConfig
{
        float windowInfluence = 0.43f;
        float lr = 0.4f;
        int scale = 8;
        bool swapRB = false;
        int totalStride = 8;
        float penaltyK = 0.055f;
        int exemplarSize = 127;
        int instanceSize = 271;
        float contextAmount = 0.5f;
        std::vector<float> ratios = { 0.33f, 0.5f, 1.0f, 2.0f, 3.0f };
        int anchorNum = int(ratios.size());
        cv::Mat anchors;
        cv::Mat windows;
        cv::Scalar avgChans;
        cv::Size imgSize = { 0, 0 };
        cv::Rect2f targetBox = { 0, 0, 0, 0 };
        int scoreSize = (instanceSize - exemplarSize) / totalStride + 1;
        float tracking_score;
        

        void update_scoreSize()
        {
            scoreSize = int((instanceSize - exemplarSize) / totalStride + 1);
        }
};


struct Params
{
    std::string model = "dasiamrpn_model.onnx";
    std::string kernel_cls1 = "dasiamrpn_kernel_cls1.onnx";
    std::string kernel_r1 = "dasiamrpn_kernel_r1.onnx";


};




class Da_treacker
{
public:
    Da_treacker(const std::string model, const std::string kernel_cls1,const std::string kernel_r1 );

    ////////////////////////////
    bool start;

    void init(cv::InputArray image, const cv::Rect& boundingBox);
    void softmax(const cv::Mat& src, cv::Mat& dst);
    void elementMax(cv::Mat& src);
    cv::Mat generateHanningWindow();
    cv::Mat generateAnchors();
    cv::Mat getSubwindow(cv::Mat& img, const cv::Rect2f& targetBox, float originalSize, cv::Scalar avgChans);
    void trackerInit(cv::Mat img);
    void trackerEval(cv::Mat img);
    trackerConfig trackState;
    ///////////////////////////////
    ObjectDetector_fp16 *detector;
    

    cv::dnn::Net siamRPN, siamKernelR1, siamKernelCL1;
    cv::Rect boundingBox_;
    cv::Mat image_;
    //auto detector_4;

    bool update(cv::InputArray image, cv::Rect& boundingBox);
    float getTrackingScore();

    Params params;



        





private:
  



    std::string model_ = "dasiamrpn_model.onnx";
    std::string kernel_cls1_ = "dasiamrpn_kernel_cls1.onnx";
    std::string kernel_r1_ = "dasiamrpn_kernel_r1.onnx";


};

