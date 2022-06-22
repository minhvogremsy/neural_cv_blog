/**
 * Copyright (c) 2021 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */



#include <chrono>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include "buffers.h"


class BoundingBox_fp16
{
public:
    int class_id = 0;
    float scores = 0.0f;
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float center_x = 0.0f;
    float center_y = 0.0f;
    bool status = false;
    cv::Mat out_mat_1;
    cv::Mat out_mat_2;
};



class ObjectDetector_fp16
{
public:
    ObjectDetector_fp16(const int input_width, const int input_heigth);

    bool LoadEngine(const std::string& model_path, cv::Mat cls1, cv::Mat r1);
    bool RebuldEngine(cv::Mat cls1, cv::Mat r1);

    std::unique_ptr<std::vector<BoundingBox_fp16>> RunInference(
        const cv::Mat& input_data,
        std::chrono::duration<double, std::milli>& time_span);

    const int Width() const;
    const int Height() const;
    const int Channels() const;

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::unique_ptr<samplesCommon::BufferManager> buffers;


    int32_t batch_size_ = 1;
    int32_t input_width_ = 271;
    int32_t input_height_ = 271;
    int32_t input_channels_ = 3;


};


