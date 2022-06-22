#include <iterator>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

#include "common.h"
#include "logger.h"

#include "object_detector_fp16.h"
//

//



ObjectDetector_fp16::ObjectDetector_fp16(const int input_width, const int input_heigth)
    : input_width_(input_width)
    , input_height_(input_heigth)
 
    
{
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
}


bool ObjectDetector_fp16::LoadEngine(const std::string &model_path,cv::Mat cls1, cv::Mat r1)
{

    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (runtime_ == nullptr)
    {
        return false;
    }

    std::vector<unsigned char> buffer;
    std::ifstream stream(model_path, std::ios::binary);
    if (!stream)
    {
        return false;
    }
    stream >> std::noskipws;
    std::copy(std::istream_iterator<unsigned char>(stream),
              std::istream_iterator<unsigned char>(),
              back_inserter(buffer));
    stream.close();


    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));

    for (int bi = 0; bi < engine_->getNbBindings(); bi++) {
            if (engine_->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine_->getBindingName(bi));
            else printf("Binding %d (%s): Output.\n", bi, engine_->getBindingName(bi));
        }




    IRefitter *refitter = createInferRefitter(*engine_, sample::gLogger.getTRTLogger());

    
    int size = (int)cls1.total() * cls1.channels();
    std::cout << "!!  cls1    size !! = " <<size << std::endl;
    Weights wt{DataType::kFLOAT, cls1.data, size};
    bool out = refitter->setWeights("Conv_15",WeightsRole::kKERNEL, wt);
    std::cout << "!!  cls1 result !! "<<out << std::endl;



    
    int size_65 = (int)r1.total() * r1.channels();
    std::cout << "!!  r1    size !! = " <<size_65 << std::endl;
    Weights wt_65{DataType::kFLOAT, r1.data, size_65 };
    bool out_65 = refitter->setWeights("Conv_12",WeightsRole::kKERNEL, wt_65);
    std::cout << "!!  r1 result !! "<<out_65 << std::endl;




    bool success = refitter->refitCudaEngine();
    std::cout << "!!  success !! " << success << std::endl;
    assert(success);
    refitter->destroy();



    if (engine_ == nullptr)
    {

        return false;
    }
    std::cout << "engine->hasImplicitBatchDimension(): " << engine_->hasImplicitBatchDimension()<< std::endl;

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());

   

    if (context_ != nullptr)
    {

        return false;
       
    }

    return true;
}

bool ObjectDetector_fp16::RebuldEngine(cv::Mat cls1, cv::Mat r1) {

    IRefitter* refitter = createInferRefitter(*engine_, sample::gLogger.getTRTLogger());


    int size = (int)cls1.total() * cls1.channels();
    std::cout << "!!  cls1    size !! = " << size << std::endl;
    Weights wt{ DataType::kFLOAT, cls1.data, size };
    bool out = refitter->setWeights("Conv_15", WeightsRole::kKERNEL, wt);
    std::cout << "!!  cls1 result !! " << out << std::endl;




    int size_65 = (int)r1.total() * r1.channels();
    std::cout << "!!  r1    size !! = " << size_65 << std::endl;
    Weights wt_65{ DataType::kFLOAT, r1.data, size_65 };
    bool out_65 = refitter->setWeights("Conv_12", WeightsRole::kKERNEL, wt_65);
    std::cout << "!!  r1 result !! " << out_65 << std::endl;




    bool success = refitter->refitCudaEngine();
    std::cout << "!!  success !! " << success << std::endl;
    assert(success);
    refitter->destroy();
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    return true;

};



std::unique_ptr<std::vector<BoundingBox_fp16>> ObjectDetector_fp16::RunInference(
    const cv::Mat &input_data,
    std::chrono::duration<double, std::milli> &time_span)
{
    

    auto results = std::make_unique<std::vector<BoundingBox_fp16>>();

    // Create RAII buffer manager object
    if (buffers == nullptr)
    {
        std::cout << "New buffers " << batch_size_ << std::endl;
        buffers = std::make_unique<samplesCommon::BufferManager>(engine_);
    }


    // Fill data buffer
    float *host_data_buffer = static_cast<float *>(buffers->getHostBuffer("input.1"));

    // Host memory for input buffer
    memcpy(host_data_buffer, input_data.data, input_data.elemSize() * input_data.total());
    
    // Memcpy from host input buffers to device input buffers
    //

    

    //
    buffers->copyInputToDevice();

    std::cout << "copyInputToDevice" << std::endl;
    const auto &start_time = std::chrono::steady_clock::now();

    auto status = context_->executeV2(buffers->getDeviceBindings().data());
    time_span =
        std::chrono::steady_clock::now() - start_time;

    if (status){
    buffers->copyOutputToHost();}

    auto detection_66 = static_cast<float*>(buffers->getHostBuffer("output0"));

    int sz_1[] = {20, 19, 19};
    cv::Mat out_1(3, sz_1, CV_32FC1, detection_66); 



    auto detection_68 = static_cast<float*>(buffers->getHostBuffer("output1"));


    int sz_2[] = {10, 19, 19};
    cv::Mat out_2(3, sz_2, CV_32FC1, detection_68); 


  

    std::cout << "context_->execute return " << status << std::endl;

    auto bounding_box = std::make_unique<BoundingBox_fp16>();
    bounding_box->status = status;
    bounding_box->out_mat_1 = out_1;
    bounding_box->out_mat_2 = out_2;


    results->emplace_back(std::move(*bounding_box));
    return results;
}



const int ObjectDetector_fp16::Width() const
{
    return input_width_;
}

const int ObjectDetector_fp16::Height() const
{
    return input_height_;
}

const int ObjectDetector_fp16::Channels() const
{
    return input_channels_;
}