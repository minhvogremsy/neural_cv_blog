#include <iterator>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

#include "common.h"
#include "logger.h"

#include "loader.h"
//

//



Model_loader::Model_loader(const int input_width, const int input_heigth)
    : input_width_(input_width)
    , input_height_(input_heigth)
 
    
{
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
}


bool Model_loader::LoadEngine(const std::string &model_path, cv::Mat params,  cv::Mat bias)
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
    int size = (int)params.total() * params.channels();
    std::cout << "!!  size !! "<<size << std::endl;
    Weights wt{DataType::kFLOAT, params.data, size};
    bool setrole = refitter->setWeights("Conv_41",WeightsRole::kKERNEL, wt);
    std::cout << "!!  setrole !! "<<setrole << std::endl;



    int size_bias = (int)bias.total() * bias.channels();
    std::cout << "!!  size !! "<<size_bias << std::endl;
    Weights wt_bias{DataType::kFLOAT, bias.data, size_bias};
    bool setrole_bias = refitter->setWeights("Conv_41",WeightsRole::kBIAS, wt_bias);
    std::cout << "!!  setrole_bias !! "<<setrole_bias << std::endl;


    // // Weights newBias_tmp;
    // // newBias_tmp.count = 512;
    // // float* newBiasLocal = new float[512];
    // // for (int i = 0; i < 64; i++)
    // // {
    // //     newBiasLocal[i] = 0.0001 * i;
    // // }

    // // newBias_tmp.values = newBiasLocal;
    // // newBias_tmp.type = DataType::kFLOAT;
    // // refitter->setWeights("Conv_41", WeightsRole::kBIAS, newBias_tmp);

    // // // const int32_t n = refitter->getMissingWeights(0, "Conv_41");
    // // // std::vector<const char*> weightsNames(n);
    // // // for (auto  i: weightsNames) {
    // // //         std::cout << i << ' ';}

    bool success = refitter->refitCudaEngine();
    assert(success);
    refitter->destroy();
    
    std::cout << "!!  success !! "<<success << std::endl;



    if (engine_ == nullptr)
    {
        // std::cout << "!!  FALSE 1   !! " << std::endl;
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





std::unique_ptr<std::vector<BoundingBox>> Model_loader::RunInference(
    const cv::Mat &input_data,
    std::chrono::duration<double, std::milli> &time_span)
{
    

    auto results = std::make_unique<std::vector<BoundingBox>>();

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

    
    
    auto status = context_->execute(batch_size_, buffers->getDeviceBindings().data());
    //auto status = context_->executeV2(buffers->getDeviceBindings().data());
    
    time_span =
        std::chrono::steady_clock::now() - start_time;

    if (status){
    buffers->copyOutputToHost();}

    auto output_1 = static_cast<float*>(buffers->getHostBuffer("output0"));


    int sz_1[] = {1,1000};
    cv::Mat out_1(2, sz_1, CV_32FC1, output_1);
    out_1 = out_1.reshape(0, {1,1000});







  

    std::cout << "context_->execute return " << status << std::endl;

    auto bounding_box = std::make_unique<BoundingBox>();
    bounding_box->status = status;
    bounding_box->out_mat_1 = out_1;




    results->emplace_back(std::move(*bounding_box));
    return results;
}



const int Model_loader::Width() const
{
    return input_width_;
}

const int Model_loader::Height() const
{
    return input_height_;
}

const int Model_loader::Channels() const
{
    return input_channels_;
}