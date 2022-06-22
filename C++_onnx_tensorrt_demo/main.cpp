#include <fstream>
#include <iostream>
#include <map>
#include <list>
#include <numeric>

#include <opencv2/opencv.hpp>


#include "common.h"
#include "logger.h"




const cv::String kKeys =
    "{help h usage ? |    | show help command.}"
    "{f file         |    | path to video file.}"
    "{s score        |0.5 | score threshold.}"
    "{w width        |360 | input model width.}"
    "{H height       |480 | input model height.}"
    "{l label        |.   | path to label file.}"
    "{o output       |    | output video file path.}"
    "{@input1         |../resnet18.trt| path to trt engine file.}"

    ;
const cv::String kWindowName = "TensorRT Efficentdet example.";

template <class T> void printType(const T&)
{
    std::cout << __PRETTY_FUNCTION__ << "\n";
}


int main(int argc, char* argv[]) {
    cv::String model_path;
    cv::String model_path_2;
    cv::String model_path_3;
    cv::String model_path_4;
    cv::CommandLineParser parser(argc, argv, kKeys);
    auto input_width = parser.get<int>("width");
    auto input_height = parser.get<int>("height");
    
    
    if (parser.has("@input1"))
    {
        model_path = parser.get<cv::String>("@input1");
    }
    else
    {
        std::cout << "No model file path." << std::endl;
        return 0;
    }





    



    


    cv::Mat frame, input_im, out1;
    frame = cv::imread("../esrgan.png");

    
    cv::dnn::Net RPN = cv::dnn::readNet("../resnet18.onnx");
    RPN.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    RPN.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
   
    //cv::Ptr<cv::dnn::Layer> class8_ab = RPN.getLayer("input.140");

    cv::Mat params =RPN.getParam(RPN.getLayerId("input.140"), 0);
    std::cout << "params  size : " << params.size  << std::endl;

    cv::Mat params_bias =RPN.getParam(RPN.getLayerId("input.140"), 1);
    std::cout << "params bias  size : " << params_bias.size  << std::endl;
  

    std::vector<cv::String> outNames_siamRPN;
    outNames_siamRPN = RPN.getLayerNames();
    for (auto i: outNames_siamRPN) {
            std::cout << " " <<i << ' ';}
    std::cout << '\n';
    



    int sz_1[] = {512,512,3,3};
    cv::Mat random_image(4, sz_1, CV_32FC1);
    cv::randu(random_image, cv::Scalar(0,0,0,0), cv::Scalar(256,256,256,256));
    std::cout << "random_image  size : " << random_image.size  << std::endl;
    random_image.convertTo(random_image, CV_32FC1, 1.0/255);
    std::cout << "random_image  size : " << random_image.size  << std::endl;

    //
    auto detector = std::make_unique<Model_loader>(input_width, input_height);
    
    detector->LoadEngine(model_path,random_image,params_bias);
    
    //
    
    int size = (int)random_image.total() * random_image.channels();
    std::cout << "!!  size  CV!! "<<size << std::endl;
    
    RPN.setParam(RPN.getLayerId("input.140"), 0, random_image);
   



    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0, cv::Size(224,224), cv::Scalar(), false, false, CV_32F); //- > cvmat


    
    RPN.setInput(blob);
    RPN.forward(out1);
    std::cout << "out1  size : " << out1.size  << std::endl;
    std::cout << "out1  size : " << blob.size  << std::endl;
    for (int i = 0; i < 10; i++) { 
                     std::cout << out1.at<float32_t>(0,i) << " "; }
    std::cout << "\n";
   


   


    int cap_width = frame.cols;
    int cap_height = frame.rows;
    auto scale_width = (double)cap_width / input_width;
    auto scale_height = (double)cap_height / input_height;


    std::cout << "scale width  : " << scale_width << std::endl;
    std::cout << "scale height : " << scale_height << std::endl;
      // Run inference.

    int i = 0;
    cv::Mat data;
    while (i < 10){ 
        std::chrono::duration<double, std::milli> inference_time_span;

        const auto& result = detector->RunInference(blob, inference_time_span);
        std::cout << "TENSORRT" <<"\n";
        for (const auto& object : *result)
        {
            data = object.out_mat_1;
            std::cout << "data  size : " << data.size  << std::endl;
            for (int i = 0; i < 10; i++) { 
                    std::cout << data.at<float32_t>(0,i) << " "; }
            std::cout << "\n";



        } 
        

        std::cout << "OPENCV" <<"\n";
        RPN.setInput(blob);
        RPN.forward(out1);
        for (int i = 0; i < 10; i++) { 
                     std::cout << out1.at<float32_t>(0,i) << " "; }
        std::cout << "\n";
       

           
           


        

        std::ostringstream time_caption;
    

        time_caption << std::fixed << std::setprecision(2) << inference_time_span.count() << " ms";
        std::cout << "time_caption : " << time_caption.str() << std::endl;


        i++;
    }

    std::cout << "model path             : " << model_path << std::endl;
    std::cout << "input width            : " << input_width << std::endl;
    std::cout << "input height           : " << input_height << std::endl;

    return 0;
}