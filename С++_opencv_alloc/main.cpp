#include <iostream>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h> 

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudafilters.hpp> 
#include <opencv2/cudawarping.hpp>

int main()
{
     //std::cout << cv::getBuildInformation() << std::endl; 
    cv::Mat frame_in;
    frame_in = cv::imread("../test_img.jpg");
    unsigned int width  = frame_in.size().width; 
    unsigned int height = frame_in.size().height;
    unsigned int chan = frame_in.channels(); 
    unsigned int pixels = width*height;
    std::cout <<"Frame size : "<<width<<" x "<<height<<", "<<pixels<<" Pixels "<<chan<<" Channels"<<std::endl;

    float resize_scale_ = 0.5;
    unsigned int rwidth = resize_scale_*width;
    unsigned int rheight = resize_scale_*height;
    unsigned int rpixels = rwidth*rheight;
    std::cout <<"Resized Frame size : "<<rwidth<<" x "<<rheight<<", "<<rpixels<<" Pixels "<<chan<<" Channels"<<std::endl;

    cv::namedWindow("frame_out", cv::WINDOW_AUTOSIZE );
    bool hasOpenGlSupport = true;
    try {
        cv::namedWindow("d_frame_out", cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);
    }
    catch(cv::Exception& e) {
	hasOpenGlSupport = false;
    }

    unsigned int frameByteSize = pixels * chan; 
    unsigned int rframeByteSize = rpixels * chan; 
    cv::Size res = cv::Size(rwidth,rheight);


// TEST STANDARD UPLOAD AND DOWNLOAD PROFILE -------------------------
    std::cout << "Using standard memory transfer" << std::endl;
    std::chrono::duration<double> diff;
    auto start_t = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat std_sd_frame_;
    cv::cuda::GpuMat std_hd_frame_cuda_;
    cv::cuda::GpuMat std_sd_frame_cuda_;

    auto end = std::chrono::high_resolution_clock::now();
    diff = end-start;
    std::cout << "STANDARD:SETUP: " << diff.count() << " s\n";
    start = std::chrono::high_resolution_clock::now();

    cv::Mat std_hd_frame_ = frame_in;
    std_hd_frame_cuda_.upload(std_hd_frame_);

    end = std::chrono::high_resolution_clock::now();
    diff = end-start;
    std::cout << "STANDARD:UPLOAD: " << diff.count() << "\n";
    start = std::chrono::high_resolution_clock::now();

    cv::cuda::resize(std_hd_frame_cuda_,std_sd_frame_cuda_,res,0,0,cv::INTER_AREA);

    end = std::chrono::high_resolution_clock::now();
    diff = end-start;
    std::cout << "STANDARD:RESIZE: " << diff.count() << "\n";
    start = std::chrono::high_resolution_clock::now();

    std_sd_frame_cuda_.download(std_sd_frame_);
    cv::imshow("frame_out", std_sd_frame_); 
    cv::waitKey(1);	

    end = std::chrono::high_resolution_clock::now();
    diff = end-start;
    std::cout << "STANDARD:ACCESS: " << diff.count() << "\n";

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_t = end-start_t;
    std::cout << "STANDARD:TOTAL: " << diff_t.count() << "\n";

// TEST UNIFIED MEMORY PROFILE --------------------------------------
    start_t = std::chrono::high_resolution_clock::now();
    start = std::chrono::high_resolution_clock::now();

    /* Unified memory */
    std::cout << "Using unified memory" << std::endl;
    void *hd_unified_ptr;
    void *sd_unified_ptr;
    cudaMallocManaged(&hd_unified_ptr, frameByteSize);
    cudaMallocManaged(&sd_unified_ptr, rframeByteSize);
    cv::Mat shd_hd_frame_(height, width, CV_8UC3, hd_unified_ptr);
    cv::Mat shd_sd_frame_(rheight, rwidth, CV_8UC3, sd_unified_ptr);
    cv::cuda::GpuMat shd_hd_frame_cuda_(height, width, CV_8UC3, hd_unified_ptr);
    cv::cuda::GpuMat shd_sd_frame_cuda_(rheight, rwidth, CV_8UC3, sd_unified_ptr);

    end = std::chrono::high_resolution_clock::now();
    diff = end-start;
    std::cout << "UNIFIED:SETUP: " << diff.count() << "\n";
    start = std::chrono::high_resolution_clock::now();

    frame_in.copyTo(shd_hd_frame_);

    end = std::chrono::high_resolution_clock::now();
    diff = end-start;
    std::cout << "UNIFIED:UPLOAD: " << diff.count() << "\n";
    start = std::chrono::high_resolution_clock::now();

    // no need to copy to device
    cv::cuda::resize(shd_hd_frame_cuda_,shd_sd_frame_cuda_,res,0,0,cv::INTER_AREA);

    end = std::chrono::high_resolution_clock::now();
    diff = end-start;
    std::cout << "UNIFIED:RESIZE: " << diff.count() << "\n";
    start = std::chrono::high_resolution_clock::now();

    if (hasOpenGlSupport)
	    cv::imshow("d_frame_out", shd_sd_frame_cuda_);
        // no need to copy back to host
    cv::imshow("frame_out", shd_sd_frame_); 
    cv::waitKey(1);	
    
    end = std::chrono::high_resolution_clock::now();
    diff = end-start;
    std::cout << "UNIFIED:ACCESS: " << diff.count() << "\n";

    end = std::chrono::high_resolution_clock::now();
    diff_t = end-start_t;
    std::cout << "UNIFIED:TOTAL: " << diff_t.count() << "\n";

    // TEST PINNED MEMORY PROFILE --------------------------------------
    start_t = std::chrono::high_resolution_clock::now();
    start = std::chrono::high_resolution_clock::now();

    /* Pinned memory. No cache */
    std::cout << "Using pinned memory" << std::endl;
    void *pdevice_hd_ptr, *phost_hd_ptr, *pdevice_sd_ptr, *phost_sd_ptr;
    cudaHostAlloc((void **)&phost_hd_ptr, frameByteSize, cudaHostAllocMapped);
    cudaHostAlloc((void **)&phost_sd_ptr, rframeByteSize, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **)&pdevice_hd_ptr, (void *) phost_hd_ptr , 0);
    cudaHostGetDevicePointer((void **)&pdevice_sd_ptr, (void *) phost_sd_ptr , 0);
    cv::Mat pshd_hd_frame_(height, width, CV_8UC3, phost_hd_ptr);
    cv::Mat pshd_sd_frame_(rheight, rwidth, CV_8UC3, phost_sd_ptr);
    cv::cuda::GpuMat pshd_hd_frame_cuda_(height, width, CV_8UC3, pdevice_hd_ptr);
    cv::cuda::GpuMat pshd_sd_frame_cuda_(rheight, rwidth, CV_8UC3, pdevice_sd_ptr);

    end = std::chrono::high_resolution_clock::now();
    diff = end-start;
    std::cout << "PINNED:SETUP: " << diff.count() << "\n";
    start = std::chrono::high_resolution_clock::now();

    frame_in.copyTo(pshd_hd_frame_);

    end = std::chrono::high_resolution_clock::now();
    diff = end-start;
    std::cout << "PINNED:UPLOAD: " << diff.count() << "\n";
    start = std::chrono::high_resolution_clock::now();

    // no need to copy to device
    cv::cuda::resize(pshd_hd_frame_cuda_,pshd_sd_frame_cuda_,res,0,0,cv::INTER_AREA);

    end = std::chrono::high_resolution_clock::now();
    diff = end-start;
    std::cout << "PINNED:RESIZE: " << diff.count() << "\n";
    start = std::chrono::high_resolution_clock::now();

    if (hasOpenGlSupport)
	    cv::imshow("d_frame_out", pshd_sd_frame_cuda_);
        // no need to copy back to host
    cv::imshow("frame_out", pshd_sd_frame_); 
    cv::waitKey(1);	
    
    end = std::chrono::high_resolution_clock::now();
    diff = end-start;
    std::cout << "PINNED:ACCESS: " << diff.count() << "\n";

    end = std::chrono::high_resolution_clock::now();
    diff_t = end-start_t;
    std::cout << "PINNED:TOTAL: " << diff_t.count() << "\n";

    cv::waitKey(2000);

    return 0;
}