// CMake_Torch.cpp : Defines the entry point for the application.
//
#include <iostream>
#include <memory>
#include <stdio.h>

#include "CMake_Torch.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>


// OPENCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#define DEFAULT_HEIGHT 720
#define DEFAULT_WIDTH 1280
#define IMG_SIZE 400//512


// PROTOTYPES
cv::Mat frame_prediction(cv::Mat frame, torch::jit::Module model);
torch::jit::Module load_model(std::string model_name);


int old_main(int argc, char* argv[]) {


	std::cout << "hello\n";

	if (argc != 3)
	{
		std::cerr << "usage: CMake_Torch <model.pt> <video path>\n";
		return -1;
	}



	// Set torch module
	torch::jit::script::Module module;
	// OPENCV
	cv::VideoCapture cap;
	cv::Mat frame;
	cap.open(argv[2]);

	if (!cap.isOpened()) {
		std::cerr << "\nCannot open video\n";
	}

	std::cout << "\nPress spacebar to terminate\n";
	// Load Model
	try {
		module = load_model(argv[1]);
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
	}

	for (;;) {
		cap.read(frame);
		if (frame.empty()) {
			std::cerr << "\nError:Blank Frame\n";
		}
		
		frame = frame_prediction(frame, module);
		cv::imshow("video", frame);

		if (cv::waitKey(1) >= 27) {
			break;
		}
	}
}


torch::jit::Module load_model(std::string model_name) {
	std::string directory = model_name;
	torch::jit::Module module = torch::jit::load(directory);
	module.to(torch::kCPU);
	module.eval();
	std::cout << "MODEL LOADED" << std::endl;
	return module;
}




cv::Mat frame_prediction(cv::Mat frame, torch::jit::Module model) {
	// Needed for Overlay
	auto total_start = std::chrono::steady_clock::now();
	double alpha = 0.1;
	double beta = (1 - alpha);
	cv::Mat frame_copy, dst;
	// Torch model input
	std::vector<torch::jit::IValue> input;
	// Mean and std
	std::vector<double> mean = { 0.406, 0.456, 0.485 };
	std::vector<double> std = { 0.225, 0.224, 0.229 };
	cv::resize(frame, frame, cv::Size(IMG_SIZE, IMG_SIZE));
	frame_copy = frame;
	frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
	// CV2 to Tensor
	torch::Tensor frame_tensor =
		torch::from_blob(frame.data, { 1, IMG_SIZE, IMG_SIZE, 3 });
	frame_tensor = frame_tensor.permute({ 0, 3, 1, 2 });
	frame_tensor = torch::data::transforms::Normalize<>(mean, std)(frame_tensor);
	frame_tensor = frame_tensor;//.to(torch::kCPU);
	input.push_back(frame_tensor);
	// Forward Pass
	std::cout << "forward" << std::endl;
	auto pred = model.forward(input).toTensor().detach();//.to(torch::kCPU);
	pred = pred.mul(100).clamp(0.255).to(torch::kU8);
	// Tensor -> CV2

	cv::Mat output_mat(cv::Size{IMG_SIZE, IMG_SIZE}, CV_8UC1, pred.data_ptr());
	cv::cvtColor(output_mat, output_mat, cv::COLOR_GRAY2RGB);
	cv::applyColorMap(output_mat, output_mat, cv::COLORMAP_TWILIGHT_SHIFTED);
	// OVERLAY ORIGINAL FRAME AND PREDICTION
	cv::addWeighted(frame_copy, alpha, output_mat, beta, 0.0, dst);
	
	std::cout << "after forward" << std::endl;
	cv::resize(frame, frame, cv::Size(DEFAULT_WIDTH, DEFAULT_HEIGHT));

	auto total_end = std::chrono::steady_clock::now();
	float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
	std::ostringstream stats_ss;
	stats_ss << std::fixed << std::setprecision(2);
	stats_ss << "Total FPS : " << total_fps;
	auto stats = stats_ss.str();
	int baseline;
	auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
	cv::rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
	cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));


	return frame;
}


int main() {

		torch::Tensor tensor = torch::rand({ 2, 3 });
		std::cout << tensor << std::endl;



}