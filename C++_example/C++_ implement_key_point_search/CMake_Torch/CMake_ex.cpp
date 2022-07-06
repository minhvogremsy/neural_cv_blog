// CMake_Torch.cpp : Defines the entry point for the application.
//
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>


using namespace cv;

using namespace std;

void processing_max_dim(cv::Mat& frame_resize, const int& processing_max_dim) {
	//Mat resize_image;
	cv::Size shape = frame_resize.size();

	double height = shape.height;
	double width = shape.width;
	double max_dim_size = max(height, width);

	if (max_dim_size > processing_max_dim) {

		if (height > width) {
			double width_s = width * (processing_max_dim / height);
			resize(frame_resize, frame_resize, cv::Size(width_s, processing_max_dim));
		}
		else
		{
			double height_s = height * (processing_max_dim / width);
			resize(frame_resize, frame_resize, cv::Size(processing_max_dim, height_s));
		}
	}

	//cout << "max_dim_size " << frame_resize.size() << "\n";

}





int main() {
	


	cv::Mat image,prev, prev_gray, prev_t;
	cv::Mat curr, curr_gray;

	cv::namedWindow("Display window");

	cv::VideoCapture cap("1.avi");
	//cap.set(cv::CAP_PROP_FPS, 30);
	//cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	//cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	std::vector<cv::Point2f> prev_pts, curr_pts;





	if (!cap.isOpened()) {

		std::cout << "cannot open camera";

	}

	cap >> prev_t;
	prev_t.copyTo(prev);
	cv::cvtColor(prev, prev_gray, COLOR_BGR2GRAY);
	cv::goodFeaturesToTrack(prev_gray, prev_pts, 500, 0.01, 30);
	

	int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
	cv::String gst_str = "appsrc !videoconvert !x264enc tune = zerolatency bitrate = 50000 speed - preset = superfast !rtph264pay !udpsink host = 192.168.1.1 port = 5000";

	cv::VideoWriter video(gst_str, cv::CAP_GSTREAMER, codec, 30, image.size());
	while (true) {

		cap >> curr;
		
		cv::cvtColor(curr, curr_gray, COLOR_BGR2GRAY);
		vector<uchar> status;
		vector<float> err;

		


		try {
			cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err);
			if (prev_pts.size() > 10 && curr_pts.size() > 10) {
				auto prev_it = prev_pts.begin();
				auto curr_it = curr_pts.begin();
				for (size_t k = 0; k < status.size(); k++) {
					if (status[k]) {
						prev_it++;
						curr_it++;
					}
					else {
						prev_it = prev_pts.erase(prev_it);
						curr_it = curr_pts.erase(curr_it);
					}
				}

				std::cout <<"prev_pts " << prev_pts.size() << std::endl;
				std:cout <<"curr_pts " << curr_pts.size() << std::endl;
				if (prev_pts.size() > 1 && curr_pts.size() > 1) {

					for (auto i = 0; i < prev_pts.size(); i++) {
						cv::circle(prev, prev_pts[i], 1, (0, 0, 255), 2);
					}
					for (auto i = 0; i < curr_pts.size(); i++) {
						cv::circle(curr, curr_pts[i], 1, (0, 0, 255), 2);
					}
				}
			}
			else {
				std::cout << "No matches found " << std::endl;
				prev_t.copyTo(prev);
				cv::cvtColor(prev, prev_gray, COLOR_BGR2GRAY);
				cv::goodFeaturesToTrack(prev_gray, prev_pts, 500, 0.01, 30);
			}
		}
		catch (...) {
			std::cout << "some kind of error " << std::endl;
			prev_t.copyTo(prev);
			cv::cvtColor(prev, prev_gray, COLOR_BGR2GRAY);
			cv::goodFeaturesToTrack(prev_gray, prev_pts, 500, 0.01, 30);

		}


	

		
		cv::imshow("curr", curr);
		cv::imshow("prev", prev);
		//video.write(image);

		char key = (char)cv::waitKey(30);

		if (key == 27) break; // break if `esc' key was pressed. 
		if (key == 'q') {
			cap >> prev_t;
			prev_t.copyTo(prev);
			cv::cvtColor(prev, prev_gray, COLOR_BGR2GRAY);
			cv::goodFeaturesToTrack(prev_gray, prev_pts, 500, 0.01, 30);
		
		}; // break if `q' key was pressed. 
		if (key == 's') {
			prev_t.copyTo(prev);
			cv::cvtColor(prev, prev_gray, COLOR_BGR2GRAY);
			cv::goodFeaturesToTrack(prev_gray, prev_pts, 500, 0.01, 30);

		};

	}
	video.release();


	return 0;

}