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

int main() {
	


	cv::Mat image;

	cv::namedWindow("Display window");

	cv::VideoCapture cap(0);
	cap.set(cv::CAP_PROP_FPS, 30);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	void* fMAT;
	void* sMAT;


	if (!cap.isOpened()) {

		std::cout << "cannot open camera";

	}
	int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
	cv::String gst_str = "appsrc !videoconvert !x264enc tune = zerolatency bitrate = 50000 speed - preset = superfast !rtph264pay !udpsink host = 192.168.1.1 port = 5000";

	cv::VideoWriter video(gst_str, cv::CAP_GSTREAMER, codec, 30, image.size());
	while (true) {

		cap >> image;
		fMAT = image.data;

		/*cv::Mat zero = cv::Mat::zeros(640 * 2, 480 * 2, CV_8UC3);
		cv::Rect rect1(0, 0, 200, 200);
		cv::Rect rect2(200, 200, 250, 250);
		Mat roi = image(rect1);
		Mat roi2 = image(rect2);

		image(rect1).copyTo(zero(cv::Range(0, 200), cv::Range(100, 300)));
		//roi.copyTo(zero(cv::Range(0, 200), cv::Range(100, 300)));
		

		roi2.copyTo(zero(cv::Range(0, 250), cv::Range(300, 550)));
		cv::imshow("zero", zero);
		cv::Size sizes = image.size();

		cv::Size sizess(640, 480);
		
		//image[] = 
		
		

		std::cout << sizes.width <<"\n";
		cv::Mat bigCube(image.size(), image.type(), fMAT);
		
		
		


		*/
		cv::imshow("Display window", image);
		//video.write(image);

		char key = (char)cv::waitKey(1);   // explicit cast
		if (key == 27) break; // break if `esc' key was pressed. 

	}
	video.release();


	return 0;

}