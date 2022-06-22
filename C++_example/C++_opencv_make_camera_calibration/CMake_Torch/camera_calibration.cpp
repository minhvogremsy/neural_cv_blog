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



void calibration() {

	std::vector<cv::String> fileNames;
	cv::glob("usb_cam/*.jpg", fileNames, false);
	cv::Size patternSize(9, 6);
	std::vector<std::vector<cv::Point2f>> q(fileNames.size());

	std::vector<std::vector<cv::Point3f>> Q;

	int checkerBoard[2] = { 9,6};

	std::vector<cv::Point3f> objp;
	for (int i = 1; i <= checkerBoard[1]; i++) {
		for (int j = 1; j <= checkerBoard[0]; j++) {
			objp.push_back(cv::Point3f(j, i, 0));
		}
	}
	std::vector<cv::Point2f> imgPoint;
	// Detect feature points
	std::size_t i = 0;
	for (auto const& f : fileNames) {
		std::cout << std::string(f) << std::endl;

		// 2. Read in the image an call cv::findChessboardCorners()
		cv::Mat img = cv::imread(fileNames[i]);
		std::cout << img.size() << std::endl;
		cv::Mat gray;

		cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

		bool patternFound = cv::findChessboardCorners(gray, patternSize, q[i]);
		//cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK

		// 2. Use cv::cornerSubPix() to refine the found corner detections
		if (patternFound) {
			cv::cornerSubPix(gray, q[i], cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
			Q.push_back(objp);
		}

		// Display
		cv::drawChessboardCorners(img, patternSize, q[i], patternFound);
		cv::imshow("chessboard detection", img);
		cv::waitKey(100);

		i++;
	}


	cv::Matx33f K(cv::Matx33f::eye());  // intrinsic camera matrix
	cv::Vec<float, 5> k(0, 0, 0, 0, 0); // distortion coefficients

	std::vector<cv::Mat> rvecs, tvecs;
	std::vector<double> stdIntrinsics, stdExtrinsics, perViewErrors;
	int flags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_K3 +
		cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_PRINCIPAL_POINT;


	cv::Size frameSize(1280, 960);

	std::cout << "Calibrating..." << std::endl;
	// 4. Call "float error = cv::calibrateCamera()" with the input coordinates
	// and output parameters as declared above...

	float error = cv::calibrateCamera(Q, q, frameSize, K, k, rvecs, tvecs, 0);

	std::cout << "Reprojection error = " << error << "\nK =\n"
		<< K << "\nk=\n"
		<< k << std::endl;

	// Precompute lens correction interpolation
	cv::Mat mapX, mapY;

	cv::Rect roi;

	//cv::Vec<float, 5> kk(-5.34885733e-02, 3.25501616e+00, -1.12829085e-02, -6.04881607e-03,
	//	-2.96335478e+01);
	//cv::Matx33f KK((2.16185016e+03, 0.00000000e+00, 6.29772855e+02),
	//	(0.00000000e+00, 2.15549755e+03, 5.11527152e+02),
	//	(0.00000000e+00, 0.00000000e+00, 1.00000000e+00));

	// Show lens corrected images




	cv::namedWindow("Display window");

	cv::VideoCapture cap(0);
	cap.set(cv::CAP_PROP_FPS, 30);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 960);
	void* fMAT;
	void* sMAT;
	cv::Mat img;

	cv::Mat K_new = cv::getOptimalNewCameraMatrix(K, k, frameSize, 1, frameSize, &roi);
	cv::initUndistortRectifyMap(K, k, cv::Matx33f::eye(), K_new, frameSize, CV_32FC1,
		mapX, mapY);
	std::chrono::time_point<std::chrono::system_clock> startTime;
	double time_counter = 0;
	int counter = 0;
	while (true) {

		cap >> img;
		fMAT = img.data;
		cv::Mat imgUndistorted;

			// 5. Remap the image using the precomputed interpolation maps.
		startTime = std::chrono::system_clock::now();
		cv::remap(img, imgUndistorted, mapX, mapY, cv::INTER_LINEAR);
		time_counter += (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - startTime).count() / 1000;
		if (counter % 100 == 0) {
			std::cout << time_counter / 100 << " ms" << std::endl;
			time_counter = 0;

		}
		counter += 1;
		imgUndistorted = imgUndistorted(roi);
		cv::imshow("Display window", imgUndistorted);
	
		char key = (char)cv::waitKey(1);   // explicit cast
		if (key == 27) break; // break if `esc' key was pressed. 

	}
}




int main() {
	

	calibration();

	return 0;

}