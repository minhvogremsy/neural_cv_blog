#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <ctime>
#include <iterator>


using namespace cv;
using namespace std;


void keypoint_matching();
void good_matching();
void find_homography();


int main(void)
{
	keypoint_matching();
	//good_matching();
	//find_homography();

	return 0;
}

void send_data(std::ostream& o, const std::vector<uchar>& v)
{
	o.write(reinterpret_cast<const char*>(v.data()), v.size());
}

void keypoint_matching()
{
	Mat src1 = imread("1.png", IMREAD_GRAYSCALE);
	Mat src2 = imread("2.png", IMREAD_GRAYSCALE);

	if (src1.empty() || src2.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Ptr<Feature2D> feature = ORB::create();

	vector<KeyPoint> keypoints1, keypoints2;

	Mat desc1, desc2;
	feature->detectAndCompute(src1, Mat(), keypoints1, desc1);
	feature->detectAndCompute(src2, Mat(), keypoints2, desc2);

	Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);

	vector<DMatch> matches;
	matcher->match(desc1, desc2, matches);
	std::cout << matches.size();
	Mat dst;
	drawMatches(src1, keypoints1, src2, keypoints2, matches, dst);

	imshow("dst", dst);
	std::vector<uchar> buf;
	cv::imencode(".jpg", dst, buf);
	std::ofstream outfile("test.jpg", std::ofstream::binary);
	send_data(outfile, buf);
	outfile.close();

	waitKey();
	destroyAllWindows();
}



void good_matching()
{
	Mat src1 = imread("1.png", IMREAD_GRAYSCALE);
	Mat src2 = imread("2.png", IMREAD_GRAYSCALE);
	//src1 = src1.reshape(-1);
	std::cout << src1.size() << std::endl;
	if (src1.empty() || src2.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Ptr<Feature2D> feature = ORB::create();

	vector<KeyPoint> keypoints1, keypoints2;

	Mat desc1, desc2;
	feature->detectAndCompute(src1, Mat(), keypoints1, desc1);
	feature->detectAndCompute(src2, Mat(), keypoints2, desc2);

	Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);

	vector<DMatch> matches;
	matcher->match(desc1, desc2, matches);
	std::cout << matches.size();

	std::sort(matches.begin(), matches.end());
	vector<DMatch> good_matches(matches.begin(), matches.begin() + 50);

	Mat dst;
	drawMatches(src1, keypoints1, src2, keypoints2, good_matches, dst,
		Scalar::all(-1), Scalar::all(-1), vector<char>(),
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imshow("dst", dst);

	cv::imwrite("put.png", dst);
	std::fflush(stdout);

	std::vector<uchar> buf;
	cv::imencode(".jpg", dst, buf);
	std::ofstream outfile("test.jpg", std::ofstream::binary);
	send_data(outfile, buf);
	outfile.close();
	

	waitKey();
	destroyAllWindows();
}

void find_homography()
{
	Mat src1 = imread("1.png", IMREAD_GRAYSCALE);
	Mat src2 = imread("2.png", IMREAD_GRAYSCALE);

	if (src1.empty() || src2.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Ptr<Feature2D> feature = ORB::create();

	vector<KeyPoint> keypoints1, keypoints2;

	Mat desc1, desc2;
	feature->detectAndCompute(src1, Mat(), keypoints1, desc1);
	feature->detectAndCompute(src2, Mat(), keypoints2, desc2);

	Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);

	vector<DMatch> matches;
	matcher->match(desc1, desc2, matches);

	std::sort(matches.begin(), matches.end());
	vector<DMatch> good_matches(matches.begin(), matches.begin() + 50);

	Mat dst;
	drawMatches(src1, keypoints1, src2, keypoints2, good_matches, dst,
		Scalar::all(-1), Scalar::all(-1), vector<char>(),
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	vector<Point2f> pts1, pts2;

	for (size_t i = 0; i < good_matches.size(); i++) {
		pts1.push_back(keypoints1[good_matches[i].queryIdx].pt);
		pts2.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(pts1, pts2, RANSAC);

	vector<Point2f> corners1, corners2;
	corners1.push_back(Point2f(0, 0));
	corners1.push_back(Point2f(src1.cols - 1.f, 0));
	corners1.push_back(Point2f(src1.cols - 1.f, src1.rows - 1.f));
	corners1.push_back(Point2f(0, src1.rows - 1.f));
	perspectiveTransform(corners1, corners2, H);

	vector<Point> corners_dst;
	for (Point2f pt : corners2) {
		corners_dst.push_back(Point(cvRound(pt.x + src1.cols), cvRound(pt.y)));
	}

	polylines(dst, corners_dst, true, Scalar(0, 255, 0), 2, LINE_AA);

	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}