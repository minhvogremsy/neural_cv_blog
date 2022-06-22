#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define SHOW
#define DEVICE  0
// #define BGR2RGB
//#define SINK_UDP
//#define SINK_SHM

int main(int argc, char* argv[]) {
    VideoCapture cap;    // Camera
    VideoWriter writer;  // gstreamer




    cout << "Press ESC to quit" << endl;
    cout << getBuildInformation() << endl;
    //    return 0;
      // CAMERA SETUP
    cap.open(DEVICE);
    if (!cap.isOpened()) {
        cerr << "Can't create camera reader" << endl;
        return -1;
    }

    const double fps = cap.get(CAP_PROP_FPS);
    const int width = cap.get(CAP_PROP_FRAME_WIDTH);
    const int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    cout << "Capture camera with " << fps << " fps, " << width << "x" << height << " px" << endl;

    void* out_unified_ptr;
    cv::Mat* mats;
    unsigned int chan = 3;
    unsigned int pixels = width * height * chan;
    mats = new cv::Mat(height, width, CV_8UC3);


    cout << endl;
    int *i,*j;
    i= new int(5);
    j = &*i;
    
    cout << *i  <<"   "<<*&*j << endl;

    std::list<int> a = {1,4,5,6,7};
    std::list<int> b = { 1,554,45,54,47};
    std::list<int> c;
   
    std::merge(a.begin(), a.end(), b.begin(), b.end(), std::inserter(c, c.end()));
    for (auto n : c) {
        std::cout << n << ", ";
    }


    

    // GSTREAMER-1.0 SETUP
#ifdef SINK_UDP
    writer.open("appsrc ! rtpvrawpay ! udpsink host=localhost port=5000", 0, fps, cv::Size(width, height), true);
#endif
#ifdef SINK_SHM
    writer.open("appsrc ! shmsink socket-path=/tmp/foo sync=true wait-for-connection=false shm-size=10000000", 0, fps, cv::Size(width, height), true);
#endif
    /*if (!writer.isOpened()) {
        cap.release();
        cerr << "Can't create gstreamer writer. Do you have the correct version installed?" << endl;
        cerr << "Print out OpenCV build information" << endl;
        cout << getBuildInformation() << endl;
        return -1;
    }*/

    cv::Mat frame, frameg1;
    int key = -1, cnt = 0;
    //std::list<cv::Mat> a;

    while (key != 27) { // 27 = ascii value of ESC
        try {
            cap >> frame;
            if (frame.empty()) {
                throw;
            }
#ifdef BGR2RGB
            cvtColor(frame, frame, cv::COLOR_BGR2RGB);
#endif
           // writer << frame;
#define CMN "[camera]"
#ifdef SHOW
            cv::Mat matd(height, width, CV_8UC3);
            static cv::Mat frameg = *mats;
            static cv::Rect rect(0, 0, 0, 0);
           
            

            std::cout << CMN << "  \n";
            //cv::flip(frameg, frameg,0);
            frame.copyTo(frameg(cv::Range(0, 480), cv::Range(0, 640)));
            //std::cout << frameg(rect).size() << "\n";
            //a.append(frameg);
            cv::rotate(frameg, frameg1, cv::ROTATE_90_CLOCKWISE);
            cv::imshow("Source image", frameg1);
#endif
        }
        catch (...) {
            cout << "Something went wrong" << endl;
            break;
        }
        key = cv::waitKey(1);
    }

    // CLEANUP
    //writer.release();
    cap.release();
    
    return 0;
}