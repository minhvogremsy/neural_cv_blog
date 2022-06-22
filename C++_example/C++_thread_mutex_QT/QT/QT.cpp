
#include "mainwindow.h"
#include "object.h"
#include <QThread>
#include <QApplication>
#include <iostream>
//////////////////////////
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core.hpp>
//////////////////////////

//////////////////////////


class SecondThread : public QThread
{
public:
    explicit SecondThread() : QThread() {}
private:
    void run() override {
        emit mySignal();
    }
signals:
    void mySignal() {
        std::cout << "MySignal start" << std::endl;
        
    };
};



class MyThread : public QThread {
private :
    void run() override {
        std::cout << "QThread start" << std::endl;
        
    }

    

};



void calc(int a, int b) {
    std::cout << "QThread calc " << a + b << std::endl;
    int* s = new int(5);
    int f = 5;
    std::cout << "S = " << *s << "   F= " << f <<  std::endl;
}


void videoProcessing() {


    std::cout << cv::getBuildInformation() << "\n";
    cv::dnn::Net siamRPN = cv::dnn::readNet("D:\\NN_ALL_C++\\GitHub\\neural_detection\\C++_example\\C++_thread_mutex\\out\\build\\x64-Release\\CMake_Torch\\dasiamrpn_model_271.onnx");
    siamRPN.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    siamRPN.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    cv::Mat buf0 = cv::imread("D:\\NN_ALL_C++\\GitHub\\neural_detection\\C++_example\\C++_thread_mutex\\out\\build\\x64-Release\\CMake_Torch\\2.jpg");
    if (buf0.empty())
    {
        std::cout << "!!! Failed imread(): image not found" << std::endl;
        // don't let the execution continue, else imshow() will crash.
    }


    cv::Mat blob;
    cv::dnn::blobFromImage(buf0, blob, 1.0, cv::Size(127, 127), cv::Scalar(), 0, false, CV_32F); //- > cvmat
    siamRPN.setInput(blob);
    std::cout << buf0.rows << "\n";
    std::cout << buf0.cols << "\n";
    cv::Mat out1;
    //     //
    std::vector<std::string> outNames_siamRPN;
    outNames_siamRPN = siamRPN.getLayerNames();
    for (auto i: outNames_siamRPN) {
                 std::cout << " " <<i << ' ';}
    std::cout << '\n';


    std::cout << std::endl;

    std::cout << "FORWARD START" << std::endl;
    siamRPN.forward(out1, "input.52");
    std::cout << "FORWARD OK" << std::endl;


}



int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    MyThread* thread = new MyThread;
    QThread* qthread = QThread::create(calc, 4, 7);
    auto sigThread = new SecondThread;
    sigThread->start();
    thread->start();

    //QThread* qthread = QThread::create(videoProcessing);
    qthread->start();
   
    
    



    w.show();
    My obj;
    return a.exec();
    



    return 0;


}