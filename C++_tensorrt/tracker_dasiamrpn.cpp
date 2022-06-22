// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/video.hpp>
#include <opencv2/core/utility.hpp>
//#include "opencv2/core/private.hpp"
#include <opencv2/core/ocl.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <typeinfo>
#include "treacker.h"

#define OPENCV //
//#define CUDNN //

using namespace cv;
template <typename T> static
T sizeCal(const T& w, const T& h)
{
    T pad = (w + h) * T(0.5);
    T sz2 = (w + pad) * (h + pad);
    return sqrt(sz2);
}

template <>
cv::Mat sizeCal(const cv::Mat& w, const cv::Mat& h)
{
    cv::Mat pad = (w + h) * 0.5;
    cv::Mat sz2 = (w + pad).mul((h + pad));

    cv::sqrt(sz2, sz2);
    return sz2;
}

Da_treacker::Da_treacker(const std::string model, const std::string kernel_cls1, const std::string kernel_r1)
    : model_(model)
    , kernel_cls1_(kernel_cls1)
    , kernel_r1_(kernel_r1)
 
    
{
        siamRPN = cv::dnn::readNet(model);
        siamKernelCL1 = cv::dnn::readNet(kernel_cls1);
        siamKernelR1 = cv::dnn::readNet(kernel_r1);
        //
        detector = new ObjectDetector_fp16(271, 271);
        //detector = std::make_unique< new ObjectDetector_fp16(271, 271)>(271, 271);// new ObjectDetector_fp16(271, 271);
        siamRPN.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        siamRPN.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        siamKernelR1.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        siamKernelR1.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        siamKernelCL1.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        siamKernelCL1.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        start = true;
}



void Da_treacker::init(cv::InputArray image, const cv::Rect& boundingBox)
{
    image_ = image.getMat().clone();

    trackState.update_scoreSize();
    trackState.targetBox = cv::Rect2f(
        float(boundingBox.x) + float(boundingBox.width) * 0.5f,  // FIXIT don't use center in Rect structures, it is confusing
        float(boundingBox.y) + float(boundingBox.height) * 0.5f,
        float(boundingBox.width),
        float(boundingBox.height)
    );
    trackerInit(image_);
}


void Da_treacker::trackerInit(cv::Mat img)
{
    Rect2f targetBox = trackState.targetBox;
    cv::Mat anchors = generateAnchors();
    trackState.anchors = anchors;

    cv::Mat windows = generateHanningWindow();

    trackState.windows = windows;
    trackState.imgSize = img.size();

    trackState.avgChans = mean(img);
    float wc = targetBox.width + trackState.contextAmount * (targetBox.width + targetBox.height);
    float hc = targetBox.height + trackState.contextAmount * (targetBox.width + targetBox.height);
    float sz = (float)cvRound(sqrt(wc * hc));

    cv::Mat zCrop = getSubwindow(img, targetBox, sz, trackState.avgChans);
    cv::Mat blob;

    cv::dnn::blobFromImage(zCrop, blob, 1.0, Size(trackState.exemplarSize, trackState.exemplarSize), Scalar(), trackState.swapRB, false, CV_32F); //- > cvmat





    

    siamRPN.setInput(blob);
    std::cout << "blob   : " << blob.size  << std::endl;
    cv::Mat out1;
    //
    std::vector<String> outNames_siamRPN;
    outNames_siamRPN = siamRPN.getLayerNames();
    for (auto i: outNames_siamRPN) {
            std::cout << " " <<i << ' ';}
    std::cout << '\n';
    

    std::cout << std::endl;

   
    siamRPN.forward(out1, "input.52");
    

    siamKernelCL1.setInput(out1);
    siamKernelR1.setInput(out1);

    cv::Mat cls1 = siamKernelCL1.forward();
    std::cout << "scale cls1  : " << cls1.size  << std::endl;

    cv::Mat r1 = siamKernelR1.forward();
    std::cout << "scale r1  : " << r1.size << std::endl;

    std::vector<int> r1_shape = { 20, 256, 4, 4 }, cls1_shape = { 10, 256, 4, 4 };
    
    
    std::vector<int> index =  siamRPN.getUnconnectedOutLayers();
    for (auto i: index) {
            std::cout << " " <<i << " - ";}
    std::cout <<"\n";

    #ifdef OPENCV
    siamRPN.setParam(siamRPN.getLayerId("input.60"), 0, r1.reshape(0, r1_shape));
    siamRPN.setParam(siamRPN.getLayerId("output1"), 0, cls1.reshape(0, cls1_shape));
    #endif

    cv::String model_path_4 = "dasiamrpn_model32_r_271.trt";
    int cls1_size = (int)cls1.total() * cls1.channels();
    int r1_size = (int)r1.total() * r1.channels();
    
    #ifdef CUDNN
    if (start) {
    detector->LoadEngine(model_path_4,  cls1.reshape(0, cls1_shape), r1.reshape(0, r1_shape)); 
    start = false;
    }
    else {
        detector->RebuldEngine(cls1.reshape(0, cls1_shape), r1.reshape(0, r1_shape));
    }
    #endif
    
    outNames_siamRPN = siamRPN.getLayerNames();
    for (auto i: outNames_siamRPN) {
            std::cout << " " <<i << ' ';}
    std::cout << '\n';

   
}

cv::Mat Da_treacker::generateHanningWindow()
{
    cv::Mat baseWindows, HanningWindows;

    createHanningWindow(baseWindows, Size(trackState.scoreSize, trackState.scoreSize), CV_32F);
    baseWindows = baseWindows.reshape(0, { 1, trackState.scoreSize, trackState.scoreSize });
    HanningWindows = baseWindows.clone();
    for (int i = 1; i < trackState.anchorNum; i++)
    {
        HanningWindows.push_back(baseWindows);
    }

    return HanningWindows;
}


cv::Mat Da_treacker::generateAnchors()
{
    int totalStride = trackState.totalStride, scales = trackState.scale, scoreSize = trackState.scoreSize;
    std::vector<float> ratios = trackState.ratios;
    std::vector<cv::Rect2f> baseAnchors;
    int anchorNum = int(ratios.size());
    int size = totalStride * totalStride;

    float ori = -(float(scoreSize / 2)) * float(totalStride);

    for (auto i = 0; i < anchorNum; i++)
    {
        int ws = int(sqrt(size / ratios[i]));
        int hs = int(ws * ratios[i]);

        float wws = float(ws) * scales;
        float hhs = float(hs) * scales;
        Rect2f anchor = { 0, 0, wws, hhs };
        baseAnchors.push_back(anchor);
    }

    int anchorIndex[4] = { 0, 0, 0, 0 };
    const int sizes[4] = { 4, (int)ratios.size(), scoreSize, scoreSize };
    cv::Mat anchors(4, sizes, CV_32F);

    for (auto i = 0; i < scoreSize; i++)
    {
        for (auto j = 0; j < scoreSize; j++)
        {
            for (auto k = 0; k < anchorNum; k++)
            {
                anchorIndex[0] = 1, anchorIndex[1] = k, anchorIndex[2] = i, anchorIndex[3] = j;
                anchors.at<float>(anchorIndex) = ori + totalStride * i;

                anchorIndex[0] = 0;
                anchors.at<float>(anchorIndex) = ori + totalStride * j;

                anchorIndex[0] = 2;
                anchors.at<float>(anchorIndex) = baseAnchors[k].width;

                anchorIndex[0] = 3;
                anchors.at<float>(anchorIndex) = baseAnchors[k].height;
            }
        }
    }

    return anchors;
}

Mat Da_treacker::getSubwindow(Mat& img, const Rect2f& targetBox, float originalSize, Scalar avgChans)
{
    Mat zCrop, dst;
    Size imgSize = img.size();
    float c = (originalSize + 1) / 2;
    float xMin = (float)cvRound(targetBox.x - c);
    float xMax = xMin + originalSize - 1;
    float yMin = (float)cvRound(targetBox.y - c);
    float yMax = yMin + originalSize - 1;

    int leftPad = (int)(fmax(0., -xMin));
    int topPad = (int)(fmax(0., -yMin));
    int rightPad = (int)(fmax(0., xMax - imgSize.width + 1));
    int bottomPad = (int)(fmax(0., yMax - imgSize.height + 1));

    xMin = xMin + leftPad;
    xMax = xMax + leftPad;
    yMax = yMax + topPad;
    yMin = yMin + topPad;

    if (topPad == 0 && bottomPad == 0 && leftPad == 0 && rightPad == 0)
    {
        img(Rect(int(xMin), int(yMin), int(xMax - xMin + 1), int(yMax - yMin + 1))).copyTo(zCrop);
    }
    else
    {
        copyMakeBorder(img, dst, topPad, bottomPad, leftPad, rightPad, BORDER_CONSTANT, avgChans);
        dst(Rect(int(xMin), int(yMin), int(xMax - xMin + 1), int(yMax - yMin + 1))).copyTo(zCrop);
    }

    return zCrop;
}


bool Da_treacker::update(InputArray image, Rect& boundingBox)
{
    image_ = image.getMat().clone();
    trackerEval(image_);
    boundingBox = {
        int(trackState.targetBox.x - int(trackState.targetBox.width / 2)),
        int(trackState.targetBox.y - int(trackState.targetBox.height / 2)),
        int(trackState.targetBox.width),
        int(trackState.targetBox.height)
    };
    return true;
}

void Da_treacker::trackerEval(Mat img)
{
    Rect2f targetBox = trackState.targetBox;

    float wc = targetBox.height + trackState.contextAmount * (targetBox.width + targetBox.height);
    float hc = targetBox.width + trackState.contextAmount * (targetBox.width + targetBox.height);

    float sz = sqrt(wc * hc);
    float scaleZ = trackState.exemplarSize / sz;

    float searchSize = float((trackState.instanceSize - trackState.exemplarSize) / 2);
    float pad = searchSize / scaleZ;
    float sx = sz + 2 * pad;

    Mat xCrop = getSubwindow(img, targetBox, (float)cvRound(sx), trackState.avgChans);

    Mat blob;
    std::vector<Mat> outs;
    std::vector<String> outNames;
    Mat delta, score;
    Mat sc, rc, penalty, pscore;

    dnn::blobFromImage(xCrop, blob, 1.0, Size(trackState.instanceSize, trackState.instanceSize), Scalar(), trackState.swapRB, false, CV_32F);


    #ifdef OPENCV


    siamRPN.setInput(blob);

    outNames = siamRPN.getUnconnectedOutLayersNames();
    for (auto i: outNames)
            std::cout << "Out " <<i << ' ';
    std::cout << '\n';

    siamRPN.forward(outs, outNames);

    delta = outs[0];
    score = outs[1];

    #endif

    #ifdef CUDNN


 
    std::chrono::duration<double, std::milli> inference_time_span_4;
    std::cout << "blob  2 : " << blob.size  << std::endl;
    
    const auto& result_4 = detector->RunInference(blob, inference_time_span_4);
    std::ostringstream time_caption_4;
    time_caption_4 << std::fixed << std::setprecision(2) << inference_time_span_4.count() << " ms";
    std::cout << "time_caption : " << time_caption_4.str() << std::endl;
     for (const auto& object : *result_4)
        {
            delta = object.out_mat_1;
            score =  object.out_mat_2;
       


        }
    #endif


    score = score.reshape(0, { 2, trackState.anchorNum, trackState.scoreSize, trackState.scoreSize });
    delta = delta.reshape(0, { 4, trackState.anchorNum, trackState.scoreSize, trackState.scoreSize });



    softmax(score, score);

    targetBox.width *= scaleZ;
    targetBox.height *= scaleZ;

    score = score.row(1);
    score = score.reshape(0, { 5, 19, 19 });

    // Post processing
    delta.row(0) = delta.row(0).mul(trackState.anchors.row(2)) + trackState.anchors.row(0);
    delta.row(1) = delta.row(1).mul(trackState.anchors.row(3)) + trackState.anchors.row(1);
    exp(delta.row(2), delta.row(2));
    delta.row(2) = delta.row(2).mul(trackState.anchors.row(2));
    exp(delta.row(3), delta.row(3));
    delta.row(3) = delta.row(3).mul(trackState.anchors.row(3));

    sc = sizeCal(delta.row(2), delta.row(3)) / sizeCal(targetBox.width, targetBox.height);
    elementMax(sc);

    rc = delta.row(2).mul(1 / delta.row(3));
    rc = (targetBox.width / targetBox.height) / rc;
    elementMax(rc);

    // Calculating the penalty
    exp(((rc.mul(sc) - 1.) * trackState.penaltyK * (-1.0)), penalty);
    penalty = penalty.reshape(0, { trackState.anchorNum, trackState.scoreSize, trackState.scoreSize });

    pscore = penalty.mul(score);
    pscore = pscore * (1.0 - trackState.windowInfluence) + trackState.windows * trackState.windowInfluence;

    int bestID[2] = { 0, 0 };
    // Find the index of best score.
    minMaxIdx(pscore.reshape(0, { trackState.anchorNum * trackState.scoreSize * trackState.scoreSize, 1 }), 0, 0, 0, bestID);
    delta = delta.reshape(0, { 4, trackState.anchorNum * trackState.scoreSize * trackState.scoreSize });
    penalty = penalty.reshape(0, { trackState.anchorNum * trackState.scoreSize * trackState.scoreSize, 1 });
    score = score.reshape(0, { trackState.anchorNum * trackState.scoreSize * trackState.scoreSize, 1 });

    int index[2] = { 0, bestID[0] };
    Rect2f resBox = { 0, 0, 0, 0 };

    resBox.x = delta.at<float>(index) / scaleZ;
    index[0] = 1;
    resBox.y = delta.at<float>(index) / scaleZ;
    index[0] = 2;
    resBox.width = delta.at<float>(index) / scaleZ;
    index[0] = 3;
    resBox.height = delta.at<float>(index) / scaleZ;

    float lr = penalty.at<float>(bestID) * score.at<float>(bestID) * trackState.lr;

    resBox.x = resBox.x + targetBox.x;
    resBox.y = resBox.y + targetBox.y;
    targetBox.width /= scaleZ;
    targetBox.height /= scaleZ;

    resBox.width = targetBox.width * (1 - lr) + resBox.width * lr;
    resBox.height = targetBox.height * (1 - lr) + resBox.height * lr;

    resBox.x = float(fmax(0., fmin(float(trackState.imgSize.width), resBox.x)));
    resBox.y = float(fmax(0., fmin(float(trackState.imgSize.height), resBox.y)));
    resBox.width = float(fmax(10., fmin(float(trackState.imgSize.width), resBox.width)));
    resBox.height = float(fmax(10., fmin(float(trackState.imgSize.height), resBox.height)));

    trackState.targetBox = resBox;
    trackState.tracking_score = score.at<float>(bestID);
}

float Da_treacker::getTrackingScore()
{
    return trackState.tracking_score;
}


void Da_treacker::softmax(const Mat& src, Mat& dst)
{
    Mat maxVal;
    cv::max(src.row(1), src.row(0), maxVal);

    src.row(1) -= maxVal;
    src.row(0) -= maxVal;

    exp(src, dst);

    Mat sumVal = dst.row(0) + dst.row(1);
    dst.row(0) = dst.row(0) / sumVal;
    dst.row(1) = dst.row(1) / sumVal;
}

void Da_treacker::elementMax(Mat& src)
{
    int* p = src.size.p;
    int index[4] = { 0, 0, 0, 0 };
    for (int n = 0; n < *p; n++)
    {
        for (int k = 0; k < *(p + 1); k++)
        {
            for (int i = 0; i < *(p + 2); i++)
            {
                for (int j = 0; j < *(p + 3); j++)
                {
                    index[0] = n, index[1] = k, index[2] = i, index[3] = j;
                    float& v = src.at<float>(index);
                    v = fmax(v, 1.0f / v);
                }
            }
        }
    }
}


