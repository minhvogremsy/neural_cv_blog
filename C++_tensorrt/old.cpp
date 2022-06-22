// std::vector<String> outNames_siamRPN;
// outNames_siamRPN = siamRPN.getLayerNames();
// for (auto i: outNames_siamRPN) {
//         std::cout << " " <<i << ' ';}
// std::cout << '\n';

// std::vector<cv::dnn::MatShape> inLayerShapes;
// std::vector<cv::dnn::MatShape> outLayerShapes;
// for (int i = 0; i < 22; i++) {
// siamRPN.getLayerShapes(cv::dnn::MatShape(), i, inLayerShapes, outLayerShapes);
// std::cout <<"\n" <<std::endl;
// std::cout << "in layer shapes: " << std::endl;
// for (const auto& n1 : inLayerShapes) {
//     std::cout << "- shape: " << std::endl;
//     for (const auto n2 : n1) {
//         std::cout << "  " + std::to_string(n2) << " ";
//     }
// }
// std::cout <<"\n" <<std::endl;
// std::cout << "out layer shapes: " << std::endl;
// for (const auto& n1 : outLayerShapes) {
//     std::cout << "- shape: " << std::endl;
//     for (const auto n2 : n1) {
//         std::cout << "  " + std::to_string(n2) << " ";
//     }
// }

// }
// 
// 
// 
// std::cout << std::endl;


    //std::vector<cv::dnn::MatShape> inLayerShapes;
    //std::vector<cv::dnn::MatShape> outLayerShapes;
    // for (int i = 0; i < 17; i++) {
    // siamRPN.getLayerShapes(cv::dnn::MatShape(), i, inLayerShapes, outLayerShapes);
    // std::cout <<"\n" <<std::endl;
    // std::cout << "in layer shapes: " << std::endl;
    // for (const auto& n1 : inLayerShapes) {
    //     std::cout << "- shape: " << std::endl;
    //     for (const auto n2 : n1) {
    //         std::cout << "  " + std::to_string(n2) << " ";
    //     }
    // }
    // std::cout <<"\n" <<std::endl;
    // std::cout << "out layer shapes: " << std::endl;
    // for (const auto& n1 : outLayerShapes) {
    //     std::cout << "- shape: " << std::endl;
    //     for (const auto n2 : n1) {
    //         std::cout << "  " + std::to_string(n2) << " ";
    //     }
    // }

    // }
    // std::cout << std::endl;
    // std::vector<int> indexs =  siamRPN.getUnconnectedOutLayers();
    // for (auto i: indexs) {
    //         std::cout << " " <<i << " - ";}
    // std::cout <<"\n";



         // std::cout << "delta  1 : " << delta.size  << std::endl;
            // std::cout << "score  1 : " << score.size  << std::endl;
            //score = score.reshape(0, { 1, 10, 19,19});
            //delta = delta.reshape(0, { 1, 20, 19,19});
            // std::cout << "delta  1 : " << delta.size  << std::endl;
            // std::cout << "score  1 : " << score.size  << std::endl;



    //cv::Mat frame = blob.clone();

    //std::cout << "blob  2 : " << blob.size  << std::endl;

    //siamRPN.setInput(blob);

    //outNames = siamRPN.getUnconnectedOutLayersNames();
    //for (auto i: outNames)
    //        std::cout << "Out " <<i << ' ';
    //std::cout << '\n';

    //siamRPN.forward(outs, outNames);


        //int size_bias = (int)cls_bias.total() * cls_bias.channels();
    //std::cout << "!!  OUT    size_bias !! = " << size_bias << std::endl;
    //Weights wt_bias{ DataType::kFLOAT, cls_bias.data, size_bias };
    //bool out_bias = refitter->setWeights("Conv_15", WeightsRole::kBIAS, wt_bias);
    //std::cout << "!!  out_bias !! " << out_bias << std::endl;

    //---------------------------------------------------------//

    //int sz_65[] = {20,256, 4, 4};
    //cv::Mat bigCube_65(4, sz_65, CV_16FC1, cv::Scalar::all(1.));


        //int  size_65_bias = (int)r1_bias.total() * r1_bias.channels();
    //std::cout << "!!  OUT    size_65_bias !! = " << size_65_bias << std::endl;
    //Weights wt_65_bias{ DataType::kFLOAT, r1_bias.data, size_65_bias };
    //bool out_bias_65 = refitter->setWeights("Conv_15", WeightsRole::kBIAS, wt_bias);
    //std::cout << "!!  out_bias !! " << out_bias_65 << std::endl;



        // //resize and normalize.
    // cv::Mat resize_im, rgb_im, normalize_im;

    // cv::resize(input_data, resize_im, cv::Size(271, 271));
    // cv::cvtColor(resize_im, rgb_im, cv::COLOR_BGR2RGB);
    // rgb_im.convertTo(normalize_im, CV_32FC3);


    //int sz[] = {256, 24, 24};
    //cv::Mat bigCube(3, sz, CV_32FC1, Scalar::all(1.));  
    //int new_sz[] = {360, 480};
    //auto new_m = normalize_im.reshape(3, new_sz);

    //normalize_im = normalize_im.reshape(3,input_width_,input_height_);
    //std::cout << "reshape" << std::endl;