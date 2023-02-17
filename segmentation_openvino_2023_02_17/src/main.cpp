// ==========================================================
// 实现功能：使用OpenVINO推理unetpp，将所有操作封装成加载模型、推理、释放，隐藏OpenVINO相关代码；
// 文件名称：openvinoInferenceUnetpp.cpp
// 相关文件：无
// 作   者：Liangliang Bai (liangliang.bai@leapting.com)
// 版   权：<Copyright(C) 2022-Leapting Technology Co.,LTD All rights reserved.>
// 修改记录：
// 日   期              版本       修改人      走读人      修改记录
// 2023.02.06   1.0.0.1   白亮亮                          None
// ==========================================================

#include "openvinoInferenceUnetpp.h"
#include <opencv2/core/utility.hpp>

int main()
{
    //1、参数初始化
    std::cout<<"-----------------------------------Test openvino demo-----------------------------------"<<std::endl;
    ltTensorParams unetppTensorParams;
    unetppTensorParams.inputTensorNames.push_back("inputs");
    unetppTensorParams.batchSize = 1;
    unetppTensorParams.outputTensorNames.push_back("outputs");
    unetppTensorParams.imgC = 3;
    unetppTensorParams.imgH = 400;  // must be 400 when using  400x640_v2.onnx
    unetppTensorParams.imgW = 640;  // must be 640 when using  400x640_v2.onnx     
    unetppTensorParams.modelFilePath = "/home/leapting/jun_ws/OPENVINO_TEST/model/400x640_v2.onnx";
    unetppTensorParams.classNum = 2;  // must be 2 when using  400x640_v2.onnx
    unetppTensorParams.deviceName = "CPU";

    // 2、 Initialize OpenVINO Runtime Core and load model using openvino
    ov::Core core;    
    std::shared_ptr<ov::Model> model;
    loadOvModel(unetppTensorParams,  core,  model);

    double tic = cv::getTickCount();
    //相关参数定义:连续推理时，将这些参数放出来，防止重复申请内存；
    std::string imgPath;
    cv::Mat orgImg;
    int orgImgW;
    int orgImgH;
    ov::Tensor inputTensor;
    high_resolution_clock::time_point beginTime;
    ov::InferRequest inferRequest;
    std::vector<cv::String> filenames; 
	cv::String folder =  "/home/leapting/jun_ws/OPENVINO_TEST/test_image/2/";
	cv::glob(folder, filenames); 
	for (size_t i = 0; i < filenames.size(); ++i)
	{
        //3、图像获取及预处理
        imgPath = filenames[i];           //  "/home/bailiangliang/OPENVINO_TEST/test_image/1/Color_1.bmp";
        std::cout<<"imgPath="<<imgPath<<std::endl;
        orgImg =  cv::imread(filenames[i]);
        orgImgW = orgImg.cols;
        orgImgH = orgImg.rows;
        beginTime = high_resolution_clock::now();    
        imagePreprocess(unetppTensorParams,  orgImg, core, model, inputTensor, inferRequest);
        
        //4、推理
        float* output = nullptr;
        ov::Tensor outputTensor;
        inference(inputTensor, inferRequest, outputTensor, &output);

        //5、图像后处理，并将后处理图映射回原图尺寸
        cv::Mat singleSegOut;
        std::string imgSavePath = "/home/leapting/jun_ws/OPENVINO_TEST/image_test_result/xxxxxxx.png";
        imagePostprocess(unetppTensorParams ,output, singleSegOut);
        cv::resize(singleSegOut, singleSegOut, cv::Size(orgImgW, orgImgH), (0, 0), (0, 0), cv::INTER_LINEAR);
        cv::imwrite(imgSavePath, singleSegOut);
        high_resolution_clock::time_point endTime = high_resolution_clock::now();       //结束时间
        milliseconds timeInterval = std::chrono::duration_cast<milliseconds>(endTime - beginTime);
        std::cout<<"Running time: "<<timeInterval.count()<<"ms"<<std::endl;
    }
    tic = (cv::getTickCount() - tic) / cv::getTickFrequency();
    std::cout<<"time spent:  "<< tic <<" seconds" <<std::endl;
    std::cout<<"images =  "<< filenames.size() <<std::endl;
    double speed = filenames.size() / tic;
    std::cout<<"speed =  "<< speed << "fps" <<std::endl;    

    return 0;
}

