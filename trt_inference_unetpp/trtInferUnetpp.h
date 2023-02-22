// ==========================================================
// 实现功能：实现使用tensorRT推理unetpp，将所有操作封装成加载模型、推理、释放，隐藏TRT相关代码；
// 文件名称：trtInferenceUnetpp.h
// 相关文件：无
// 作   者：Liangliang Bai (liangliang.bai@leapting.com)
// 版   权：<Copyright(C) 2022-Leapting Technology Co.,LTD All rights reserved.>
// 修改记录：
// 日   期       版本     修改人   走读人  修改记录
// 2022.12.21   1.0.0.1  白亮亮           None
// ==========================================================
#pragma once

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include<vector>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include "sampleUtils.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
//#include <opencv2/dnn/dnn.hpp>
#include <time.h>    //用于测试时间
/*
clock_t startTime, endTime;    
startTime = clock();
endTime = clock();
cout << "Running Time: " << endTime - startTime<<" ms" << endl;
*/
#include <chrono>     //用于测试时间，该方式更精准
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
// high_resolution_clock::time_point beginTime = high_resolution_clock::now();    //起始时间
// high_resolution_clock::time_point endTime = high_resolution_clock::now();       //结束时间
// milliseconds timeInterval = std::chrono::duration_cast<milliseconds>(endTime - beginTime);
// std::cout<<"Running time: "<<timeInterval.count()<<"ms"<<std::endl;

#ifndef INPUT
#define INPUT
#endif

#ifndef OUTPUT
#define OUTPUT
#endif

using namespace std;


// 实现功能：LT_TENSOR_PARAMS结构体，保存张量的常用参数；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     tensorFormat:模型格式(onnx/trt/pth)，当前仅支持TRT；
//     imgH                 :推理时输入网络的图像高，int32_t即signed int；
//     imgW                :推理时输入网络的图像宽；
//     imgC                 :推理时输入网络的图像通道数
//     batchSize       :输入的batch_size大小
//     dlaCore           :Specify the DLA core(深度学习加速核) to run network on.
//     classNum       :类别数
//     Int8                   :推理精度
//     Fp16                 :推理精度
//     inputTensorNames   :模型输入节点的名称
//     outputTensorNames:模型输出节点的名称
//     modelFilePath            :模型保存路径
// outputParas:
//     None
// returnValue:None
// ----------------------------------->parameters<----------------------------------
typedef struct LT_TENSOR_PARAMS
{
	std::string tensorFormat{"TRT"};                                                        
	int32_t imgH{512};                                                                                 
	int32_t imgW{512};                                                                                   
	int32_t imgC{3};                                                                                         
	int32_t batchSize{ 1 };                                                                           
	int32_t dlaCore{ -1 };                                                                                
	int32_t classNum{3};                                                                              

	bool Int8{ false };
	bool Fp16{ false };
	std::vector<std::string> inputTensorNames{"inputs"};             //输入节点的名字
	std::vector<std::string> outputTensorNames{"outputs"};      //输出节点的名字
	std::string modelFilePath{"./"};                                                          //硬盘上的模型名字全称(路径加名字)

} ltTensorParams;


// 实现功能：将单通道图变成三通道图；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     singleChannelImg：单通道图；
// outputParas:
//     colorImg：三通道彩色图；
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void singleChannel2threeeChannel(INPUT const cv::Mat& singleChannelImg, OUTPUT cv::Mat& colorImg);


// 实现功能：保存TRT模型，该功能未测试；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     engine：待保存模型;
//     fileName：模型保存路径;
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
bool saveEngine(const ICudaEngine& engine, const std::string& fileName);


// 实现功能：加载TRT模型；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     engine：TRT模型所在路径;
//     DLACore：指定显卡
//     err：输出流对象，用于输出提示纤信息
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
ICudaEngine* loadEngine(INPUT const std::string& engine, INPUT const int& DLACore, INPUT std::ostream& err);


// 实现功能：将单张图像由HWC转CHW；注意：只支持cv::Vec3b格式的Mat(uchar数字，一个像素占一个字节)；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     hwcImage：待转换的HWC格式原图;
// outputParas:
//     chwImage：转换后的CHW格式图像；
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void HWC2CHW(INPUT const cv::Mat& hwcImage, OUTPUT uint8_t * chwImage);    // void HWC2CHW(INPUT const cv::Mat& hwcImage, OUTPUT uint8_t chwImage [])



// 实现功能：图像前处理；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     srcImg：待转换的HWC格式原图;
//     unetppTensorParams：ltTensorParams参数
// outputParas:
//     hostInputBuffer：处理后的数据(CHW形状、float32格式)，当batch_size为1时可直接送入网络预测；
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void imagePreprocess(INPUT cv::Mat& srcImg, INPUT const ltTensorParams unetppTensorParams, OUTPUT float* hostInputBuffer);



// 实现功能：加载TRT模型并创建会话；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     unetppTensorParams：ltTensorParams参数
// outputParas:
//     buffers:创建好的samplesCommon::BufferManager对象的指针；
//     context:创建好的context；
//     hostInputBuffer：处理后的数据(CHW形状、float32格式)，当batch_size为1时可直接送入网络预测；
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void loadModel(INPUT const ltTensorParams& unetppTensorParams, OUTPUT void ** buffers, OUTPUT void** context, OUTPUT float** hostInputBuffer);


// 实现功能：TRT推理(包括数据从HOST搬运到DEVICE，推理后再将数据从DEVICE搬运到HOST)；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     unetppTensorParams：ltTensorParams参数
//     buffers:创建好的samplesCommon::BufferManager对象的指针；
//     context:创建好的context；
// outputParas:
//     output：推理后的结果；
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void trtInference(INPUT const ltTensorParams& unetppTensorParams, INPUT void * buffer, void * context, OUTPUT float**output);


// 实现功能：图像后处理；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     unetppTensorParams：ltTensorParams参数
//     output：推理后的结果；
// outputParas:
//     m_seg_out:后处理之后的索引图
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void imagePostprocess(INPUT const ltTensorParams& unetppTensorParams ,INPUT float* output, OUTPUT cv::Mat& m_seg_out);



// 实现功能：释放自己开辟的内存；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     buffer：开辟的内存地址；
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void freeModel(INPUT void* buffer);


// 实现功能：trt推理，输入图像路径，输出单通道分割图；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgPath:本地待推理图像路径
// outputParas:
//     singleSegOut:推理后的单通道索引图
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void trtInferenceOffLine(INPUT const std::string& imgPath, OUTPUT cv::Mat& singleSegOut);


// 实现功能：trt推理，输入内存中的待推理图像，输出单通道分割图；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     orgImg:内存中的待推理图像路径
// outputParas:
//     singleSegOut:推理后的单通道索引图
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void trtInferenceOnLine(INPUT cv::Mat& orgImg, OUTPUT cv::Mat& singleSegOut);








