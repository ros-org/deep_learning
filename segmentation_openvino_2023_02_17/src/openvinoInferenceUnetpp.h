// ==========================================================
// 实现功能：使用OpenVINO推理unetpp，将所有操作封装成加载模型、推理、释放，隐藏OpenVINO相关代码；
// 文件名称：openvinoInferenceUnetpp.h
// 相关文件：无
// 作   者：Liangliang Bai (liangliang.bai@leapting.com)
// 版   权：<Copyright(C) 2022-Leapting Technology Co.,LTD All rights reserved.>
// 修改记录：
// 日   期              版本       修改人      走读人      修改记录
// 2023.02.06   1.0.0.1   白亮亮                          None
// ==========================================================

#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include "format_reader_ptr.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

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
	std::string tensorFormat{"OV"};       //当前仅作为一种标识，预留参数(用于标准化推理框架)        
    std::string deviceName{"CPU"};      //当前仅作为一种标识，预留参数(用于标准化推理框架)                                            
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


// 实现功能：Print algoritmn version；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     None
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void getVersion();


// 实现功能：将单张图像由HWC转CHW；注意：只支持cv::Vec3b格式的Mat(uchar数字，一个像素占一个字节)；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     hwcImage：待转换的HWC格式原图;
// outputParas:
//     chwImage：转换后的CHW格式图像；
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void HWC2CHW(INPUT const cv::Mat& hwcImage, OUTPUT uint8_t * chwImage);


// 实现功能：加载OpenVINO模型；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     unetppTensorParams：ltTensorParams参数
// outputParas:
//     core:OpenVINO Runtime Core；
//     model:加载后的模型
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void loadOvModel(INPUT const ltTensorParams& unetppTensorParams, 
                                       OUTPUT ov::Core& core, 
                                       OUTPUT std::shared_ptr<ov::Model>& model);


// 实现功能：图像前处理(使用opencv的常规操作完成预处理)；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     srcImg：待转换的HWC格式原图;
//     unetppTensorParams：ltTensorParams参数
// outputParas:
//     hostInputBuffer：处理后的数据(CHW形状、float32格式)，当batch_size为1时可直接送入网络预测；
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void imagePreprocess(INPUT cv::Mat& srcImg, 
                                               INPUT const ltTensorParams unetppTensorParams, 
											   OUTPUT float* hostInputBuffer);


// 实现功能：图像预处理(使用openvino提供的接口实现)；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     unetppTensorParams：ltTensorParams参数;
//     orgImg:输入的待推理图像;
//     core:OpenVINO Runtime Core；
//     model:加载后的模型;
// outputParas:
//     model:加载后的模型
//     inputTensor:OpenVINO官方提供的tensor数据类型，需要将我们自己的数据封装进该数据，然后送入网络推理；
//     inferRequest:infer request；
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void imagePreprocess(INPUT const ltTensorParams& unetppTensorParams, 
											   INPUT cv::Mat& orgImg, 
											   INPUT ov::Core& core, 
											   INPUT std::shared_ptr<ov::Model>& model,
											   OUTPUT ov::Tensor& inputTensor, 
											   OUTPUT ov::InferRequest& inferRequest);


// 实现功能：使用OpenVINO推理；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     inputTensor:OpenVINO官方提供的tensor数据类型，需要将我们自己的数据封装进该数据，然后送入网络推理；
//     inferRequest:已经在imagePreprocess函数中创建的infer request；
// outputParas:
//     model:加载后的模型
//     outputTensor:OpenVINO官方提供的tensor数据类型，推理后的结果保存在该数据中，需要自己从中解析；
//     output:获取的结果数据地址
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void inference(INPUT ov::Tensor& inputTensor, 
							   INPUT ov::InferRequest& inferRequest, 
							   INPUT ov::Tensor& outputTensor, 
							   OUTPUT float** output);


// 实现功能：图像后处理；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     unetppTensorParams：ltTensorParams参数
//     output：推理后的结果；
// outputParas:
//     m_seg_out:后处理之后的索引图
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void imagePostprocess(INPUT const ltTensorParams& unetppTensorParams ,
                                                 INPUT float* output, 
                                                 OUTPUT cv::Mat& m_seg_out);


/*
@brief 用 OpenVino 模型推理，输入内存中的待推理图像，输出单通道分割图；
@param orgImg 内存中的待推理图像路径
@param unetppTensorParams 模型相关参数
@param buffer ？？？
@param context ？？？
@param hostInputBuffer ？？？
@param singleSegOut 推理后的单通道索引图
@return None
*/ 
void inference_online_openvino(INPUT cv::Mat& orgImg, INPUT const ltTensorParams& unetppTensorParams, 
															 INPUT void* buffer, INPUT void* context, INPUT float* hostInputBuffer, 
															 OUTPUT cv::Mat& singleSegOut);
