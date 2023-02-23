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


// 实现功能：Print algoritmn version；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     None
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void getVersion()
{
    const char* algVersion =" v1.00.000";
    std::cout<<"Algorithmn version is:"<<algVersion<<std::endl;
    slog::info << ov::get_openvino_version() << slog::endl;
}


// 实现功能：将单张图像由HWC转CHW；注意：只支持cv::Vec3b格式的Mat(uchar数字，一个像素占一个字节)；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     hwcImage：待转换的HWC格式原图;
// outputParas:
//     chwImage：转换后的CHW格式图像；
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void HWC2CHW(INPUT const cv::Mat& hwcImage, OUTPUT uint8_t * chwImage)    // void HWC2CHW(INPUT const cv::Mat& hwcImage, OUTPUT uint8_t chwImage [])
{
	int imgC = hwcImage.channels();
	int imgH = hwcImage.rows;
	int imgW = hwcImage.cols;

	for (int c = 0; c < imgC; ++c)
	{
		for (int h = 0; h < imgH; ++h)
		{
			for (int w = 0; w < imgW; ++w)
			{
				int dstIdx = c * imgH * imgW + h * imgW + w;
				int srcIdx = h * imgW * imgC + w * imgC + c;
				chwImage[dstIdx] =  hwcImage.at<cv::Vec3b>(h, w)[c];   
			}
		}
	}
}


// 实现功能：加载OpenVINO模型；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     unetppTensorParams：ltTensorParams参数
// outputParas:
//     core:OpenVINO Runtime Core；
//     model:加载后的模型
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void loadOvModel(INPUT const ltTensorParams& unetppTensorParams, OUTPUT ov::Core& core, OUTPUT std::shared_ptr<ov::Model>& model)
{
    // Read a model 
    model = core.read_model(unetppTensorParams.modelFilePath);
    printInputAndOutputsInfo(*model);

    // 注意：当前仅支持单输入和单输出头
    OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
    OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");
}


// 实现功能：图像前处理(使用opencv的常规操作完成预处理)；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     srcImg：待转换的HWC格式原图;
//     unetppTensorParams：ltTensorParams参数
// outputParas:
//     hostInputBuffer：处理后的数据(CHW形状、float32格式)，当batch_size为1时可直接送入网络预测；
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void imagePreprocess(INPUT cv::Mat& srcImg, INPUT const ltTensorParams unetppTensorParams, OUTPUT float* hostInputBuffer)
{
	//定义中间变量
	cv::Mat imgResize, imgCvt;

	//resize操作
	cv::resize(srcImg, imgResize, cv::Size(unetppTensorParams.imgW, unetppTensorParams.imgH), (0, 0), (0, 0), cv::INTER_LINEAR);

	//BGR转RGB
	cv::cvtColor(imgResize, imgCvt, cv::COLOR_BGR2RGB);

	//HWC转CHW
	uint8_t chwImg[3*416*512];
	HWC2CHW(imgCvt, chwImg);

	//归一化
	for (int i = 0; i < unetppTensorParams.imgC *unetppTensorParams. imgH * unetppTensorParams.imgW; i++)
	{
		hostInputBuffer[i] =  chwImg[i]/255.0;
	}
}


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
                                               OUTPUT ov::InferRequest& inferRequest)
{
    ov::element::Type input_type = ov::element::u8;
    ov::Shape input_shape = {size_t(unetppTensorParams.batchSize), size_t(orgImg.rows), size_t(orgImg.cols), size_t(orgImg.channels())};
    //注意：inputTensor第三个参数接收void*数据(可以是待推理图像数据段地址),所以输入uchar*也没问题
    inputTensor = ov::Tensor(input_type, input_shape, orgImg.data);
    
    // Configure preprocessing 
    static int pppInitializeTimes = 0;
    if(0 == pppInitializeTimes)
    {
        static ov::preprocess::PrePostProcessor ppp(model);
        
        // Set input tensor information:
        const ov::Layout tensor_layout{"NHWC"};
        ppp.input().tensor().set_shape(input_shape).set_element_type(input_type).set_layout(tensor_layout);
        
        // 要把输入改为 f32 类型，必须用 convert_element_type，而不能用 ppp.input().tensor().set_element_type，否则无效。
        ppp.input().preprocess().convert_element_type(ov::element::f32);
        ppp.input().preprocess().scale(255.0f);  // 做归一化处理，除以 255
        
        // Adding explicit preprocessing steps:
        ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
        
        // Here we suppose model has 'NCHW' layout for input
        ppp.input().model().set_layout("NCHW");
        
        // Set output tensor information: - precision of tensor is supposed to be 'f32'
        ppp.output().tensor().set_element_type(ov::element::f32);
        
        // Apply preprocessing modifying the original 'model'
        model = ppp.build();

        // Loading a model to the device 
        ov::CompiledModel compiledModel = core.compile_model(model, unetppTensorParams.deviceName);

        //Create an infer request 
        inferRequest = compiledModel.create_infer_request();
        pppInitializeTimes ++;
    }
}


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
void inference(INPUT ov::Tensor& inputTensor, INPUT ov::InferRequest& inferRequest, INPUT ov::Tensor& outputTensor, OUTPUT float** output)
{
    // Prepare input 
    inferRequest.set_input_tensor(inputTensor);

    //  Do inference synchronously 
    inferRequest.infer();

    //  Process output
    outputTensor = inferRequest.get_output_tensor();
    * output = (float*)outputTensor.data();    
}


// 实现功能：图像后处理；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     unetppTensorParams：ltTensorParams参数
//     output：推理后的结果；
// outputParas:
//     m_seg_out:后处理之后的索引图
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void imagePostprocess(INPUT const ltTensorParams& unetppTensorParams ,INPUT float* output, OUTPUT cv::Mat& m_seg_out)
{
	m_seg_out = cv::Mat::zeros(unetppTensorParams.imgH, unetppTensorParams.imgW, CV_8UC1);
	for (int h = 0;h < unetppTensorParams.imgH;h++) 
	{
		for (int w = 0;w < unetppTensorParams.imgW;w++) 
		{
			float val_max = output[h*unetppTensorParams.imgW+w];
			unsigned char arg_max = 0;
			for (int c = 1;c < unetppTensorParams.classNum; c++) 
			{
				if (output[h*unetppTensorParams.imgW+w+unetppTensorParams.imgW*unetppTensorParams.imgH*c] > val_max)
				{
					arg_max = c;                                                                                                                                                                              //更新arg_max
					val_max = output[h*unetppTensorParams.imgW+w+unetppTensorParams.imgW*unetppTensorParams.imgH*c];                //更新当前最大值
				}
			}
			m_seg_out.at<uchar>(h,w) = int(arg_max)*60;
		}
	}
}


/*
@brief 用 OpenVino 对图片进行前处理、推理和后处理。输出一个单通道分割图。
@param orgImg 内存中的待推理图像， 是一个 cv::Mat 对象。
@param unetppTensorParams 模型相关参数。
@param core 是 OpenVINO 的 Runtime Core。
@param model 是一个指针，指向一个训练好的分割模型。
@param inputTensor 一个 OpenVINO 张量，输入的图片会被存入该张量中。
@param inferRequest 一个 OpenVINO 的推理请求。
@param output 一个 float* 指针，用于临时存放 OpenVINO 的推理结果。
@param outputTensor 一个 OpenVINO 张量，用于临时存放推理结果。
@param singleSegOut 推理后的单通道索引图。
@return None
*/ 
void inference_online_openvino(INPUT cv::Mat& orgImg, INPUT const ltTensorParams& unetppTensorParams, 
															 INPUT ov::Core& core, INPUT std::shared_ptr<ov::Model>& model, 
															 INPUT ov::Tensor& inputTensor, INPUT ov::InferRequest& inferRequest, 
															 INPUT float* output, INPUT ov::Tensor& outputTensor, 
															 OUTPUT cv::Mat& singleSegOut)
{
	// 1. 预处理。
	int orgImgW = orgImg.cols;
	int orgImgH = orgImg.rows;
	// high_resolution_clock::time_point beginTime = high_resolution_clock::now();    
	imagePreprocess(unetppTensorParams, orgImg, core, model, inputTensor, inferRequest);

	// 2. 推理
	inference(inputTensor, inferRequest, outputTensor, &output);

	// 3. 图像后处理，并将后处理图映射回原图尺寸
	imagePostprocess(unetppTensorParams ,output, singleSegOut);
	cv::resize(singleSegOut, singleSegOut, cv::Size(orgImgW, orgImgH), (0, 0), (0, 0), cv::INTER_LINEAR);

	// high_resolution_clock::time_point endTime = high_resolution_clock::now();       //结束时间
	// milliseconds timeInterval = std::chrono::duration_cast<milliseconds>(endTime - beginTime);
	// std::cout<<"Running time: "<<timeInterval.count()<<" ms"<<std::endl;  // 后续程序稳定后，可以注释掉计时的部分。
	
	// 以下 2 行备用。如果 CPU 占用率过高，可以在推理完一帧图片之后，让出 CPU 300ms，降低占用率。
	// std::this_thread::sleep_for(std::chrono::milliseconds(300)); 
	// std::cout<<"sleep_for: 300 ms" <<std::endl;
	
}
