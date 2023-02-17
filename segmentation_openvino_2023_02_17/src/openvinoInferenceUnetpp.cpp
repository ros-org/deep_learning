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
															 OUTPUT cv::Mat& singleSegOut)
{
	//1、参数初始化
	// ltTensorParams unetppTensorParams;
	// unetppTensorParams.inputTensorNames.push_back("inputs");
	// unetppTensorParams.batchSize = 1;
	// unetppTensorParams.outputTensorNames.push_back("outputs");
	// unetppTensorParams.imgC = 3;
	// unetppTensorParams.imgH = 416;
	// unetppTensorParams.imgW = 512;
	// unetppTensorParams.modelFilePath = "/home/bailiangliang/deep_learning_deploy_on_pc/unetpp/epoch_128.trt";
	// unetppTensorParams.classNum = 3;

	//2、创建TRT引擎和会话
	// void * buffer = nullptr;
	// void * context = nullptr;
	// float* hostInputBuffer=nullptr;
	// loadModel(unetppTensorParams, &buffer, &context, &hostInputBuffer);
	
	//3、图像读取与前处理
	// cv::Mat orgImg =  cv::imread(imgPath);
	imagePreprocess(orgImg, unetppTensorParams, hostInputBuffer);
	
	//4、TRT引擎推理
	float *output;    
	// 暂时注释掉下面这行的 trt 推理。
	// trtInference(unetppTensorParams, buffer,  context, &output);

	//5、图像后处理，并将后处理图映射回原图尺寸
	cv::Mat colorImg;
	// std::string imgSavePath = "/home/bailiangliang/deep_learning_deploy_on_pc/unetpp/test_image/xxx.bmp";
	// 暂时注释掉下面这行的后处理。
    // imagePostprocess(unetppTensorParams ,output, singleSegOut);
	cv::resize(singleSegOut, singleSegOut, cv::Size(orgImg.cols, orgImg.rows), (0, 0), (0, 0), cv::INTER_LINEAR);
	// singleChannel2threeeChannel(singleSegOut, colorImg);    
	// cv::imwrite(imgSavePath, colorImg);

	//6、输入上面处理后的三通道索引图，获取矩形中心和组件边缘直线
	#if 0    //获取两个矩形
	cv::Point cen;
	cv::Mat display;
	get_ObjRectCenter(colorImg, cen, display);
	#endif

	#if 0    //获取直线
	cv::Mat display;
	float line_angle = get_ObjLineAngle(colorImg, display);
	#endif
	
	//6、内存释放
	// freeModel(buffer,  context);
}
