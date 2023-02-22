// ==========================================================
// 实现功能：实现使用tensorRT推理unetpp，将所有操作封装成加载模型、推理、释放，隐藏TRT相关代码；
// 文件名称：trtInferenceUnetpp.cpp
// 相关文件：无
// 作   者：Liangliang Bai (liangliang.bai@leapting.com)
// 版   权：<Copyright(C) 2022-Leapting Technology Co.,LTD All rights reserved.>
// 修改记录：
// 日   期       版本     修改人   走读人  修改记录
// 2022.12.21   1.0.0.1  白亮亮           None
// ==========================================================
#pragma once

#include"trtInferUnetpp.h"

// 实现功能：将单通道图变成三通道图；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     singleChannelImg：单通道图；
// outputParas:
//     colorImg：三通道彩色图；
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void singleChannel2threeeChannel(INPUT const cv::Mat& singleChannelImg, OUTPUT cv::Mat& colorImg)
{
    colorImg = cv::Mat::zeros(singleChannelImg.rows,singleChannelImg.cols, CV_8UC3);
    vector<cv::Mat> channels;
    for (int i=0;i<3;i++)
    {
        channels.push_back(singleChannelImg);
    }
    merge(channels,colorImg);
	return;
}


// 实现功能：保存TRT模型，该功能未测试；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     engine：待保存模型;
//     fileName：模型保存路径;
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
bool saveEngine(const ICudaEngine& engine, const std::string& fileName)
{
	std::ofstream engineFile(fileName, std::ios::binary);
	if (!engineFile)
	{
		std::cout << "Cannot open engine file: " << fileName << std::endl;
		return false;
	}

	IHostMemory* serializedEngine = engine.serialize();
	if (serializedEngine == nullptr)
	{
		std::cout << "Engine serialization failed" << std::endl;
		return false;
	}

	engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
	return !engineFile.fail();
}

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
ICudaEngine* loadEngine(INPUT const std::string& engine, INPUT const int& DLACore, INPUT std::ostream& err)
{
	std::ifstream engineFile(engine, std::ios::binary);
	if (!engineFile)
	{
		err << "Error opening engine file: " << engine << std::endl;
		return nullptr;
	}

	engineFile.seekg(0, engineFile.end);
	long int fsize = engineFile.tellg();
	engineFile.seekg(0, engineFile.beg);

	std::vector<char> engineData(fsize);
	engineFile.read(engineData.data(), fsize);
	if (!engineFile)
	{
		err << "Error loading engine file: " << engine << std::endl;
		return nullptr;
	}

	IRuntime* runtime = createInferRuntime(sample::gLogger.getTRTLogger());
	if (DLACore != -1)
	{
		runtime->setDLACore(DLACore);
	}

	return runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
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


// 实现功能：图像前处理；
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
void loadModel(INPUT const ltTensorParams& unetppTensorParams, OUTPUT void ** buffers, OUTPUT void** context, OUTPUT float** hostInputBuffer)
{
	// 1、从TRT文件读取模型数据
	nvinfer1::ICudaEngine* engineNew = loadEngine(unetppTensorParams.modelFilePath, 0, sample::gLogError);
	if (!engineNew) 
	{
		std::cout<<"TRT模型加载失败"<<std::endl;
	}
	else
	{
		std::cout<<"TRT模型加载成功"<<std::endl;
	}

	// 2、创建 RAII buffer 管理对象
	shared_ptr<nvinfer1::ICudaEngine> mEngineNew2 = shared_ptr<nvinfer1::ICudaEngine>(engineNew, samplesCommon::InferDeleter());
	samplesCommon::BufferManager* buffer = new samplesCommon::BufferManager (mEngineNew2);
	

	//3、创建context
	nvinfer1::IExecutionContext* _context = engineNew->createExecutionContext();
	if (!_context) 
	{
		std::cout<<"_context失败"<<std::endl;
	}
	else
	{
		std::cout<<"_context成功"<<std::endl;
	}
	
	*hostInputBuffer = static_cast<float*>(buffer->getHostBuffer(unetppTensorParams.inputTensorNames[0]));    //根据trt输入tensor的名称分配内存host buffer

	*buffers = buffer;
	*context = _context;
}


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
void trtInference(INPUT const ltTensorParams& unetppTensorParams, INPUT void * buffer, void * context, OUTPUT float**output)
{
	//对void*强制转换
	samplesCommon::BufferManager* buffers = static_cast<samplesCommon::BufferManager*> (buffer);
	nvinfer1::IExecutionContext* _context = static_cast<nvinfer1::IExecutionContext*> (context);

	//5、 Memcpy from host input buffers to device input buffers
	buffers->copyInputToDevice();

	//6、TensorRT execution is typically asynchronous, so enqueue the kernels on a CUDA stream:
	bool status = _context->executeV2(buffers->getDeviceBindings().data());
	if(status)
	{
		std::cout<<"Trt inference success."<<std::endl;
	}
	else
	{
		std::cout<<"Trt inference fail"<<std::endl;
	}

	//7、Memcpy from device output buffers to host output buffers
	buffers->copyOutputToHost();

	//8、Get Results from Buffers
	*output = static_cast<float*>(buffers->getHostBuffer(unetppTensorParams.outputTensorNames[0]));
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


// 实现功能：释放自己开辟的内存；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     buffer：开辟的内存地址；
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void freeModel(INPUT void* buffer)
{
    float* ptr = (float*) buffer;
	if (nullptr != buffer)
	{
		delete buffer;
		buffer = nullptr;
	}
}


// 实现功能：trt推理，输入图像路径，输出单通道分割图；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgPath:本地待推理图像路径
// outputParas:
//     singleSegOut:推理后的单通道索引图
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void trtInferenceOffLine(INPUT const std::string& imgPath, OUTPUT cv::Mat& singleSegOut)
{
	//1、参数初始化
	ltTensorParams unetppTensorParams;
	unetppTensorParams.inputTensorNames.push_back("inputs");
	unetppTensorParams.batchSize = 1;
	unetppTensorParams.outputTensorNames.push_back("outputs");
	unetppTensorParams.imgC = 3;
	unetppTensorParams.imgH = 416;
	unetppTensorParams.imgW = 512;
	unetppTensorParams.modelFilePath = "/home/bailiangliang/deep_learning_deploy_on_pc/unetpp/epoch_128.trt";
	unetppTensorParams.classNum = 3;

	//2、创建TRT引擎和会话
	void * buffer = nullptr;
	void * context = nullptr;
	float* hostInputBuffer;
	loadModel(unetppTensorParams, &buffer, &context, &hostInputBuffer);
	
	//3、图像读取与前处理
	//std::string imgPath = "/home/bailiangliang/deep_learning_deploy_on_pc/unetpp/test_image/2/Color_38.bmp";
	cv::Mat orgImg =  cv::imread(imgPath);
	int orgImgW = orgImg.cols;
	int orgImgH = orgImg.rows;

	imagePreprocess(orgImg, unetppTensorParams, hostInputBuffer);
	
	//4、TRT引擎推理
	float *output;    
	trtInference(unetppTensorParams, buffer,  context, &output);

	//5、图像后处理，并将后处理图映射回原图尺寸
	cv::Mat singleSegOut, colorImg;
	std::string imgSavePath = "/home/bailiangliang/deep_learning_deploy_on_pc/unetpp/test_image/xxx.bmp";
	imagePostprocess(unetppTensorParams ,output, singleSegOut);
	cv::resize(singleSegOut, singleSegOut, cv::Size(orgImgW, orgImgH), (0, 0), (0, 0), cv::INTER_LINEAR);
	singleChannel2threeeChannel(singleSegOut, colorImg);    
	cv::imwrite(imgSavePath, colorImg);

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
	freeModel(buffer);
}


// 实现功能：trt推理，输入内存中的待推理图像，输出单通道分割图；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     orgImg:内存中的待推理图像路径
// outputParas:
//     singleSegOut:推理后的单通道索引图
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void trtInferenceOnLine(INPUT cv::Mat& orgImg, OUTPUT cv::Mat& singleSegOut)
{
	//1、参数初始化
	ltTensorParams unetppTensorParams;
	unetppTensorParams.inputTensorNames.push_back("inputs");
	unetppTensorParams.batchSize = 1;
	unetppTensorParams.outputTensorNames.push_back("outputs");
	unetppTensorParams.imgC = 3;
	unetppTensorParams.imgH = 416;
	unetppTensorParams.imgW = 512;
	unetppTensorParams.modelFilePath = "/home/bailiangliang/deep_learning_deploy_on_pc/unetpp/epoch_128.trt";
	unetppTensorParams.classNum = 3;

	//2、创建TRT引擎和会话
	void * buffer = nullptr;
	void * context = nullptr;
	float* hostInputBuffer;
	loadModel(unetppTensorParams, &buffer, &context, &hostInputBuffer);
	
	//3、图像读取与前处理
	//std::string imgPath = "/home/bailiangliang/deep_learning_deploy_on_pc/unetpp/test_image/2/Color_38.bmp";
	//cv::Mat orgImg =  cv::imread(imgPath);
	int orgImgW = orgImg.cols;
	int orgImgH = orgImg.rows;

	imagePreprocess(orgImg, unetppTensorParams, hostInputBuffer);
	
	//4、TRT引擎推理
	float *output;    
	trtInference(unetppTensorParams, buffer,  context, &output);

	//5、图像后处理，并将后处理图映射回原图尺寸
	cv::Mat singleSegOut, colorImg;
	std::string imgSavePath = "/home/bailiangliang/deep_learning_deploy_on_pc/unetpp/test_image/xxx.bmp";
	imagePostprocess(unetppTensorParams ,output, singleSegOut);
	cv::resize(singleSegOut, singleSegOut, cv::Size(orgImgW, orgImgH), (0, 0), (0, 0), cv::INTER_LINEAR);
	singleChannel2threeeChannel(singleSegOut, colorImg);    
	cv::imwrite(imgSavePath, colorImg);

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
	freeModel(buffer);
}
