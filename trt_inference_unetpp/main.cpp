// ==========================================================
// 实现功能：使用trtInferUnetpp.cpp中封装后的TRT接口推理测试；
// 文件名称：main.cpp
// 相关文件：无
// 作   者：Liangliang Bai (liangliang.bai@leapting.com)
// 版   权：<Copyright(C) 2022-Leapting Technology Co.,LTD All rights reserved.>
// 修改记录：
// 日   期       版本     修改人   走读人  修改记录
// 2022.12.21   1.0.0.1  白亮亮           None
// ==========================================================
#pragma once
#include"trtInferUnetpp.h"
#include"postprocessIndexImage.h"

void testTrtInference()
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
	std::string imgPath = "/home/bailiangliang/deep_learning_deploy_on_pc/unetpp/test_image/2/Color_38.bmp";
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

	#if 1    //获取直线
	cv::Mat display;
	float line_angle = get_ObjLineAngle(colorImg, display);
	#endif
	
	//6、内存释放
	freeModel(buffer);
}


void testPostprocessIndexImage()
{
	std::vector<cv::String> filenames; 
	cv::String folder =  "/home/bailiangliang/deep_learning_deploy_on_pc/unetpp/segTestResult/2/";
	cv::glob(folder, filenames); 
	for (size_t i = 0; i < filenames.size(); ++i)
	{
		std::cout << filenames[i] << std::endl;
		cv::Mat src = cv::imread(filenames[i]);
		
		// 获取两个矩形
		#if 0
		cv::Point cen;
	    cv::Mat display;
	    get_ObjRectCenter(src, cen, display);
		#endif

		// 获取直线
		#if 1
		cv::Mat display;
		float line_angle = get_ObjLineAngle(src, display);
		#endif
	}

}


int main()
{
	testTrtInference();
	cin.get();
}
