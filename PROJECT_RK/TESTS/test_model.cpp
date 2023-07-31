// ==========================================================
// 实现功能：模型批量离线测试、模糊图测试
// 文件名称：test_model.cpp
// 相关文件：无
// 作   者：Liangliang Bai (liangliang.bai@leapting.com)
// 版   权：<Copyright(C) 2023-Leapting Technology Co.,LTD All rights reserved.>
// 修改记录：
// 日   期             版本       修改人   走读人  
// 2023.05.25         2.0.3      白亮亮

// 修改记录：无
// ==========================================================
#include <stdio.h>
#include "Configer.hpp"
#include "Yolo.hpp"
#include "common.h"
#include "ProcessMgr.hpp"
#include <iostream>
#include <fstream>
#include "Configer.hpp"
#include<thread>
#include"segResPostProcessing.h"
#include"imageCropping.h"

using namespace std;


// 实现功能：测试自适应切图离线测试
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgPath:待裁剪图像路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void testAdaptiveCropImg(IN const std::string& imgPath)
{
    cv::Mat srcImage;
    srcImage = cv::imread(imgPath);
    float a[6] = { 10,20,210,120,0.995, 0 };
    std::vector<float*> detRes;
    detRes.push_back(a);
    int* ptr = nullptr;
    adaptiveImageCropping imgCrop(srcImage, detRes, 2, 720, 1280, ptr);
    imgCrop.adaptiveCropImage(0, 100, 20, 20, 720, 1280);
    cv::imwrite("./testImage/4.png", imgCrop.mDstImage);
}


// 实现功能：天气模型离线批量测试
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgDir:待检测图所在文件夹;
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void weatherModelOfflineTest(IN const std::string& imgDir)
{
    //模型初始化
    Configer *pConfiger = Configer::GetInstance();
    std::cout << "version:" << pConfiger->verison() << std::endl;

    std::cout<<"Loading weather classification model..."<<std::endl;
    CLA_CFG_t* m_p_cla_weather_cfg;
    m_p_cla_weather_cfg = pConfiger->get_cla_weather_cfg();
    Classify mCla_weather;
    int ret = mCla_weather.init(m_p_cla_weather_cfg);

    //图像预处理 
    int col_center = 640;
    int row_center = 360 + 130;
    cv::Mat frame, im_classify_weather_part, im_classify_weather;
    std::vector<String> imgNamesPath;
    glob(imgDir, imgNamesPath, false);  //调用opncv中的glob函数，将遍历路径path，将该路径下的全部文件名的绝对路径存进imgNmaes

    for(int imgNumIdx=0; imgNumIdx<imgNamesPath.size(); ++imgNumIdx)
    {
        std::cout<<"当前图片路径:"<<imgNamesPath[imgNumIdx]<<std::endl;
        frame = imread(imgNamesPath[imgNumIdx],cv::IMREAD_UNCHANGED);
        std::cout<<"图像尺寸为:"<<frame.cols<<" "<<frame.rows<<",请检查是否要进行resize<"<<std::endl;
        cv::resize(frame, frame, cv::Size(1280, 720), (0, 0), (0, 0), cv::INTER_LINEAR);
        im_classify_weather_part = frame(cv::Rect(col_center-m_p_cla_weather_cfg->feed_w/2, row_center-m_p_cla_weather_cfg->feed_h/2, m_p_cla_weather_cfg->feed_w, m_p_cla_weather_cfg->feed_h)).clone();
        cv::resize(im_classify_weather_part, im_classify_weather, cv::Size(m_p_cla_weather_cfg ->feed_w,m_p_cla_weather_cfg ->feed_h), (0, 0), (0, 0), cv::INTER_LINEAR);
        
        int classifyRes;
        uint8_t chwImg[3*m_p_cla_weather_cfg ->feed_w*m_p_cla_weather_cfg ->feed_h];
        HWC2CHW(im_classify_weather, chwImg);
        int ret = mCla_weather.run(chwImg, "CHW", classifyRes);
        std::cout<<"推理结果="<<classifyRes<<std::endl;
    }
}


// 实现功能：清洁度模型离线批量测试
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgDir:待检测图所在文件夹;
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void cleanlinessModelOfflineTest(IN const std::string& imgDir)
{
    //模型初始化
    Configer *pConfiger = Configer::GetInstance();
    std::cout << "version:" << pConfiger->verison() << std::endl;

    std::cout<<"Loading cleanliness model..."<<std::endl;
    CLA_CFG_t* m_p_cla_cleanliness_cfg;
    m_p_cla_cleanliness_cfg = pConfiger->get_cla_cfg();
    Classify mCla_cleanliness;
    int ret = mCla_cleanliness.init(m_p_cla_cleanliness_cfg);

    //图像预处理 
    int col_center;
    int row_center;
    uint8_t chwImg[3*m_p_cla_cleanliness_cfg ->feed_w*m_p_cla_cleanliness_cfg ->feed_h];
    cv::Mat frame2, frame,im_classify_cleanliness_part, im_classify_cleanliness;
    std::vector<String> imgNamesPath;
    glob(imgDir, imgNamesPath, false);  //调用opncv中的glob函数，将遍历路径path，将该路径下的全部文件名的绝对路径存进imgNmaes

    for(int imgNumIdx=0; imgNumIdx<imgNamesPath.size(); ++imgNumIdx)
    {
        std::cout<<"当前图片路径:"<<imgNamesPath[imgNumIdx]<<std::endl;
        frame = imread(imgNamesPath[imgNumIdx],cv::IMREAD_UNCHANGED);
        std::cout<<"图像尺寸为:"<<frame.cols<<" "<<frame.rows<<",请检查是否要进行resize"<<std::endl;
        // cv::resize(frame, frame, cv::Size(1280, 720), (0, 0), (0, 0), cv::INTER_LINEAR);
        col_center = frame.cols/2;
        row_center = frame.rows/2 + 150;
        im_classify_cleanliness_part = frame(cv::Rect(col_center-m_p_cla_cleanliness_cfg->feed_w/2, row_center-m_p_cla_cleanliness_cfg->feed_h/2, m_p_cla_cleanliness_cfg->feed_w, m_p_cla_cleanliness_cfg->feed_h)).clone();
        cv::resize(im_classify_cleanliness_part, im_classify_cleanliness, cv::Size(m_p_cla_cleanliness_cfg ->feed_w,m_p_cla_cleanliness_cfg ->feed_h), (0, 0), (0, 0), cv::INTER_LINEAR);
        
        int classifyRes;
        HWC2CHW(im_classify_cleanliness, chwImg);
        int ret = mCla_cleanliness.run(chwImg, "CHW", classifyRes);
        std::cout<<"推理结果="<<classifyRes<<std::endl;
    }
}


// 实现功能：检测模型1 离线批量测试,并切图用于检测2
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgDir:待检测图所在文件夹;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
int testDetection1(IN const std::string& imgDir, IN const std::string& imgSaveDir)
{
    //模型初始化    
    Configer *pConfiger = Configer::GetInstance();
    std::cout<<"Loading detection model 1..."<<std::endl;
    YOLO_CFG_t *m_p_yolo_cfg = pConfiger->get_yolo_cfg();
    YOLO_CFG_t *m_p_yolo_cfg2 = pConfiger->get_yolo_cfg2();
    Yolo mYolo;
    int ret = -1;
    ret = mYolo.init(m_p_yolo_cfg);
    std::cout<<"Loaded detection model 1"<<std::endl;

    //图像预处理 
    int infeSuccessImgIdx = 0;
    int croppedSccessImgNum = 0;
    int bridgeLengthThres = 100;
    int lowerBridgeLengthThres = 100;
    vector<float *> res;
    uint8_t chwImgDet[3*m_p_yolo_cfg->feed_w*m_p_yolo_cfg->feed_h]; 
    cv::Mat frame,im_detect, im_detect_resized, im_detect_rgb;
    std::string croppedImgPath;
    std::vector<String> imgNamesPath;
    std::vector<String> imgNamesPath2;
    glob(imgDir, imgNamesPath, false);  
    for(int imgNumIdx=0; imgNumIdx<imgNamesPath.size(); ++imgNumIdx)
    {
        std::cout<<"当前图片路径:"<<imgNamesPath[imgNumIdx]<<std::endl;
        frame = imread(imgNamesPath[imgNumIdx],cv::IMREAD_UNCHANGED);
        std::cout<<"图像尺寸为:"<<frame.cols<<" "<<frame.rows<<",请检查是否要进行resize"<<std::endl;
        if(!frame.empty())
        {
            // 前处理
            im_detect = frame;
            cv::resize(im_detect, im_detect_resized, cv::Size(m_p_yolo_cfg->feed_w, m_p_yolo_cfg->feed_h), (0, 0), (0, 0), cv::INTER_LINEAR);
            cv::cvtColor(im_detect_resized, im_detect_rgb, cv::COLOR_BGR2RGB);
            HWC2CHW(im_detect_rgb, chwImgDet);

            // 模型推理
            int ret = mYolo.run(chwImgDet, "CHW", res);
            if(0 != ret)
            {
                std::cout<<"检测模型1推理异常,请检查..."<<std::endl;
            }
            else
            {
                std::cout<<"检测模型1推理成功!"<<std::endl;
            }

            // 6.5、根据推理输出的结果，对类别0和1根据目标宽度先过滤一遍并统计每个类别([bridge,lowerBridge])目标数；
            int bridgeNum = 0;                                          
            int lowerBridgeNum = 0;
            int xxx = 0;   
            int xxx2 = 0;                                
            for (int i = 0; i < res.size(); ++i) 
            {
                if (res[i][5] == 0. && (res[i][2]-res[i][0]>bridgeLengthThres))                                      
                {
                    bridgeNum++;
                }
                
                if (res[i][5] == 1. && (res[i][2]-res[i][0]>lowerBridgeLengthThres))                                       
                {
                    lowerBridgeNum++;
                }

                if (res[i][5] == 2.)                                       
                {
                    xxx++;
                }

                if (res[i][5] == 3.)                                       
                {
                    xxx2++;
                }
            }

            std::cout<<"imgNumIdx="<<imgNumIdx<<" 检测到下桥架目标数:"<<lowerBridgeNum<<std::endl;
            if(bridgeNum >0)
            {
                infeSuccessImgIdx++;
                std::cout<<"infeSuccessImgIdx="<<infeSuccessImgIdx<<std::endl;

                std::cout<<"模型1检测到有桥架,准备切图!"<<std::endl;

                // 切图：调用自适应切图类进行切图
                int * a = new int();
                adaptiveImageCropping imgCrop(frame, res, 1, m_p_yolo_cfg->feed_h, m_p_yolo_cfg->feed_w, a);
                imgCrop.adaptiveCropImage(0, bridgeLengthThres, 5,5,m_p_yolo_cfg2->feed_h, m_p_yolo_cfg2->feed_w);
                croppedImgPath = imgSaveDir+std::to_string(imgNumIdx)+".jpg";
                if(!imgCrop.mDstImage.empty())
                {
                    croppedSccessImgNum++;
                    std::cout<<"croppedSccessImgNum="<<croppedSccessImgNum<<std::endl;
                    cv::imwrite(croppedImgPath, imgCrop.mDstImage);
                    std::cout<<"成功切到用于检测2的图像!"<<std::endl;
                }
                else
                {
                    imgNamesPath2.push_back("当前图未切到");
                    imgNamesPath2.push_back(imgNamesPath[imgNumIdx]);
                }
            }
            else
            {
                imgNamesPath2.push_back(imgNamesPath[imgNumIdx]);
            }
        }
    }

    std::cout<<"批量检测结束"<<std::endl;
    for(int i = 0; i<imgNamesPath2.size(); ++i)
    {
        std::cout<<imgNamesPath2[i]<<std::endl;
    }
    std::cout<<"以上图未切到或未检测到桥架区域"<<std::endl;
    return 0;
}


// 实现功能：检测模型2 离线批量测试:输入的推理图是检测1截取后的图
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgDir:待检测图所在文件夹;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
int testDetection2(IN const std::string& imgDir, IN const std::string& imgSaveDir)
{
    //模型初始化    
    Configer *pConfiger = Configer::GetInstance();
    std::cout<<"Loading detection model 2..."<<std::endl;
    YOLO_CFG_t *m_p_yolo_cfg2 = pConfiger->get_yolo_cfg2();
    Yolo mYolo2;
    int ret = -1;
    ret = mYolo2.init(m_p_yolo_cfg2);
    std::cout<<"Loaded detection model 2"<<std::endl;

    vector<float *> res2;
    uint8_t chwImgDet2[3*m_p_yolo_cfg2->feed_w*m_p_yolo_cfg2->feed_h]; 
    cv::Mat frame,im_detect2, im_detect_resized2, im_detect_rgb2;
    std::string imgSavePath;
    std::vector<String> imgNamesPath;
    std::vector<String> imgNamesPath2;
    glob(imgDir, imgNamesPath, false);  
    for(int imgNumIdx=0; imgNumIdx<imgNamesPath.size(); ++imgNumIdx)
    {
        std::cout<<"当前图片路径:"<<imgNamesPath[imgNumIdx]<<std::endl;
        frame = imread(imgNamesPath[imgNumIdx],cv::IMREAD_UNCHANGED);
        std::cout<<"图像尺寸为:"<<frame.cols<<" "<<frame.rows<<",请检查是否要进行resize"<<std::endl;
        imgSavePath = imgSaveDir+std::to_string(imgNumIdx)+".jpg";
        if(!frame.empty())
        {
            // 前处理
            im_detect2 = frame;
            cv::resize(im_detect2, im_detect_resized2, cv::Size(m_p_yolo_cfg2->feed_w, m_p_yolo_cfg2->feed_h), (0, 0), (0, 0), cv::INTER_LINEAR);
            cv::cvtColor(im_detect_resized2, im_detect_rgb2, cv::COLOR_BGR2RGB);
            HWC2CHW(im_detect_rgb2, chwImgDet2);

            // 7.4、模型推理
            int ret = mYolo2.run(chwImgDet2, "CHW", res2);
            if(0 != ret)
            {
                std::cout<<"检测模型2推理异常,请检查..."<<std::endl;
            }
            else
            {
                std::cout<<"检测模型2推理成功!"<<std::endl;
            }

            // 7.5、根据推理输出的结果，统计每个类别([no fracture,fracture])目标数
            int fractureNum = 0;                                       
            for (int i = 0;i < res2.size(); ++i) 
            {
                if (res2[i][5] == 1.)                                    
                {
                    fractureNum++;
                }
            }

            if (res2.size() > 0)               
            {
                mYolo2.show_res(im_detect_resized2,res2); 
                cv::imwrite(imgSavePath, im_detect_resized2);
            }
            else
            {
                imgNamesPath2.push_back(imgNamesPath[imgNumIdx]);
            }
        }
    }

    std::cout<<"批量检测结束"<<std::endl;
    for(int i = 0; i<imgNamesPath2.size(); ++i)
    {
        std::cout<<imgNamesPath2[i]<<std::endl;
    }
    std::cout<<"以上图未检测到目标"<<std::endl;
    return 0;
}


// 实现功能：模糊图判断-Laplacian：Laplacian梯度是另一种求图像梯度的方法
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgPath:待检测图路径;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
int fuzzyImgJudge_Laplacian(IN const std::string& imgPath, IN const std::string& imgSaveDir)
{
    size_t pathLength = imgPath.length();
    size_t lastPos = imgPath.find_first_of("/", pathLength);
    size_t nameIndex = imgPath.find(".", 0);
    string imageName = imgPath.substr(15, nameIndex-lastPos);
    std::cout<<"imageName="<<imageName<<std::endl;
    string newImagePath = imgSaveDir+imageName;

    cv::Mat srcImg = cv::imread(imgPath);
	Mat imageGrey, imageSobel;
	cvtColor(srcImg, imageGrey, COLOR_RGB2GRAY);
	Laplacian(imageGrey, imageSobel, CV_16U);
 
	//图像的平均灰度
	double meanValue = 0.0;
	meanValue = mean(imageSobel)[0];
 
	//double转string
	stringstream meanValueStream;
	string meanValueString;
	meanValueStream << meanValue;
	meanValueStream >> meanValueString;
	meanValueString = "Articulation(Laplacian Method): " + meanValueString;
	putText(srcImg, meanValueString, Point(20, 50), cv::FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 255, 25), 2);
    cv::imwrite(newImagePath, srcImg);
}


// 实现功能：模糊图判断-Laplacian 测试函数
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgDir:待检测图所在文件夹;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
int fuzzyImgJudgeTest_Laplacian(IN const std::string& imgDir, IN const std::string& imgSaveDir)
{
    std::vector<String> imgNamesPath;
    glob(imgDir, imgNamesPath, false);  
    for(int imgNumIdx=0; imgNumIdx<imgNamesPath.size(); ++imgNumIdx)
    {
        std::cout<<"测试图索引："<<imgNumIdx<<std::endl;
        std::cout<<"当前图片路径:"<<imgNamesPath[imgNumIdx]<<std::endl;
        fuzzyImgJudge_Laplacian(imgNamesPath[imgNumIdx], imgSaveDir);
    }
}


// 实现功能：模糊图判断-Tenengrad梯度方法:衡量的指标是经过Sobel算子处理后的图像的平均灰度值，值越大，代表图像越清晰。
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgPath:待检测图路径;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
int fuzzyImgJudge_Tenengrad(IN const std::string& imgPath, IN const std::string& imgSaveDir)
{
    size_t pathLength = imgPath.length();
    size_t lastPos = imgPath.find_first_of("/", pathLength);
    size_t nameIndex = imgPath.find(".", 0);
    string imageName = imgPath.substr(15, nameIndex-lastPos);
    std::cout<<"imageName="<<imageName<<std::endl;
    string newImagePath = imgSaveDir+imageName;
    
	Mat imageGrey, imageSobel;
    cv::Mat srcImg = cv::imread(imgPath);
	cvtColor(srcImg, imageGrey, COLOR_RGB2GRAY);
	Sobel(imageGrey, imageSobel, CV_16U, 1, 1);
 
	//图像的平均灰度
	double meanValue = 0.0;
	meanValue = mean(imageSobel)[0];
 
	//double转string
	stringstream meanValueStream;
	string meanValueString;
	meanValueStream << meanValue;                    // 将meanValue放入输入流中
	meanValueStream >> meanValueString;              // 从sstream中抽取前面插入的meanValue，赋给meanValueString
	meanValueString = "Articulation(Sobel Method): " + meanValueString;
	putText(srcImg, meanValueString, Point(20, 50), cv::FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 255, 25), 2);
	cv::imwrite(newImagePath, srcImg);

}


// 实现功能：模糊图判断-Tenengrad梯度方法 测试函数
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgDir:待检测图所在文件夹;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
int fuzzyImgJudgeTest_Tenengrad(IN const std::string& imgDir, IN const std::string& imgSaveDir)
{
    std::vector<String> imgNamesPath;
    glob(imgDir, imgNamesPath, false);  
    for(int imgNumIdx=0; imgNumIdx<imgNamesPath.size(); ++imgNumIdx)
    {
        std::cout<<"测试图索引："<<imgNumIdx<<std::endl;
        std::cout<<"当前图片路径:"<<imgNamesPath[imgNumIdx]<<std::endl;
        fuzzyImgJudge_Tenengrad(imgNamesPath[imgNumIdx], imgSaveDir);
    }
}


// 实现功能：模糊图判断-Variance求方差方法:对焦清晰的图像相比对焦模糊的图像，它的数据之间的灰度差异应该更大，即它的方差应该较大，
// 可以通过图像灰度数据的方差来衡量图像的清晰度，方差越大，表示清晰度越好 
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgPath:待检测图路径;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
int fuzzyImgJudge_Variance(IN const std::string& imgPath, IN const std::string& imgSaveDir)
{
    size_t pathLength = imgPath.length();
    size_t lastPos = imgPath.find_first_of("/", pathLength);
    size_t nameIndex = imgPath.find(".", 0);
    string imageName = imgPath.substr(15, nameIndex-lastPos);
    std::cout<<"imageName="<<imageName<<std::endl;
    string newImagePath = imgSaveDir+imageName;

	Mat srcImg,imageGrey, meanValueImage, meanStdValueImage;
    srcImg = imread(imgPath);
	cvtColor(srcImg, imageGrey, COLOR_RGB2GRAY);
 
	//求灰度图像的标准差
	meanStdDev(imageGrey, meanValueImage, meanStdValueImage);
	double meanValue = 0.0;
	meanValue = meanStdValueImage.at<double>(0, 0);
 
	//double to string
	stringstream meanValueStream;
	string meanValueString;
	meanValueStream << meanValue*meanValue;
	meanValueStream >> meanValueString;
	meanValueString = "Articulation(Variance Method): " + meanValueString;

	putText(srcImg, meanValueString, Point(20, 50), cv::FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 255, 25), 2);
    cv::imwrite(newImagePath, srcImg);
}


// 实现功能：模糊图判断-Variance求方差方法 测试函数
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgDir:待检测图所在文件夹;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
int fuzzyImgJudgeTest_Variance(IN const std::string& imgDir, IN const std::string& imgSaveDir)
{
    std::vector<String> imgNamesPath;
    glob(imgDir, imgNamesPath, false);  
    for(int imgNumIdx=0; imgNumIdx<imgNamesPath.size(); ++imgNumIdx)
    {
        std::cout<<"测试图索引："<<imgNumIdx<<std::endl;
        std::cout<<"当前图片路径:"<<imgNamesPath[imgNumIdx]<<std::endl;
        fuzzyImgJudge_Variance(imgNamesPath[imgNumIdx], imgSaveDir);
    }
}


// 实现功能：模糊图判断-SMD(灰度方差)函数 
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgPath:待检测图路径;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
int fuzzyImgJudge_SMD(IN const std::string& imgPath, IN const std::string& imgSaveDir)
{
    size_t pathLength = imgPath.length();
    size_t lastPos = imgPath.find_first_of("/", pathLength);
    size_t nameIndex = imgPath.find(".", 0);
    string imageName = imgPath.substr(15, nameIndex-lastPos);
    std::cout<<"imageName="<<imageName<<std::endl;
    string newImagePath = imgSaveDir+imageName;

	Mat srcImg,imageGrey, smd_image_x, smd_image_y, G;
    srcImg = imread(imgPath);

	if (srcImg.channels() == 3)
    {
		cv::cvtColor(srcImg, imageGrey, COLOR_RGB2GRAY);
	}
 
	cv::Mat kernel_x(3, 3, CV_32F, cv::Scalar(0));
	kernel_x.at<float>(1, 2) = -1.0;
	kernel_x.at<float>(1, 1) = 1.0;
	cv::Mat kernel_y(3, 3, CV_32F, cv::Scalar(0));
	kernel_y.at<float>(0, 1) = -1.0;
	kernel_y.at<float>(1, 1) = 1.0;
	cv::filter2D(imageGrey, smd_image_x, imageGrey.depth(), kernel_x);
	cv::filter2D(imageGrey, smd_image_y, imageGrey.depth(), kernel_y);
 
	smd_image_x = cv::abs(smd_image_x);
	smd_image_y = cv::abs(smd_image_y);
	G = smd_image_x + smd_image_y;
 
	float result = cv::mean(G)[0];

    //double to string
	stringstream meanValueStream;
	string meanValueString;
	meanValueStream << result;
	meanValueStream >> meanValueString;
	meanValueString = "Articulation(SMD Method): " + meanValueString;

	putText(srcImg, meanValueString, Point(20, 50), cv::FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 255, 25), 2);
    cv::imwrite(newImagePath, srcImg);

    return 0;
}


// 实现功能：模糊图判断-SMD(灰度方差) 测试函数
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgDir:待检测图所在文件夹;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
int fuzzyImgJudgeTest_SMD(IN const std::string& imgDir, IN const std::string& imgSaveDir)
{
    std::vector<String> imgNamesPath;
    glob(imgDir, imgNamesPath, false);  
    for(int imgNumIdx=0; imgNumIdx<imgNamesPath.size(); ++imgNumIdx)
    {
        std::cout<<"测试图索引："<<imgNumIdx<<std::endl;
        std::cout<<"当前图片路径:"<<imgNamesPath[imgNumIdx]<<std::endl;
        fuzzyImgJudge_SMD(imgNamesPath[imgNumIdx], imgSaveDir);
    }
}


// 实现功能：模糊图判断-Brenner梯度函数
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgPath:待检测图路径;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
int fuzzyImgJudge_Brenner(IN const std::string& imgPath, IN const std::string& imgSaveDir)
{ 
    size_t pathLength = imgPath.length();
    size_t lastPos = imgPath.find_first_of("/", pathLength);
    size_t nameIndex = imgPath.find(".", 0);
    string imageName = imgPath.substr(15, nameIndex-lastPos);
    std::cout<<"imageName="<<imageName<<std::endl;
    string newImagePath = imgSaveDir+imageName;

	Mat srcImg,imageGrey;
    srcImg = imread(imgPath);

	if (3 == srcImg.channels())
    {
		cv::cvtColor(srcImg, imageGrey, COLOR_RGB2GRAY);
	}
 
	double result = .0f;
	for (int i = 0; i < imageGrey.rows; ++i)
    {
		uchar *data = imageGrey.ptr<uchar>(i);
		for (int j = 0; j < imageGrey.cols - 2; ++j)
        {
			result += pow(data[j + 2] - data[j], 2);
		}
	}
    result = result/imageGrey.total();

    //double to string
	stringstream meanValueStream;
	string meanValueString;
	meanValueStream << result;
	meanValueStream >> meanValueString;
	meanValueString = "Articulation(Brenner Method): " + meanValueString;

	putText(srcImg, meanValueString, Point(20, 50), cv::FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 255, 25), 2);
    cv::imwrite(newImagePath, srcImg);

    return 0;
}


// 实现功能：模糊图判断-Brenner梯度 测试函数
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgDir:待检测图所在文件夹;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
int fuzzyImgJudgeTest_Brenner(IN const std::string& imgDir, IN const std::string& imgSaveDir)
{
    std::vector<String> imgNamesPath;
    glob(imgDir, imgNamesPath, false);  
    for(int imgNumIdx=0; imgNumIdx<imgNamesPath.size(); ++imgNumIdx)
    {
        std::cout<<"测试图索引："<<imgNumIdx<<std::endl;
        std::cout<<"当前图片路径:"<<imgNamesPath[imgNumIdx]<<std::endl;
        fuzzyImgJudge_Brenner(imgNamesPath[imgNumIdx], imgSaveDir);
    }
}


// 实现功能：模糊图判断-SMD2(灰度方差乘积)函数
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgPath:待检测图路径;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
int fuzzyImgJudge_SMD2(IN const std::string& imgPath, IN const std::string& imgSaveDir)
{
    size_t pathLength = imgPath.length();
    size_t lastPos = imgPath.find_first_of("/", pathLength);
    size_t nameIndex = imgPath.find(".", 0);
    string imageName = imgPath.substr(15, nameIndex-lastPos);
    std::cout<<"imageName="<<imageName<<std::endl;
    string newImagePath = imgSaveDir+imageName;

	Mat srcImg, imageGrey, smd_image_x, smd_image_y, G;
    srcImg = imread(imgPath);

	if (srcImg.channels() == 3)
    {
		cv::cvtColor(srcImg, imageGrey, COLOR_RGB2GRAY);
	}
 
	cv::Mat kernel_x(3, 3, CV_32F, cv::Scalar(0));
	kernel_x.at<float>(1, 2) = -1.0;
	kernel_x.at<float>(1, 1) = 1.0;
	cv::Mat kernel_y(3, 3, CV_32F, cv::Scalar(0));
	kernel_y.at<float>(1, 1) = 1.0;
	kernel_y.at<float>(2, 1) = -1.0;
	cv::filter2D(imageGrey, smd_image_x, imageGrey.depth(), kernel_x);
	cv::filter2D(imageGrey, smd_image_y, imageGrey.depth(), kernel_y);
 
	smd_image_x = cv::abs(smd_image_x);
	smd_image_y = cv::abs(smd_image_y);
	cv::multiply(smd_image_x, smd_image_y, G);
	float result = cv::mean(G)[0];

    //double to string
	stringstream meanValueStream;
	string meanValueString;
	meanValueStream << result;
	meanValueStream >> meanValueString;
	meanValueString = "Articulation(SMD2 Method): " + meanValueString;

	putText(srcImg, meanValueString, Point(20, 50), cv::FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 255, 25), 2);
    cv::imwrite(newImagePath, srcImg);

    return 0;
}


// 实现功能：模糊图判断-SMD2(灰度方差乘积)方法 测试函数
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgDir:待检测图所在文件夹;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
int fuzzyImgJudgeTest_SMD2(IN const std::string& imgDir, IN const std::string& imgSaveDir)
{
    std::vector<String> imgNamesPath;
    glob(imgDir, imgNamesPath, false);  
    for(int imgNumIdx=0; imgNumIdx<imgNamesPath.size(); ++imgNumIdx)
    {
        std::cout<<"测试图索引："<<imgNumIdx<<std::endl;
        std::cout<<"当前图片路径:"<<imgNamesPath[imgNumIdx]<<std::endl;
        fuzzyImgJudge_SMD2(imgNamesPath[imgNumIdx], imgSaveDir);
    }
}


// 实现功能：模糊图判断-能量梯度函数
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgPath:待检测图路径;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
int fuzzyImgJudge_energyGradient(IN const std::string& imgPath, IN const std::string& imgSaveDir)
{
	size_t pathLength = imgPath.length();
    size_t lastPos = imgPath.find_first_of("/", pathLength);
    size_t nameIndex = imgPath.find(".", 0);
    string imageName = imgPath.substr(15, nameIndex-lastPos);
    std::cout<<"imageName="<<imageName<<std::endl;
    string newImagePath = imgSaveDir+imageName;

	Mat srcImg, imageGrey, smd_image_x, smd_image_y, G;
    srcImg = imread(imgPath);

	if (srcImg.channels() == 3)
    {
		cv::cvtColor(srcImg, imageGrey, COLOR_RGB2GRAY);
	}
 
	cv::Mat kernel_x(3, 3, CV_32F, cv::Scalar(0));
	kernel_x.at<float>(1, 2) = -1.0;
	kernel_x.at<float>(1, 1) = 1.0;
	cv::Mat kernel_y(3, 3, CV_32F, cv::Scalar(0));
	kernel_y.at<float>(1, 1) = 1.0;
	kernel_y.at<float>(2, 1) = -1.0;
	cv::filter2D(imageGrey, smd_image_x, imageGrey.depth(), kernel_x);
	cv::filter2D(imageGrey, smd_image_y, imageGrey.depth(), kernel_y);
 
	cv::multiply(smd_image_x, smd_image_x, smd_image_x);
	cv::multiply(smd_image_y, smd_image_y, smd_image_y);
	G = smd_image_x + smd_image_y;
	float result = cv::mean(G)[0];

    //double to string
	stringstream meanValueStream;
	string meanValueString;
	meanValueStream << result;
	meanValueStream >> meanValueString;
	meanValueString = "Articulation(Energy gradient Method): " + meanValueString;

	putText(srcImg, meanValueString, Point(20, 50), cv::FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 255, 25), 2);
    cv::imwrite(newImagePath, srcImg);

    return 0;
}


// 实现功能：模糊图判断-能量梯度方法 测试函数
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgDir:待检测图所在文件夹;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
int fuzzyImgJudgeTest_energyGradient(IN const std::string& imgDir, IN const std::string& imgSaveDir)
{
    std::vector<String> imgNamesPath;
    glob(imgDir, imgNamesPath, false);  
    for(int imgNumIdx=0; imgNumIdx<imgNamesPath.size(); ++imgNumIdx)
    {
        std::cout<<"测试图索引："<<imgNumIdx<<std::endl;
        std::cout<<"当前图片路径:"<<imgNamesPath[imgNumIdx]<<std::endl;
        fuzzyImgJudge_energyGradient(imgNamesPath[imgNumIdx], imgSaveDir);
    }
}


// 实现功能：模糊图判断-EAV点锐度算法
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgPath:待检测图所在文件夹;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------  
int fuzzyImgJudge_eav(IN const std::string& imgPath, IN const std::string& imgSaveDir)
{
	size_t pathLength = imgPath.length();
    size_t lastPos = imgPath.find_first_of("/", pathLength);
    size_t nameIndex = imgPath.find(".", 0);
    string imageName = imgPath.substr(15, nameIndex-lastPos);
    std::cout<<"imageName="<<imageName<<std::endl;
    string newImagePath = imgSaveDir+imageName;

	Mat srcImg, imageGrey, smd_image_x, smd_image_y, G;
    srcImg = imread(imgPath);
 
	if (srcImg.channels() == 3)
    {
		cv::cvtColor(srcImg, imageGrey, COLOR_RGB2GRAY);
	}
 
	double result = .0f;
	for (int i = 1; i < imageGrey.rows-1; ++i)
    {
		uchar *prev = imageGrey.ptr<uchar>(i - 1);
		uchar *cur = imageGrey.ptr<uchar>(i);
		uchar *next = imageGrey.ptr<uchar>(i + 1);
		for (int j = 0; j < imageGrey.cols; ++j)
        {
			result += (abs(prev[j - 1] - cur[i])*0.7 + abs(prev[j] - cur[j]) + abs(prev[j + 1] - cur[j])*0.7 +
				abs(next[j - 1] - cur[j])*0.7 + abs(next[j] - cur[j]) + abs(next[j + 1] - cur[j])*0.7 +
				abs(cur[j - 1] - cur[j]) + abs(cur[j + 1] - cur[j]));
		}
	}
	result = result / imageGrey.total();

    stringstream meanValueStream;
	string meanValueString;
	meanValueStream << result;
	meanValueStream >> meanValueString;
	meanValueString = "Articulation(Eav Method): " + meanValueString;

	putText(srcImg, meanValueString, Point(20, 50), cv::FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 255, 25), 2);
    cv::imwrite(newImagePath, srcImg);

    return 0;
}


// 实现功能：模糊图判断-EAV点锐度算法 测试函数
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgDir:待检测图所在文件夹;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------  
int fuzzyImgJudgeTest_eav(IN const std::string& imgDir, IN const std::string& imgSaveDir)
{
    std::vector<String> imgNamesPath;
    glob(imgDir, imgNamesPath, false);  
    for(int imgNumIdx=0; imgNumIdx<imgNamesPath.size(); ++imgNumIdx)
    {
        std::cout<<"测试图索引："<<imgNumIdx<<std::endl;
        std::cout<<"当前图片路径:"<<imgNamesPath[imgNumIdx]<<std::endl;
        fuzzyImgJudge_eav(imgNamesPath[imgNumIdx], imgSaveDir);
    }
}


// ----------------------------------->parameters<----------------------------------
// inputParas :
//     gray_img:待检测图;
// outputParas:
//     blur_mean：模糊均值
//     blur_ratio:模糊因子
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void compute_blur_IQA(IN cv::Mat &gray_img, OUT float &blur_mean, OUT float &blur_ratio)
{
	//计算水平/竖直差值获取梯度图
	cv::Mat grad_h, grad_v;
	cv::Mat kernel_h = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
	kernel_h.at<float>(0, 1) = -1;
	kernel_h.at<float>(2, 1) = 1;
	cv::filter2D(gray_img, grad_h, CV_32FC1, kernel_h);
	cv::Mat kernel_v = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
	kernel_v.at<float>(1, 0) = -1;
	kernel_v.at<float>(1, 2) = 1;
	cv::filter2D(gray_img, grad_v, CV_32FC1, kernel_v);
 
	//获取候选边缘点
	//筛选条件：D_h > D_mean
	float mean = static_cast<float>(cv::mean(grad_v)[0]);
	cv::Mat mask = grad_h > mean;
	mask = mask / 255;
	mask.convertTo(mask, CV_32FC1);
	cv::Mat C_h;
	cv::multiply(grad_h, mask, C_h);
 
	//进一步筛选边缘点
	//筛选条件：C_h(x,y) > C_h(x,y-1) and C_h(x,y) > C_h(x,y+1)
	cv::Mat edge = cv::Mat::zeros(C_h.rows, C_h.cols, CV_8UC1);
	for (int i = 1; i < C_h.rows-1; ++i){
		float *prev = C_h.ptr<float>(i - 1);
		float *cur = C_h.ptr<float>(i);
		float *next = C_h.ptr<float>(i + 1);
		uchar *data = edge.ptr<uchar>(i);
		for (int j = 0; j < C_h.cols; ++j){
			if (prev[j] < cur[j] && next[j] < cur[j]){
				data[j] = 1;
			}
		}
	}
 
	//检测边缘点是否模糊
	//获取inverse blur
	cv::Mat A_h = grad_h / 2;
	cv::Mat BR_h=cv::Mat(gray_img.size(),CV_32FC1);
	gray_img.convertTo(gray_img, CV_32FC1);
	cv::absdiff(gray_img, A_h, BR_h);
	cv::divide(BR_h, A_h, BR_h);
	cv::Mat A_v = grad_v / 2;
	cv::Mat BR_v;
	cv::absdiff(gray_img, A_v, BR_v);
	cv::divide(BR_v, A_v, BR_v);
 
	cv::Mat inv_blur = cv::Mat::zeros(BR_v.rows, BR_v.cols, CV_32FC1);
	for (int i = 0; i < inv_blur.rows; ++i){
		float *data_v = BR_v.ptr<float>(i);
		float *data = inv_blur.ptr<float>(i);
		float *data_h = BR_h.ptr<float>(i);
		for (int j = 0; j < inv_blur.cols; ++j){
			data[j] = data_v[j]>data_h[j] ? data_v[j] : data_h[j];
		}
	}
	//获取最终模糊点
	cv::Mat blur = inv_blur < 0.1 / 255;
	blur.convertTo(blur, CV_32FC1);
 
	//计算边缘模糊的均值和比例
	int sum_inv_blur = cv::countNonZero(inv_blur);
	int sum_blur = cv::countNonZero(blur);
	int sum_edge = cv::countNonZero(edge);
	blur_mean = static_cast<float>(sum_inv_blur) / sum_blur;
	blur_ratio = static_cast<float>(sum_blur) / sum_edge;
}
 

// ----------------------------------->parameters<----------------------------------
// inputParas :
//     gray_img:待检测图的灰度图;
// outputParas:
//     noise_mean：噪点均值
//     noise_ratio:噪点因子
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void compute_noise_IQA(IN cv::Mat &gray_img, OUT float &noise_mean, OUT float &noise_ratio)
{
	//均值滤波去除噪声对边缘检测的影响
	cv::Mat blur_img;
	cv::blur(gray_img, blur_img, cv::Size(3, 3));
 
	//进行竖直方向边缘检测
	cv::Mat grad_h, grad_v;
	cv::Mat kernel_h = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
	kernel_h.at<float>(0, 1) = -1;
	kernel_h.at<float>(2, 1) = 1;
	cv::filter2D(gray_img, grad_h, CV_32FC1, kernel_h);
	cv::Mat kernel_v = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
	kernel_v.at<float>(1, 0) = -1;
	kernel_v.at<float>(1, 2) = 1;
	cv::filter2D(gray_img, grad_v, CV_32FC1, kernel_v);
 
	//筛选候选点
	//水平/竖直梯度的均值
	float D_h_mean = .0f, D_v_mean = .0f;
	D_h_mean = static_cast<float>(cv::mean(grad_h)[0]);
	D_v_mean = static_cast<float>(cv::mean(grad_v)[0]);
 
	//获取候选噪声点
	cv::Mat N_cand = cv::Mat::zeros(gray_img.rows, gray_img.cols, CV_32FC1);
	for (int i = 0; i < gray_img.rows; ++i)
    {
		float *data_h = grad_h.ptr<float>(i);
		float *data_v = grad_v.ptr<float>(i);
		float *data = N_cand.ptr<float>(i);
		for (int j = 0; j < gray_img.cols; ++j)
        {
			if (data_v[j] < D_v_mean && data_h[j] < D_h_mean)
            {
				data[j] = data_v[j]>data_h[j] ? data_v[j] : data_h[j];
			}
		}
	}
 
	//最终的噪声点
	float N_cand_mean = static_cast<float>(cv::mean(N_cand)[0]);
	cv::Mat mask = (N_cand>N_cand_mean)/255;
	mask.convertTo(mask, CV_32FC1);
	cv::Mat N;
	cv::multiply(N_cand, mask, N);
 
	//计算噪声的均值和比率
	float sum_noise = static_cast<float>(cv::sum(N)[0]);
	int sum_noise_cnt = cv::countNonZero(N);
	noise_mean = sum_noise / (sum_noise_cnt + 0.0001);
	noise_ratio = static_cast<float>(sum_noise_cnt) / N.total();
}
 

// 实现功能：对模糊和噪声进行无参考图像质量评估
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgPath:待检测图的路径;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------  
int blurNoiseIQA(IN const std::string& imgPath, IN const std::string& imgSaveDir)
{
	size_t pathLength = imgPath.length();
    size_t lastPos = imgPath.find_first_of("/", pathLength);
    size_t nameIndex = imgPath.find(".", 0);
    string imageName = imgPath.substr(15, nameIndex-lastPos);
    std::cout<<"imageName="<<imageName<<std::endl;
    string newImagePath = imgSaveDir+imageName;

	Mat image, gray_img;
    image = imread(imgPath);
	gray_img=cv::Mat(image.size(),CV_8UC1);
	if (image.channels() == 3)
    {
		cv::cvtColor(image, gray_img, COLOR_RGB2GRAY);
	}
 
	//1、模糊检测
	float blur_mean = 0.f, blur_ratio = 0.f;
	compute_blur_IQA(gray_img, blur_mean, blur_ratio);
 
	//2、噪声点检测
	float noise_mean = 0.f, noise_ratio = 0.f;
	compute_noise_IQA(gray_img, noise_mean, noise_ratio);
 
	//3、噪声和模糊的组合
	double result = 1 - (blur_mean + 0.95*blur_ratio + 0.3*noise_mean + 0.75*noise_ratio);

    stringstream meanValueStream;
	string meanValueString;
	meanValueStream << result;
	meanValueStream >> meanValueString;
	meanValueString = "Articulation(NoiseIQA Method): " + meanValueString;

	putText(image, meanValueString, Point(20, 50), cv::FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 255, 25), 2);
    cv::imwrite(newImagePath, image);

    return 0;
}


// 实现功能：对模糊和噪声进行无参考图像质量评估 测试函数
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     imgDir:待检测图所在文件夹;
//     imgSaveDir:检测完的效果图保存文件夹路径
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------   
int test_blurNoiseIQA(IN const std::string& imgDir, IN const std::string& imgSaveDir)
{
    std::vector<String> imgNamesPath;
    glob(imgDir, imgNamesPath, false);  
    for(int imgNumIdx=0; imgNumIdx<imgNamesPath.size(); ++imgNumIdx)
    {
        std::cout<<"测试图索引："<<imgNumIdx<<std::endl;
        std::cout<<"当前图片路径:"<<imgNamesPath[imgNumIdx]<<std::endl;
        blurNoiseIQA(imgNamesPath[imgNumIdx], imgSaveDir);
    }
}


int main()
{
    // std::string imgPath = "./testImage/1.png";         //待裁剪图路径
    // testAdaptiveCropImg(imgPath);                      //测试自动切图功能

    std::string imgDir = "/userdata/data/";               //待测试图所在文件夹
    std::string imgSaveDir = "/userdata/test_dir/";       //测试后的效果图保存文件夹
    // weatherModelOfflineTest(imgDir);                   //天气模型离线批量测试
    // cleanlinessModelOfflineTest(imgDir);               //清洁度模型离线批量测试
    // testDetection1(imgDir, imgSaveDir);                //检测模型1离线批量测试
    testDetection2(imgDir, imgSaveDir);                //检测模型2离线批量测试

    // fuzzyImgJudgeTest_Tenengrad(imgDir, imgSaveDir);   //模糊图判断-Tenengrad梯度方法 测试函数
    
    // 经测试这种方式很不靠谱
    // fuzzyImgJudgeTest_Laplacian(imgDir, imgSaveDir);   //模糊图判断-Laplacian求梯度方法 测试函数
    
    // fuzzyImgJudgeTest_Variance(imgDir, imgSaveDir);    //模糊图判断-Variance求方差方法 测试函数
     
    // 这种方式有效果，但是计算时间太长，几乎1s1张；
    // test_blurNoiseIQA(imgDir, imgSaveDir);             //模糊图判断-blurNoiseIQA 测试函数

    // fuzzyImgJudgeTest_eav(imgDir, imgSaveDir);

    // fuzzyImgJudgeTest_energyGradient(imgDir, imgSaveDir);

    // fuzzyImgJudgeTest_SMD2(imgDir, imgSaveDir);

    // fuzzyImgJudgeTest_Brenner(imgDir, imgSaveDir);

    // fuzzyImgJudgeTest_SMD(imgDir, imgSaveDir);
    

    std::cout<<"测试结束."<<std::endl;
}



