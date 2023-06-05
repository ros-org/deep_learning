/*==========================================================
模块名  ：自适应切图程序头文件；
裁剪方法：输入原图+检测模型推理结果->将指定类别的框坐标进行上采样，扩大几个像素->判断是否到边缘->切图
          ->计算自适应填充像素个数(宽高比与下个模型推理尺寸宽高比保持一致，防止resize导致图像扭曲)->自适应填充像素生成新的图
文件名  ：imageCropping.h
相关文件：无
作者    ：Liangliang Bai (liangliang.bai@leapting.com)
版权    ：<Copyright(C) 2022- Huzhou leapting Technology Co., Ltd. All rights reserved.>
修改记录：
日  期        版本     修改人   走读人
2023.04.25    1.0.0.1  白亮亮
==========================================================*/

#pragma once
#ifndef __IMAGECROPPING__
#define __IMAGECROPPING__

#include"common.h"
#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>

class adaptiveImageCropping
{
private:
        int* mPtr;                      //备用
        std::vector<float*> mDetRes;    //上一次检测模型的推理结果
        cv::Mat mSrcImage;              //上一次推理所使用的原图(未经过resize等任何操作)
        int mClsNum;                    //上次检测的类别数
        int mInfeHeight;                //上次推理输入网络的图像高(已经过resize等前处理操作)
        int mInfeWidth;                 //上次推理输入网络的图像宽(已经过resize等前处理操作)

public:
        cv::Mat mDstImage;              //下一次推理要使用的原图

public:
        adaptiveImageCropping() = default;
        adaptiveImageCropping(IN const adaptiveImageCropping& imgCrop);
        adaptiveImageCropping(IN const cv::Mat& srcImage, IN const std::vector<float*>&  detRes, IN const unsigned int& clsNum, IN const unsigned int& mInfeHeight, IN const unsigned int&  mInfeWidth, IN int* ptr);
        adaptiveImageCropping& operator= (IN const adaptiveImageCropping& imgCrop);
        ~adaptiveImageCropping();
        int adaptiveCropImage(IN const unsigned int& targetLabelCategory, IN const  unsigned int& thresLength, IN const unsigned int& offsetHeight, IN const unsigned int&  offsetWidth, IN const unsigned int& dstImgHeight, IN const unsigned int& dstImgWidth);
        int cropImage(IN const int& offsetWidth, IN const int& offsetHeight, IN const  unsigned int& dstImgHeight, IN const unsigned int& dstImgWidth, OUT cv::Mat& dstImage);
        int countTargetNum(OUT std::vector<int>& targetsNum);
        int getTargetCoord(IN const unsigned int& targetLabelCategory, IN const unsigned  int& thresLength, OUT std::vector<float>& targetCoord);

private:
        int targetCoordUpSampling(IN_OUT std::vector<float>& targetCoord);
        int dilationTargetCoord(IN const unsigned int& offsetHeight, IN const unsigned  int& offsetWidth, IN_OUT std::vector<float>& targetCoord);
        int adaptivaFillImage(IN const cv::Mat& srcImage, IN const unsigned int&  dstImgHeight, IN const unsigned int& dstImgWidth, OUT cv::Mat& imgFilled);
};


#endif
