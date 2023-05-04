/*==========================================================
模块名  ：对分割结果进行处理，计算桥架两个边缘的角度；
文件名  ：segResPostProcessing.h
相关文件：无
作者    ：Liangliang Bai (liangliang.bai@leapting.com)
版权    ：<Copyright(C) 2022- Suzhou leapting Technology Co., Ltd. All rights reserved.>
修改记录：
日  期        版本     修改人   走读人  修改记录

2022.04.12    1.0.0.1  白亮亮           将分割结果图上的分割轮廓提取出来并分别拟合线
==========================================================*/

#pragma once
#ifndef __SEGRESPOSTPROCESSING__
#define __SEGRESPOSTPROCESSING__
#endif // !__SEGRESPOSTPROCESSING__

#include"LT_VISION_COMMON.h"
#include<cmath>
#define PI 3.1415926


enum SelectShapeType
{
	SELECT_AREA,			  //选中区域面积
	SELECT_RECTANGULARITY,	  //选中区域矩形度
	SELECT_WIDTH,			  //选中区域宽度（平行于坐标轴）
	SELECT_HEIGHT,			  //选中区域高度（平行于坐标轴）
	SELECT_ROW,				  //选中区域中心行索引
	SELECT_COLUMN,			  //选中区域中心列索引
	SELECT_RECT2_LEN1,		  //选中区域最小外接矩形的一半长度
	SELECT_RECT2_LEN2,		  //选中区域最小外接矩形的一半宽度
	SELECT_RECT2_PHI		  //选中区域最小外接矩形的方向
};

enum SelectOperation
{
	SELECT_AND,		          //与
	SELECT_OR		          //或
};


/*
* ---------------------------------------->获取算法版本号<---------------------------------------- *
* Parameters:
*     input：
*         None
*     output：
*         char* versionNum：当前版本号；
* returnValue:
*     None
* ---------------------------------------->获取算法版本号<---------------------------------------- *
*/
void getAlgVersion(OUTPUT char* versionNum);


/*
* ---------------------------------------->分割结果后处理<---------------------------------------- *
* Parameters:
*     input：
*         cv::Mat& image：输入图像，分割出来的索引图(像素值是0、1、2，...);
*     output：
*         float& angle：桥架两边缘之间的角度；
* returnValue:
*     LT::ErrorValue：返回值类型，错误捕获；
* ---------------------------------------->分割结果后处理<---------------------------------------- *
*/
LT::ErrorValue postProcessingSegRes(INPUT cv::Mat& image, OUTPUT float& angle);


/*
* ------------------------------->实现类似于halcon的select_shape算子<------------------------------- *
* Parameters:
*     input：
*         cv::Mat src：输入图像，图像的格式是8位单通道的图像，并且被解析为二值图像;
*         INPUT std::vector<SelectShapeType> types:要检查的形状特征；
*         INPUT SelectOperation operation：各个要素的链接类型（与、或）；
*         INPUT std::vector<double> mins：对应特征的最小值，其个数必须等于types；
*	      INPUT std::vector<double> maxs：对应特征的最大值，其个数必须等于types；
*     output：
*         cv::Mat& dst：按条件选择后的输出图像；
* returnValue:
*     None
* ------------------------------->实现类似于halcon的select_shape算子<------------------------------- *
*/
void selectShape(INPUT cv::Mat src,
	INPUT std::vector<SelectShapeType> types,
	INPUT SelectOperation operation,
	INPUT std::vector<double> mins,
	INPUT std::vector<double> maxs,
	OUTPUT cv::Mat& dst);

