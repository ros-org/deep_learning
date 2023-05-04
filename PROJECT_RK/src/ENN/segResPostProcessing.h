/*==========================================================
ģ����  ���Էָ������д��������ż�������Ե�ĽǶȣ�
�ļ���  ��segResPostProcessing.h
����ļ�����
����    ��Liangliang Bai (liangliang.bai@leapting.com)
��Ȩ    ��<Copyright(C) 2022- Suzhou leapting Technology Co., Ltd. All rights reserved.>
�޸ļ�¼��
��  ��        �汾     �޸���   �߶���  �޸ļ�¼

2022.04.12    1.0.0.1  ������           ���ָ���ͼ�ϵķָ�������ȡ�������ֱ������
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
	SELECT_AREA,			  //ѡ���������
	SELECT_RECTANGULARITY,	  //ѡ��������ζ�
	SELECT_WIDTH,			  //ѡ�������ȣ�ƽ���������ᣩ
	SELECT_HEIGHT,			  //ѡ������߶ȣ�ƽ���������ᣩ
	SELECT_ROW,				  //ѡ����������������
	SELECT_COLUMN,			  //ѡ����������������
	SELECT_RECT2_LEN1,		  //ѡ��������С��Ӿ��ε�һ�볤��
	SELECT_RECT2_LEN2,		  //ѡ��������С��Ӿ��ε�һ����
	SELECT_RECT2_PHI		  //ѡ��������С��Ӿ��εķ���
};

enum SelectOperation
{
	SELECT_AND,		          //��
	SELECT_OR		          //��
};


/*
* ---------------------------------------->��ȡ�㷨�汾��<---------------------------------------- *
* Parameters:
*     input��
*         None
*     output��
*         char* versionNum����ǰ�汾�ţ�
* returnValue:
*     None
* ---------------------------------------->��ȡ�㷨�汾��<---------------------------------------- *
*/
void getAlgVersion(OUTPUT char* versionNum);


/*
* ---------------------------------------->�ָ�������<---------------------------------------- *
* Parameters:
*     input��
*         cv::Mat& image������ͼ�񣬷ָ����������ͼ(����ֵ��0��1��2��...);
*     output��
*         float& angle���ż�����Ե֮��ĽǶȣ�
* returnValue:
*     LT::ErrorValue������ֵ���ͣ����󲶻�
* ---------------------------------------->�ָ�������<---------------------------------------- *
*/
LT::ErrorValue postProcessingSegRes(INPUT cv::Mat& image, OUTPUT float& angle);


/*
* ------------------------------->ʵ��������halcon��select_shape����<------------------------------- *
* Parameters:
*     input��
*         cv::Mat src������ͼ��ͼ��ĸ�ʽ��8λ��ͨ����ͼ�񣬲��ұ�����Ϊ��ֵͼ��;
*         INPUT std::vector<SelectShapeType> types:Ҫ������״������
*         INPUT SelectOperation operation������Ҫ�ص��������ͣ��롢�򣩣�
*         INPUT std::vector<double> mins����Ӧ��������Сֵ��������������types��
*	      INPUT std::vector<double> maxs����Ӧ���������ֵ��������������types��
*     output��
*         cv::Mat& dst��������ѡ�������ͼ��
* returnValue:
*     None
* ------------------------------->ʵ��������halcon��select_shape����<------------------------------- *
*/
void selectShape(INPUT cv::Mat src,
	INPUT std::vector<SelectShapeType> types,
	INPUT SelectOperation operation,
	INPUT std::vector<double> mins,
	INPUT std::vector<double> maxs,
	OUTPUT cv::Mat& dst);

