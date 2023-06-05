/*==========================================================
ģ����  �������Ӿ�ͨ��ͷ�ļ���
�ļ���  ��LT_VISION_COMMON.h
����ļ�����
����    ��Liangliang Bai (liangliang.bai@leapting.com)
��Ȩ    ��<Copyright(C) 2022- Suzhou leapting Technology Co., Ltd. All rights reserved.>
�޸ļ�¼��
��  ��        �汾     �޸���   �߶���  �޸ļ�¼

2022.04.13    1.0.0.1  ������           ���й��̻���ͷ�ļ���Ҫ������ͷ�ļ�������ÿ���ļ���includeһ�Σ�
==========================================================*/
#ifndef __LT_VISION_COMMON__
#define __LT_VISION_COMMON__
#endif

#ifndef INPUT
#define INPUT
#endif

#ifndef OUTPUT
#define OUTPUT
#endif

#include<iostream>
#include<thread>
// #include<opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>        
#include <opencv2/highgui/highgui.hpp> 

namespace LT
{
    // ǿö���࣬���ڲ������;
    enum class ErrorValue : int
    {
        SUCCESS = 0,                  //��������ȷ���
        IS_VARIABLE_EMPTY,            //����ı���Ϊ��
        VARIABLE_INCORRECT,           //���ܴ���ı�������
        CODE_STREAM_MODE_ERROR,       //���������ģʽ���󣬽�֧��1/2/3/4;
        CURRENT_FRAME_EMPTY,          //��ǰ֡δ��ȡ��
        CAMERA_NOT_OPEN,              //�����ʧ��
        CAMERA_OPENED_SUCCESS,        //����򿪳ɹ�
        GET_CURRENT_FRAME_SUCCESS,    //��ȡ��ǰ֡�ɹ�
        FAILED_TO_GET_CURRENT_FRAME,  //��ȡ��ǰ֡ʧ��
        IMAGE_EMPTY,                  //ͼ��Ϊ��
        WRONG_LABEL_NUM,              //�ָ�����żܸ�������
        POINTS_NUM_WRONG              //用于拟合线的点集数量太少
    };


    // �㷨���������� �ṹ��
    typedef struct LT_BRIDGE_ANGLE
    {
        float angle;
        //...
        int nStructSize;

        LT_BRIDGE_ANGLE() :angle(-999), nStructSize(sizeof(LT_BRIDGE_ANGLE)) {}
    } BridgeAngle;
}