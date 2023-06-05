// ==========================================================
// 实现功能：视觉框架主线程
// 文件名称：ProcessMgr.hpp
// 相关文件：无
// 作   者：Liangliang Bai (liangliang.bai@leapting.com)
// 版   权：<Copyright(C) 2023-Leapting Technology Co.,LTD All rights reserved.>
// 修改记录：
// 日   期             版本       修改人   走读人  
// 2022.9.28          2.0.2      白亮亮
// ==========================================================

#ifndef __PROCESS_MGR_HPP__
#define __PROCESS_MGR_HPP__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include "CapProcess.hpp"
#include "pthread.h"
#include "Configer.hpp"
#include "Yolo.hpp"
#include "Segmentation.hpp"
#include "Uart.hpp"
#include"getCurrentTime.h"
#include"classify.h"
#include"version.h"
#include"imageCropping.h"


class ProcessMgr                                                // 定义主线程类 wgn
{
public:
    ProcessMgr(){};                                             // 构造函数
    ~ProcessMgr(){};                                            // 析构函数
    static ProcessMgr *GetInstance();                           // 获取实例， wgn
    int init();                                                 // 开始/初始化  wgn
    int start();
    int run();
    static void *start_thread(void *);                          // 开启线程 wgn
    static ProcessMgr *mpProcessMgr;                            // 成员指针 wgn
    
private:                                   
    int saveImage(INPUT std::string& saveDir, INPUT Mat& im, INPUT int& cnt, INPUT const int& saveFrequency);                        // 保存图像
    void writeMsgToLogfile(const std::string& strMsg,  unsigned char info);    //将消息写入日志文件
    void writeMsgToLogfile2(const std::string& strMsg,  float info);           //将消息写入日志文件,专用于写入float数据
    void writeMsgToLogfile(const std::string& strMsg,  char info[32]);
    void getCleannessQuaWeights(INPUT const int& cla_num);                     //获取清洁度量化每个类别的清洁度贡献值和对应的权重
    void getCurrentWeight(INPUT const int& currentClaRes, OUTPUT float& x, OUTPUT float& w);
    bool m_b_detect;                                            //检测模型运行标志(大模型)
    bool m_b_detect2;                                           //检测模型运行标志(小模型)
    bool m_b_seg;                                               //分割模型运行标志
    bool m_b_cla;                                               //分类模型运行标志
    bool m_b_weatherClassification;                             //是否运行天气分类模型           
    CapProcess m_CapProcess;
    pthread_t m_tid;                                            // 线程id 号，
    Configer *mpConfiger;                                       //模型参数实例指针
    YOLO_CFG_t *m_p_yolo_cfg;                                   //检测模型yolo参数指针
    YOLO_CFG_t *m_p_yolo_cfg2;                                  //检测模型2 yolo参数指针
    Yolo mYolo;                                                 //检测模型对象
    Yolo mYolo2;                                                //检测模型2对象

    SEG_CFG_t *m_p_seg_cfg;                                     //分割模型unetpp参数指针
    Segmentation mSeg;                                          //分割模型unetpp对象
    bool m_bdebug;                                              //调试模式标志
    int m_debug_fd;                                             //调试日志文件（由open打开的文件）
    
    Uart m_uart;                                                //串口消息发送类

    CLA_CFG_t* m_p_cla_weather_cfg;                             //天气分类参数指针，wgn
    Classify mCla_weather;                                      //天气分类模型对象，wgn

    CLA_CFG_t* m_p_cla_cfg;                                     //清洁度分类参数指针
    Classify mCla;                                              //清洁度分类模型对象
    std::vector<float> X, W;                                    //清洁度所有类别的清洁度贡献值及对应的权重
    float minCleannessValue;                                    // 清洁度最小值
    float maxCleannessValue;                                    // 清洁度最大值

};


// 函数功能：获取当前时间(时间格式是：年月日时分秒)
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     logBuff：时间字符串
// outputParas:
//     None
// returnValue:
//     None
// ----------------------------------->parameters<----------------------------------
void getCurrentTime(OUTPUT char* logBuff);


// 函数功能：int转字符串
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     integer:int数据
// outputParas:
//     None
// returnValue:
//     str：int转为string格式数据
// ----------------------------------->parameters<----------------------------------
std::string intToString(INPUT int& integer);


// 函数功能：根据类别数获取清洁度每个类别的清洁度贡献值及对应的权重
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     minCleannessValue:清洁度最小值
//     maxCleannessValue：清洁度最大值
//     cla_num：分类类别数
// outputParas:
//     X：所有类别的清洁度贡献值
//     W：所有类别的清洁度贡献值对应的权重
// returnValue:
//     None
// ----------------------------------->parameters<----------------------------------
void getCleannessQuaWeights(INPUT const float& minCleannessValue, INPUT const float& maxCleannessValue, INPUT const int& cla_num, OUTPUT std::vector<float>& X, OUTPUT std::vector<float>& W);


// 函数功能：根据当前推理结果获取当前对应的 清洁度贡献值 及 对应的权重
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     X：所有类别的清洁度贡献值
//     W：所有类别的清洁度贡献值对应的权重
//     currentClaRes：当前推理结果(类别索引值)
// outputParas:
//     x：当前推理结果所对应的清洁度贡献值
//     W：当前推理结果所对应的清洁度权重
// returnValue:
//     None
// ----------------------------------->parameters<----------------------------------
void getCurrentWeight(INPUT const std::vector<float>& X, INPUT const std::vector<float>& W, INPUT const int& currentClaRes, OUTPUT float& x, OUTPUT float& w);


// 实现功能：将单张图像由HWC转CHW；注意：只支持cv::Vec3b格式的Mat(uchar数字，一个像素占一个字节)；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     hwcImage：待转换的HWC格式原图;
// outputParas:
//     chwImage：转换后的CHW格式图像；
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void HWC2CHW(INPUT const cv::Mat& hwcImage, OUTPUT uint8_t * chwImage);


#endif


