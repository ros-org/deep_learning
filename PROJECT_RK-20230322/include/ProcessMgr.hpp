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

class ProcessMgr                                                // 定义主线程类 wgn
{
public:
    ProcessMgr(){};                                             // 构造函数
    ~ProcessMgr(){};                                            // 析构函数
    static ProcessMgr *GetInstance();                           // 获取实例， wgn
    int init();                                                 // 开始/初始化  wgn
    int start();
    // int quit();
    int run();
    static void *start_thread(void *);                          // 开启线程 wgn
    static ProcessMgr *mpProcessMgr;                            // 成员指针 wgn
    
private:                                   
    int save_image(Mat &im,char *title);                        // 保存图像
    void writeMsgToLogfile(const std::string& strMsg,  unsigned char info);    //将消息写入日志文件
    void writeMsgToLogfile2(const std::string& strMsg,  float info);           //将消息写入日志文件,专用于写入float数据
    bool m_b_detect;                                            //检测模型标志
    bool m_b_seg;                                               //分割模型标志
    bool m_b_cla;                                               //分类模型标志
    bool m_b_weatherClassification;                             //是否运行天气分类模型           
    // bool m_binit;
    CapProcess m_CapProcess;
    pthread_t m_tid;                                            // 线程id 号，
    Configer *mpConfiger;                                       //模型参数实例指针
    YOLO_CFG_t *m_p_yolo_cfg;                                   //检测模型yolo参数指针
    Yolo mYolo;                                                 //检测模型对象

    SEG_CFG_t *m_p_seg_cfg;                                     //分割模型unetpp参数指针
    Segmentation mSeg;                                          //分割模型unetpp对象
    bool m_bdebug;                                              //调试模式标志
    int m_debug_fd;                                             //调试日志文件（由open打开的文件）
    
    Uart m_uart;                                                //串口消息发送类

    CLA_CFG_t* m_p_cla_weather_cfg;                   //天气分类参数指针，wgn
    Classify mCla_weather;                            //天气分类模型对象，wgn

    CLA_CFG_t* m_p_cla_cfg;                           //清洁度分类参数指针
    Classify mCla;                                    //清洁度分类模型对象

};

#endif


