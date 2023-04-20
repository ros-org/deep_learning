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

class ProcessMgr
{
public:
    ProcessMgr(){};
    ~ProcessMgr(){};
    static ProcessMgr *GetInstance();
    int init();
    int start();
    int quit();
    int run();
    static void *start_thread(void *);
    static ProcessMgr *mpProcessMgr;
    
private:
    int save_image(Mat &im,char *title);
    void writeMsgToLogfile(const std::string& strMsg,  unsigned char info);    //将消息写入日志文件
    void writeMsgToLogfile2(const std::string& strMsg,  float info);                      //将消息写入日志文件,专用于写入float数据
    bool m_b_detect;                                              //检测模型标志
    bool m_b_seg;                                                    //分割模型标志
    bool m_b_cla;                                                     //分类模型标志
    bool m_b_weatherClassification;              //是否运行天气分类模型           
    // bool m_binit;
    CapProcess m_CapProcess;
    pthread_t m_tid;
    Configer *mpConfiger;                                    //模型参数实例指针
    YOLO_CFG_t *m_p_yolo_cfg;                      //检测模型yolo参数指针
    Yolo mYolo;                                                          //检测模型对象

    SEG_CFG_t *m_p_seg_cfg;                           //分割模型unetpp参数指针
    Segmentation mSeg;                                       //分割模型unetpp对象
    bool m_bdebug;                                                //调试模式标志
    int m_debug_fd;                                                //调试日志文件（由open打开的文件）
    
    Uart m_uart;                                                       //串口消息发送类

    CLA_CFG_t* m_p_cla_cfg;                           //清洁度分类参数指针
    Classify mCla;                                                    //清洁度分类模型对象

};

#endif


