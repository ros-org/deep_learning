// ==========================================================
// 实现功能：相机线程，实现读图、写图、转头等操作
// 文件名称：CapProcess.hpp
// 相关文件：无
// 作   者：Liangliang Bai (liangliang.bai@leapting.com)
// 版   权：<Copyright(C) 2023-Leapting Technology Co.,LTD All rights reserved.>
// 修改记录：
// 日   期             版本       修改人   走读人  
// 2022.9.28          2.0.2      白亮亮
// ==========================================================
#ifndef __CAP_PROCESS_HPP__
#define __CAP_PROCESS_HPP__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <iostream>

#include <time.h>
#include "dhnetsdk.h"
#include "dhconfigsdk.h"
#include"avglobal.h"
#include<iostream>
#include<cstring>
#include <unistd.h>

#include<opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "common.h"

#ifndef INPUT
#define INPUT
#endif

#ifndef OUTPUT
#define OUTPUT
#endif

using namespace std;
using namespace cv;

class CapProcess                          // 相机线程类
{
public:
    CapProcess(){};                       // 构造函数
    ~CapProcess(){};                      // 析构函数
    // static CapProcess *GetInstance(); 
    int init();             
    int run(Mat &frame); 
    int quit();
    int start();
    static void *start_thread(void *param);  // 开启相机线程

    // 专门与外界进行信息交互的接口
    void getMsgFromMainThread(INPUT unsigned char& signalValue);   // signalValue 相机线程从主线程得到的信号值  

private:
    void InitPtz();
    static void CALLBACK DisConnectFunc(long lLoginID, char *pchDVRIP, int nDVRPort, long dwUser);
    static void CALLBACK HaveReConnect(long lLoginID, char *pchDVRIP, int nDVRPort, long dwUser);
    void ptzControl(INPUT const int& verticalAngle , INPUT const int& horizontalAngle);



public:
    static CapProcess *mpCapProcess;             // 相机线程的成员指针
    
private:
    bool m_binit;                                //  相机线程成员指针
    VideoCapture m_captrue;                      // 打开相机
    int m_interval;                 //帧间隔
    int total_fram_num;             //从摄像头获取的图像总帧数
    Mat m_frame;                    //最新帧，获取到后就拷贝到缓存帧
    Mat m_frame_rcv;                //缓存帧
    Mat m_frame_run;                //运行帧，运行的时候去拷贝缓存帧，最新帧-》缓存/缓存帧-》运行帧用一把互斥锁，保证了放的时候不取，取的时候不放;
    pthread_t m_tid;
    pthread_mutex_t m_mutex;        //锁头
    int m_frame_h;                  //原图Height，不是resize后的Height
    int m_frame_w;                  //原图Width
    int m_frame_size;               //一张 源彩色图 的像素个数（H×W×3）
    bool m_start;                   //该参数值由程序决定，当预测的线程要想获得有效图像，必须为true
    bool m_bdebug;                  //控制图片来自本地还是摄像头(true:load from local disk; false:get image from camera)
    unsigned char mainThreadMsg;    //从主线程传来的云台控制消息 


    BOOL g_bNetSDKInitFlag;         // 初始化函数返回值，为0说明初始化失败
    LLONG g_lLoginHandle;           // 登录函数的返回值，为0说明登录失败或未登录状态
    char g_szDevIp[32];             // 相机IP 地址
    WORD g_nPort = 37777;           // tcp 连接端口,需与登录设备页面 tcp 端口配置一致
    char g_szUserName[64];          // 用户名
    char g_szPasswd[64];            // 登录密码
};

#endif


