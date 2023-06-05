// ==========================================================
// 实现功能：视觉框架主线程
// 文件名称：ProcessMgr.cpp
// 相关文件：无
// 作   者：Liangliang Bai (liangliang.bai@leapting.com)
// 版   权：<Copyright(C) 2023-Leapting Technology Co.,LTD All rights reserved.>
// 修改记录：
// 日   期             版本       修改人   走读人  
// 2022.9.28          2.0.2      白亮亮

// 修改记录：
// 2023-05-24:检测到下坡，将发送1次消息改为发多次消息
//            分割后可以通过则屏蔽一端时间(防止在过桥的过程出现异常结果)
// ==========================================================

#include "ProcessMgr.hpp"
#include "segResPostProcessing.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <numeric>
#include <chrono>                       //用于测试时间，该方式更精准

#define cameraPtzResetFrequence 200     //相机云台归零频率
#define bridgeLengthThres 280           //桥架长度阈值
#define lowerBridgeLengthThres 250      //下坡类别目标长度
#define ignoreDownBredgeInfo 100        //忽略下桥架消息次数
#define minCt 500000                    //一次检测使用的最小时间，小于这个时间则阻塞

using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

//外面的对象调用GetInstance函数，可以将该全局对象赋给外面的对象
ProcessMgr *ProcessMgr::mpProcessMgr = NULL;
ProcessMgr *ProcessMgr::GetInstance()
{
    if (mpProcessMgr == NULL) 
    {
        mpProcessMgr = new ProcessMgr();                         //new 创建对象，并调用构造函数，
        mpProcessMgr->init();
    }
    return mpProcessMgr;                                         // 返回主线程类对象地址，
}


int ProcessMgr::init()
{
    int ret = -1;                                                // 初始化一个变量，
    char* portName = "/dev/ttyS3";                               // RKNN 烧写系统后 dev下的端口号，没有 ttyS3 说明系统有问题，
    m_uart = Uart(portName);                                     // 通信端口号，
    m_bdebug = true;                                             // 调试模式标志(true则会写日志和保存部分结果图)
    m_b_detect = true;                                           // 检测模型(大模型)是否运行标志
    if(m_b_detect)
    {
        m_b_detect2 = true;                                      // 检测模型(小模型)是否运行标志，该模型不能手动开关，切记
        m_b_seg = true;                                          // 分割模型是否运行标志，该模型不能手动开关关，切记
    }
    else
    {
        m_b_detect2 = false;                                          
        m_b_seg = false;
    }
    m_b_weatherClassification = true;                            // 是否运行天气分类
    m_b_cla = false;                                             // 清洁度分类标志
    minCleannessValue = 10.f;                                    // 清洁度最小值
    maxCleannessValue = 90.f;                                    // 清洁度最大值

    if (m_bdebug == true) 
    {
        m_debug_fd = open("/userdata/output/debug.txt",O_RDWR | O_TRUNC | O_CREAT);
        CHECK_EXPR(m_debug_fd < 0,-1);

        char *data = "Debug ..................................\n";
        int size = strlen(data);
        int nwrite = write(m_debug_fd,data,strlen(data));
        CHECK_EXPR(nwrite != size,-1);
    } 
    char visionFrameVersion[32];
    getVisionFrameVersion(visionFrameVersion);  
    writeMsgToLogfile("当前视觉框架版本号:",visionFrameVersion); 

    mpConfiger = Configer::GetInstance(); 
    
    // 启动消息线程
    ret = m_uart.init();                                         
    if(0 != ret)
    {
        std::cout<<"消息线程启动失败,请检查..."<<std::endl;
        writeMsgToLogfile2("消息线程启动失败,请检查...", ret);
    }
    CHECK_EXPR(ret != 0,-1);

    m_p_yolo_cfg = mpConfiger->get_yolo_cfg();
    if (m_b_detect) 
    {   
        char detectionModelVersion[32]; 
        getDetectionModelVersion(detectionModelVersion);
        writeMsgToLogfile("检测模型1版本号:",detectionModelVersion); 

        std::cout<<"Loading detection model..."<<std::endl;
        ret = mYolo.init(m_p_yolo_cfg);
        if(0 != ret)
        {
            std::cout<<"检测模型1初始化失败,请检查..."<<std::endl;
            writeMsgToLogfile2("检测模型1初始化失败,请检查...", ret);
        }
        else
        {
            std::cout<<"检测模型1初始化成功!"<<std::endl;
            writeMsgToLogfile2("检测模型1初始化成功!", ret);
        }
        CHECK_EXPR(ret != 0,-1);
    }

    m_p_yolo_cfg2 = mpConfiger->get_yolo_cfg2();
    if (m_b_detect2) 
    {   
        char detection2ModelVersion[32]; 
        getDetection2ModelVersion(detection2ModelVersion);
        writeMsgToLogfile("检测模型2版本号:",detection2ModelVersion); 

        std::cout<<"Loading detection2 model..."<<std::endl;
        ret = mYolo2.init(m_p_yolo_cfg2);
        if(0 != ret)
        {
            std::cout<<"检测模型2初始化失败,请检查..."<<std::endl;
            writeMsgToLogfile2("检测模型2初始化失败,请检查...", ret);
        }
        else
        {
            std::cout<<"检测模型2初始化成功!"<<std::endl;
            writeMsgToLogfile2("检测模型2初始化成功!", ret);
        }
        CHECK_EXPR(ret != 0,-1);
    }

    m_p_seg_cfg = mpConfiger->get_seg_cfg();
    if (m_b_seg) 
    {
        char segmentationMdelVersion[32];
        getSegmentationModelVersion(segmentationMdelVersion);
        writeMsgToLogfile("分割模型版本号:", segmentationMdelVersion);

        std::cout<<"Loading segementation model..."<<std::endl;
        ret = mSeg.init(m_p_seg_cfg);
        if(0 != ret)
        {
            std::cout<<"分割模型初始化失败,请检查..."<<std::endl;
            writeMsgToLogfile2("分割模型初始化失败,请检查...", ret);
        }
        else
        {
            std::cout<<"分割模型初始化成功!"<<std::endl;
            writeMsgToLogfile2("分割模型初始化成功!", ret);
        }
        CHECK_EXPR(ret != 0,-1);                       
    }

    m_p_cla_weather_cfg = mpConfiger->get_cla_weather_cfg();
    if (m_b_weatherClassification) 
    {
        char weatherModelVersion[32];
        getWeatherModelVersion(weatherModelVersion);
        writeMsgToLogfile("天气模型版本号:", weatherModelVersion);

        std::cout<<"Loading weather classification model..."<<std::endl;
        ret = mCla_weather.init(m_p_cla_weather_cfg);
        if(0 != ret)
        {
            std::cout<<"天气分类模型初始化失败,请检查..."<<std::endl;
            writeMsgToLogfile2("天气分类模型初始化失败,请检查...", ret);
        }
        else
        {
            std::cout<<"天气分类模型初始化成功!"<<std::endl;
            writeMsgToLogfile2("天气分类模型初始化成功!", ret);
        }
        CHECK_EXPR(ret != 0,-1);
    }

    m_p_cla_cfg = mpConfiger->get_cla_cfg();
    if (m_b_cla) 
    {
        char cleannessModelVersion[32];
        getClealinessModelVersion(cleannessModelVersion);
        writeMsgToLogfile("清洁度模型版本号:", cleannessModelVersion);

        std::cout<<"Loading cleanness classification model..."<<std::endl;
        ret = mCla.init(m_p_cla_cfg);
        if(0 != ret)
        {
            std::cout<<"清洁度模型初始化失败,请检查..."<<std::endl;
            writeMsgToLogfile2("清洁度模型初始化失败,请检查...", ret);
        }
        else
        {
            std::cout<<"清洁度模型初始化成功!"<<std::endl;
            writeMsgToLogfile2("清洁度模型初始化成功!", ret);
        }
        CHECK_EXPR(ret != 0,-1);

        //清洁度量化参数初始化(根据类别数生成每个类别的清洁度贡献值和权重)
        getCleannessQuaWeights(m_p_cla_cfg->cls_num-1);
    }

    // 启动图像线程：在该函数中开启了获取图像的线程
    ret = m_CapProcess.init();                                   
    if(0 != ret)
    {
        std::cout<<"相机线程启动失败,请检查..."<<std::endl;
        writeMsgToLogfile2("相机线程启动失败,请检查...", ret);
    }
    CHECK_EXPR(ret != 0,-1);

    return 0;
}



void *ProcessMgr::start_thread(void *param)
{
    ProcessMgr *pProcessMgr = (ProcessMgr *)param;
    pProcessMgr->run();
    return NULL;
}


int ProcessMgr::start()
{
    int res = pthread_create(&m_tid,NULL,start_thread,(void*)this);
    return 0;
}



int ProcessMgr::run()
{
    Mat frame;                                                  //拷贝缓冲区的图片，用于检测和分割
    int col_center;                                             //frame的中心col坐标
    int row_center;                                             //frame的中心row坐标
    Mat im_classify_weather_part;                               //将天气分类的图切一块
    Mat im_classify_weather_part_resize;                        //天气图，将im_classify_weather_part进行resize
    //天气图，将im_classify_weather_part_resize由HWC转CHW 
    uint8_t chwImgWeather[3*m_p_cla_weather_cfg ->feed_w*m_p_cla_weather_cfg ->feed_h];                                                  
    Mat im_classify_cleanliness_part;                           //将清洁度分类的图切一块
    Mat im_classify_cleanliness_part_resize;                    //清洁度图，将im_classify_cleanliness_part进行resize操作
    //清洁度图，将im_classify_cleanliness_part_resize由HWC转CHW
    uint8_t chwImgCleanness[3*m_p_cla_cfg->feed_w*m_p_cla_cfg->feed_h];
    //检测图，将im_detect_rgb由HWC转CHW 
    uint8_t chwImgDet[3*m_p_yolo_cfg->feed_w*m_p_yolo_cfg->feed_h]; 
    //检测图2，将im_detect_rgb2由HWC转CHW 
    uint8_t chwImgDet2[3*m_p_yolo_cfg2->feed_w*m_p_yolo_cfg2->feed_h]; 
    //用于检测的图                      
    Mat im_detect,im_detect2, im_detect_resized, im_detect_resized2, im_detect_rgb,im_detect_rgb2;     
    //检测图，将im_seg_rgb由HWC转CHW
    uint8_t chwImgSeg[3*m_p_seg_cfg->feed_w*m_p_seg_cfg->feed_h];
    Mat im_seg, im_seg_resized, im_seg_rgb;                     //用于分割的图
    int cla_weather_cnt = 0;                                    //天气分类运行次数
    int cla_clean_cnt = 0;                                      //清洁度分类运行次数
    int det_cnt = 0;                                            //检测模型运行次数
    int det_cnt2 = 0;                                           //检测模型2运行次数
    int seg_cnt = 0;                                            //分割模型运行次数
    int inerval_cnt = 1;                                        //预测帧率(从缓冲区拿到的图片中每隔inerval_cnt检测一次)间隔
    int cnt = 0;                                                //从缓冲区真正拿到图的数量
    int infe_image_num = 0;                                     //用于推理的图像张数
    float angleThresValue = 10.;                                //角度阈值，当角度不小于该值，认为无法通过
    Timer timer;
    int classifyRes = -999;                                     //定义天气分类的类别结果
    int cleanlinessOutput = -999;                               //清洁度单次推理结果。 最脏的类别为：1；越干净类别数越大；
    float cleanlinessOutputs, cleanlinessOutputs2;              //清洁度多次推理结果中间值，当收到相机掉头信号时将结果清零(即清洁度仅计算单趟清洁的结果)
    float x,  w;                                                //当前清洁度推理结果类别 所对应的清洁度贡献值、权重
    float cleannessQuaRes;                                      //清洁度量化结果
    Mat seg_res;                                                //分割后的结果图（索引图）
    char buffer[1024];                                          //用于存储被格式化后的路径等字符串                                           
    unsigned char detInfo = 0x00;                               //发送给驱动板的检测(桥架有断裂)消息
    unsigned char segInfo = 0x00;                               //发送给驱动板的分割(角度太大)消息
    unsigned char speedDown = 0x64;                             //发送给驱动板的 要下坡了，请减速
    unsigned char restoreSpeed = 0x65;                          //发送给驱动板的 下坡结束，请恢复到原来的速度
    bool lastDownhillStatus = false;                            //上轮检测到的 是否下桥状态，和最新轮的下桥架状态做对比，结果不一致，则发消息并更新状态 
    bool latestDownhillStatus = false;                          //本轮检测到的是否下桥架状态
    bool lastfractureStatus = false;                            //上次是否断裂的状态
    bool latestFractureStatus = false;                          //最新是否断裂状态
    bool lastAngleStatus = false;                               //上次角度是否太大的状态
    bool latestAngleStatus = false;                             //当前角度是否太大的状态
    unsigned char signalFromMsgThread = 255;                    //从消息线程获取到的消息
    unsigned char msgsTomsgThread[4] = {254, 90, 254, 254};     //存储检测的各种信息的数组，当其中有值发生变化就立刻发送到消息线程，254是视觉询问驱动板专用数值(当所以值为254时，认为一切正常)；
    high_resolution_clock::time_point beginTime;                //当前图片检测开始起始时间
    high_resolution_clock::time_point endTime;                  //当前图片检测结束结束时间
    milliseconds timeInterval;                                  //当前图片检测时间间隔
    int downBridgeSleepTimes = 99999;                           //减速、加速计数
    bool bSaveOrgImage = true;                                  //是否保存原始图像  
    unsigned char cameraStatus = 0;                             //相机方向状态1,0代表清洗机没有动，不用管相机方向
    unsigned char cameraStatus2 = 0;                            //相机方向状态2
    vector<float *> res;                                        //存放第一个检测模型的推理结果
    vector<float *> res2;                                       //存放第一个检测模型的推理结果
    int bridgeNumWeights[3] = {0,0,0};                          //先进先出，记录最新三次是否检测到桥架，检测到了将最新值置1，否则置0
    int lowerBridgeNumWeights[3] = {0,0,0};                     //先进先出，记录最新三次是否检测到下桥架，检测到了将最新值置1，否则置0
    bool speedDownStatus = false;                               //减速状态，已减速则设置为true，已恢复速度则设置为false
    int sentSpeedDownInfoStatus = -999;                         //发送减速/恢复速度的状态(防止仅发送1次对方收不到)，1则持续发减速，0则持续发恢复速度
    while (1) 
    {
        //注意：模块3的顺序不能写在模块1和2之前；
        //----------------->1、从缓存区获取图像并保存原始图像<------------------//
        // 1.1、从缓存获取图像
        bool have_data = m_CapProcess.run(frame);
        if (have_data == false) 
        {
            usleep(5000);
            continue;
        }

        // 1.2、保存原始图像到路径"/userdata/autoImageAcq/"
        if(bSaveOrgImage)
        {
            std::string imageDir = "/userdata/autoImageAcq/";
            saveImage(imageDir, frame, cnt, 4);
        }
        //----------------->1、从缓存区获取图像并保存原始图像<------------------//


        //--------------> 2、间隔inerval_cnt次取到的图 进行检测<---------------//
        cnt++;
        if (cnt % inerval_cnt != 0) 
        {
            continue;
        }
        //--------------> 2、间隔inerval_cnt次取到的图 进行检测<---------------//


        // 主线程发消息到消息线程的CT必须大于消息线程的CT
        beginTime = high_resolution_clock::now();
        writeMsgToLogfile2("===========================运行次数============================", infe_image_num);
        infe_image_num++;
        
        
        //--------------------------->3、消息交互<---------------------------//
        // 3、1、发消息：主动先发消息给消息线程,切记，发完消息就将相应的值重置为默认值。
        m_uart.getMsgFromMainThread(msgsTomsgThread);
        writeMsgToLogfile("当前给开发板发送的消息是", msgsTomsgThread[0]);
        writeMsgToLogfile("当前给开发板发送的消息是", msgsTomsgThread[1]);
        writeMsgToLogfile("当前给开发板发送的消息是", msgsTomsgThread[2]);
        writeMsgToLogfile("当前给开发板发送的消息是", msgsTomsgThread[3]);
        if(detInfo == msgsTomsgThread[3] || segInfo == msgsTomsgThread[3])
        {
            //上次检测到断裂，掉头运行需要先减速，减速日时阻塞一段时间直到返回到第一次检到测断裂的位置之外
            std::cout<<"准备减速"<<std::endl;
            // usleep(7000000);
        }
        // 重置上次的检测结果
        msgsTomsgThread[2] = 254;
        msgsTomsgThread[3] = 254;

        // 3.2、收消息：从消息线程获取驱动板发来的消息
        m_uart.sendMsgToMainThread(signalFromMsgThread);
        writeMsgToLogfile("从消息线程获取的转头消息", signalFromMsgThread);

        // 3.3、发消息：将从消息线程拿到的转头消息发给相机线程
        if(255 != signalFromMsgThread)
        {
            // 3.3.1、捕捉相机的方向状态，当相机状态改变，说明单趟重新开始，则清洁度累积数值清零并重新计算；
            cameraStatus2 = signalFromMsgThread;
            if(cameraStatus != cameraStatus2)
            {
                cleanlinessOutputs = 0.0;
                cleanlinessOutputs2 = 0.0;
                writeMsgToLogfile("相机方向状态改变", signalFromMsgThread);
            }
            cameraStatus = cameraStatus2;

            // 3.3.2、调用相机的对外消息接口，并将该消息写入到相机线程，然后在相机线程中控制云台运动
            if(infe_image_num % cameraPtzResetFrequence == 0)
            {
                //强制修改主线程发给相机线程的信息，用于相机定时垂直方向归零
                if(1 == signalFromMsgThread)
                {
                    signalFromMsgThread = 3;
                }
                else if(2 == signalFromMsgThread)
                {
                    signalFromMsgThread = 4;
                }
            }
            m_CapProcess.getMsgFromMainThread(signalFromMsgThread);
            writeMsgToLogfile("将相机方向消息发给相机和图像线程", signalFromMsgThread);
            
            // 3.3.3、相机线程取完方向消息后需要重新置为异常值255
            signalFromMsgThread = 255;  
        }
        //--------------------------->3、消息交互<---------------------------//


        //------------------------->判断图像是否模糊<-------------------------//
        // Do something
        //------------------------->判断图像是否模糊<-------------------------//


        // 注意：将所有天气分为多个类别，可以工作的类别排布在后面（100-199），不可工作的类别排布在前面（0-99）
        //      0(100)-->晴天，1(101)-->下雪，2(102)-->下雨，3(103)-->沙尘暴;
        //--------------------------->4、天气分类<---------------------------//
        if(m_b_weatherClassification)
        {  
            std::cout<<"------------------Running weather classification model------------------"<<cla_weather_cnt<<std::endl;  
            writeMsgToLogfile2("--------------Running weather classification model--------------", cla_weather_cnt);
            cla_weather_cnt++;

            timer.start();
            // 4.1、图像预处理 
            col_center = frame.cols/2;
            row_center = frame.rows/2 + 130;
            im_classify_weather_part = frame(cv::Rect(col_center-m_p_cla_weather_cfg ->feed_w/2, row_center-m_p_cla_weather_cfg ->feed_h/2, m_p_cla_weather_cfg ->feed_w, m_p_cla_weather_cfg ->feed_h)).clone();
            cv::resize(im_classify_weather_part, im_classify_weather_part_resize, cv::Size(m_p_cla_weather_cfg ->feed_w,m_p_cla_weather_cfg ->feed_h), (0, 0), (0, 0), cv::INTER_LINEAR);
            HWC2CHW(im_classify_weather_part_resize, chwImgWeather);
            cv::imwrite("/userdata/output/watherImgCropped.jpg", im_classify_weather_part_resize);
            
            // 4.2天气分类推理
            int ret = mCla_weather.run(chwImgWeather, "CHW", classifyRes);
            if(0 != ret)
            {
                std::cout<<"天气模型推理异常,请检查..."<<std::endl;
                writeMsgToLogfile2("天气模型推理异常,请检查...", ret);
            }
            else
            {
                std::cout<<"天气模型推理成功!"<<std::endl;
                writeMsgToLogfile2("天气模型推理成功!", ret);
            }
            CHECK_EXPR(ret != 0,-1);
            timer.end("Cla_weather");
 
            // 4.3、将天气分类结果写入到日志
            // classifyRes = 0;
            if(0 == classifyRes)
            {
                writeMsgToLogfile2("天气分类结果:晴天", float(classifyRes));
            }
            else if(1 == classifyRes)
            {
                writeMsgToLogfile2("天气分类结果:下雪", float(classifyRes));
            }
            else if(2 == classifyRes)
            { 
                writeMsgToLogfile2("天气分类结果:下雨", float(classifyRes));
            }
            else
            {
                writeMsgToLogfile2("天气分类结果:沙尘暴", float(classifyRes));
            }

            // 4.4、实时发送天气分类结果到消息线程
            msgsTomsgThread[0] = uchar(classifyRes+100);
            writeMsgToLogfile("天气检测发送结果到驱动板:", msgsTomsgThread[0]);                                 
            m_b_weatherClassification = false;    //天气每次上电只需要检测一次
        }
        //--------------------------->4、天气分类<---------------------------//


        //------------------------>5、分类：清洁度检测<-----------------------//
        if(m_b_cla)
        {
            std::cout<<"--------------------Running cleanliness classification model----------------------"<<cla_clean_cnt<<std::endl;
            writeMsgToLogfile2("--------------Running cleanliness classification model--------------", cla_clean_cnt);
            cla_clean_cnt++;
            timer.start();

            // 5.1图像预处理
            col_center = frame.cols/2;
            row_center = frame.rows/2 + 150;
            im_classify_cleanliness_part = frame(cv::Rect(col_center-m_p_cla_cfg->feed_w/2, row_center-m_p_cla_cfg->feed_h/2, m_p_cla_cfg->feed_w, m_p_cla_cfg->feed_h)).clone();
            cv::resize(im_classify_cleanliness_part, im_classify_cleanliness_part_resize, cv::Size(m_p_cla_cfg->feed_w, m_p_cla_cfg->feed_h), (0, 0), (0, 0), cv::INTER_LINEAR);
            HWC2CHW(im_classify_cleanliness_part_resize, chwImgCleanness);

            //5.2清洁度分类推理
            int ret = mCla.run(chwImgCleanness, "CHW", cleanlinessOutput);
            if(0 != ret)
            {
                std::cout<<"清洁度模型推理异常,请检查..."<<std::endl;
                writeMsgToLogfile2("清洁度模型推理异常,请检查...", ret);
            }
            else
            {
                std::cout<<"清洁度模型推理成功!"<<std::endl;
                writeMsgToLogfile2("清洁度模型推理成功!", ret);
            }
            CHECK_EXPR(ret != 0,-1);
            timer.end("Cla_cleanliness");
            
            if(cleanlinessOutput < m_p_cla_cfg->cls_num-1)
            {
                // 5.3、统计单趟清洁度，清洗机运行一趟时清洁度是实时变化的，当第二趟开始，重新进行计算；
                getCurrentWeight(cleanlinessOutput, x,  w);
                cleanlinessOutputs += x*w;
                cleanlinessOutputs2 += w;
                cleannessQuaRes = cleanlinessOutputs/cleanlinessOutputs2;

                // 5.4、实时发送清洁度到消息线程
                msgsTomsgThread[1] = uchar(int(cleannessQuaRes));
                writeMsgToLogfile2("当前清洁度", int(cleannessQuaRes));
            }  
            else
            {
                msgsTomsgThread[1] = 255;
                writeMsgToLogfile2("当前图片异常，清洁度不进行统计", int(cleannessQuaRes));
            }
            writeMsgToLogfile2("当前清洁度预测类别", float(cleanlinessOutput));
        }
        //------------------------>5、分类：清洁度检测<-----------------------//


        //---------------------------->6、检测1<----------------------------//
        if (true == m_b_detect) 
        {
            // 6.1、模型运行标志
            writeMsgToLogfile2("-------------->Running object detect model 1<---------------:", det_cnt);
            det_cnt++;
            std::cout<<"-------------->Running object detect model 1---------------"<<std::endl;

            // 6.2、变量设置、初始化等操作
            res.clear();
            downBridgeSleepTimes++;

            // 6.3、图像前处理
            timer.start();
            im_detect = frame;
            cv::resize(im_detect, im_detect_resized, cv::Size(m_p_yolo_cfg->feed_w, m_p_yolo_cfg->feed_h), (0, 0), (0, 0), cv::INTER_LINEAR);
            cv::cvtColor(im_detect_resized, im_detect_rgb, cv::COLOR_BGR2RGB);
            HWC2CHW(im_detect_rgb, chwImgDet);
            
            // 6.4、模型推理
            int ret = mYolo.run(chwImgDet, "CHW", res);
            if(0 != ret)
            {
                std::cout<<"检测模型1推理异常,请检查..."<<std::endl;
                writeMsgToLogfile2("检测模型1推理异常,请检查...", ret);
            }
            else
            {
                std::cout<<"检测模型1推理成功!"<<std::endl;
                writeMsgToLogfile2("检测模型1推理成功!", ret);
            }
            CHECK_EXPR(ret != 0,-1);
            timer.end("Detect1");
                        
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
            
            // 统计是否下桥的权重，当三次中有两次结果一样，则准备更新下桥状态
            lowerBridgeNumWeights[0] = lowerBridgeNumWeights[1];
            lowerBridgeNumWeights[1] = lowerBridgeNumWeights[2];
            if(lowerBridgeNum>0)
            {
                lowerBridgeNumWeights[2] = 1;
            }
            else
            {
                lowerBridgeNumWeights[2] = 0;
            }

            // 6.5、下桥状态判断，连续3次检测结果之和发生改变则更新下桥状态、发消息; 
            // 6.5.1、统计是下坡还是非下坡                            
            if(lowerBridgeNumWeights[0]+lowerBridgeNumWeights[1]+lowerBridgeNumWeights[2] >= 2)
            {
                latestDownhillStatus = true;
            }
            else
            {
                latestDownhillStatus = false;
            }
            
            // 6.5.2、减速参数更新
            if(false == lastDownhillStatus && true == latestDownhillStatus)
            {
                if((downBridgeSleepTimes > ignoreDownBredgeInfo) && (false == speedDownStatus))
                {
                    downBridgeSleepTimes = 0;
                    speedDownStatus = true; 
                    sentSpeedDownInfoStatus = 1;  
                    lastDownhillStatus = latestDownhillStatus;                
                }
            }
            
            // 6.5.3、恢复速度参数更新；注：该模块不能将downBridgeSleepTimes重新置0；
            if(true == lastDownhillStatus && false == latestDownhillStatus)
            {
                if((downBridgeSleepTimes > ignoreDownBredgeInfo) && (true == speedDownStatus))
                {
                    speedDownStatus = false;
                    sentSpeedDownInfoStatus = 0;
                    lastDownhillStatus = latestDownhillStatus; 
                }
            }

            // 6.5.4、发送减速消息
            if(1 == sentSpeedDownInfoStatus)
            {
                msgsTomsgThread[2] = speedDown;
                writeMsgToLogfile("发送减速的消息：请减速", speedDown); 
            }
            
            // 6.5.5、发送恢复速度消息
            if(0 == sentSpeedDownInfoStatus)
            {
                msgsTomsgThread[2] = restoreSpeed;
                writeMsgToLogfile("发送恢复速度的消息给消息线程:请恢复速度", restoreSpeed);
            }

            // 6.6、预留类别
            if(xxx>0)
            {
                // Do something
            }

            // 6.7、预留类别2；注：需要将异常图(起始站、低头图、背景图)进行了抑制
            if(xxx2>0)
            {
                // Do something
            }

            // 统计桥架区域的权重，当连续三次中有两次结果一样，则准备裁图
            bridgeNumWeights[0] = bridgeNumWeights[1];
            bridgeNumWeights[1] = bridgeNumWeights[2];
            if(bridgeNum>0)
            {
                bridgeNumWeights[2] =1;
                writeMsgToLogfile2("------------------------------------检测到桥架-------------------------------------", 0);
                std::cout<<"------------------------------------检测到桥架-------------------------------------"<<std::endl;
            }
            else
            {
                bridgeNumWeights[2] =0;
                writeMsgToLogfile2("---------未检测到桥架---------", 0);
                std::cout<<"---------未检测到桥架---------"<<std::endl;
            }

            // 6.8、本轮(连续三次检测有两次检测到桥架区域)检测到桥架则启动检测模型2、裁图；
            if(bridgeNumWeights[0]+bridgeNumWeights[1]+bridgeNumWeights[2] >= 2)
            {
                writeMsgToLogfile("模型1检测到有桥架,准备切图!", m_b_detect2);
                std::cout<<"模型1检测到有桥架,准备切图!"<<std::endl;

                // 6.9、切图：调用自适应切图类进行切图
                int * a = new int();
                adaptiveImageCropping imgCrop(frame, res, 1, m_p_yolo_cfg->feed_h, m_p_yolo_cfg->feed_w, a);
                imgCrop.adaptiveCropImage(0, bridgeLengthThres, 5,5,m_p_yolo_cfg2->feed_h, m_p_yolo_cfg2->feed_w);
                if(!imgCrop.mDstImage.empty())
                {
                    m_b_detect2 = true;
                    im_detect2 = imgCrop.mDstImage;
                    im_seg = imgCrop.mDstImage;
                    writeMsgToLogfile("成功切到用于检测2的图像!", m_b_detect2);
                    std::cout<<"成功切到用于检测2的图像!"<<std::endl;
                }
                else
                {
                    m_b_detect2 = false;
                    m_b_seg = false;
                    lastfractureStatus = false;
                    lastAngleStatus = false;
                    writeMsgToLogfile("未切到用于检测2的图像!", m_b_detect2);
                    std::cout<<"未切到用于检测2的图像"<<std::endl;
                }
            }
            else
            {
                m_b_detect2 = false;
                m_b_seg = false;
                lastfractureStatus = false;
                lastAngleStatus = false;
            }
 
            // 6.10、保存日志及检测结果图
            if (m_bdebug == true && det_cnt < 1024)
            {
                if (res.size() > 0)               
                {
                    mYolo.show_res(im_detect_resized,res);                                                                                                                           //画框并打印检测结果
                    snprintf(buffer,sizeof(buffer),"/userdata/output/lt_det_out_%d.jpg",det_cnt);
                    cv::imwrite(buffer,im_detect_resized); 
                    
                    for (int i = 0; i < res.size(); ++i) 
                    {  
                        snprintf(buffer,sizeof(buffer),"[detect %d] %f %f %f %f %f %f\n",det_cnt,
                                res[i][0],res[i][1],res[i][2],res[i][3],
                                res[i][4],res[i][5]);
                        int size = strlen(buffer);
                        int nwrite = write(m_debug_fd,buffer,strlen(buffer));
                        CHECK_EXPR(nwrite != size,-1);
                    }                                                                                                                                          //保存检测图片
                }
            }
            
        }
        //---------------------------->6、检测1<----------------------------//
 


        //---------------------------->7、检测2<----------------------------//
        if(m_b_detect && m_b_detect2)
        {
            // 7.1、模型运行标志
            writeMsgToLogfile2("-------------->Running object detect model 2<---------------:", det_cnt2);
            det_cnt2++;
            std::cout<<"-------------->Running object detect model 2<---------------"<<std::endl;

            // 7.2、变量设置、初始化等操作
            res2.clear();
            
            // 7.3、图像前处理
            timer.start();
            cv::resize(im_detect2, im_detect_resized2, cv::Size(m_p_yolo_cfg2->feed_w, m_p_yolo_cfg2->feed_h), (0, 0), (0, 0), cv::INTER_LINEAR);
            cv::cvtColor(im_detect_resized2, im_detect_rgb2, cv::COLOR_BGR2RGB);
            HWC2CHW(im_detect_rgb2, chwImgDet2);

            // 7.4、模型推理
            int ret = mYolo2.run(chwImgDet2, "CHW", res2);
            if(0 != ret)
            {
                std::cout<<"检测模型2推理异常,请检查..."<<std::endl;
                writeMsgToLogfile2("检测模型2推理异常,请检查...", ret);
            }
            else
            {
                std::cout<<"检测模型2推理成功!"<<std::endl;
                writeMsgToLogfile2("检测模型2推理成功!", ret);
            }
            CHECK_EXPR(ret != 0,-1);
            timer.end("Detect2");

            // 7.5、根据推理输出的结果，统计每个类别([no fracture,fracture])目标数
            int fractureNum = 0;                                       
            for (int i = 0;i < res2.size(); ++i) 
            {
                if (res2[i][5] == 1.)                                    
                {
                    fractureNum++;
                }
            }

            // 7.6、是否有断裂 状态判断，两次状态不一致则发消息(这样可以做到 只发一次断裂/无断裂的消息给驱动板）                          
            if(fractureNum > 0)
            {
                latestFractureStatus = true;
                m_b_seg = false;
                lastAngleStatus = false;
                writeMsgToLogfile2("**************************************检测到断裂,机器准备返回*************************************", 0);
                std::cout<<"**************************************检测到断裂*************************************"<<std::endl;
            }
            else
            {
                latestFractureStatus = false;
                m_b_seg = true;
                writeMsgToLogfile("检测到桥架区域无断裂,将分割模型标识符置为true", m_b_seg);
                std::cout<<"**********未检测到断裂**********"<<std::endl;
            }

            if(lastfractureStatus != latestFractureStatus)
            {
                if(lastfractureStatus==false && latestFractureStatus==true)
                {
                    msgsTomsgThread[3] = detInfo;
                    writeMsgToLogfile("发送检测结果到消息线程：上次无断裂，这次有断裂", detInfo);
                }
                else
                {
                    writeMsgToLogfile2("发送检测结果到消息线程：上次断裂，这次无断裂", 666);
                }

                lastfractureStatus = latestFractureStatus;
            }
            
            // 7.7、保存日志及检测结果图
            if (m_bdebug == true && det_cnt2 < 1024)
            {
                if (res2.size() > 0)               
                {
                    mYolo2.show_res(im_detect_resized2,res2);                                                                                                                           //画框并打印检测结果
                    snprintf(buffer,sizeof(buffer),"/userdata/output/lt_det_out2_%d.jpg",det_cnt2);
                    cv::imwrite(buffer,im_detect_resized2); 
                    
                    for (int i = 0;i < res.size();i++) 
                    {  
                        snprintf(buffer,sizeof(buffer),"[detect2 %d] %f %f %f %f %f %f\n",det_cnt2,
                                res2[i][0],res2[i][1],res2[i][2],res2[i][3], res2[i][4],res2[i][5]);
                        int size = strlen(buffer);
                        int nwrite = write(m_debug_fd,buffer,strlen(buffer));
                        CHECK_EXPR(nwrite != size,-1);
                    }                                                                                                                                            //保存检测图片
                }
            }
        }
        //---------------------------->7、检测2<----------------------------//



        //---------------------------->8、分割<-----------------------------//
        if (m_b_detect && m_b_detect2 && m_b_seg) 
        {
            // 8.1、模型运行标志
            writeMsgToLogfile2("------------>Running seg model<------------", seg_cnt);
            seg_cnt++;
            std::cout<<"------------>Running seg model<------------"<<std::endl;

            // 8.2、图像前处理
            timer.start();
            cv::resize(im_seg, im_seg_resized, cv::Size(m_p_seg_cfg->feed_w, m_p_seg_cfg->feed_h), (0, 0), (0, 0), cv::INTER_LINEAR);
            cv::cvtColor(im_seg_resized, im_seg_rgb, cv::COLOR_BGR2RGB);
            HWC2CHW(im_seg_rgb, chwImgSeg);

            // 8.3、推理
            int ret = mSeg.run(chwImgSeg, "CHW", seg_res);
            if(0 != ret)
            {
                std::cout<<"分割模型推理异常,请检查..."<<std::endl;
                writeMsgToLogfile2("分割模型推理异常,请检查...", ret);
            }
            else
            {
                std::cout<<"分割模型推理成功!"<<std::endl;
                writeMsgToLogfile2("分割模型推理成功!", ret);
            }
            CHECK_EXPR(ret != 0,-1);
            timer.end("Seg");

            // 8.4、分割后处理：计算桥架角度
            float angle = -999.;
            #if 1
            ret = (int)postProcessingSegRes(seg_res,angle);
            writeMsgToLogfile2("分割后计算角度结果", angle);

            // 8.5、根据角度值，给驱动板发消息及状态更新
            if(angle >= angleThresValue)
            {
                latestAngleStatus = true;
            }
            else
            {
                latestAngleStatus = false;
            }

            if(false == lastAngleStatus && true == latestAngleStatus)
            {
                msgsTomsgThread[3] = segInfo;
                writeMsgToLogfile("发送分割角度结果的消息(上次可以通过，本次无法通过)到消息线程", segInfo);
            }
            
            if(true == lastAngleStatus && false == latestAngleStatus)
            {
                std::cout<<"++++++++++++++++++++++++++++发送分割角度结果的消息(上次无法通过，本次可以通过)到消息线程++++++++++++++++++++++++++++"<<std::endl;
                writeMsgToLogfile2("++++++++++++++++++++++++++++发送分割角度结果的消息(上次无法通过，本次可以通过)到消息线程++++++++++++++++++++++++++++", 666);
                sleep(20);
            }

            if(lastAngleStatus==false && latestAngleStatus==false)
            {
                // 检测到桥架了，且没断、角度也不大,可以通过，在通过桥架的过程，不做任何操作
                std::cout<<"++++++++++++++++++++++++++++分割到可以通过，进行阻塞++++++++++++++++++++++++++++"<<std::endl;
                writeMsgToLogfile2("++++++++++++++++++++++++++++分割到可以通过，进行阻塞++++++++++++++++++++++++++++", 0);
                sleep(20);
            }
            lastAngleStatus = latestAngleStatus;

            #endif

            // 8.6、写日志及将分割结果图保存到硬盘上
            if (m_bdebug == true && seg_cnt < 1024) 
            {
                snprintf(buffer,sizeof(buffer),"[seg %d]angle:%f\n",seg_cnt,angle);
                int size = strlen(buffer);
                int nwrite = write(m_debug_fd,buffer,strlen(buffer));
                CHECK_EXPR(nwrite != size,-1);

                snprintf(buffer,sizeof(buffer),"/userdata/output/lt_seg_out_%d.jpg",seg_cnt);
                mSeg.draw_seg(im_seg_resized,seg_res);
                cv::imwrite(buffer,im_seg_resized);
                
                // 将分割结果图画在原图上(该方法更耗时)
                /*
                timer.start();
                std::vector<std::vector<cv::Point>> seg_res_contour;
                std::vector<cv::Vec4i> hreir;
                cv::findContours(seg_res, seg_res_contour, hreir, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE,cv::Point());
                drawContours(im_seg, seg_res_contour, -1, cv::Scalar(255, 0, 0),1, 8);
                snprintf(buffer,sizeof(buffer),"/userdata/output/segMask_%d.jpg",debug_seg_cnt++);
                cv::imwrite(buffer,im_seg);
                */
            }

        }
        //---------------------------->8、分割<-----------------------------//


        //注意：当主线程比消息线程快的时候，就无法保证上次的消息真正发出去，所以此处要稍微阻塞一下主线程
        //-------------------------->9、阻塞主线程<--------------------------//
        //9、1.计算上次流程的总时间
        endTime = high_resolution_clock::now();      
        timeInterval = std::chrono::duration_cast<milliseconds>(endTime - beginTime);
        std::cout<<"Running time: "<<timeInterval.count()<<"ms"<<std::endl;

        //9.2、阻塞主线程（当主线程运行一次的时间小于300ms，则阻塞）
        if(timeInterval.count()<minCt)
        {
            usleep(minCt-timeInterval.count());  //注意：usleep单位是微妙
        }
        //-------------------------->9、阻塞主线程<--------------------------//            
    }

    return 0;
}



// 实现功能：将单张图像由HWC转CHW；注意：只支持cv::Vec3b格式的Mat(uchar数字，一个像素占一个字节)；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     hwcImage：待转换的HWC格式原图;
// outputParas:
//     chwImage：转换后的CHW格式图像；
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void HWC2CHW(INPUT const cv::Mat& hwcImage, OUTPUT uint8_t * chwImage)    // void HWC2CHW(INPUT const cv::Mat& hwcImage, OUTPUT uint8_t chwImage [])
{
	int imgC = hwcImage.channels();
	int imgH = hwcImage.rows;
	int imgW = hwcImage.cols;

	for (int c = 0; c < imgC; ++c)
	{
		for (int h = 0; h < imgH; ++h)
		{
			for (int w = 0; w < imgW; ++w)
			{
				int dstIdx = c * imgH * imgW + h * imgW + w;
				int srcIdx = h * imgW * imgC + w * imgC + c;
				chwImage[dstIdx] =  hwcImage.at<cv::Vec3b>(h, w)[c];   
			}
		}
	}
}


void ProcessMgr::getCleannessQuaWeights(INPUT const int& cla_num)
{
    // 检查传入的参数是否异常
    if(cla_num<2)
    {
        std::cout<<"请检查分类类别数:类别数不能小于2..."<<std::endl;
    }

    float x;
    float w;
    for(int i = 0; i<cla_num; ++i)
    {
        x = minCleannessValue+(maxCleannessValue-minCleannessValue)/(cla_num-1)*i;
        w = 1.0/(i+1);
        X.push_back(x);
        W.push_back(w);
    }

    //检查输出的参数是否异常
    //Do something
}


void ProcessMgr::getCurrentWeight(INPUT const int& currentClaRes, OUTPUT float& x, OUTPUT float& w)
{
    x = X.at(currentClaRes);
    w = W.at(currentClaRes);
}


void ProcessMgr::writeMsgToLogfile(const std::string& strMsg,  unsigned char info)
{
    tm_YMDHMS currentTime;
	struct tm * localTime;
	time_t nowtime;
	time(&nowtime);                              //得到当前系统时间
	localTime = localtime(&nowtime);             //将nowtime变量中的日历时间转化为本地时间，存入到指针为p的时间结构体中
    currentTime.changeTmToYmdhms(*localTime);    //Change tm format data into tm_YMDHMS;
	
    char* logBuff = new char[1024];
    snprintf(logBuff,1024,"%d-%d-%d-%d-%d-%d: %s:%d;\n", currentTime.tm_year,currentTime.tm_mon,currentTime.tm_mday,currentTime.tm_hour, currentTime.tm_min,currentTime.tm_sec,strMsg.c_str(), int(info));
    int nwrite = write(m_debug_fd,logBuff,strlen(logBuff));
    if(nullptr != logBuff)
    {
        delete [] logBuff;
        logBuff = nullptr;
    }
}


void ProcessMgr::writeMsgToLogfile2(const std::string& strMsg,  float info)
{
    tm_YMDHMS currentTime;
	struct tm * localTime;
	time_t nowtime;
	time(&nowtime);                              //得到当前系统时间
	localTime = localtime(&nowtime);             //将nowtime变量中的日历时间转化为本地时间，存入到指针为p的时间结构体中
    currentTime.changeTmToYmdhms(*localTime);    //Change tm format data into tm_YMDHMS;
	
    char* logBuff = new char[1024];
    snprintf(logBuff,1024,"%d-%d-%d-%d-%d-%d: %s   %f;\n", currentTime.tm_year,currentTime.tm_mon,currentTime.tm_mday,currentTime.tm_hour, currentTime.tm_min,currentTime.tm_sec,strMsg.c_str(), info);
    int nwrite = write(m_debug_fd,logBuff,strlen(logBuff));
    if(nullptr!=logBuff)
    {
        delete [] logBuff;
        logBuff = nullptr;
    }
}



void ProcessMgr::writeMsgToLogfile(const std::string& strMsg,  char info[32])
{
    tm_YMDHMS currentTime;
	struct tm * localTime;
	time_t nowtime;
	time(&nowtime);                              //得到当前系统时间
	localTime = localtime(&nowtime);             //将nowtime变量中的日历时间转化为本地时间，存入到指针为p的时间结构体中
    currentTime.changeTmToYmdhms(*localTime);    //Change tm format data into tm_YMDHMS;
	
    char* logBuff = new char[1024];
    snprintf(logBuff,1024,"%d-%d-%d-%d-%d-%d: %s %s;\n", currentTime.tm_year,currentTime.tm_mon,currentTime.tm_mday,currentTime.tm_hour, currentTime.tm_min,currentTime.tm_sec,strMsg.c_str(), info);
    int nwrite = write(m_debug_fd,logBuff,strlen(logBuff));
    if(nullptr!=logBuff)
    {
        delete [] logBuff;
        logBuff = nullptr;
    }
}



int ProcessMgr::saveImage(INPUT std::string& saveDir, INPUT Mat& im, INPUT int& cnt, INPUT const int& saveFrequency)
{
    std::string saveImagePath = saveDir + intToString(cnt)+ ".jpg";
    if(0 == cnt%saveFrequency)
    {
        cv::imwrite(saveImagePath, im);   
    }

    // 将当前最新图片序号保存在文件中，方便下次读取并继续保存
    // Do something
} 


// 函数功能：获取当前时间(时间格式是：年月日时分秒)
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     logBuff：时间字符串
// outputParas:
//     None
// returnValue:
//     None
// ----------------------------------->parameters<----------------------------------
void getCurrentTime(OUTPUT char* logBuff)
{
    tm_YMDHMS currentTime;
	struct tm * localTime;
	time_t nowtime;
	time(&nowtime);                              //得到当前系统时间
	localTime = localtime(&nowtime);             //将nowtime变量中的日历时间转化为本地时间，存入到指针为p的时间结构体中
    currentTime.changeTmToYmdhms(*localTime);    //Change tm format data into tm_YMDHMS;
    snprintf(logBuff,1024,"%d_%d_%d_%d_%d_%d-", currentTime.tm_year,currentTime.tm_mon,currentTime.tm_mday,currentTime.tm_hour, currentTime.tm_min,currentTime.tm_sec);
}


// 函数功能：int转字符串
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     integer:int数据
// outputParas:
//     None
// returnValue:
//     str：int转为string格式数据
// ----------------------------------->parameters<----------------------------------
std::string intToString(INPUT int& integer)
{
	char buf[32] = {0};
	snprintf(buf, sizeof(buf), "%u", integer);
 
	std::string str = buf;
	return str;
}


