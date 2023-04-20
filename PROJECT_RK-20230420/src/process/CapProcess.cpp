#include "CapProcess.hpp"
#include "../include/common.h"


void *CapProcess::start_thread(void *param)
{
    CapProcess *pProcessMgr = (CapProcess *)param;
    pProcessMgr->start();
    return NULL;
}


int CapProcess::init()
{
    m_start = false;                                                      //当是空图时为false则预测线程也不会取图成功，同时还可以保证每一张图至多被预测线程预测一次;
    m_bdebug = true;                                                      //控制图片来自本地还是摄像头(true:加载硬盘上的图片检测; false:从摄像头获取图片进行检测) 
    m_interval = 20;                                                      //帧间隔，摄像头隔m_interval帧向缓存中放图
    total_fram_num = 0;                                                   //总帧数初始化（从摄像头获取到的图的总帧数）
    mainThreadMsg = 255;                                                  //从主线程获取到的云台消息

    //----------------------------->云台控制参数初始化<-----------------------------//
    g_bNetSDKInitFlag = FALSE;                                                                              // 初始化函数返回值，为0说明初始化失败
    g_lLoginHandle = 0L;                                                                                             // 登录函数的返回值，为0说明登录失败或未登录状态
    memcpy(g_szDevIp, "192.168.1.108", sizeof("192.168.1.108"));         // 相机IP地址
    g_nPort = 37777;                                                                                                      // tcp 连接端口,需与期望登录设备页面 tcp 端口配置一致
    memcpy(g_szUserName, "admin", sizeof("admin"));                      // 用户名
    memcpy(g_szPasswd, "Litian123", sizeof("Litian123"));                // 登录密码

    InitPtz();
    //----------------------------->云台控制参数初始化<-----------------------------//

    String source = "rtsp://admin:Litian123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0";
    m_captrue.set(cv::CAP_PROP_BUFFERSIZE, 2);
    m_captrue.set(cv::CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));    //视频流格式
    m_captrue.open(source);
    CHECK_EXPR(m_captrue.isOpened() == false,-1);
    cout << "CapProcess::init finished!" << endl; 
    pthread_mutex_init(&m_mutex,NULL);
    int res = pthread_create(&m_tid,NULL,start_thread,(void*)this);

    return 0;
}

// 初始化云台转动相关参数及设备
void CapProcess::InitPtz()
{
    // 初始化SDK
    CapProcess::g_bNetSDKInitFlag = CLIENT_Init(CapProcess::DisConnectFunc, 0);
    if (FALSE == g_bNetSDKInitFlag)
    {
        printf("Initialize client SDK fail; \n");
        return;
    }
    else
    {
        printf("Initialize client SDK done; \n");
    }

    // 获取 SDK 版本信息
    DWORD dwNetSdkVersion = CLIENT_GetSDKVersion();
    printf("NetSDK version is [%d]\n", dwNetSdkVersion);

    // 设置断线重连回调接口,设置过断线重连成功回调函数后,当设备出现断线情况,SDK内部会自动进行重连操作;此操作为可选操作,但建议用户进行设置
    CLIENT_SetAutoReconnect(&HaveReConnect, 0);
    // 设置登录超时时间和尝试次数(可选操作)
    int nWaitTime = 5000;                                    // 登录请求响应超时时间设置为 5s
    int nTryTimes = 3;                                       // 登录时尝试建立链接 3 次
    CLIENT_SetConnectTime(nWaitTime, nTryTimes);

    // (可选操作)设置更多网络参数,NET_PARAM 的 nWaittime,nConnectTryNum 成员与CLIENT_SetConnectTime 接口设置的登录设备超时时间和尝试次数意义相同
    NET_PARAM stuNetParm = {0};
    stuNetParm.nConnectTime = 3000;                          // 登录时尝试建立链接的超时时间
    stuNetParm.nPicBufSize = 4*1024*1024;                    // unit byte
    CLIENT_SetNetworkParam(&stuNetParm);

    NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY stInparam;
    memset(&stInparam, 0, sizeof(stInparam));
    stInparam.dwSize = sizeof(stInparam);
    strncpy(stInparam.szIP, g_szDevIp, sizeof(stInparam.szIP) - 1);
    strncpy(stInparam.szPassword, g_szPasswd, sizeof(stInparam.szPassword) - 1);
    strncpy(stInparam.szUserName, g_szUserName, sizeof(stInparam.szUserName) - 1);
    stInparam.nPort = g_nPort;
    stInparam.emSpecCap = EM_LOGIN_SPEC_CAP_TCP;

    NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY stOutparam;
    memset(&stOutparam, 0, sizeof(stOutparam));
    stOutparam.dwSize = sizeof(stOutparam);

    while(0 == g_lLoginHandle)
    {
        std::cout<<"准备登录设备"<<std::endl;
        g_lLoginHandle = CLIENT_LoginWithHighLevelSecurity(&stInparam, &stOutparam);
        if(0 == g_lLoginHandle)
        {
            // 根据错误码,可以在 dhnetsdk.h 中找到相应的解释,此处打印的是 16 进制,头文件中是十进制,其中的转换需注意
            // 例如: #define NET_NOT_SUPPORTED_EC(23) // 当前 SDK 未支持该功能,对应的错误码为 0x80000017, 23 对应的 16 进制为 0x17
            printf("CLIENT_LoginWithHighLevelSecurity %s[%d]Failed!Last Error[%x]\n" ,g_szDevIp ,g_nPort ,CLIENT_GetLastError());
        }
        else
        {
            printf("CLIENT_LoginWithHighLevelSecurity %s[%d] Success\n" ,g_szDevIp ,g_nPort);

            // 用户初次登录设备,需要初始化一些数据才能正常实现业务功能,建议登录后等待一小段时间,具体等待时间因设备而异。
            sleep(2);

            // 设置云台初始化角度
            std::cout<<"第一次初始化转动云台"<<std::endl;
            // ptzControl(2070, 10);    //1P机器
            // ptzControl(730, 10);     //1P机器
            ptzControl(2470, 10);       //1.5P机器
            sleep(2);
        }
    }
}

// 云台转动控制
void CapProcess::ptzControl(INPUT const int& verticalAngle , INPUT const int& horizontalAngle)
{
    //云台转动
    if(0 != CapProcess::g_lLoginHandle)
    {
        bool ptzControl = FALSE;
        int nChannelId = 0;
        // 云台转动参数说明：
        //     参数1：CLIENT_LoginWithHighLevelSecurity的返回值;
        //     参数2：视频通道号,从 0 开始递增的整数;
        //     参数3：控制命令类型(枚举值，控制着云台是要进行变焦/转动/光圈/...),该参数和后面的3个参数强相关;
        //     参数4：如参数3是DH_EXTPTZ_FASTGOTO(快速定位)，则代表水平坐标(0-8192);如参数3是DH_EXTPTZ_EXACTGOTO(三维精确定位),则该参数是水平角度(0-3600);
        //     参数5：如参数3是DH_EXTPTZ_FASTGOTO(快速定位)，则代表垂直坐标(0-8192);如参数3是DH_EXTPTZ_EXACTGOTO(三维精确定位),则该参数是垂直较度(0-900);
        //     参数6：如参数3是DH_EXTPTZ_FASTGOTO(快速定位)，代表变倍(4);如参数3是DH_EXTPTZ_EXACTGOTO(三维精确定位),代表变倍(1-128);
        //     参数7：停止标志。对云台八方向操作及镜头操作命令有效,进行其他操作时,本参数应填充 FALSE;
        //     参数8：支持扩展控制命令参数，支持8种控制命令，具体见NetSdk编程手册78页;
        ptzControl = CLIENT_DHPTZControlEx(CapProcess::g_lLoginHandle, nChannelId, DH_EXTPTZ_FASTGOTO, horizontalAngle, verticalAngle, 1, FALSE);
        if (FALSE == ptzControl)
        {
            std::cout<<"云台转动失败"<<std::endl;
        }
        else
        {
            std::cout<<"云台转动成功"<<std::endl;
        }
    }
    else
    {
        std::cout<<"请检查云台是否初始化或连接是否正常!"<<std::endl;
    }
}

//专门与外界进行信息交互的接口：获取主线程发来的消息
void CapProcess::getMsgFromMainThread(INPUT unsigned char& signalValue)
{
    mainThreadMsg = signalValue;
    if(1 == mainThreadMsg)
    {
        //转云台(往左)
        // ptzControl(730, 10);     // 1P机器
        ptzControl(930, 10);        // 1.5P机器
    }
    else if(2 == mainThreadMsg)
    {
        //转云台(往右)
        // ptzControl(2070, 10);    // 1P机器
        ptzControl(2470, 10);       // 1.5P机器
    }
    else
    {
        //云台不动
        //Do something
    }
    
    mainThreadMsg = 255;
}

//云台转动回调函数
void CALLBACK CapProcess::DisConnectFunc(long lLoginID, char *pchDVRIP, int nDVRPort, long dwUser)
{
    printf("Call DisConnectFunc\n");
    printf("lLoginID[0x%x]", lLoginID);
    if (NULL != pchDVRIP)
    {
        printf("pchDVRIP[%s]\n", pchDVRIP);
    }
    printf("nDVRPort[%d]\n", nDVRPort);
    printf("dwUser[%p]\n", dwUser);
    printf("\n");
}

//云台转动回调函数
void CALLBACK CapProcess::HaveReConnect(long lLoginID, char *pchDVRIP, int nDVRPort, long dwUser)
{
    printf("Call HaveReConnect\n");
    printf("lLoginID[0x%x]", lLoginID);
    if (NULL != pchDVRIP)
    {
        printf("pchDVRIP[%s]\n", pchDVRIP);
    }
    printf("nDVRPort[%d]\n", nDVRPort);
    printf("dwUser[%p]\n", dwUser);
    printf("\n");
}


int CapProcess::start()
{
    // 获取一次摄像头的图，得到其宽/高尺寸。摄像头宽高尺寸并不固定，所以最好不要手动写成固定的尺寸;
    m_captrue >> m_frame;
    if (m_frame.empty()) 
    {
        m_start = false;
        std::cout<<"请检查相机是否连接正确！"<<std::endl;
        return -1;
    }
    m_frame_h = m_frame.rows;
    m_frame_w = m_frame.cols;
    m_frame_rcv = Mat::zeros(m_frame_h, m_frame_w, CV_8UC3);
    m_frame_run = Mat::zeros(m_frame_h, m_frame_w, CV_8UC3);
    m_frame_size = m_frame_h * m_frame_w * 3;


    Mat im_test_cap;
    int size;    
    if (m_bdebug == true) 
    {
        std::cout<<"Read local image...*.jpg"<<std::endl;
        Mat im_test = cv::imread("/userdata/data/rain.jpg", cv::IMREAD_UNCHANGED); 
        std::cout<<"Check:离线图像通道数="<<im_test.channels()<<std::endl;
        // 这不是一个多余操作。防止本地上的图和摄像头获取的图的尺寸不一样，所以额外加一个resize操作
        if(m_frame_w == im_test.cols &&  m_frame_h==im_test.rows)
        {
            im_test_cap = im_test;
        }
        else
        {
            cv::resize(im_test, im_test_cap, cv::Size(m_frame_w, m_frame_h), (0, 0), (0, 0), cv::INTER_LINEAR);
            std::cout<<"???????????????离线图片尺寸和相机实时获取的图像尺寸不相等???????????????"<<std::endl;
        }
        size = im_test_cap.cols * im_test_cap.rows * 3;
    }


    while (m_frame.empty() == false)
    {
        m_captrue >> m_frame;                 
        if (m_frame.empty()) 
        {
            m_start = false;
            // return -1;
        }
        total_fram_num++;                                         //每次取的帧如果不为空，则帧数加1
        // std::cout<<"图像获取线程运行中，已获得图像数："<<total_fram_num<<std::endl;  
        
        //将离线图写入缓存
        if (m_bdebug == true) 
        {
            pthread_mutex_lock(&m_mutex);
            memcpy(m_frame_rcv.data, im_test_cap.data, size);
            // std::cout<<"将本地的图片写入缓存"<<std::endl;
            m_start = true;
            pthread_mutex_unlock(&m_mutex);
        } 
        //将在线图写入缓存
        else 
        {
            if(total_fram_num % m_interval == 0)
            {
                pthread_mutex_lock(&m_mutex);
                memcpy(m_frame_rcv.data, m_frame.data, m_frame_size);
                m_start = true;
                // std::cout<<"将摄像头上的图片写入缓存"<<std::endl;
                pthread_mutex_unlock(&m_mutex);
            }
        }
    }

    return 0;
}


int CapProcess::run(Mat &frame)
{
    static int cnt = 0;
    if (m_start == false) 
    {
        return false;
    }

    pthread_mutex_lock(&m_mutex);
    memcpy(m_frame_run.data, m_frame_rcv.data, m_frame_size);
    frame = m_frame_run;
    m_start = false;
    pthread_mutex_unlock(&m_mutex);

    cnt++;

    return true;
}


int CapProcess::quit()
{
    return 0;
}
