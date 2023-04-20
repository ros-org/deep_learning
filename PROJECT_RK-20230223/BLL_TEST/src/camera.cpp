// ==========================================================
// 实现功能：调用DaHua的SDK进行图形获取&&云台控制功能；
// 文件名称：camera.cpp
// 相关文件：无
// 作   者：Liangliang Bai (liangliang.bai@leapting.com)
// 版   权：<Copyright(C) 2022-Leapting Technology Co.,LTD All rights reserved.>
// 修改记录：
// 日   期       版本     修改人   走读人  修改记录
// 2022.09.28   1.0.0.1  白亮亮           None
// ==========================================================

// 步骤1: 完成 SDK 初始化流程。
// 步骤2: 初始化成功后,调用 CLIENT_LoginWithHighLevelSecurity 登录设备。
// 步骤3: 调用 CLIENT_SetSnapRevCallBack 设置抓图回调函数,当 SDK 收到设备端发送过来的抓图数据时,会调用 fSnapRev 回调函数回调图片信息及图片数据给用户。
// 步骤4: 调用 CLIENT_SnapPictureEx 发送抓图命令给前端设备,在 fSnapRev 回调函数中等待设备回复的图片信息。
// 步骤5: 调用 CLIENT_Logout,注销用户。
// 步骤6: SDK 功能使用完后,调用 CLIENT_Cleanup 释放 SDK 资源。

#include"camera.h"

// #pragma comment(lib, "dhnetsdk.lib")
// #pragma comment(lib, "dhconfigsdk.lib")

static BOOL g_bNetSDKInitFlag = FALSE;          // 初始化函数返回值，为0说明初始化失败
static LLONG g_lLoginHandle = 0L;               // 登录函数的返回值，为0说明登录失败或未登录状态
static char g_szDevIp[32] = "192.168.1.108";
static WORD g_nPort = 37777;                    // tcp 连接端口,需与期望登录设备页面 tcp 端口配置一致
static char g_szUserName[64] = "admin";
static char g_szPasswd[64] = "Litian123";
static short g_nCmdSerial = 0;                  // 抓图序列号
cv::VideoCapture m_captrue;
cv::Mat m_frame;


void InitTest()
{
    // 初始化SDK
    g_bNetSDKInitFlag = CLIENT_Init(DisConnectFunc, 0);
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
    std::cout<<"CLIENT_SetAutoReconnect"<<std::endl;
    // 设置登录超时时间和尝试次数(可选操作)
    int nWaitTime = 5000;                                    // 登录请求响应超时时间设置为 5s
    int nTryTimes = 3;                                       // 登录时尝试建立链接 3 次
    CLIENT_SetConnectTime(nWaitTime, nTryTimes);
    std::cout<<"CLIENT_SetConnectTime"<<std::endl;

    // (可选操作)设置更多网络参数,NET_PARAM 的 nWaittime,nConnectTryNum 成员与CLIENT_SetConnectTime 接口设置的登录设备超时时间和尝试次数意义相同
    NET_PARAM stuNetParm = {0};
    stuNetParm.nConnectTime = 3000;                          // 登录时尝试建立链接的超时时间
    stuNetParm.nPicBufSize = 4*1024*1024;                    // unit byte
    CLIENT_SetNetworkParam(&stuNetParm);
    std::cout<<"CLIENT_SetNetworkParam"<<std::endl;

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
        std::cout<<"g_lLoginHandle="<<g_lLoginHandle<<std::endl;
        if(0 == g_lLoginHandle)
        {
            // 根据错误码,可以在 dhnetsdk.h 中找到相应的解释,此处打印的是 16 进制,头文件中是十进制,其中的转换需注意
            // 例如: #define NET_NOT_SUPPORTED_EC(23) // 当前 SDK 未支持该功能,对应的错误码为 0x80000017, 23 对应的 16 进制为 0x17
            printf("CLIENT_LoginWithHighLevelSecurity %s[%d]Failed!Last Error[%x]\n" ,g_szDevIp ,g_nPort ,CLIENT_GetLastError());
        }
        else
        {
            printf("CLIENT_LoginWithHighLevelSecurity %s[%d] Success\n" ,g_szDevIp ,g_nPort);
        }
        // 用户初次登录设备,需要初始化一些数据才能正常实现业务功能,建议登录后等待一小段时间,具体等待时间因设备而异。
        sleep(1);

    }

    std::cout<<"Opencv获取图像..."<<std::endl;
    // cv::VideoCapture m_captrue;
    cv::String source = "rtsp://admin:Litian123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0";
    m_captrue.open(source);
    std::cout<<"m_captrue.isOpened()="<<m_captrue.isOpened()<<std::endl;
    if(m_captrue.isOpened() == false)
    {
        std::cout<<"Opencv获取图像失败"<<std::endl;
    }
    else
    {
        std::cout<<"Opencv获取图像成功"<<std::endl;
    }
}


void RunTest()
{
    if (FALSE == g_bNetSDKInitFlag)
    {
        return;
    }
    if (0 == g_lLoginHandle)
    {
        return;
    }

    //云台转动
    if(0 != g_lLoginHandle)
    {
        bool ptzControl = FALSE;
        int nChannelId = 0;
        sleep(5);
        ptzControl = CLIENT_DHPTZControlEx2(g_lLoginHandle, nChannelId, DH_EXTPTZ_FASTGOTO, 120, 100, 1, FALSE);
        if (FALSE == ptzControl)
        {
            std::cout<<"云台转动失败"<<std::endl;
        }
        else
        {
            std::cout<<"云台转动成功"<<std::endl;
        }
        sleep(5);
        CLIENT_DHPTZControlEx2(g_lLoginHandle, nChannelId, DH_EXTPTZ_FASTGOTO, 5, 5, 1, FALSE);
    }



    //*********************************************************************抓图方式1***********************************************************************
    // 抓图回调函数原形(pBuf内存由SDK内部申请释放); EncodeType 编码类型，10：表示jpeg图片 0：mpeg4的i帧;
    // 声明于dhnetsdk头文件中的函数指针，该类型指针指向返回值为void，参数为(LLONG lLoginID, BYTE *pBuf, UINT RevLen, UINT EncodeType, DWORD CmdSerial, LDWORD dwUser)的函数，
    //typedef void (CALLBACK *fSnapRev)(LLONG lLoginID, BYTE *pBuf, UINT RevLen, UINT EncodeType, DWORD CmdSerial, LDWORD dwUser);
    //定义于camera.cpp文件的SnapRev函数就是返回值为void，参数为(LLONG lLoginID, BYTE *pBuf, UINT RevLen, UINT EncodeType, DWORD CmdSerial, LDWORD dwUser);
    //void CALLBACK SnapRev            (LLONG lLoginID, BYTE *pBuf, UINT RevLen, UINT EncodeType, DWORD CmdSerial, LDWORD dwUser);
    //该函数声明于dhnetsdk头文件，其第一个参数的类型就是fSnapRev;
    //CLIENT_NET_API void CALL_METHOD CLIENT_SetSnapRevCallBack(fSnapRev OnSnapRevMessage, LDWORD dwUser);
    // 设置抓图回调函数
    CLIENT_SetSnapRevCallBack(SnapRev, NULL);


    // 发送抓图命令给前端设备
    int nChannelId = 0;                       // 事例中默认通道 ID 为 0、抓图模式为抓一幅图,用户可根据实际情况自行选择
    int nSnapType = 0;                        // 抓图模式;-1:表示停止抓图, 0:表示请求一帧, 1:表示定时发送请求, 2:表示连续请求
    SNAP_PARAMS stuSnapParams;
    stuSnapParams.Channel = nChannelId;
    stuSnapParams.mode = nSnapType;
    stuSnapParams.CmdSerial = ++g_nCmdSerial; // 请求序列号,有效值范围 0~65535,超过范围会被截断为 unsigned short
    if (FALSE == CLIENT_SnapPictureEx(g_lLoginHandle, &stuSnapParams))
    {
        printf("CLIENT_SnapPictureEx Failed!Last Error[%x]\n", CLIENT_GetLastError());
        return;
    }
    else
    {
        printf("CLIENT_SnapPictureEx succ\n");
    }

    //使用opencv获取图
    m_captrue >> m_frame;
    if(m_frame.empty())
    {
        std::cout<<"m_frame.empty()==true"<<std::endl;
    }
    else
    {
        std::cout<<"m_frame.empty()==false"<<std::endl;
    }
    cv::String imagePath = "/userdata/aaaaaaaaaaaaaa.jpg";
    cv::imwrite(imagePath,m_frame);
    //*********************************************************************抓图方式1***********************************************************************

}


void EndTest()
{
    printf("input any key to quit!\n");
    getchar();
    // 退出设备
    if (0 != g_lLoginHandle)
    {
        if(FALSE == CLIENT_Logout(g_lLoginHandle))
        {
            printf("CLIENT_Logout Failed!Last Error[%x]\n", CLIENT_GetLastError());
        }
        else
        {
            g_lLoginHandle = 0;
        }
    }

    // 清理初始化资源
    if (TRUE == g_bNetSDKInitFlag)
    {
        CLIENT_Cleanup();
        g_bNetSDKInitFlag = FALSE;
    }

}


//*********************************************************************************
// 常用回调集合定义

void CALLBACK DisConnectFunc(long lLoginID, char *pchDVRIP, int nDVRPort, long dwUser)
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


void CALLBACK HaveReConnect(long lLoginID, char *pchDVRIP, int nDVRPort, long dwUser)
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


//注意：好戏该函数可以获取到图像的二进制流，但是对这段内存排布不清楚，无法解码成cv::Mat;另外，如何持续获取图片未测试，只是根据文档提示好像是支持的，但是官方技术人员说不支持;
void CALLBACK SnapRev(LLONG lLoginID, BYTE *pBuf, UINT RevLen, UINT EncodeType, DWORD CmdSerial, LDWORD dwUser)
{
    printf("[SnapRev] -- receive data!\n");
    if(lLoginID == g_lLoginHandle)
    {
        if (NULL != pBuf && RevLen > 0)
        {
            std::cout<<"sizeof(pBuf)="<<sizeof(pBuf)<<std::endl;
            char szPicturePath[256] = "";
            time_t stuTime;
            time(&stuTime);
            char szTmpTime[128] = "";
            strftime(szTmpTime, sizeof(szTmpTime) - 1, "%y%m%d_%H%M%S",
            gmtime(&stuTime));
            snprintf(szPicturePath, sizeof(szPicturePath)-1, "/userdata/%d_%s.jpg", CmdSerial, szTmpTime);
            FILE* pFile = fopen(szPicturePath, "wb");
            if (NULL == pFile)
            {
                std::cout<<"fopen打开图片文件失败"<<std::endl;
            }
            int nWrite = 0;
            while(nWrite != RevLen)
            {
                // size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream)  //把ptr所指向的数组中的数据写入到给定流stream中。返回实际写入的数据块数目;
                // 参数：ptr--这是指向要被写入的元素数组的指针; size--这是要被写入的每个元素的大小，以字节为单位; nmemb--这是元素的个数，每个元素的大小为 size 字节;
                //      stream-- 这是指向 FILE 对象的指针，该 FILE 对象指定了一个输出流。
                nWrite += fwrite(pBuf + nWrite, 1, RevLen - nWrite, pFile);
            }
            fclose(pFile);
        }
    }
}


bool chao_StreamFileToImage(std::string filename, cv::Mat &image)
{
	const char *filenamechar = filename.c_str();
	FILE *fpr = fopen(filenamechar, "rb");
	if (fpr == NULL)
	{
		fclose(fpr);
		return false;
	}
	int channl(0);
	int imagerows(0);
	int imagecols(0);
	fread(&channl, sizeof(char), 1, fpr);    //第一个字节 通道
	fread(&imagerows, sizeof(char), 4, fpr); //四个字节存 行数
	fread(&imagecols, sizeof(char), 4, fpr); //四个字节存 列数
	if (channl == 3)
	{
		image = cv::Mat::zeros(imagerows,imagecols, CV_8UC3);
		char* pData = (char*)image.data;
		for (int i = 0; i < imagerows*imagecols; i++)
		{
			fread(&pData[i * 3], sizeof(char), 1, fpr);
			fread(&pData[i * 3 + 1], sizeof(char), 1, fpr);
			fread(&pData[i * 3 + 2], sizeof(char), 1, fpr);
		}
	}
	else if (channl == 1)
	{
		image = cv::Mat::zeros(imagerows, imagecols, CV_8UC1);
		char* pData = (char*)image.data;
		for (int i = 0; i < imagerows*imagecols; i++)
		{
			fread(&pData[i], sizeof(char), 1, fpr);
		}
	}
	fclose(fpr);
	return true;
}


//*********************************************************************************
// 常用函数定义
int GetIntInput(char *szPromt, int& nError)
{
    long int nGet = 0;
    char* pError = NULL;
    printf(szPromt);
    char szUserInput[32] = "";
    std::cin.getline(szUserInput, 32);
    // gets(szUserInput);
    nGet = strtol(szUserInput, &pError, 10);
    if ('\0' != *pError)
    {
        // 入参有误
        nError = -1;
    }
    else
    {
        nError = 0;
    }
    return nGet;
}


void GetStringInput(const char *szPromt , char *szBuffer)
{
    printf(szPromt);
    int len = strlen(szPromt);
    std::cin.getline(szBuffer, len);
}



void getMsgFromMainThread(OUTPUT unsigned char& signalValue)
{
    mainThreadMsg = signalValue;
    if(1 == mainThreadMsg)
    {
        //转云台(往左)
        //Do something
        std::cout<<"====================================云台往左转====================================================="<<std::endl;
    }
    else if(2 == mainThreadMsg)
    {
        //转云台(往右)
        //Do something
        std::cout<<"====================================云台往右转====================================================="<<std::endl;
    }
    else
    {
        //云台不动
        std::cout<<"====================================禁止云台转动====================================================="<<std::endl;
    }
    std::cout<<"图像线程获取到的转云台的消息"<<(int)mainThreadMsg<<std::endl;
    
    signalValue = 255;
    mainThreadMsg = 255;
}
