#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>
#include <cstring>
#include "dhnetsdk.h"
#include "dhconfigsdk.h"
#include"ptzControl.h"
#include<iostream>

//宏定义设备的用户名和密码
#define USERNAME    "admin"
#define PASSWORD    "Litian123"
 
char Ip[32] = "192.168.1.108";
char UserName[64] = "admin";
char Passwd[64] = "Litian123";
unsigned int Port = 37777;
long LoginHandle = 0L;

//*********************************************************************************
// 常用回调集合声明
// 设备断线回调函数
// 不建议在该回调函数中调用 SDK 接口
// 通过 CLIENT_Init 设置该回调函数,当设备出现断线时,SDK 会调用该函数
void CALLBACK DisConnectFunc(long lLoginID, char *pchDVRIP, int nDVRPort, long dwUser);
// 断线重连成功回调函数
// 不建议在该回调函数中调用 SDK 接口
// 通过 CLIENT_SetAutoReconnect 设置该回调函数,当已断线的设备重连成功时,SDK 会调用该函数
void CALLBACK HaveReConnect(long lLoginID, char *pchDVRIP, int nDVRPort, long dwUser);
//*********************************************************************************

/*
// Ptz 控制信息结构体
typedef struct tagPtzControlInfo
{
    tagPtzControlInfo():m_iCmd(-1), m_bStopFlag(false){}
    tagPtzControlInfo(int iCmd, const std::string& sDescription, bool bStopFlag):m_iCmd(iCmd), m_sDescription(sDescription), m_bStopFlag(bStopFlag){}
    int m_iCmd;
    std::string m_sDescription;
    bool m_bStopFlag; // 部分 Ptz 操作,start 后需要调用相应的 stop 操作
}PtzControlInfo;
*/
int main_TangTao(void)
{
    printf("****************code start****************\n");

    bool SDKInitFlag = CLIENT_Init(DisConnectFunc, 0);  // 初始化 SDK
    if(FALSE == SDKInitFlag)
    {
        printf("Initialize client SDK fail; \n");
        return 0;
    }
    else
    {
        printf("Initialize client SDK done; \n");
    }
    
    // 开启日志
    LOG_SET_PRINT_INFO logPrintInfo = {0};
	logPrintInfo.dwSize = sizeof(LOG_SET_PRINT_INFO);
	BOOL openLogFlag = CLIENT_LogOpen(&logPrintInfo);
    if (TRUE == openLogFlag)
    {
    // 成功
    printf("Success call CLIENT_LogOpen\n");
    }
    else
    {
    // 失败
    printf("Fail call CLIENT_LogOpen\n");
    }

    unsigned int NetSdkVersion = CLIENT_GetSDKVersion();  // 获取 SDK 版本信息
    printf("NetSDK version is [%d]\n", NetSdkVersion);

    // 设置断线重连回调接口,设置过断线重连成功回调函数后,当设备出现断线情况,SDK内部会自动进行重连操作
    CLIENT_SetAutoReconnect(&HaveReConnect, 0);
    // 登录时尝试建立链接 3 次登录超时时间5000ms和尝试次数3
    CLIENT_SetConnectTime(5000, 3);


    NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY stInparam;
    memset(&stInparam, 0, sizeof(stInparam));
    stInparam.dwSize = sizeof(stInparam);
    strncpy(stInparam.szIP, Ip, sizeof(stInparam.szIP) - 1);
    strncpy(stInparam.szPassword, Passwd, sizeof(stInparam.szPassword) - 1);
    strncpy(stInparam.szUserName, UserName, sizeof(stInparam.szUserName) - 1);
    stInparam.nPort = Port;
    stInparam.emSpecCap = EM_LOGIN_SPEC_CAP_TCP;

    NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY stOutparam;
    memset(&stOutparam, 0, sizeof(stOutparam));
    stOutparam.dwSize = sizeof(stOutparam);

    while(0 == LoginHandle)
    {
        LoginHandle = CLIENT_LoginWithHighLevelSecurity(&stInparam, &stOutparam); // 登录设备
        std::cout<<"登录函数返回值:"<<LoginHandle<<std::endl;
        if(0 == LoginHandle)
        {
            printf("CLIENT_LoginWithHighLevelSecurity %s[%d]Failed!Last Error[%x]\n", Ip, Port, CLIENT_GetLastError());
        }
        else
        {
            printf("CLIENT_LoginWithHighLevelSecurity %s[%d] Success\n", Ip, Port);
        }
        // 用户初次登录设备,需要初始化一些数据才能正常实现业务功能,建议登录后等待一小段时间,具体等待时间因设备而异
        sleep(1);
        printf("start move\n");
    }
/*
    // 获取云台能力集
    char szBuffer[2048] = "";
    int nError = 0;
    if (FALSE == CLIENT_QueryNewSystemInfo(LoginHandle, CFG_CAP_CMD_PTZ, 0, szBuffer, (unsigned int)sizeof(szBuffer), &nError))
    {
        printf("CLIENT_QueryNewSystemInfo Failed, cmd[CFG_CAP_CMD_PTZ], LastError[%x]\n" , CLIENT_GetLastError());
        return;
    }
    CFG_PTZ_PROTOCOL_CAPS_INFO stuPtzCapsInfo ={sizeof(CFG_PTZ_PROTOCOL_CAPS_INFO)};
    if (FALSE == CLIENT_ParseData(CFG_CAP_CMD_PTZ, szBuffer, &stuPtzCapsInfo, sizeof(stuPtzCapsInfo), NULL))
    {
        printf("CLIENT_ParseData Failed, cmd[CFG_CAP_CMD_PTZ], Last Error[%x]\n", CLIENT_GetLastError());
        return;
    }
*/
    int nChannelId = 0;
    if (FALSE == CLIENT_DHPTZControlEx(LoginHandle, nChannelId, DH_EXTPTZ_FASTGOTO, 100, 100, 1, FALSE))
    {
        printf("CLIENT_DHPTZControlEx2 Failed, cLastChoose->GetCmd()[%x]!Last Error[%x]\n", DH_EXTPTZ_STARTPANCRUISE, CLIENT_GetLastError());
    }
}

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

