#ifndef __CAMERA__
#define __CAMERA__

#include <stdio.h>
#include <time.h>
#include "dhnetsdk.h"
#include "dhconfigsdk.h"
#include"avglobal.h"
#include<iostream>
#include<cstring>
#include <unistd.h>
#include<string>
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>        
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/videoio.hpp>

#ifndef INPUT
#define INPUT
#endif
#ifndef OUTPUT
#define OUTPUT
#endif

//*********************************************************************************
// 常用回调集合声明
// 设备断线回调函数
// 不建议在该回调函数中调用 SDK 接口
// 通过 CLIENT_Init 设置该回调函数,当设备出现断线时,SDK 会调用该函数。
// void CALLBACK DisConnectFunc(LLONG lLoginID, char *pchDVRIP, LONG nDVRPort, DWORD dwUser);
void CALLBACK DisConnectFunc(long lLoginID, char *pchDVRIP, int nDVRPort, long dwUser);

// 断线重连成功回调函数
// 不建议在该回调函数中调用 SDK 接口
// 通过 CLIENT_SetAutoReconnect 设置该回调函数,当已断线的设备重连成功时,SDK 会调用该函数。
void CALLBACK HaveReConnect(long lLoginID, char *pchDVRIP, int nDVRPort, long dwUser);
// 抓图回调函数
// 不建议在该回调函数中调用 SDK 接口
// 通过 CLIENT_SetSnapRevCallBack 设置该回调函数,当前端设备有抓图数据发送过来时,SDK 会调用该函数
void CALLBACK SnapRev(LLONG lLoginID, BYTE *pBuf, UINT RevLen, UINT EncodeType, DWORD CmdSerial, LDWORD dwUser);
//*********************************************************************************
// 常用函数声明
// 获取输入的整形
int GetIntInput(char *szPromt, int& nError);

// 获取输入的字符串
void GetStringInput(const char *szPromt , char *szBuffer);
//*********************************************************************************

void InitTest();
void RunTest();
void EndTest();

void getMsgFromMainThread(OUTPUT unsigned char& signalValue);

#endif