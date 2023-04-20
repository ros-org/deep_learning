// ==========================================================
// 实现功能：读取消息的类头文件。1、从驱动板收消息并发送给主线程(然后由主线程分发给相机线程); 2、从主线程获取检测的各种消息，并发送给驱动板;
// 文件名称：Uart.hpp
// 相关文件：无
// 作   者：Liangliang Bai (liangliang.bai@leapting.com)
// 版   权：<Copyright(C) 2022-Leapting Technology Co.,LTD All rights reserved.>
// 修改记录：
// 日   期       版本     修改人   走读人  修改记录
// 2022.09.28   1.0.0.1  白亮亮           None
// ==========================================================

#ifndef __UART_HPP__
#define __UART_HPP__

#include<stdio.h>  
#include<stdlib.h>  
#include<unistd.h>  
#include<sys/signal.h>   
#include<errno.h>  
#include <termios.h>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "common.h"

#define INPUT
#define OUTPUT

using namespace std;

class Uart
{
public:
       
private:
    // bool m_binit;
    // struct termios m_options;
    int m_fd;                                                                     //打开的串口文件句柄
    unsigned char m_bufferSend[8];                                                //发送消息的buffer，存储发送给开发板的数据
    int m_len;                                                                    //和驱动板交互时，一包消息的长度
    static int m_waitFlag;                                                        //收消息的标志,值为0就去读发来的消息，读完重新置为异常值-999
    pthread_mutex_t m_uartMutex;                                                  //互斥锁
    pthread_mutex_t m_uartMutex2;                                                 //互斥锁
    pthread_t m_uartTid;                                                          //线程标识                                         
    char* m_portName;                                                             //串口端口号，字符串，"/dev/ttyS3"：代表端口号3;
    unsigned char m_mesageFromDriverCard;                                         //收到的从驱动板发来的具体消息值
    unsigned char m_messageFromMainThread;                                        //从主线程获取到的消息，消息线程会将该消息发给驱动板
    

public:
    //只要显示的定义任何一种构造函数，系统就不会再自动生成这样默认的(空实现)构造函数。如果希望有一个这样的无参构造函数，则需要显示地写出来;
    Uart(){};                                                                     //无参构造
    Uart(INPUT char* portName);                                                   //有参构造
    ~Uart();
    int init();                                                                   //初始化函数
    int deinit();                                                                 //释放init函数中创建的资源
    //线程间交互的消息只有：1、收发消息线程收主线程发来的待发送消息;
    //                  2、收发消息的线程无需将转头的消息发送给主线程（主线程收到转头消息后，再将其分发给相机线程）;
    void sendMsgToMainThread(OUTPUT unsigned char& signalValue);                  //线程数据交互函数，将消息从消息线程发到主线程  
    void getMsgFromMainThread(INPUT unsigned char& signalValue);                    //线程数据交互函数，将消息从主线程发到消息线程   

private:
    int run();
    unsigned short ByteCrc16(INPUT unsigned char *buffer, INPUT unsigned short size);
    int set_opt(INPUT const int& fd, INPUT const int& nSpeed, INPUT const int& nBits, INPUT const char& nEvent, INPUT const int& nStop);
    int open_port_v1(INPUT char *dev_file);
    int send_base(INPUT unsigned char *pdata,INPUT const int& len);
    int send_to_stm32(INPUT unsigned char *pdata, INPUT const int& len); 
    int receive_from_stm32(OUTPUT unsigned char *pdata,INPUT const int& len);
    int receive_base(unsigned char *pdata,int len);
    static void signal_handler_IO (INPUT int status) ;                                  //类的static函数在类内声明，在类外定义时就不能再加static了
    static void* start_thread(INPUT void* param);                                 //由于pthread_create函数的参数类型的原因，导致该成员函数必须是static函数，其中void*参数是为了接收this用的;
    
};


#endif