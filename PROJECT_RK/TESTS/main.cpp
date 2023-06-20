// ==========================================================
// 实现功能：启动视觉框架主函数
// 文件名称：main.cpp
// 相关文件：无
// 作   者：Liangliang Bai (liangliang.bai@leapting.com)
// 版   权：<Copyright(C) 2023-Leapting Technology Co.,LTD All rights reserved.>
// 修改记录：
// 日   期             版本       修改人   走读人  
// 2022.9.28          2.0.2      白亮亮

// 修改记录：
// 2023-05-25:将模型测试等其他测试代码写到另外一个文件
// ==========================================================
#include <stdio.h>
#include "Configer.hpp"
#include "Yolo.hpp"
#include "common.h"
#include "ProcessMgr.hpp"
#include <iostream>
#include <fstream>
#include "Configer.hpp"



int main(int argc, char **argv)
{
#if 1
    // 由于C语言pthread_create函数的原因，导致在C语言中启动线程的方式看着很难受（不直接优雅美观）
    ProcessMgr *ProcessMgr = ProcessMgr::GetInstance();
    ProcessMgr->start();

    // 阻塞主线程
    while (1)
    {
        pause();
    }
#else
    // Do something
#endif
    return 0;
}