#include<iostream>
#include "include/testFuncs.h"    //注意：include的时候要么是绝对路径，要么是基于makefile文件的路径，否则编译会无法通过
#include"camera.h"
#include"ptzControl.h"

int main(int argc,char **argv)
{
    func_1();
    std::cout<<"XXXXXXXXXXXXXXXXX"<<std::endl;

    // main_TangTao();

    InitTest();
    RunTest();
    EndTest();

    return 0;
}
