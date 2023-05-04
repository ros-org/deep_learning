#ifndef __DETECT_PROCESS_HPP__
#define __DETECT_PROCESS_HPP__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>


class DetectProcess
{
public:
    DetectProcess(){};
    ~DetectProcess(){};
    static DetectProcess *GetInstance();
    int init();
    int start();
    int quit();
    static DetectProcess *mpDetectProcess;
    
private:
    bool m_binit;
    

};


#endif