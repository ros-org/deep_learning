#ifndef __SEG_PROCESS_HPP__
#define __SEG_PROCESS_HPP__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>

class SegProcess
{
public:
    SegProcess(){};
    ~SegProcess(){};
    static SegProcess *GetInstance();
    int init();
    int start();
    int quit();
    static SegProcess *mpSegProcess;
    
private:
    bool m_binit;
    

};

#endif