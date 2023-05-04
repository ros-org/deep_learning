#include "SegProcess.hpp"


SegProcess *SegProcess::mpSegProcess = NULL;

SegProcess *SegProcess::GetInstance()
{
    if (mpSegProcess == NULL) {
        mpSegProcess = new SegProcess();
        mpSegProcess->init();
    }
    return mpSegProcess;
}

int SegProcess::init()
{
    return 0;
}

int SegProcess::start()
{

    return 0;
}

int SegProcess::quit()
{
    return 0;
}


