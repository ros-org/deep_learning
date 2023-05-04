#include "DetectProcess.hpp"


DetectProcess *DetectProcess::mpDetectProcess = NULL;

DetectProcess *DetectProcess::GetInstance()
{
    if (mpDetectProcess == NULL) {
        mpDetectProcess = new DetectProcess();
        mpDetectProcess->init();
    }
    return mpDetectProcess;
}

int DetectProcess::init()
{
    return 0;
}

int DetectProcess::start()
{
    return 0;
}

int DetectProcess::quit()
{
    return 0;
}