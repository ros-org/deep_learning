#include"version.h"
#include"common.h"


// 视觉框架版本号
int getVisionFrameVersion(OUT char version[32])
{
    const char* visionFrame = "v1.0.6";
    memcpy(version, visionFrame, 32);
    return 0;
}


// 检测模型1版本号
int getDetectionModelVersion(OUT char version[32])
{
    const char* detectionModelVersion = "v1.0.0";
    memcpy(version, detectionModelVersion, 32);
    return 0;
}


// 检测模型2版本号
int getDetection2ModelVersion(OUT char version[32])
{
    const char* detection2ModelVersion = "v1.0.0";
    memcpy(version, detection2ModelVersion, 32);
    return 0;
}


// 分割模型版本号
int getSegmentationModelVersion(OUT char version[32])
{
    const char* segmentationModelVersion = "v1.0.0";
    memcpy(version, segmentationModelVersion, 32);
    return 0;
}


// 天气模型版本
int getWeatherModelVersion(OUT char version[32])
{
    const char* weatherModelVersion = "v1.0.0";
    memcpy(version, weatherModelVersion, 32);
    return 0;
}


// 清洁度模型版本
int getClealinessModelVersion(OUT char version[32])
{
    const char* clealinessModelVersion = "v1.0.0";
    memcpy(version, clealinessModelVersion, 32);
    return 0;
}