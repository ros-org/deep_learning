#include"version.h"
#include"common.h"

int getVisionFrameVersion(OUT char version[32])
{
    const char* visionFrame = "v1.0.1";
    memcpy(version, visionFrame, 32);
    return 0;
}


int getDetectionModelVersion(OUT char version[32])
{
    const char* detectionModelVersion = "v1.0.0";
    memcpy(version, detectionModelVersion, 32);
    return 0;
}



int getDetection2ModelVersion(OUT char version[32])
{
    const char* detection2ModelVersion = "v1.0.0";
    memcpy(version, detection2ModelVersion, 32);
    return 0;
}


int getSegmentationModelVersion(OUT char version[32])
{
    const char* segmentationModelVersion = "v1.0.0";
    memcpy(version, segmentationModelVersion, 32);
    return 0;
}


int getWeatherModelVersion(OUT char version[32])
{
    const char* weatherModelVersion = "v1.0.0";
    memcpy(version, weatherModelVersion, 32);
    return 0;
}


int getClealinessModelVersion(OUT char version[32])
{
    const char* clealinessModelVersion = "v1.0.0";
    memcpy(version, clealinessModelVersion, 32);
    return 0;
}