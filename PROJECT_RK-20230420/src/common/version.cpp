#include"version.h"
#include"common.h"

int getVisionFrameVersion(OUT char version[32])
{
    const char* visionFrame = "v2.0.0";
    memcpy(version, visionFrame, 32);
}


int getDetectionModelVersion(OUT char version[32])
{
    const char* detectionModelVersion = "";
    memcpy(version, detectionModelVersion, 32);
}



int getDetection2ModelVersion(OUT char version[32])
{
    const char* detection2ModelVersion = "";
    memcpy(version, detection2ModelVersion, 32);
}


int getSegmentationModelVersion(OUT char version[32])
{
    const char* segmentationModelVersion = "";
    memcpy(version, segmentationModelVersion, 32);
}


int getWeatherModelVersion(OUT char version[32])
{
    const char* weatherModelVersion = "";
    memcpy(version, weatherModelVersion, 32);
}


int getClealinessModelVersion(OUT char version[32])
{
    const char* clealinessModelVersion = "";
    memcpy(version, clealinessModelVersion, 32);
}