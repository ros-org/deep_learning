#ifndef __VERSION_H__
#define __VERSION_H__
#include"common.h"

int getVisionFrameVersion(OUT char version[32]);

int getDetectionModelVersion(OUT char version[32]);

int getDetection2ModelVersion(OUT char version[32]);

int getSegmentationModelVersion(OUT char version[32]);

int getWeatherModelVersion(OUT char version[32]);

int getClealinessModelVersion(OUT char version[32]);

#endif