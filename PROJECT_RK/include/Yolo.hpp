#ifndef __YOLO_HPP__
#define __YOLO_HPP__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include "detection_output.h"
#include "common.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "rknn_api.h"
#include "Configer.hpp"

using namespace std;
using namespace cv;

class Yolo
{
public:
    Yolo(){};
    ~Yolo();

    int init(YOLO_CFG_t *pcfg);
    int run(unsigned char *p_feed_data, const std::string& imgFmt, vector<float *> &output);
    void show_res(Mat &img,vector<float *> &res);

private:
    
    int m_feed_height;
    int m_feed_width;
    int m_cls_num;
    unsigned char *m_model;
    rknn_input_output_num m_io_num;
    rknn_context m_ctx;

    rknn_tensor_attr *m_p_input_attrs;
    rknn_tensor_attr *m_p_output_attrs;
    DetectionOutput mDetectionOutput;
    rknn_output m_outputs; 
    float *m_buffer_out;
};

#endif