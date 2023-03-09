#ifndef __SEGMENTATION_HPP__
#define __SEGMENTATION_HPP__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include "common.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "rknn_api.h"
#include "Configer.hpp"

using namespace std;
using namespace cv;

class Segmentation
{
public:
    Segmentation(){};
    ~Segmentation();

    int init(SEG_CFG_t *pcfg);
    int run(unsigned char *p_feed_data,Mat &im_seg);
    void draw_seg(Mat &img, Mat &im_seg);
    int post_process(Mat &im_seg);

private:
    bool m_binit;
    unsigned char *m_model;
    SEG_CFG_t m_Cfg;
    rknn_input_output_num m_io_num;
    rknn_context m_ctx;

    rknn_tensor_attr *m_p_input_attrs;
    rknn_tensor_attr *m_p_output_attrs;
    rknn_output m_outputs; 

    // unsigned char *m_p_seg_data;
    Mat m_seg_out;
};

#endif