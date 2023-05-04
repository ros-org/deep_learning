#ifndef __CLASSIFY__
#define __CLASSIFY__

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
#include "enn_common.hpp"

using namespace std;
using namespace cv;

class Classify
{
public:
    Classify(){};
    ~Classify();

    int init(CLA_CFG_t *pcfg);
    int run(unsigned char *p_feed_data, const std::string& imgFmt, int& output);
    void post_process(vector<float *> &res);

private:
    
    int m_feed_height;
    int m_feed_width;
    int m_cls_num;
    unsigned char *m_model;
    rknn_input_output_num m_io_num;
    rknn_context m_ctx;

    rknn_tensor_attr *m_p_input_attrs;
    rknn_tensor_attr *m_p_output_attrs;
    rknn_output m_outputs; 
};

#endif


