#ifndef __ENN_COMMON_HPP__
#define __ENN_COMMON_HPP__

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

class ENN
{
public:
    static unsigned char *load_model(char *filename, int *model_size);
    static void printRKNNTensor(rknn_tensor_attr *attr);
    static int get_model_io_attrs(rknn_context &ctx,
                                  rknn_input_output_num &io_num,
                                  rknn_tensor_attr **input_attrs,
                                  rknn_tensor_attr **output_attrs);
};


#endif