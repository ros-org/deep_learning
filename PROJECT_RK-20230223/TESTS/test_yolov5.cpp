/*-------------------------------------------
                Includes
-------------------------------------------*/
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
#include "Yolo.hpp"
//#include "public.h"

using namespace std;
using namespace cv;



/*-------------------------------------------
                  Functions
-------------------------------------------*/
#if 1
static void printRKNNTensor(rknn_tensor_attr *attr) {
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n", 
            attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0], 
            attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if(fp) {
        fclose(fp);
    }
    return model;
}
/*
2.11444 17.4693 511.517 162.358 0.94902 0                                                                                                
476.149 33.2115 512.055 63.4473 0.909804 1   
*/

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main()
{
    const int img_width = 512;
    const int img_height = 288;
    const int img_channels = 3;
	int num_cls = 4;
	DetectionOutput *pDetectionOutput = new DetectionOutput();
	pDetectionOutput->init(img_width,img_height,num_cls);
	
    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;
	//Timer *pTimer = new Timer();
	
    const char *model_path = "/userdata/models/yolov5.rknn";
    const char *img_path = "/userdata/data/test.jpg";

    // Load image
    cv::Mat orig_img = cv::imread(img_path);
    
    cv::Mat img = orig_img.clone();
    if(!orig_img.data) 
    {
        printf("cv::imread %s fail!\n", img_path);
        return -1;
    }
    cal_time_start();
    if(orig_img.cols != img_width || orig_img.rows != img_height) {
        printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, img_width, img_height);
        cv::resize(orig_img, img, cv::Size(img_width, img_height), (0, 0), (0, 0), cv::INTER_LINEAR);
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cal_time_end("==================resize cvtColor");

    // Load RKNN Model
    model = load_model(model_path, &model_len);
    ret = rknn_init(&ctx, model, model_len, 0);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(output_attrs[i]));
    }

    // Set Input Data
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = img.cols*img.rows*img.channels();
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = img.data;

    cal_time_start();
    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if(ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }
    cal_time_end("==================rknn_inputs_set");
	
	rknn_output outputs;
	outputs.want_float = 0;
	outputs.index = 0;
	outputs.is_prealloc = 1;

    int dim_sub = 7;
	int out_size = output_attrs[0].n_elems;
	float *pOutput = NULL;
	outputs.buf = (unsigned char *)malloc(sizeof(unsigned char) * out_size);
	outputs.size = out_size * sizeof(unsigned char);
	pOutput = (float *)malloc(sizeof(float) * out_size);
	memset(pOutput,'\0',sizeof(float) * out_size);
	int dim = out_size / dim_sub;
	unsigned char (*ptr)[dim_sub] = (unsigned char (*)[dim_sub])outputs.buf;
	float (*ptrDecode)[dim_sub] = (float (*)[dim_sub])pOutput;
	
	unsigned char threshold = 100;
	vector<float *>  res;
    float div_val = 1/255.;
	for (int i = 0;i < 1;i++) {
		cal_time_start();
		ret = rknn_run(ctx, nullptr);
		if(ret < 0) {
			printf("rknn_run fail! ret=%d\n", ret);
			return -1;
		}
		// cal_time_end("rknn_run");

		// cal_time_start();
		ret = rknn_outputs_get(ctx, 1, &outputs, NULL);
		if(ret < 0) {
			printf("rknn_outputs_get fail! ret=%d\n", ret);
			return -1;
		}
		// cal_time_end("rknn_outputs_get");

		// Post Process
		// cal_time_start();
		for (int i = 0;i < dim;i++) {
			if (ptr[i][4] > threshold) {
				for (int j =  0;j < dim_sub;j++) {
					ptrDecode[i][j] = ptr[i][j] * div_val;
				}
			}
		}
		// cal_time_end("uchar to float");

		// cal_time_start();
		res = pDetectionOutput->run((float *)pOutput);	
		// cal_time_end("pDetectionOutput->run");
        cal_time_end("ALL time");
    }
        

	for (int i = 0;i < res.size();i++) {
		cv::rectangle(img,Point(res[i][0],res[i][1]),Point(res[i][2],res[i][3]),cv::Scalar(0,0,255),1,1,0);
	}
    cv::imwrite("/userdata/data/yolov5_test_result.jpg",img);
	pDetectionOutput->print_bboxes(res);

    // Release
    if(ctx >= 0) {
        rknn_destroy(ctx);
    }
    if(model) {
        free(model);
    }
    return 0;
}
#endif