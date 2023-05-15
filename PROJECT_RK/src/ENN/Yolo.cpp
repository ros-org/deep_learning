#include "Yolo.hpp"
#include "enn_common.hpp"


int Yolo::init(YOLO_CFG_t *pcfg)
{
    CHECK_EXPR(pcfg == NULL,-1);

    mDetectionOutput.init(pcfg->feed_w,pcfg->feed_h,pcfg->cls_num);
    m_feed_height = pcfg->feed_h;
    m_feed_width = pcfg->feed_w;
    m_cls_num = pcfg->cls_num;    // Load RKNN Model

    int model_len = 0;
    bool exists = check_file_exists(pcfg->model);
    CHECK_EXPR(exists == false,-1);

    m_model = ENN::load_model(pcfg->model, &model_len);
    
    int ret = rknn_init(&m_ctx, m_model, model_len, 0);
    CHECK_EXPR(ret < 0,-1);

#if 0
    // Get Model Input Output Info
    ret = rknn_query(m_ctx, RKNN_QUERY_IN_OUT_NUM, &m_io_num, sizeof(m_io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", m_io_num.n_input, m_io_num.n_output);

    printf("input tensors:\n");
    // rknn_tensor_attr input_attrs[m_io_num.n_input];

    m_p_input_attrs = (rknn_tensor_attr *)malloc(sizeof(rknn_tensor_attr) * m_io_num.n_input);
    CHECK_EXPR(m_p_input_attrs == NULL,-1);

    memset(m_p_input_attrs, 0, sizeof(rknn_tensor_attr) * m_io_num.n_input);
    for (int i = 0; i < m_io_num.n_input; i++) {
        m_p_input_attrs[i].index = i;
        ret = rknn_query(m_ctx, RKNN_QUERY_INPUT_ATTR, &(m_p_input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        ENN::printRKNNTensor(&(m_p_input_attrs[i]));
    }

    printf("output tensors:\n");
    // rknn_tensor_attr output_attrs[m_io_num.n_output];

    m_p_output_attrs = (rknn_tensor_attr *)malloc(sizeof(rknn_tensor_attr) * m_io_num.n_output);
    CHECK_EXPR(m_p_output_attrs == NULL,-1);

    memset(m_p_output_attrs, 0, sizeof(rknn_tensor_attr)* m_io_num.n_output);
    for (int i = 0; i < m_io_num.n_output; i++) {
        m_p_output_attrs[i].index = i;
        ret = rknn_query(m_ctx, RKNN_QUERY_OUTPUT_ATTR, &(m_p_output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        ENN::printRKNNTensor(&(m_p_output_attrs[i]));
    }
#else
    ret = ENN::get_model_io_attrs(m_ctx,m_io_num,&m_p_input_attrs,&m_p_output_attrs);
    CHECK_EXPR(ret != 0,-1);
#endif

    /**********config output*************/
	m_outputs.want_float = 0;
	m_outputs.index = 0;
	m_outputs.is_prealloc = 1;

    // rknn_tensor_attr input_attrs[m_io_num.n_input];

    int dim_sub = m_cls_num + 5;
	int out_size = m_p_output_attrs[0].n_elems;
    int dim = out_size / dim_sub;

	// float *pOutput = NULL;
	m_outputs.buf = (unsigned char *)malloc(sizeof(unsigned char) * out_size);
	m_outputs.size = out_size * sizeof(unsigned char);
	m_buffer_out = (float *)malloc(sizeof(float) * out_size);
	memset(m_buffer_out,'\0',sizeof(float) * out_size);

    return 0;
}

int Yolo::run(unsigned char *p_feed_data, const std::string& imgFmt, vector<float *> &output)
{
    // Set Input Data
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = m_feed_height * m_feed_width * 3;
    inputs[0].buf = p_feed_data;

    if("CHW" == imgFmt)
    {
        inputs[0].fmt = RKNN_TENSOR_NCHW;   //设置预测时tensor的格式NCHW，推理时需要先将图片由hwc转chw
    }
    else if("HWC" == imgFmt)
    {
        inputs[0].fmt = RKNN_TENSOR_NHWC;   //设置预测时tensor的格式NHWC，这样就不需要做hwc转chw操作了
    }
    else
    {
        std::cout<<"当前推理图像输入格式错误,仅支持 HWC和CHW 两种..."<<std::endl;
    }

    int ret = rknn_inputs_set(m_ctx, m_io_num.n_input, inputs);
    CHECK_EXPR(ret < 0,-1);
	
    int dim_sub = m_cls_num + 5;
	int out_size = m_p_output_attrs[0].n_elems;
    int dim = out_size / dim_sub;
    //ptrDecode和m_buffer_out指向同一块内存，下面往ptrDecode这块内存写数据，然后将m_buffer_out送入mDetectionOutput.run函数；
	float (*ptrDecode)[dim_sub] = (float (*)[dim_sub])m_buffer_out;
    //将m_outputs.buf(是void*)这段内存强转成一个数组
	unsigned char (*ptr)[dim_sub] = (unsigned char (*)[dim_sub])m_outputs.buf;
    memset((unsigned char *)m_outputs.buf,0,out_size);

	unsigned char threshold = 100;
	vector<float *>  res;
    float div_val = 1/255.;
    

    Timer timer;
	for (int i = 0; i < 1; i++) 
    {
		// cal_time_start();
        timer.start();
		ret = rknn_run(m_ctx, nullptr);
		CHECK_EXPR(ret < 0,-1);
		// cal_time_end("rknn_run");

        
		cal_time_start();
		ret = rknn_outputs_get(m_ctx, 1, &m_outputs, NULL);
		CHECK_EXPR(ret < 0,-1);

		// Post Process
        memset((unsigned char *)ptrDecode,0,out_size*sizeof(float));
		for (int i = 0;i < dim;i++) 
        {
			if (ptr[i][4] > threshold) 
            {
				for (int j =  0;j < dim_sub;j++) 
                {
					ptrDecode[i][j] = ptr[i][j] * div_val;
				}
			}
		}

		res = mDetectionOutput.run((float *)m_buffer_out);	
    }
    output = res;
    return 0;
}

void Yolo::show_res(Mat &img,vector<float *> &res)
{    
    double rGrayValue[2]={120, 255};
    double gGrayValue[2]={0, 255};
    double bGrayValue[2]={0, 255};
    for (int i = 0; i < res.size();++i) 
    {
        if(0 == res[i][5])
        {
            cv::rectangle(img,Point(res[i][0],res[i][1]),Point(res[i][2],res[i][3]),cv::Scalar(0,0,255),1,1,0);
        }
        else if(1 == res[i][5])
        {
            cv::rectangle(img,Point(res[i][0],res[i][1]),Point(res[i][2],res[i][3]),cv::Scalar(0,255,0),1,1,0);
        }
        else if(2 == res[i][5])
        {
            cv::rectangle(img,Point(res[i][0],res[i][1]),Point(res[i][2],res[i][3]),cv::Scalar(255,0,0),1,1,0);
        }
        else if(3 == res[i][5])
        {
            cv::rectangle(img,Point(res[i][0],res[i][1]),Point(res[i][2],res[i][3]),cv::Scalar(255,255,0),1,1,0);
        }
        else if(4 == res[i][5])
        {
            cv::rectangle(img,Point(res[i][0],res[i][1]),Point(res[i][2],res[i][3]),cv::Scalar(255,0,255),1,1,0);
        }
        else
        {
            cv::rectangle(img,Point(res[i][0],res[i][1]),Point(res[i][2],res[i][3]),cv::Scalar(0,255,255),1,1,0);
        } 
    }
    
    mDetectionOutput.print_bboxes(res);
}


Yolo::~Yolo()
{
    if(m_ctx >= 0) {
        rknn_destroy(m_ctx);
    }
    if(m_model) {
        free(m_model);
    }

}
